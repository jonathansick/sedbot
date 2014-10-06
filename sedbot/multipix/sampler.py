#!/usr/bin/env python
# encoding: utf-8
"""
Sample for Multi Pixel Gibbs MCMC.
"""

from collections import OrderedDict
# import pprint
import json
import numpy as np

MODEL = None


class MultiPixelGibbsBgSampler(object):
    """MH-in-Gibbs sampler for three level hierarchical models of
    pixel parameters, global parameters, and a linearly sampled background
    parameter.

    Parameters
    ----------
    model_initializer : function
        Function that, when called, creates a model instance.
        This is done to enable multi-processing.
    n_cpu : int
        Number of CPUs to run on.
    """
    def __init__(self, model_initializer,
                 n_cpu=1):
        super(MultiPixelGibbsBgSampler, self).__init__()
        self._n_cpu = n_cpu
        self._model_initializer = model_initializer

        # Set up a local model instance
        self._model = self._model_initializer()

        # Set up a pixel compute pool
        self._map = self._init_pixel_compute_pool()

    def _init_pixel_compute_pool(self):
        """Initialize an ipython.parallel pool for processing pixels."""
        # Put the model in the global namespace for parallel processing
        global MODEL
        MODEL = self._model_initializer()

        # Setup up the pool
        if self._n_cpu == 1:
            map_fcn = map  # single processing
        else:
            map_fcn = None  # TODO
        return map_fcn

    def sample(self, n_iter,
               theta_prop=None,
               phi_prop=None,
               theta0=None,
               phi0=None,
               B0=None):
        """Sample for `n_iter` Gibbs steps.

        Parameters
        ----------
        n_iter : int
            Number of Gibbs steps (number of full iterations over all
            parameters)
        theta_prop : ndarray
            Standard deviations of Gaussian proposal distribtions for
            theta (pixel) parameters
        phi_prop : ndarray
            Standard deviations of Gaussian proposal distributions for
            phi (global) parameters. Leave as ``None`` to use the model's
            defaults
        """
        global MODEL

        # initialize memory
        i0 = self._init_chains(n_iter, theta0, phi0, B0)

        for i in xrange(i0, i0 + n_iter):
            print "Iteration {0:d}".format(i)
            # Sample for all pixels
            args = []
            for ipix in xrange(self._model.n_pix):
                _post0 = self.pix_lnpost[i - 1, ipix]
                _theta0 = self.theta[i - 1, ipix, :]
                _phi0 = self.phi[i - 1, :]
                _B0 = self.B[i - 1, :]
                args.append((ipix, _post0, _theta0, _phi0, _B0, theta_prop))
            results = self._map(pixel_mh_sampler, args)
            for ipix, result in enumerate(results):
                lnpost, theta, blob, theta_accept = result
                self.pix_lnpost[i, ipix] = lnpost
                self.theta[i, ipix, :] = theta
                if blob is not None:
                    for k, v in blob.iteritems():
                        self.blobs[i][k][ipix] = v
                    else:
                        # repeat previous blob values
                        for k in self.blobs.dtype.fields:
                            self.blobs[i][k][ipix] = self.blobs[i - 1][k][ipix]
                print theta_accept
                self._theta_n_accept[ipix, :] += theta_accept

            # TODO Update the background
            self.B[i, :] = self.B[i - 1, :]

            # TODO Update the global posterior probability with these pixel
            # values and new background
            pass

            # TODO Sample the global parameters
            self.phi[i, :] = self.phi[i - 1, :]

    def _init_chains(self, n_iter, theta0, phi0, B0):
        """Initialize memory

        Note this could be adapted to extend existing chains.
        """
        # Make parameter chains
        self.theta = np.empty((n_iter + 1,
                               self._model.n_pix,
                               self._model.n_theta),
                              dtype=np.float)
        self.theta.fill(np.nan)
        self.theta[0, :, :] = theta0

        self.phi = np.empty((n_iter + 1,
                             self._model.n_phi),
                            dtype=np.float)
        self.phi.fill(np.nan)
        self.phi[0, :] = phi0

        self.B = np.empty((n_iter + 1,
                           self._model.n_bands),
                          dtype=np.float)
        self.B.fill(np.nan)
        self.B[0, :] = B0

        self.lnpost = np.empty(n_iter + 1, dtype=np.float)
        self.lnpost.fill(np.nan)

        self.pix_lnpost = np.empty((n_iter + 1, self._model.n_pix),
                                   dtype=np.float)
        self.pix_lnpost.fill(np.nan)

        # Make a structured array for the complex blob data
        self.blobs = np.empty(n_iter + 1, dtype=self._model.blob_dtype)
        self.blobs.fill(np.nan)

        # Initialize the starting point of the chains
        print "theta0", theta0
        print "phi0", phi0
        print "B0", B0
        global_lnp, pixel_lnp, pixel_blobs \
            = self._model.sample_global(theta0, phi0, B0)
        self.lnpost[0] = global_lnp
        for ipix, lnp in enumerate(pixel_lnp):
            self.pix_lnpost[0, ipix] = lnp
        for ipix, blobs in enumerate(pixel_blobs):
            for k, v in blobs.iteritems():
                self.blobs[0][k][ipix] = v

        # Track acceptance of individual parameters for each pixel
        self._theta_n_accept = np.zeros((self._model.n_pix,
                                         self._model.n_theta),
                                        dtype=np.int)
        self._phi_n_accept = np.zeros(self._model.n_phi, dtype=np.int)

        return 1

    @property
    def median_theta_faccept(self):
        """Median acceptance fraction across all pixels for each parameter"""
        n_samples = float(self.theta.shape[0])
        faccept = np.median(self._theta_n_accept / n_samples, axis=0)
        return faccept

    @property
    def phi_faccept(self):
        """Acceptance fraction of phi parameters."""
        n_samples = float(self.theta.shape[0])
        return self._phi_n_accept / n_samples

    def print_faccept(self):
        """Printed report of the acceptance fraction statistics"""
        faccept = self.median_theta_faccept
        # FIXME Model should have public accessort to param names
        d = OrderedDict(zip(self._model._theta_params, faccept))
        # pp = pprint.PrettyPrinter(indent=4)
        # pp.pprint(d)
        print("theta")
        print(json.dumps(d, indent=4))
        print("phi")
        d = OrderedDict(zip(self._model._phi_params, self.phi_faccept))
        print(json.dumps(d, indent=4))


def pixel_mh_sampler(args):
    """Execute MH-in-Gibbs sampling for a single pixel.

    This function is design to be called by a ``map`` function, hence all
    arguments are wrapped in a single tuple, ``args``.

    Parameters
    ----------
    ipix : int
        Pixel index
    post0 : float
        Ln Posterior probability *for the pixel* given initial values of
        theta0 and phi0.
    theta0 : ndarray
        Initial values of theta parameters.
    phi0 : ndarray
        Initial values of the phi parameters.
    B0 : ndarray
        Initial values of the B, background, parameters.
    theta_prop : ndarray
        Array of Gaussian proposal standard deviations.

    Returns
    -------
    post : float
        Final posterior probability *for this pixel*.
    theta : ndarray
        Theta parameters from this Gibbs step.
    blob : dict
        Stellar population metadata for this pixel given the updated `theta`
        parameters.
    n_accept : ndarray
        Array of length ``n_theta`` whose values are 1 for parameters
        that were updated, and zeros for values that remain constant from
        the previous iteration.
    """
    global MODEL
    ipix, post0, theta0, phi0, B0, theta_prop = args
    # print "post0", post0
    # print "theta0", theta0
    # print "phi0", phi0
    # print "B0", B0
    theta = np.copy(theta0)
    post = post0
    # MH on each parameter
    blob = None
    n_accept = np.zeros(theta.shape[0], dtype=np.int)
    for i in xrange(theta0.shape[0]):
        # Gaussian proposal for parameter i, only
        theta_new = theta.copy()
        theta_new[i] += theta_prop[i] * np.random.randn()
        lnpost_new, blob_new = MODEL.sample_pixel(theta_new, phi0, B0, ipix)
        ln_r = lnpost_new - post
        # print ipix, i, ln_r
        reject = True
        if ~np.isfinite(ln_r):
            pass
        elif ln_r >= 0.:
            reject = False
        else:
            x = np.random.rand(0., 1.)
            if x < np.exp(ln_r):
                reject = False
        if not reject:
            # adopt new point
            theta = theta_new
            blob = blob_new
            post = lnpost_new
            n_accept[i] = 1
    return post, theta, blob, n_accept
