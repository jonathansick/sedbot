#!/usr/bin/env python
# encoding: utf-8
"""
Sample for Multi Pixel Gibbs MCMC.
"""

import numpy as np


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
        self._m = self._init_pixel_compute_pool()

    def _init_pixel_compute_pool(self):
        """Initialize an ipython.parallel pool for processing pixels."""
        # Put the model in the global namespace for parallel processing
        global MODEL
        MODEL = self._model_initialier()

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
            # Sample for all pixels
            args = []
            for ipix in xrange(self._model.n_pix):
                _post0 = self.lnpost[i0 - 1]
                _theta0 = self.theta[i0 - 1, ipix, :]
                _phi0 = self.phi[i0 - 1, :]
                _B0 = self.B[i0 - 1, :]
                theta_prop = None
                args.append((ipix, _post0, _theta0, _phi0, _B0, theta_prop))
            results = self._M(pixel_mh_sampler, args)

    def _init_chains(self, n_iter, theta0, phi0, B0):
        """Initialize memory

        Note this could be adapted to extend existing chains.
        """
        # Make parameter chains
        self.theta = np.empty((n_iter,
                               self._model.n_pix,
                               self._model._n_theta),
                              dtype=np.float)
        self.theta.fill(np.nan)
        self.theta[0, :, :] = theta0

        self.phi = np.empty((n_iter,
                             self._model._n_phi),
                            dtype=np.float)
        self.phi.fill(np.nan)
        self.phi[0, :] = phi0

        self.B = np.empty((n_iter,
                           self._model._n_bands),
                          dtype=np.float)
        self.B.fill(np.nan)
        self.B[0, :] = B0

        self.lnpost = np.empty(n_iter, dtype=np.float)
        self.lnpost.fill(np.nan)
        # FIXME compute inital global posterior

        self.pix_lnpost = np.empty((n_iter, self._model.n_pix),
                                   dtype=np.float)
        self.pix_lnpost.fill(np.nan)
        # FIXME compute initial pixel lnposts

        # Make a structured array for the complex blob data
        self.blobs = np.empty(n_iter, dtype=self._model.blob_dtype)
        self.blobs.fill(np.nan)
        # FIXME compute initial blob data

        # Initialize chains for individiual pixels
        for ipix in xrange(self._model.n_pix):
            lnprob, blob_dict = self._model.sample_pixel(theta0[ipix, :],
                                                         phi0, ipix)

        return 1


def pixel_mh_sampler(args):
    """Execute MH-in-Gibbs sampling for a single pixel.

    This function is design to be called by a ``map`` function, hence all
    arguments are wrapped in a single tuple, ``args``.

    Paramters
    ---------
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
    blob : list
        Stellar population metadata for this pixel given the updated `theta`
        parameters.
    """
    global MODEL
    pix, post0, theta0, phi0, B0, theta_prop = args
    theta = np.copy(theta0)
    for i in xrange(theta0.shape[0]):
        # MH on each parameter
        pass
    return post, theta, blob
