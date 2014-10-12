#!/usr/bin/env python
# encoding: utf-8
"""
Sample for Multi Pixel Gibbs MCMC.
"""

from collections import OrderedDict
# import pprint
import json
import numpy as np

import fsps

from astropy.utils.console import ProgressBar
from astropy.table import Table, hstack, Column

from sedbot.photconv import mjy_to_luminosity
from .chain import MultiPixelChain


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
               B0=None,
               chain=None):
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
            defaults.
        theta0 : ndarray
            Initial values for the theta parameters, a ``(n_pix, n_theta)``
            array.
        phi0 : ndarray
            Initial values for the phi parameters, a ``(n_phi)`` array.
        B0 : ndarray
            Initial values for the background, a ``(n_band)`` array.
            Background is in units of flux per arcsec^2.
        chain : :class:`sedbot.chain.MultiPixelChain`
            A previously-built chain, whose last sample will be used
            as the starting points for this sampling (in liue of setting
            `theta0`, `phi0` and `B0` manually).
        """
        global MODEL

        if chain is not None:
            theta0 = np.empty((self._model.n_pix, self._model.n_theta),
                              dtype=np.float)
            for i, n in enumerate(self._model.theta_params):
                theta0[:, i] = chain[n][-1]

            phi0 = np.empty(self._model.n_phi, dtype=np.float)
            for i, n in enumerate(self._model.phi_params):
                phi0[i] = chain[n][-1]

            B0 = np.empty(self._model.n_bands, dtype=np.float)
            for i, n in enumerate(self._model.observed_bands):
                name = "B_{0}".format(n)
                B0[i] = chain[name][-1]
        else:
            assert theta0 is not None
            assert phi0 is not None
            assert B0 is not None

        # initialize memory
        i0 = self._init_chains(n_iter, theta0, phi0, B0)

        with ProgressBar(n_iter) as bar:
            for i in xrange(i0, i0 + n_iter):
                # Sample for all pixels
                args = []
                for ipix in xrange(self._model.n_pix):
                    _post0 = self.pix_lnpost[i - 1, ipix]
                    _theta0 = self.theta[i - 1, ipix, :]
                    _phi0 = self.phi[i - 1, :]
                    _B0 = self.B[i - 1, :]
                    args.append((ipix, _post0, _theta0, _phi0, _B0,
                                 theta_prop))
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
                    self._theta_n_accept[ipix, :] += theta_accept

                # Update the background
                model_seds = self.blobs[i]['model_sed']
                B_new, global_lnp, pixel_lnp, pixel_blobs \
                    = self._model.update_background(self.theta[i, :, :],
                                                    self.phi[i - 1, :],
                                                    model_seds)
                self.B[i, :] = B_new

                # Sample the global parameters
                phi_new, global_lnp, pixel_lnps, pixel_blobs, n_accept \
                    = self._global_mh(i,
                                      global_lnp,
                                      self.phi[i - 1, :],
                                      self.theta[i, :, :],
                                      self.B[i, :],
                                      phi_prop)
                # fill in new values to chain.
                self.phi[i, :] = phi_new
                self.lnpost[i] = global_lnp
                self._phi_n_accept += n_accept
                # Update ln post and blob data for all pixels too
                # it was done for the pixel-only step, but these values have
                # changed given the new background, etc.
                if pixel_lnps is not None:
                    for ipix, lnp in enumerate(pixel_lnps):
                        self.pix_lnpost[i, ipix] = lnp
                if pixel_blobs is not None:
                    for ipix, blobs in enumerate(pixel_blobs):
                        for k, v in blobs.iteritems():
                            self.blobs[i][k][ipix] = v

                bar.update()

    def _global_mh(self, i, lnpost0, phi, theta, B, phi_prop):
        """Perform MH-in-Gibbs at the global level."""
        phi = np.copy(phi)
        lnpost = lnpost0
        lnpost_pixel = None
        blobs = None
        n_accept = np.zeros(phi.shape[0], dtype=np.int)
        for j in xrange(phi.shape[0]):
            # Gaussian proposal for parameter j, only
            phi_new = np.copy(phi)
            phi_new[j] += phi_prop[j] * np.random.randn()
            global_lnp, pixel_lnp, pixel_blobs \
                = self._model.sample_global(theta,
                                            phi_new,
                                            B)
            ln_r = global_lnp - lnpost0
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
                phi = phi_new
                lnpost = global_lnp
                lnpost_pixel = pixel_lnp
                blobs = pixel_blobs
                n_accept[j] = 1
        return phi, lnpost, lnpost_pixel, blobs, n_accept

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
        d = OrderedDict(zip(self._model.theta_params, faccept))
        # pp = pprint.PrettyPrinter(indent=4)
        # pp.pprint(d)
        print("theta")
        print(json.dumps(d, indent=4))
        print("phi")
        d = OrderedDict(zip(self._model.phi_params, self.phi_faccept))
        print(json.dumps(d, indent=4))

    @property
    def table(self):
        """An :class:`astropy.table.Table` with the chain."""
        msuns = np.array([fsps.get_filter(n).msun_ab
                          for n in self._model.computed_bands])
        theta_f_accept = json.dumps(dict(zip(self._model.theta_params,
                                             self.median_theta_faccept)))
        phi_f_accept = json.dumps(dict(zip(self._model.phi_params,
                                           self.phi_faccept)))
        meta = OrderedDict((
            ('theta_f_accept', theta_f_accept),
            ('phi_f_accept', phi_f_accept),
            ('obs_bands', self._model.observed_bands),
            ('compute_bands', self._model.computed_bands),
            ('msun_ab', msuns),
            ('band_indices', self._model.band_indices),
            ('theta_params', self._model.theta_params),
            ('phi_params', self._model.phi_params),
            ('sed', self._model._seds),
            ('sed_err', self._model._errs),
            ('pixels', self._model.pixel_metadata),
            ('area', self._model._areas)))
        # Make tables for individual chains; stack later
        # FIXME should axis order be changed for theta throughout the sampler?
        # or I can just continue to swarp aces here
        theta_table = Table(np.swapaxes(self.theta, 1, 2),
                            names=self._model.theta_params,
                            meta=meta)
        phi_table = Table(self.phi, names=self._model.phi_params)
        background_names = ["B_{0}".format(n)
                            for n in self._model.observed_bands]
        B_table = Table(self.B, names=background_names)
        blob_table = Table(self.blobs)
        tbl = MultiPixelChain(hstack((theta_table,
                                      phi_table,
                                      B_table,
                                      blob_table)))

        # Add M/L computations for each computed band.
        for i, (band_name, msun) in enumerate(zip(self._model.computed_bands,
                                                  msuns)):
            logLsol = mjy_to_luminosity(tbl['model_sed'][:, :, i],
                                        msun,
                                        np.atleast_2d(tbl['d']).T)
            ml = tbl['logMstar'] - logLsol
            colname = "logML_{0}".format(band_name)
            tbl.add_column(Column(name=colname, data=ml))

        return tbl


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
