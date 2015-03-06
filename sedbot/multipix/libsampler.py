#!/usr/bin/env python
# encoding: utf-8
"""
Sampler for Multi Pixel Gibbs modelling, using a library-based SP
estimates rather than MCMC.
"""

import numpy as np

from sedbot.library.marginalizer import LibraryEstimator


class MultiPixelLibraryGibbsBgSampler(object):
    """Library-in-Gibbs sampler for two-level hierarchical models of pixel
    parameters and a linearly sampled background parameter.

    Parameters
    ----------
    model : :class:`sedbot.multipix.libmodels.LibraryModel`
        A LibraryModel instance that encapsulates the library SEDs and the
        observed SEDs.
    """
    def __init__(self, model):
        super(MultiPixelLibraryGibbsBgSampler, self).__init__()
        self._model = model

    def estimate_theta(self, B):
        """Estimate the stellar population parameters.

        B : ndarray
            Initial values for the background, a ``(n_band)`` array.
            Background is in units of flux per arcsec^2.
        """
        n_pixels = self._model.n_pix
        n_dim = self._model.n_theta
        theta = np.empty((n_pixels, n_dim), dtype=np.float)
        blob = np.empty((n_pixels, len(self._model.meta_params)),
                        dtype=np.float)
        ml = np.empty((n_pixels, len(self._model.library_bands)),
                      dtype=np.float)
        sed = np.empty((n_pixels, len(self._model.library_bands)),
                       dtype=np.float)

        # Compute SP parameters for each SED, with background subtracted
        for i in xrange(n_pixels):
            estimator = LibraryEstimator(
                self.model._seds[i, :] - B, self.model._errs[i, :],
                self._obs_bands, self.model.d,
                self.model.library_file, self.model.library_group,
                ncpu=1)
            # marginal estimate of model parameters
            for j, name in enumerate(self._model.theta_params):
                theta[i, j] = estimator.estimate(name, p=(0.5,))[0]
            # marginal estimate of metadata parameters
            for j, name in enumerate(self._model.meta_params):
                blob[i, j] = estimator.estimate_meta(name, p=(0.5,))[0]
            # marginal estimate of M/L
            for j, band in enumerate(self._model.library_bands):
                ml[i, j] = estimator.estimate_ml(band, p=(0.5,))[0]
            # marginal estimate of SED
            for j, band in enumerate(self._model.library_bands):
                sed[i, j] = estimator.estimate_flux(band, p=(0.5,))[0]

        return theta, blob, ml, sed

    def sample(self, n_iter,
               theta0=None,
               B0=None,
               chain=None):
        """Sample for `n_iter` Gibbs steps.

        Parameters
        ----------
        n_iter : int
            Number of Gibbs steps (number of full iterations over all
            parameters)
        theta0 : ndarray
            Initial values for the theta parameters, a ``(n_pix, n_theta)``
            array.
        B0 : ndarray
            Initial values for the background, a ``(n_band)`` array.
            Background is in units of flux per arcsec^2.
        chain : :class:`sedbot.chain.MultiPixelChain`
            A previously-built chain, whose last sample will be used
            as the starting points for this sampling (in liue of setting
            `theta0`, `phi0` and `B0` manually).
        """
        if chain is not None:
            theta0 = np.empty((self._model.n_pix, self._model.n_theta),
                              dtype=np.float)
            for i, n in enumerate(self._model.theta_params):
                theta0[:, i] = chain[n][-1]

            B0 = np.empty(self._model.n_bands, dtype=np.float)
            for i, (n, instr) in enumerate(zip(self._model.observed_bands,
                                               self._model.instruments)):
                name = "B__{0}__{1}".format(instr, n)
                B0[i] = chain[name][-1]
        else:
            assert theta0 is not None
            assert B0 is not None

        # Initialize chains
        i0 = self._init_chains(n_iter, theta0, B0)

        with ProgressBar(n_iter) as bar:
            for i in xrange(i0, i0 + n_iter):
                # Sample stellar populations
                # TODO estimate_theta should return a blob
                theta_i, blob_i = self.estimate_theta(self.B[i - 1, :])

                # TODO insert theta and blob into the chain
                pass

                # TODO background



    def _init_chains(self, n_iter, theta0, B0):
        """Initialize memory for the chain."""
        self.theta = np.empty((n_iter + 1,
                               self._model.n_pix,
                               self._model.n_theta),
                              dtype=np.float)
        self.theta.fill(np.nan)
        self.theta[0, :, :] = theta0

        self.B = np.empty((n_iter + 1,
                           self._model.n_bands),
                          dtype=np.float)
        self.B.fill(np.nan)
        self.B[0, :] = B0

        # blobs are for SP computed values, but are not input parameters
        self.blobs = np.empty(n_iter + 1, dtype=self._model.blob_dtype)
        self.blobs.fill(np.nan)

        return 1
