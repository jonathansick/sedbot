#!/usr/bin/env python
# encoding: utf-8
"""
Sampler for Multi Pixel Gibbs modelling, using a library-based SP
estimates rather than MCMC.
"""

import numpy as np

from astropy.utils.console import ProgressBar

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

    def sample(self, n_iter,
               B0=None,
               chain=None):
        """Sample for `n_iter` Gibbs steps.

        Parameters
        ----------
        n_iter : int
            Number of Gibbs steps (number of full iterations over all
            parameters)
        B0 : ndarray
            Initial values for the background, a ``(n_band)`` array.
            Background is in units of flux per arcsec^2.
        chain : :class:`sedbot.chain.MultiPixelChain`
            A previously-built chain, whose last sample will be used
            as the starting points for this sampling (in liue of setting
            `theta0`, `phi0` and `B0` manually).
        """
        if chain is not None:
            B0 = np.empty(self._model.n_bands, dtype=np.float)
            for i, (n, instr) in enumerate(zip(self._model.observed_bands,
                                               self._model.instruments)):
                name = "B__{0}__{1}".format(instr, n)
                B0[i] = chain[name][-1]
        else:
            assert B0 is not None

        # Initialize chains
        self._init_chains(n_iter)

        with ProgressBar(n_iter) as bar:
            for i in xrange(n_iter):
                # Sample stellar populations
                self._estimate_theta(i, B=B0)
                B0 = None

                # TODO background

                bar.update()

    def _init_chains(self, n_iter):
        """Initialize memory for the chain."""
        # FSPS SP Parameter estimates
        self.theta_chain = np.empty((n_iter + 1,
                                     self._model.n_pix,
                                     self._model.n_theta),
                                    dtype=np.float)
        self.theta_chain.fill(np.nan)

        # Background estimates
        self.B_chain = np.empty((n_iter + 1,
                                 self._model.n_bands),
                                dtype=np.float)
        self.B_chain.fill(np.nan)

        # FSPS computed metadata values
        self.blob_chain = np.empty((n_iter + 1,
                                    self._model.n_pix,
                                    len(self._model.meta_params)),
                                   dtype=np.float)

        # M/L ratios in all library bands
        self.ml_chain = np.empty((n_iter + 1,
                                  self._model.n_pix,
                                  len(self._model.library_bands)),
                                 dtype=np.float)

        # Model SED
        self.sed_chain = np.empty((n_iter + 1,
                                   self._model.n_pix,
                                   len(self._model.library_bands)),
                                  dtype=np.float)

    def _estimate_theta(self, k, B=None):
        """Estimate the stellar population parameters.

        Parameters
        ----------
        B : ndarray
            Optional values for the background, a ``(n_band)`` array.
            Background is in units of flux per arcsec^2.
            If `None` then the background is read from the previous value of
            the background chain.
        """
        n_pixels = self._model.n_pix

        if B is None:
            B = self.B_chain[k - 1, :]

        # Compute SP parameters for each SED, with background subtracted
        for i in xrange(n_pixels):
            le = LibraryEstimator(
                self.model._seds[i, :] - B, self.model._errs[i, :],
                self._obs_bands, self.model.d,
                self.model.library_file, self.model.library_group,
                ncpu=1)
            # marginal estimate of model parameters
            for j, name in enumerate(self._model.theta_params):
                self.theta_chain[k, i, j] = le.estimate(name, p=(0.5,))[0]
            # marginal estimate of metadata parameters
            for j, name in enumerate(self._model.meta_params):
                self.blob_chain[k, i, j] = le.estimate_meta(name, p=(0.5,))[0]
            # marginal estimate of M/L
            for j, band in enumerate(self._model.library_bands):
                self.ml_chain[k, i, j] = le.estimate_ml(band, p=(0.5,))[0]
            # marginal estimate of SED
            for j, band in enumerate(self._model.library_bands):
                self.sed_chain[k, i, j] = le.estimate_flux(band, p=(0.5,))[0]

    def table(self):
        """A :class:`MultiPixelChain` made the Gibbs sampler."""
        pass
