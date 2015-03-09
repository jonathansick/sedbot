#!/usr/bin/env python
# encoding: utf-8
"""
Sampler for Multi Pixel Gibbs modelling, using a library-based SP
estimates rather than MCMC.
"""

from collections import OrderedDict

import numpy as np

from astropy.table import Table, hstack
from astropy.utils.console import ProgressBar

from sedbot.chain import MultiPixelChain


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
        self.model = model

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
            B0 = np.empty(self.model.n_bands, dtype=np.float)
            for i, (n, instr) in enumerate(zip(self.model.observed_bands,
                                               self.model.instruments)):
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

                # Estimate background given current SP state
                self._estimate_background(i)

                bar.update()

    def _init_chains(self, n_iter):
        """Initialize memory for the chain."""
        # FSPS SP Parameter estimates
        self.theta_chain = np.empty((n_iter + 1,
                                     self.model.n_pix,
                                     self.model.n_theta),
                                    dtype=np.float)
        self.theta_chain.fill(np.nan)

        # Background estimates
        self.B_chain = np.empty((n_iter + 1,
                                 self.model.n_bands),
                                dtype=np.float)
        self.B_chain.fill(np.nan)

        # FSPS computed metadata values
        self.blob_chain = np.empty((n_iter + 1,
                                    self.model.n_pix,
                                    len(self.model.meta_params)),
                                   dtype=np.float)

        # M/L ratios in all library bands
        self.ml_chain = np.empty((n_iter + 1,
                                  self.model.n_pix,
                                  len(self.model.library_bands)),
                                 dtype=np.float)

        # Model SED
        self.sed_chain = np.empty((n_iter + 1,
                                   self.model.n_pix,
                                   len(self.model.library_bands)),
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
        n_pixels = self.model.n_pix

        if B is None:
            B = self.B_chain[k - 1, :]

        # Compute SP parameters for each SED, with background subtracted
        for i in xrange(n_pixels):
            le = self.model.estimator_for_pixel(i, B)
            # marginal estimate of model parameters
            for j, name in enumerate(self.model.theta_params):
                self.theta_chain[k, i, j] = le.estimate(name, p=(0.5,))[0]
            # marginal estimate of metadata parameters
            for j, name in enumerate(self.model.meta_params):
                self.blob_chain[k, i, j] = le.estimate_meta(name, p=(0.5,))[0]
            # marginal estimate of M/L
            for j, band in enumerate(self.model.library_bands):
                self.ml_chain[k, i, j] = le.estimate_ml(band, p=(0.5,))[0]
            # marginal estimate of SED
            for j, band in enumerate(self.model.library_bands):
                self.sed_chain[k, i, j] = le.estimate_flux(band, p=(0.5,))[0]

    def _estimate_background(self, k):
        """Estimate the background given the current stellar population
        parameter state.

        The background update is *always* done after the theta update.

        Parameters
        ----------
        k : int
            Index of the current Gibbs step.
        """
        # Get the model SED matching the observed bands
        model_seds = self.sed_chain[k, :, self.model.band_indices]
        B_new = self.model.estimate_background(model_seds)
        self.B_chain[k, :] = B_new

    @property
    def table(self):
        """A :class:`MultiPixelChain` representing the Gibbs chain.

        The intention is for the library sampler to have a data output
        similar to the MH-in-Gibbs sampler.
        """
        meta = OrderedDict((
            ('observed_bands', self.model.observed_bands),
            ('instruments', self.model.instruments),
            ('computed_bands', self.model.library_bands),
            ('msun_ab', self.model.msun_ab),
            ('band_indices', self.model.band_indices),
            ('theta_params', self.model.theta_params),
            ('sed', self.model._seds),
            ('sed_err', self.model._errs),
            ('pixels', self.model.pixel_metadata),
            ('area', self.model._areas),
            ('d', self.model.d)))

        # Make tables for theta chain, B chain, and blob chain (that includes
        # SED and M/L datasets)
        # TODO we should probably just change the chain dtypes instead
        # original table - (step, pixel, parameter)
        # output table - (step, parameter, pixel)
        theta_table = Table(np.swapaxes(self.theta_chain, 1, 2),
                            names=self.mdoel.theta_params,
                            meta=meta)

        background_names = ["B__{0}__{1}".format(n, b)
                            for n, b in zip(self.model.instruments,
                                            self.model.observed_bands)]
        B_table = Table(self.B, names=background_names)

        meta_table = Table(np.swapaxes(self.blob_chain, 1, 2),
                           names=self.model.meta_params)

        # output table (step, band, pixel)
        sed_table = Table(np.swapaxes(self.sed_chain, 1, 2),
                          names=self.model.library_bands)

        # output table (step, band, pixel)
        # name columns to be different from flux
        ml_names = ["logML_{0}".format(b) for b in self.model.library_bands]
        ml_table = Table(np.swapaxes(self.ml_chain, 1, 2),
                         names=ml_names)

        tbl = MultiPixelChain(hstack((theta_table,
                                      B_table,
                                      meta_table,
                                      sed_table,
                                      ml_table)))

        return tbl
