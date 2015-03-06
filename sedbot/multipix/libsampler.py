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

    def initialize_theta(self, B0):
        """Initialize the stellar population parameters.

        This can be used as an input for :meth:`sample`.

        B0 : ndarray
            Initial values for the background, a ``(n_band)`` array.
            Background is in units of flux per arcsec^2.
        """
        n_pixels = self._model.n_pix
        n_dim = self._model.n_theta
        theta = np.empty((n_pixels, n_dim), dtype=np.float)

        # Compute SP parameters for each SED
        for i in xrange(n_pixels):
            estimator = LibraryEstimator(
                self.model._seds[i, :], self.model._errs[i, :],
                self._obs_bands, self.model.d,
                self.model.library_file, self.model.library_group,
                ncpu=1)
            for j, name in enumerate(self.model.theta_params):
                theta[i, j] = estimator.estimate(self, name, p=(0.5,))[0]

        return theta

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
