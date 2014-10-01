#!/usr/bin/env python
# encoding: utf-8
"""
Sample for Multi Pixel Gibbs MCMC.
"""


class MultiPixelGibbsBgSampler(object):
    """MH-in-Gibbs sampler for three level hierarchical models of
    pixel parameters, global parameters, and a linearly sampled background
    parameter.

    Parameters
    ----------
    seds : ndarray
        SEDs in µJy, shape ``(npix, nbands)``.
    seds : ndarray
        SEDs uncertainties in µJy, shape ``(npix, nbands)``.
    bands : list
        List of bandpass names corresponding to the ``seds``
    compute_bands : list
        List of bandpasses to compute and included in the chain metadata.
    model_initializer : function
        Function that, when called, creates a model instance.
        This is done to enable multi-processing.
    n_cpu : int
        Number of CPUs to run on.
    """
    def __init__(self, seds, sed_errs, bands,
                 model_initializer,
                 compute_bands=None,
                 n_cpu=1):
        super(MultiPixelGibbsBgSampler, self).__init__()
        self._seds = seds
        self._errs = sed_errs
        self._sed_bands = bands
        if compute_bands is not None:
            self._compute_bands = compute_bands
        else:
            self._compute_bands = self._sed_bands
        self._n_cpu = n_cpu
        self._model_initializer = model_initializer

    def sample(self, n_iter,
               theta_prop=None,
               phi_prop=None):
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
        pass
