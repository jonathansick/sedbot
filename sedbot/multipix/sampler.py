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
        if self._n_cpu == 1:
            map_fcn = map  # single processing
        else:
            map_fcn = None  # TODO
        return map_fcn

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
