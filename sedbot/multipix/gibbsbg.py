#!/usr/bin/env python
# encoding: utf-8
"""
Gibbs sampling of a hierarchical model with multiple pixels having independent
star formation histories and a global scalar background that effects all
pixels.
"""


class MultiPixelGibbsBgModeller(object):
    """Runs a Gibbs sampler that alternates between sampling pixels and
    sampling a background bias that uniformly affects all pixels.

    Parameters
    ----------
    seds : ndarray
        An ``(n_pixel, n_band)`` shape array of SEDs for each pixel, in
        units of µJy.
    sed_errs : ndarray
        An ``(n_pixel, n_band)`` shape array of SEDs uncertainties for each
        pixel, in units of µJy.
    pixel_lnpost : instance
        A lnpost function for pixel-level MCMC.
    global_lnpost : instance
        A lnpost function for global-level MCMC.
    theta_init_sigma : ndarray
        Array of standard deviations to create a start distribution for the
        per-pixel theta parameters.
    phi_init_sigma : ndarray
        Array of standard devitions to create a start distribution for the
        global phi parameters (aside from B).
    n_pixel_walkers : int
        Number of emcee walkers for pixel-level MCMC
    n_global_walkers : int
        Number of emcee walkers for global-level MCMC
    """
    def __init__(self, seds, sed_errs,
                 pixel_lnpost, global_lnpost,
                 theta_init_sigma, phi_init_sigma,
                 n_pixel_walkers=100,
                 n_global_walkers=100):
        super(MultiPixelGibbsBgModeller, self).__init__()
        self._obs_seds = seds
        self._obs_errs = sed_errs
        self._pixel_lnpost = pixel_lnpost
        self._theta_init_sigma = theta_init_sigma
        self._phi_init_sigma = phi_init_sigma
        self._n_pixel_walkers = n_pixel_walkers
        self._n_global_walkers = n_global_walkers

    def sample(self,
               theta0, bg0, d0,
               n_iters,
               n_pixel_steps=10,
               n_cpu=None):
        """Make samples

        Parameters
        ----------
        theta0 : ndarray
            Initial position of the stellar population (per-pixel) parameters
        bg0 : ndarry
            Initial position of the background estimate
        d0 : ndarray
            Initial position of the distance estimate
        n_iters : int
            Number of Gibbs samples to make
        n_pixel_steps : int
            Number of steps to make in pixel space
        n_global_steps : int
            Number of steps to make in the global parameter space.
        n_cpu : int
            Number of parallel cores to run on
        """
        # Initialize the posterior chains
        for j in xrange(n_iters):
            # Step 1. Send jobs out to each pixel to sample the stellar pop
            # in each pixel given the last parameter estimates.
            # get the parameter chain for each pixel and the model flux
            self._sample_pixel_posterior()

            # Step 2. Sample a new distance.
            self._sample_global_posterior()

            # Step 3. Sample the background
            self._recompute_background()

    def _sample_pixel_posterior(self):
        """Sample the parameters at the level of each pixel."""
        # Set up an emcee run for each pixel
        pass

    def _sample_global_posterior(self):
        """Sample from parameters at the global level in a single MCMC run."""
        pass

    def _recompute_background(self):
        """Linear estimate of the background."""
        pass
