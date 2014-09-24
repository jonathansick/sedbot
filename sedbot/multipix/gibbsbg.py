#!/usr/bin/env python
# encoding: utf-8
"""
Gibbs sampling of a hierarchical model with multiple pixels having independent
star formation histories and a global scalar background that effects all
pixels.
"""

import numpy as np
import emcee
from IPython.parallel import Client


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
    bands : list
        List of python-fsps band names, corresponding to the ``seds`` array.
    pixel_lnpost_class : class
        A class that, when initialized, provides a callable pixel posterior
        probability function.
    global_lnpost_class : class
        A class that, when initialized, provides a callable global posterior
        probability function.
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
    fsps_compute_bands : list
        (Optional) list of all bands to compute and persist with the chain.
    """
    def __init__(self, seds, sed_errs, bands,
                 pixel_lnpost_class, global_lnpost_class,
                 theta_init_sigma, phi_init_sigma,
                 n_pixel_walkers=100,
                 n_global_walkers=100,
                 fsps_compute_bands=None):
        super(MultiPixelGibbsBgModeller, self).__init__()
        self._obs_seds = seds
        self._obs_errs = sed_errs
        self._obs_bands = bands
        self._fsps_compute_bands = fsps_compute_bands

        self._pixel_lnpost_class = pixel_lnpost_class
        self._theta_init_sigma = theta_init_sigma
        self._phi_init_sigma = phi_init_sigma
        self._n_pixel_walkers = n_pixel_walkers
        self._n_global_walkers = n_global_walkers

        self._n_pix = self._obs_seds.shape[0]

        # Hack, just get priors from pixel_lnpost; in principle we want to
        # store a different set of priors for each pixel
        self._pixel_priors = [self._pixel_lnpost._priors] * self._n_pix

    def sample(self,
               theta0, bg0, d0,
               n_iters,
               n_pixel_steps=10,
               n_global_steps=10,
               parallel=False):
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
        parallel : bool
            True if iPython.parallel should be used to distribute work.
            Note that the cluster should be setup with a
            ``ipcluster start -n <n cpu>`` command.
        """
        # Always make a local ln posterior function
        self._init_local_lnpost()

        if parallel:
            # Build pool of posterior calculators
            self._build_pool()

        # Pre-allocate memory for the posterior chains (theta, phi, B)
        # The final 1 is for the zeroth sample
        n_samples = n_iters * (self._n_pixel_walkers * n_pixel_steps
                               + self._n_global_walkers * n_global_steps
                               + 1) + 1
        self._theta_chain = np.empty((n_samples,
                                      self._pixel_lnpost.ndim,
                                      self._n_pix),
                                     np.float)
        self._B_chain = np.empty((n_samples, self._pixel_lnpost.nbands),
                                 np.float)
        self._phi_chain = np.empty((n_samples, self._pixel_lnpost.ndim_phi),
                                   np.float)

        # Ask the posterior class to initialize the blob
        # We make a different blob chain for each pixel.
        self._blobs = [PIXEL_LNPOST.init_blob_chain(self._n_pixel_walkers,
                                                    n_samples)
                       for i in xrange(self._n_pix)]

        # Make an initial point for the local posterior chain
        # **for each pixel**
        # Shape of chain: (nwalkers, nlinks, dim)
        # Shape of p0: (nwalkers, dim)
        # Shape of flatchain: (nlinks, dim)
        for i in xrange(self._obs_seds.shape[0]):
            p0 = self._theta_init_sigma \
                * np.random.rand(PIXEL_LNPOST.ndim * self._n_pixel_walkers)\
                .reshape((self._n_pixel_walkers, PIXEL_LNPOST.ndim))
            self._theta_chain[0, :, i] = p0

        # Make an initial point for the global posterior chain
        # TODO

        # Initial guess is the zeroth elements of these posterior arrays
        self._last_i = 0

        # Gibbs Sampling
        for j in xrange(n_iters):
            # Step 1. Send jobs out to each pixel to sample the stellar pop
            # in each pixel given the last parameter estimates.
            # get the parameter chain for each pixel and the model flux
            self._sample_pixel_posterior(n_pixel_steps, parallel)

            # Step 2. Sample a new distance.
            self._sample_global_posterior()

            # Step 3. Sample the background
            self._recompute_background()

    def _init_pool(self, n_cpu):
        """Build an ipython.parallel pool of CPUs to compute likelihoods.

        Recall that the number of nodes is defined
        """
        c = Client()
        self._pool = c[:]  # a DirectView object

        # Here we send objects to each compute server. These make up the
        # environment for using python-fsps
        self._pool.push({"PIXEL_LNPOST_CLASS": self._pixel_lnpost_class,
                         "OBS_BANDS": self._obs_bands,
                         "FSPS_COMPUTE_BANDS": self._fsps_compute_bands,
                         "init_pixel_lnpost": init_pixel_lnpost,
                         "sample_phi": sample_phi})

        # Sync import statements
        self._pool.execute("import numpy as np")
        self._pool.execute("import emcee")
        self._pool.execute("import fsps")
        self._pool.execute("import fsps")
        self._pool.execute("from sedbot.photconv import abs_ab_mag_to_mjy")
        self._pool.execute("from sedbot.zinterp import bracket_logz")
        self._pool.execute("from sedbot.zinterp import interp_logz")

        # initialize the pixel posterior
        self._pool.execute("PIXEL_LNPOST = None")
        self._pool.execute("init_pixel_lnpost(PIXEL_LNPOST_CLASS, OBS_BANDS, "
                           "fsps_compute_bands=FSPS_COMPUTE_BANDS)")

    def _init_local_lnpost(self):
        init_pixel_lnpost(self._pixel_lnpost_class, self._obs_bands,
                          fsps_compute_bands=self._fsps_compute_bands)

    def _sample_pixel_posterior(self, n_steps, parallel):
        """Sample the parameters at the level of each pixel."""
        # Grab last values of B and phi
        B_i = self._B[self._last_i, :]
        phi_i = self._phi[self._last_i, :]
        # Construct arguments for sample_phi() for each pixel
        args = []
        for i in xrange(self._n_pix):
            obs_sed = self._obs_seds[i, :]
            obs_err = self._obs_errs[i, :]
            priors = self._pixel_priors[i]
            p0 = self._theta_chain[self._last_i:, i]  # prev set of walkers
            arg = (obs_sed, obs_err, priors, B_i, phi_i, p0,
                   self._n_walkers, n_steps)
            args.append(arg)
        if not parallel:
            results = map(sample_phi, args)
        else:
            results = self._pool.map(sample_phi, args)
        # Append results to the flatchains
        for i, result in enumerate(results):
            chain, blobs = result
            j = self._last_i + 1
            k = self._last_i + 1 + n_steps
            self._theta_chain[j:k, :, i] = chain
            # persist the blobs using the posterior function
            # i.e. there is a separate blob chain for each pixel
            PIXEL_LNPOST.append_blobs(j, self._blobs[i], blobs)

        self._last_i += n_steps

    def _sample_global_posterior(self, n_steps):
        """Sample from parameters at the global level in a single MCMC run."""
        pass

        self._last_sample_index += n_steps

    def _recompute_background(self):
        """Linear estimate of the background."""
        self._last_sample_index += 1


PIXEL_LNPOST = None


def init_pixel_lnpost(lnpost_cls, fsps_params, obs_bands,
                      fsps_compute_bands=None):
    """Initialize the per-pixel ln posterior probability functions on each
    server (for parallel computing; or just for the local module for
    non-parallel computing.
    """
    global PIXEL_LNPOST
    PIXEL_LNPOST = lnpost_cls(fsps_params, obs_bands,
                              fsps_compute_bands=fsps_compute_bands)


def sample_phi(args):
    """Run an emcee sampler in the theta (per pixel) space.

    This sampler is extracted into separate function so that it may be called
    by a parallel map function.

    Arguments
    ---------
    obs_sed : ndarray
        Observed SED (µJy)
    obs_errs : ndarray
        SED uncertainties (µJy)
    priors : dict
        Dictionary of prior functions
    B : ndarray
        Current sample of B, the background
    phi : ndarray
        Current sample of phi, the global parameters
    p0 : ndarray
        Position of the walkers in theta at the previous Gibbs step.
    n_walkers : int
        Number of emcee walkers.
    n_steps : int
        Number of emcee steps.

    Returns
    -------
    flatchain : ndarray
        The flattened chain.
    blobs : ndarray
        Metadata associated with the flatchain. This is a list of length
        ``n_steps``, where each item is of length ``n_walkers``.
    """
    global PIXEL_LNPOST
    obs_sed, obs_errs, priors, B, phi, p0, n_walkers, n_steps = args
    PIXEL_LNPOST.reset_pixel(obs_sed, obs_errs, priors)
    sampler = emcee.EnsembleSampler(
        n_walkers, PIXEL_LNPOST.ndim, PIXEL_LNPOST, args=(B, phi))
    sampler.run_mcmc(p0, n_steps)
    flatchain = sampler.flatchain
    blobs = sampler.blobs
    return flatchain, blobs
