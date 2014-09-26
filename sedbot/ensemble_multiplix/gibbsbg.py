#!/usr/bin/env python
# encoding: utf-8
"""
Gibbs sampling of a hierarchical model with multiple pixels having independent
star formation histories and a global scalar background that effects all
pixels.
"""

import numpy as np
from numpy.random import randn
import emcee
from IPython.parallel import Client

GLOBAL_LNPOST = None


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
    fsps_params : dict
        Parameters to initialize the python-fsps StellarPopulation engine.
    n_walkers : int
        Number of emcee walkers to use in both pixel and global levels.
    n_emcee_steps : int
        Number of steps to make in each emcee call.
    fsps_compute_bands : list
        (Optional) list of all bands to compute and persist with the chain.
    """
    def __init__(self, seds, sed_errs, bands,
                 pixel_lnpost_class, global_lnpost_class,
                 theta_priors, phi_priors,
                 theta_init_sigma, phi_init_sigma,
                 fsps_params,
                 n_walkers=100,
                 n_emcee_steps=10,
                 fsps_compute_bands=None):
        super(MultiPixelGibbsBgModeller, self).__init__()
        self._obs_seds = seds
        self._obs_errs = sed_errs
        self._obs_bands = bands
        self._fsps_compute_bands = fsps_compute_bands
        self._fsps_params = fsps_params

        self._theta_priors = theta_priors
        self._phi_priors = phi_priors
        self._pixel_lnpost_class = pixel_lnpost_class
        self._theta_init_sigma = theta_init_sigma
        self._phi_init_sigma = phi_init_sigma
        self._n_walkers = n_walkers
        self._n_emcee_steps = n_emcee_steps

        self._n_pix = self._obs_seds.shape[0]

    def sample(self,
               theta0, B0, phi0,
               n_iters,
               parallel=False):
        """Make samples

        Parameters
        ----------
        theta0 : ndarray
            Initial position of the stellar population (per-pixel) parameters.
            Should be of shape ``(npixel, ndim)``.
        B0 : ndarry
            Initial position of the background estimate.
            Should be of shape ``(nband,)``.
        phi0 : ndarray
            Initial position of the phi-level parameters.
            Should be of shape ``(ndim,)``.
        n_iters : int
            Number of Gibbs samples to make
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
        # The first 1 is for the p0 initialization sample
        n_samples = 1 + n_iters * (self._n_emcee_steps  # theta step
                                   + self._n_emcee_steps  # phi step
                                   + 1)  # B step
        self._theta_chain = np.empty((self._n_walkers,
                                      n_samples,
                                      PIXEL_LNPOST.ndim,
                                      self._n_pix),
                                     dtype=np.float)
        self._B_chain = np.empty((n_samples,
                                  PIXEL_LNPOST.nbands),
                                 np.float)
        self._phi_chain = np.empty((self._n_walkers,
                                    n_samples,
                                    GLOBAL_LNPOST.ndim_phi),
                                   dtype=np.float)

        # Ask the posterior class to initialize the blob
        # We make a different blob chain for each pixel.
        self._blobs = [PIXEL_LNPOST.init_blob_chain(n_samples)
                       for i in xrange(self._n_pix)]

        # Make an initial point for the local posterior chain
        # **for each pixel**
        for i in xrange(self._obs_seds.shape[0]):
            for j in xrange(self._n_walkers):
                p0 = self._theta_init_sigma * randn(PIXEL_LNPOST.ndim) \
                    + theta0[j, :]
                self._theta_chain[j, 0, :, i] = p0[:]

        # Make an initial point for the global posterior chain
        for j in xrange(self._n_walkers):
            p0 = self._phi_init_sigma * randn(GLOBAL_LNPOST.ndim) + phi0
            self._phi_chain[j, 0, :] = p0[:]

        # Set initial point for the B chain
        self._B_chain[0, :] = B0

        # Initial guess is the zeroth elements of these posterior arrays
        self._last_i = 0

        # Gibbs Sampling
        for j in xrange(n_iters):
            # Step 1. Send jobs out to each pixel to sample the stellar pop
            # in each pixel given the last parameter estimates.
            # get the parameter chain for each pixel and the model flux
            self._sample_pixel_posterior(self._n_emcee_steps, parallel)

            # Step 2. Sample a new distance.
            self._sample_global_posterior(self._n_emcee_steps)

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
                         "FSPS_PARAMS": self._fsps_params,
                         "init_pixel_lnpost": init_pixel_lnpost,
                         "sample_theta": sample_theta})

        # Sync import statements
        self._pool.execute("import numpy as np")
        self._pool.execute("import emcee")
        self._pool.execute("import fsps")
        self._pool.execute("from sedbot.photconv import abs_ab_mag_to_mjy")
        self._pool.execute("from sedbot.zinterp import bracket_logz")
        self._pool.execute("from sedbot.zinterp import interp_logz")

        # initialize the pixel posterior
        self._pool.execute("PIXEL_LNPOST = None")
        self._pool.execute("init_pixel_lnpost(PIXEL_LNPOST_CLASS, "
                           "FSPS_PARAMS, OBS_BANDS, "
                           "fsps_compute_bands=FSPS_COMPUTE_BANDS)")

    def _init_local_lnpost(self):
        init_pixel_lnpost(self._pixel_lnpost_class,
                          self._fsps_params,
                          self._obs_bands,
                          fsps_compute_bands=self._fsps_compute_bands)

    def _init_global_lnpost(self):
        global GLOBAL_LNPOST
        GLOBAL_LNPOST = self._global_lnpost_class(
            self._fsps_params,
            self._obs_bands, self._obs_seds, self._obs_errs,
            self._phi_priors)

    def _sample_pixel_posterior(self, n_steps, parallel):
        """Sample the parameters at the level of each pixel."""
        # Grab last values of B and phi
        B_i = self._B_chain[self._last_i, :]
        # FIXME Always use zeroth walker from phi chain
        # or should I choose random walker. Note sure how to
        # 'align' the chains so a theta walker matches a phi walker
        phi_i = self._phi_chain[0, self._last_i, :]
        # Construct arguments for sample_theta() for each pixel
        args = []
        for i in xrange(self._n_pix):
            obs_sed = self._obs_seds[i, :]
            obs_err = self._obs_errs[i, :]
            priors = self._theta_priors[i]
            p0 = self._theta_chain[:, self._last_i, :, i]  # prev pos for pixel
            print "pixel sample p0.shape", p0.shape
            arg = (obs_sed, obs_err, priors, B_i, phi_i, p0,
                   self._n_walkers, n_steps)
            args.append(arg)
        if not parallel:
            results = map(sample_theta, args)
        else:
            results = self._pool.map(sample_theta, args)
        # Append results to the chain
        for i, result in enumerate(results):
            chain, blobs = result
            j = self._last_i + 1
            k = j + n_steps
            self._theta_chain[:, j:k, :, i] = chain
            # persist the blobs using the posterior function
            # i.e. there is a separate blob chain for each pixel
            PIXEL_LNPOST.append_blobs(j, self._blobs[i], blobs)
        # repeat B and phi in their respective chains
        self._phi_chain[:, j:k, :] = self._phi_chain[:, j - 1, :]
        self._B_chain[j:k, :] = self._B_chain[j - 1, :]

        self._last_i += n_steps

    def _sample_global_posterior(self, n_steps):
        """Sample from parameters at the global level in a single MCMC run."""
        # Get previous state of chain
        # FIXME Choose zeroth walker from theta chain
        thetas_i = self._theta_chain[0, self._last_i, :, :]
        B_i = self._B_chain[self._last_i, :]
        p0 = self._phi[self._last_i, :]

        # Run sampler
        sampler = emcee.EnsembleSampler(
            self._n_global_walkers, GLOBAL_LNPOST.ndim, GLOBAL_LNPOST,
            args=(thetas_i, B_i))
        sampler.run_mcmc(p0, n_steps)
        chain = sampler.chain

        # Insert posterior samples into chain
        j = self._last_i + 1
        k = j + n_steps
        self._phi_chain[:, j:k, :] = chain
        # repeat B and theta in their respective chains
        self._theta_chain[:, j:k, :, :] = self._theta_chain[:, j - 1, :, :]
        self._B_chain[j:k, :] = self._B_chain[j - 1, :]
        # Replicate blob data too (needed for background estimate)
        self._replicate_blob_chain(j - 1, range(j, k))

        self._last_i += n_steps * self._n_global_walkers

    def _recompute_background(self):
        """Linear estimate of the background.

        The background is modelled, for each pixel, as

        .. math::
           \langle B \rangle_n = F_n - f(\theta_{n, i}, \phi_i)

        Given Gaussian uncertainties in $F_n$, these estimates of
        $\langle B \rangle_n$ are themselves generated from a normal
        distribution:

        .. math::
           N\left(\frac{\sum_n \frac{\langle B \rangle_n}{\sigma_n^2}}
           {\sum_n \frac{1}{\sigma_n^2}}, \frac{1}{\sum_n \sigma_n^{-2}}\right)

        (see http://en.wikipedia.org/wiki/
        Weighted_arithmetic_mean#Dealing_with_variance)

        Hence we can produce the next sample of $B$ by simply drawing from
        this normal distribution.
        """
        B_est = np.empty((self._n_pix, PIXEL_LNPOST.nbands),
                         dtype=np.float)
        B_est.fill(np.nan)
        for i in self._n_pix:
            # Compute background estimate
            # the lnpost needs to compute this because it knows
            # - where in the blob the model_sed is stored
            # - how to map modelel_sed to obs_sed bandpasses
            B_est[i, :] = PIXEL_LNPOST.estimate_backgrounds(
                self._obs_seds[i, :], self._blobs[i][self._i_last])

        # Estimate Normal Distribution to sample from
        # (vectors of length n_bands)
        # FIXME check axis
        means = np.sum(B_est / self._obs_errs ** 2., axis=0) \
            / np.sum(self._obs_errs ** -2., axis=0)
        sigmas = np.sqrt(1. / np.sum(self._obs_errs ** -2., axis=0))

        # Sample from Gaussian to draw
        B_proposed = sigmas * np.random.randn(PIXEL_LNPOST.nbands) + means
        j = self._i_last + 1
        self._B_chain[j, :] = B_proposed

        # Repeat theta and phi values in the chain
        self._phi_chain[: j, :] = self._phi_chain[:, j - 1, :]
        self._theta_chain[:, j, :, :] = self._theta_chain[:, j - 1, :, :]
        # Replicate blob data too
        self._replicate_blob_chain(j - 1, [j])

        self._last_i += 1

    def _replicate_blob_chain(self, source_step, target_steps):
        for j in xrange(self._n_pix):
            self._blobs[j][target_steps] = source_step


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


def sample_theta(args):
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
    chain : ndarray
        The posterior chain, ``(n_walkers, n_steps, ndim)``.
    blobs : ndarray
        Metadata associated with the chain. This is a list of length
        ``n_steps``, where each item is of length ``n_walkers``.
    """
    global PIXEL_LNPOST
    obs_sed, obs_errs, priors, B, phi, p0, n_walkers, n_steps = args
    PIXEL_LNPOST.reset_pixel(obs_sed, obs_errs, priors)
    sampler = emcee.EnsembleSampler(
        n_walkers, PIXEL_LNPOST.ndim, PIXEL_LNPOST, args=(B, phi))
    sampler.run_mcmc(p0, n_steps)
    chain = sampler.chain
    blobs = sampler.blobs
    return chain, blobs
