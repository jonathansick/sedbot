#!/usr/bin/env python
# encoding: utf-8
"""
Models to use with the Gibbs multipixel sampler.
"""

import itertools

import numpy as np

import fsps

from sedbot.photconv import abs_ab_mag_to_mjy
from sedbot.zinterp import bracket_logz, interp_logz

# Module-level stellar population
SP = None


class MultiPixelBaseModel(object):
    """Baseclass for multipixel models.

    Parameters
    ----------
    pset : dict
        Initialization arguments to :class:`fsps.StellarPopulation`, as a
        dictionary.
    """
    def __init__(self, pset=None):
        super(MultiPixelBaseModel, self).__init__()
        # Containers for SEDs of all pixels
        self._seds = None
        self._errs = None

        # Dict of band index, background level
        # Use this so that certain bands can always have a fixed background
        # level that bypasses ``update_background``.
        self._fixed_bg = {}

        # Lists of names of parameters in theta and phi levels
        self._theta_params = None
        self._phi_params = None

        # Band names in the SED
        self._obs_bands = []
        self._compute_bands = []

        # Prior function dictionaries
        self._theta_priors = []  # list for each pixel
        self._phi_priors = {}  # dict keyed by prior key

        # Function or class method for computing pixel likelihoods
        self._lnlike_fcn = interp_z_likelihood

        # Map processing function
        self._M = map  # or an ipython cluster

        # Set up the module-level stellar population engine
        global SP
        if pset is None:
            pset = {}
        SP = fsps.StellarPopulation(**pset)

    @property
    def band_indices(self):
        return np.array([self._compute_bands.index(b)
                         for b in self._obs_bands])

    @property
    def n_pix(self):
        return self._seds.shape[0]

    @property
    def n_bands(self):
        return self._seds.shape[1]

    @property
    def n_theta(self):
        """Number of theta parameters."""
        return len(self._theta_params)

    @property
    def n_phi(self):
        """Number of phi parameters."""
        return len(self._phi_params)

    @property
    def blob_dtype(self):
        """Dtype for the blob data generated by this model for each pixel."""
        dt = [('logMstar', np.float, self.n_pix),
              ('logMdust', np.float, self.n_pix),
              ('logLbol', np.float, self.n_pix),
              ('logSFR', np.float, self.n_pix),
              ('logAge', np.float, self.n_pix),
              ('model_sed', np.float, (self.n_pix, self.n_bands))]
        return np.dtype(dt)

    def sample_pixel(self, theta_i, phi, B, ipix):
        """Compute the ln-prob of a proposed step in local parameter
        space (theta) for a single pixel, ``ipix``.

        Parameters
        ----------
        theta_i : ndarray
            Parameters in theta space for this pixel
        phi : ndarray
            Global parameters
        B : ndarray
            Background, ``(nbands,)``.
        ipix : int
            Specify the pixel index

        Returns
        -------
        lnprob : float
            Ln probability.
        blob : ndarray
            Structured array with metadata for this pixel sample.
        """
        lnprior = self._pixel_ln_prior(theta_i, ipix)
        lnprior += self._global_ln_prior(phi)
        if not np.isfinite(lnprior):
            return -np.inf, np.nan

        lnL, blob = self._lnlike_fcn((theta_i,
                                      self._theta_params,
                                      phi,
                                      self._phi_params,
                                      B,
                                      self._seds[ipix, :],
                                      self._errs[ipix, :],
                                      self._obs_bands,
                                      self._compute_bands))
        lnp = lnprior + lnL
        if not np.isfinite(lnp):
            return -np.inf, np.nan
        else:
            return lnL, blob

    def sample_global(self, theta, phi, B):
        """Compute the ln-prob of a proposed step in global parameter
        space (phi).

        Parameters
        ----------
        theta : ndarray
            Parameters in theta space for all pixels,
            ``(nparams, npix)`` shape.
        phi : ndarray
            Global parameters
        B : ndarray
            Background, ``(nbands,)``.

        Returns
        -------
        lnprob : float
            Ln probability
        pixel_lnprobs : list
            Ln probability for individual pixels
        blobs : list
            Blobs for individual pixels
        """
        lnpriors = []
        for ipix in xrange(self.n_pix):
            lnprior = self._pixel_ln_prior(theta[ipix, :], ipix)
            if not np.isfinite(lnprior):
                return -np.inf, np.inf, None
            lnpriors.append(lnprior)

        args = []
        for ipix in xrange(self.n_pix):
            args.append((theta[ipix, :],
                         self._theta_params,
                         phi,
                         self._phi_params,
                         B,
                         self._seds[ipix, :],
                         self._errs[ipix, :],
                         self._obs_bands,
                         self._compute_bands))
        results = self._M(self._lnlike_fcn, args)

        lnprob = self._global_ln_prior(phi)
        pixel_lnprobs = []
        blobs = []
        for lnprior, result in zip(lnpriors, results):
            lnL, blob = result
            lnprob += lnL + lnprior
            pixel_lnprobs.append(lnprob)
            blobs.append(blob)
        return lnprob, pixel_lnprobs, blobs

    def _pixel_ln_prior(self, theta, ipix):
        """Compute prior probabilities for a pixel sample

        Excludes global priors.
        """
        lnp = 0
        for i, name in enumerate(self._theta_params):
            lnp += self._theta_priors[ipix][name](theta[i])
        return lnp

    def _global_ln_prior(self, phi):
        """Compute prior probabilities of global parameters."""
        lnp = 0
        for i, name in enumerate(self._phi_params):
            lnp += self._phi_priors[name](phi[i])
        return lnp

    def update_background(self, theta, phi, model_seds):
        """Recompute the scalar background from a Normal distribution
        of model observation residuals.

        .. note:: Bandpasses with backgrounds fixed via the ``_fixed_bg``
                  attribute are respected.

        Parameters
        ----------
        theta : ndarray
            An `(n_pixels,)` structured array with pixel-level parameters
        phi : ndarray
            A `(1,)` structured array with the global-level parameters.
        model_seds : ndarray
            A ``(nbands, npix)`` array of model SEDs from the blob data
            of the previous step. This lets us skip the task of recomputing
            SEDs before estimating a new background.

        Returns
        -------
        B : ndarray
            New background vector, shape ``(n_bands,)``.
        pixel_lnprobs : list
            Ln probability for individual pixels
        blobs : list
            Blobs for individual pixels
        """
        # Reduce the model SEDs to just the observed bands
        model = model_seds[self.band_indices, :]
        all_residuals = self._seds - model
        residuals = all_residuals.mean(axis=0)  # FIXME?
        print residuals

        # Sample new values of B (for each bandpass) from a normal dist.
        # TODO

        # reset B for any images with fixed background
        # TODO

        # recompute the ln probability
        return self.sample_global(theta, phi)


def interp_z_likelihood(args):
    """Generic single-pixel likeihood function where a single metallicity
    is linearly interpolated.
    """
    global SP

    theta, theta_names, phi, phi_names, B, sed, err, sed_bands, compute_bands \
        = args

    meta1 = np.empty(5, dtype=float)
    meta2 = np.empty(5, dtype=float)

    band_indices = np.array([compute_bands.index(b) for b in sed_bands])

    # Evaluate the ln-likelihood function by interpolating between
    # metallicty bracket
    for name, val in itertools.chain(zip(theta_names, theta),
                                     zip(phi_names, phi)):
        if name == 'logmass':
            logm = val
        elif name == 'logZZsol':
            logZZsol = val
        elif name == 'logtau':
            SP.params['tau'] = 10. ** val
        elif name in SP.params.all_params:  # list of parameter names
            SP.params[name] = val

    zmet1, zmet2 = bracket_logz(logZZsol)
    # Compute fluxes with low metallicity
    SP.params['zmet'] = zmet1
    f1 = abs_ab_mag_to_mjy(
        SP.get_mags(tage=13.8, bands=compute_bands),
        phi[0])
    meta1[0] = SP.stellar_mass
    meta1[1] = SP.dust_mass
    meta1[2] = SP.log_lbol
    meta1[3] = SP.sfr
    meta1[4] = SP.log_age
    # Compute fluxes with high metallicity
    SP.params['zmet'] = zmet2
    f2 = abs_ab_mag_to_mjy(
        SP.get_mags(tage=13.8, bands=compute_bands),
        phi[0])
    meta2[0] = SP.stellar_mass
    meta2[1] = SP.dust_mass
    meta2[2] = SP.log_lbol
    meta2[3] = SP.sfr
    meta2[4] = SP.log_age

    # Interpolate and scale the SED by mass
    model_mjy = 10. ** logm * interp_logz(zmet1, zmet2, logZZsol, f1, f2)

    # Interpolate metadata between the two metallicities
    meta = interp_logz(zmet1, zmet2, logZZsol, meta1, meta2)

    # Compute likelihood
    L = -0.5 * np.sum(
        np.power((model_mjy[band_indices] + B - sed) / err, 2.))

    # Scale statistics by the total mass
    blob = {"logMstar": logm + np.log10(meta[0]),  # log solar masses,
            "logMdust": logm + np.log10(meta[1]),  # log solar masses
            "logLbol": logm * meta[2],  # log solar luminosities
            "logSFR": logm * meta[3],  # star formation rate, M_sun / yr
            "logAge": meta[4],
            "model_sed": model_mjy}  # note: background no included

    return L, blob


class ThreeParamSFH(MultiPixelBaseModel):
    """Model based on a three-parameter star formation history.

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
    pset : dict
        Initialization arguments to :class:`fsps.StellarPopulation`, as a
        dictionary.
    """
    def __init__(self, seds, sed_errs, sed_bands,
                 theta_priors=None,
                 phi_priors=None,
                 compute_bands=None,
                 pset=None):
        super(ThreeParamSFH, self).__init__(pset=pset)
        self._seds = seds
        self._errs = sed_errs
        self._obs_bands = sed_bands
        if compute_bands is None:
            self._compute_bands = self._obs_bands
        else:
            self._compute_bands = compute_bands
        self._theta_priors = theta_priors
        self._phi_priors = phi_priors

        self._theta_params = ['logmass', 'logZZsol',
                              'sf_start', 'logtau', 'const',
                              'dust1', 'dust2']
        self._phi_params = ['d']
