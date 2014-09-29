#!/usr/bin/env python
# encoding: utf-8
"""
Models to use with the Gibbs multipixel sampler.
"""

import itertools

import numpy as np

# import fsps

from sedbot.photconv import abs_ab_mag_to_mjy
from sedbot.zinterp import bracket_logz, interp_logz

# Module-level stellar population
SP = None


class MultiPixelBaseModel(object):
    """Baseclass for multipixel models."""
    def __init__(self):
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

        # Prior function dictionaries
        self._theta_priors = []  # list for each pixel
        self._phi_priors = {}  # dict keyed by prior key

        # Function or class method for computing pixel likelihoods
        self._lnlike_fcn = interp_z_likelihood

        # Map processing function
        self._M = map  # or an ipython cluster

    def sample_pixel(self, theta_i, phi, ipix):
        """Compute the ln-prob of a proposed step in local parameter
        space (theta) for a single pixel, ``ipix``.

        Parameters
        ----------
        theta_i : ndarray
            Parameters in theta space for this pixel
        phi : ndarray
            Global parameters
        ipix : int
            Specify the pixel index

        Returns
        -------
        lnprob : float
            Ln probability.
        blob : ndarray
            Structured array with metadata for this pixel sample.
        """
        lnprior = self._ln_prior(theta_i, phi, ipix)
        if not np.isfinite(lnprior):
            return -np.inf, np.nan

        lnL, blob = self._lnlike_fcn((theta_i,
                                      self._theta_params,
                                      phi,
                                      self._phi_params,
                                      self._seds[ipix, :],
                                      self._errs[ipix, :]))
        lnp = lnprior + lnL
        if not np.isfinite(lnp):
            return -np.inf, np.nan
        else:
            return lnL, blob

    def sample_global(self, theta, phi):
        """Compute the ln-prob of a proposed step in global parameter
        space (phi).

        Parameters
        ----------
        theta : ndarray
            Parameters in theta space for all pixels,
            ``(nparams, npix)`` shape.
        phi : ndarray
            Global parameters
        ipix : int
            Specify the pixel index

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
        for ipix in xrange(self._npix):
            lnprior = self._ln_prior(theta[:, ipix], phi, ipix)
            if not np.isfinite(lnprior):
                return -np.inf
            lnpriors.append(lnprior)

        args = []
        for ipix in xrange(self._npix):
            args.append((theta[:, ipix],
                         self._theta_params,
                         phi,
                         self._phi_params,
                         self._seds[ipix, :],
                         self._errs[ipix, :]))
        results = self._M(self._lnlike_fcn, args)

        lnprob = 0.
        pixel_lnprobs = []
        blobs = []
        for lnprior, result in zip(lnpriors, results):
            lnL, blob = result
            lnprob += lnL + lnprior
            pixel_lnprobs.append(lnprob)
            blobs.append(blobs)
        return lnprob, pixel_lnprobs, blobs

    def update_background(self, theta, phi):
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

        Returns
        -------
        B : ndarray
            New background vector, shape ``(n_bands,)``.
        """
        pass


def interp_z_likelihood(args):
    """Generic single-pixel likeihood function where a single metallicity
    is linearly interpolated.
    """
    global SP

    theta, theta_names, phi, phi_names, sed, err, sed_bands, compute_bands \
        = args

    meta1 = np.empty(5, dtype=float)
    meta2 = np.empty(5, dtype=float)

    band_indices = np.array([compute_bands.index(b) for b in sed_bands])

    # Evaluate the ln-likelihood function by interpolating between
    # metallicty bracket
    for name, val in itertools.chain(zip(theta, theta_names),
                                     zip(phi, phi_names)):
        if name == 'logmass':
            logm = val
        elif name == 'logZZsol':
            logZZsol = val
        elif name == 'logtau':
            SP.params['tau'] = 10. ** val
        elif name == 'B':
            B = np.array(val)
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
    log_m_star = logm + np.log10(meta[0])  # log solar masses
    log_m_dust = logm + np.log10(meta[1])  # log solar masses
    log_lbol = logm * meta[2]  # log solar luminosities
    log_sfr = logm * meta[3]  # star formation rate, M_sun / yr
    log_age = meta[4]  # log(age / yr)
    blob = (log_m_star, log_m_dust, log_lbol, log_sfr, log_age,
            model_mjy)

    return L, blob
