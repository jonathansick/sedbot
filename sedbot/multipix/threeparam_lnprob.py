#!/usr/bin/env python
# encoding: utf-8
"""
Three-parameter star formation history for the multi-pixel model.

Instantiate one posterior function per compute core.
"""

import numpy as np

import fsps

from sedbot.photconv import abs_ab_mag_to_mjy
from sedbot.zinterp import bracket_logz, interp_logz


class ThreeParamLnProb(object):
    """
    FIXME add addition of background.

    Parameters in theta space are:

    0. logmass
    1. logZZsol
    2. logtau
    3. const
    4. sf_start
    5. dust1
    6. dust2

    Parameters in phi are:

    0. d
    """
    def __init__(self, fsps_params, obs_bands,
                 obs_sed=None, obs_err=None, priors=None,
                 fsps_compute_bands=None):
        super(ThreeParamLnProb, self).__init__()
        self._sp = fsps.StellarPopulation(**fsps_params)
        self._bands = obs_bands
        self._obs_sed = obs_sed
        self._obs_err = obs_err
        self._priors = priors
        self._param_names = ['logmass',
                             'logZZsol',
                             'logtau',
                             'const',
                             'sf_start',
                             'dust1',
                             'dust2']
        # pre-allocate memory
        self._prior_vals = np.empty(self.ndim)
        # Placeholders for metadata at high and low metallicity brackets
        self._meta1 = np.empty(5)
        self._meta2 = np.empty(5)

        # Possible to compute SED in more bands than necessary
        if fsps_compute_bands is None:
            self._compute_bands = self._bands
            self._band_indices = np.ones(len(self._bands))
        else:
            self._compute_bands = fsps_compute_bands
            self._band_indices = np.array([self._compute_bands.index(b)
                                           for b in self._bands])

    @property
    def ndim(self):
        return len(self._param_names)

    def reset_pixel(self, obs_sed, obs_err, priors):
        """Change observation and priors for a new pixel."""
        self._obs_sed = obs_sed
        self._obs_err = obs_err
        self._priors = priors

    def __call__(self, theta, B, phi):
        """Compute the ln posterior probability."""
        # Physically we'd expect dust1 > dust2 if young stars are more embedded
        if theta[5] < theta[6]:
            return -np.inf, np.nan

        # Evaluate priors
        lnprior = 0.
        for i, name in enumerate(self.param_names):
            lnprior += self._priors[name](theta[i])
        if not np.isfinite(lnprior):
            return -np.inf, np.nan

        # Evaluate the ln-likelihood function by interpolating between
        # metallicty bracket
        logm = theta[0]  # logmass
        logZZsol = theta[1]
        self._sp.params['tau'] = 10. ** theta[2]
        self._sp.params['const'] = theta[3]
        self._sp.params['sf_start'] = theta[4]
        self._sp.params['dust1'] = theta[5]
        self._sp.params['dust2'] = theta[6]
        zmet1, zmet2 = bracket_logz(logZZsol)
        # Compute fluxes with low metallicity
        self._sp.params['zmet'] = zmet1
        f1 = abs_ab_mag_to_mjy(
            self._sp.get_mags(tage=13.8, bands=self._compute_bands),
            phi[0])
        self._meta1[0] = self._sp.stellar_mass
        self._meta1[1] = self._sp.dust_mass
        self._meta1[2] = self._sp.log_lbol
        self._meta1[3] = self._sp.sfr
        self._meta1[4] = self._sp.log_age
        # Compute fluxes with high metallicity
        self._sp.params['zmet'] = zmet2
        f2 = abs_ab_mag_to_mjy(
            self._sp.get_mags(tage=13.8, bands=self._compute_bands),
            phi[0])
        self._meta2[0] = self._sp.stellar_mass
        self._meta2[1] = self._sp.dust_mass
        self._meta2[2] = self._sp.log_lbol
        self._meta2[3] = self._sp.sfr
        self._meta2[4] = self._sp.log_age

        # Interpolate and scale the SED by mass
        model_mjy = 10. ** logm * interp_logz(zmet1, zmet2, logZZsol, f1, f2)

        # Interpolate metadata between the two metallicities
        meta = interp_logz(zmet1, zmet2, logZZsol, self._meta1, self._meta2)

        # Compute likelihood
        L = -0.5 * np.sum(
            np.power((model_mjy[self._band_indices] + B
                     - self._obs_sed) / self._obs_err, 2.))

        # Compute ln posterior probability
        lnpost = lnprior + L
        if not np.isfinite(lnpost):
            lnpost = -np.inf

        # Scale statistics by the total mass
        log_m_star = theta[0] + np.log10(meta[0])  # log solar masses
        log_m_dust = theta[0] + np.log10(meta[1])  # log solar masses
        log_lbol = theta[0] * meta[2]  # log solar luminosities
        log_sfr = theta[0] * meta[3]  # star formation rate, M_sun / yr
        log_age = meta[4]  # log(age / yr)
        blob = (lnpost, log_m_star, log_m_dust, log_lbol, log_sfr, log_age,
                model_mjy)
        return lnpost, blob
