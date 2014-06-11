#!/usr/bin/env python
# encoding: utf-8
"""
Model based on the five-parameter star formation history (with mass, distance,
metallicity and attenuation as additional parameters).

Model components are:

1. `logmass`: log10 solar mass
2. `log(Z/Z_sol)`: logarithmic metallicity
3. `d`: distance in parsecs
4. `tau`: e-folding time
5. `const`: constant star forming rate component
6. `sf_start`: time of oldest star formation, in Gyr after the Big Bang.
7. `tburst`: time of star formation burst (in Gyr after Big Bang).
8. `fburst`: fraction of final stellar mass generated in burst.
9. `dust2`: attenuation due to dust (with Calzetti 2000 curve).

In this model, MCMC sampling is done for each of these nine parameters.
"""

import numpy as np

from sedbot.photconv import abs_ab_mag_to_mjy
from sedbot.zinterp import bracket_logz, interp_logz

# Number of model dimensions
NDIM = 9


def ln_prob(theta, obs_mjy, obs_sigma, bands, sp, prior_funcs):
    """ln-probability function
    
    Returns
    -------
    lnpost : float
        Posterior probability value
    blob : tuple
        FSPS metadata for this likelihood call. Includes data on stellar mass,
        dust mass, as well as the modelled SED (in µJy).
    """
    # Placeholders for metadata at high and low metallicity brackets
    meta1 = np.empty(5)
    meta2 = np.empty(5)

    # Evaluate priors
    prior_p = sum(lnp(x) for x, lnp in zip(theta, prior_funcs))
    if not np.isfinite(prior_p):
        return -np.inf, 0.

    # Evaluate the ln-likelihood function by interpolating between metallicty
    # bracket
    logm = theta[0]  # logmass
    logZZsol = theta[1]
    d = theta[2]
    sp.params['tau'] = theta[3]
    sp.params['const'] = theta[4]
    sp.params['sf_start'] = theta[5]
    sp.params['tburst'] = theta[6]
    sp.params['fburst'] = theta[7]
    sp.params['dust2'] = theta[8]
    zmet1, zmet2 = bracket_logz(logZZsol)
    # Compute fluxes with low metallicity
    sp.params['zmet'] = zmet1
    f1 = abs_ab_mag_to_mjy(sp.get_mags(tage=13.8, bands=bands), d)
    meta1[0] = sp.stellar_mass
    meta1[1] = sp.dust_mass
    meta1[2] = sp.log_lbol
    meta1[3] = sp.sfr
    meta1[4] = sp.log_age
    # Compute fluxes with high metallicity
    sp.params['zmet'] = zmet2
    f2 = abs_ab_mag_to_mjy(sp.get_mags(tage=13.8, bands=bands), d)
    meta2[0] = sp.stellar_mass
    meta2[1] = sp.dust_mass
    meta2[2] = sp.log_lbol
    meta2[3] = sp.sfr
    meta2[4] = sp.log_age
    model_mjy = interp_logz(zmet1, zmet2, logZZsol, f1, f2)
    meta = interp_logz(zmet1, zmet2, logZZsol, meta1, meta2)

    L = -0.5 * np.sum(
        np.power((10. ** logm * model_mjy - obs_mjy) / obs_sigma, 2.))

    # Compute ln posterior probability
    # L, model_sed = ln_like(theta, obs_mjy, obs_sigma, bands, sp)
    lnpost = prior_p + L

    # Scale statistics by the total mass
    log_m_star = theta[0] + np.log10(meta[0])  # log solar masses
    log_m_dust = theta[0] + np.log10(meta[1])  # log solar masses
    log_lbol = theta[0] * meta[2]  # log solar luminosities
    log_sfr = theta[0] * meta[3]  # star formation rate, M_sun / yr
    log_age = meta[4]  # log(age / yr)
    blob = (log_m_star, log_m_dust, log_lbol, log_sfr, log_age, model_mjy)
    return lnpost, blob
