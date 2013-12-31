#!/usr/bin/env python
# encoding: utf-8
"""
Model based on the five-parameter star formation history (with mass, distance,
metallicity and attenuation as additional parameters).

Model components are:

1. `mass`: solar masses
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

from sedbot.photconv import ab_to_mjy
from sedbot.zinterp import bracket_logz, interp_logz

# Number of model dimensions
NDIM = 9


def ln_like(theta, obs_mjy, obs_sigma, bands, sp):
    """ln-likelihood function"""
    m = theta[0]
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
    f1 = ab_to_mjy(sp.get_mags(tage=13.8, bands=bands), d)
    # Compute fluxes with high metallicity
    sp.params['zmet'] = zmet2
    f2 = ab_to_mjy(sp.get_mags(tage=13.8, bands=bands), d)
    model_mjy = interp_logz(zmet1, zmet2, logZZsol, f1, f2)
    L = -0.5 * np.sum(np.power((m * model_mjy - obs_mjy) / obs_sigma, 2.))
    return L


def ln_prob(theta, obs_mjy, obs_sigma, bands, sp, prior_funcs):
    """ln-probability function"""
    prior_p = sum(lnp(x) for x, lnp in zip(theta, prior_funcs))
    if not np.isfinite(prior_p):
        return -np.inf, 0.
    lnpost = prior_p + ln_like(theta, obs_mjy, obs_sigma, bands, sp)
    # Scale statistics by the total mass
    m_star = theta[0] * sp.stellar_mass  # solar masses
    m_dust = theta[0] * sp.dust_mass  # solar masses
    lbol = theta[0] * 10. ** sp.log_lbol  # solar luminosities
    sfr = theta[0] * sp.sfr  # star formation rate, M_sun / yr
    age = sp.log_age  # log(age / yr)
    return lnpost, (m_star, m_dust, lbol, sfr, age)
