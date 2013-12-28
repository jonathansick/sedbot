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
from sedbot.modeltools import reset_seed_limits
from sedbot.zinterp import bracket_logz, interp_logz

# Number of model dimensions
NDIM = 9


def init_chain(n_walkers, d0, m0, limits, **pset):
    """Init chain position given the number of walkers.
    
    Parameters
    ----------
    n_walkers : int
        Number of `emcee` walkers.
    d0 : float
        Initial guess at the distance in parsecs
    m0 : float
        Initial guess at mass.
    limits : dict
        Dictionary of lower and upper bounds of each parameter.
    """
    p0 = np.random.randn(NDIM * n_walkers).reshape((n_walkers, NDIM))

    # mass start point
    p0[:, 0] = p0[:, 0] + m0
    if 'mass' in limits:
        reset_seed_limits(p0[:, 0], *limits['mass'])

    # metallicity start point
    p0[:, 1] = 0.1 * p0[:, 1] - 0.
    if 'logZZsol' in limits:
        reset_seed_limits(p0[:, 1], *limits['logZZsol'])

    # distance start point
    p0[:, 2] = 10. * p0[:, 2] + d0
    if 'd' in limits:
        reset_seed_limits(p0[:, 2], *limits['d'])

    # tau start point
    p0[:, 3] = p0[:, 3] + 10.
    if 'tau' in limits:
        reset_seed_limits(p0[:, 3], *limits['tau'])

    # const start point
    p0[:, 4] = p0[:, 4] + 2.
    if 'const' in limits:
        reset_seed_limits(p0[:, 4], *limits['const'])

    # sf_start start point
    p0[:, 5] = p0[:, 5] + 2.
    if 'sf_start' in limits:
        reset_seed_limits(p0[:, 5], *limits['sf_start'])

    # tburst start point
    p0[:, 6] = p0[:, 6] + 6.
    if 'tburst' in limits:
        reset_seed_limits(p0[:, 6], *limits['tburst'])

    # fburst start point
    p0[:, 7] = 0.1 * p0[:, 7] + 0.2
    if 'fburst' in limits:
        reset_seed_limits(p0[:, 7], *limits['fburst'])

    # dust2 attenuation start point
    p0[:, 8] = 0.1 * p0[:, 8] + 0.2
    if 'dust2' in limits:
        reset_seed_limits(p0[:, 8], *limits['dust2'])

    return p0


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
        return -np.inf
    return prior_p + ln_like(theta, obs_mjy, obs_sigma, bands, sp)
