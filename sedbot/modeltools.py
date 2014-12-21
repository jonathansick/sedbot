#!/usr/bin/env python
# encoding: utf-8
"""
Utilities for working with models (the astrophysics complement to `mcmctools`).
"""

import numpy as np

from sedbot.photconv import abs_ab_mag_to_micro_jy
from sedbot.zinterp import bracket_logz, interp_logz


def mock_dataset(sp, bands, d0, m0, logZZsol, mag_sigma, apply_errors=False):
    """Generate a mock SED given the FSPS stellar population model.

    Parameters
    ----------
    sp : :class:`fsps.StellarPopulation`
        A python-fsps StellarPopulation instance with parameters pre-set.
    bands : iterable
        A list of bandpass names, as strings (see python-fsps documentation.
    d0 : float
        Distance in parsecs.
    m0 : float
        Mass of stellar population (solar masses).
    logZZsol : float
        Metallicity of stellar population, :math:`log(Z/Z_\odot)`. This
        parameters, rather than the FSPS `zmet` parameter is used so that
        a stellar population of an arbitrary metallicity can be logarithmically
        interpolated from two bracketing isochrones.
    mag_sigma : float or (nbands,) iterable
        Photometric uncertainty of each bandpass, in magnitudes. If a single
        float is passed then that uncertainty is used for each bandpass.
        Otherwise, it must be an array of uncertainties matching the number
        of bands.
    apply_errors : bool
        If true, then Gaussian errors, specified by `mag_sigma` will be applied
        to the SED.

    Returns
    -------
    mock_mjy : ndarray
        SED, in micro-Janskies.
    mock_sigma : ndarray
        SED uncertainties, in micro-Janskies.
    """
    zmet1, zmet2 = bracket_logz(logZZsol)
    sp.params['zmet'] = zmet1
    f1 = abs_ab_mag_to_micro_jy(sp.get_mags(tage=13.8, bands=bands), d0)
    sp.params['zmet'] = zmet2
    f2 = abs_ab_mag_to_micro_jy(sp.get_mags(tage=13.8, bands=bands), d0)
    mock_mjy = m0 * interp_logz(zmet1, zmet2, logZZsol, f1, f2)
    if isinstance(mag_sigma, float):
        mag_sigma = np.ones(len(bands)) * mag_sigma
    mock_sigma = (mock_mjy * mag_sigma) / 1.0875
    if apply_errors:
        nbands = len(bands)
        mock_mjy += mock_sigma * np.random.normal(loc=0.0, scale=1.0,
                                                  size=nbands)
    return mock_mjy, mock_sigma
