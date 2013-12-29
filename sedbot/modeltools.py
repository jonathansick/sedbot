#!/usr/bin/env python
# encoding: utf-8
"""
Utilities for setting up models and working with emcee samplers.
"""

import time
import numpy as np


from sedbot.photconv import ab_to_mjy
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
    f1 = ab_to_mjy(sp.get_mags(tage=13.8, bands=bands), d0)
    sp.params['zmet'] = zmet2
    f2 = ab_to_mjy(sp.get_mags(tage=13.8, bands=bands), d0)
    mock_mjy = interp_logz(zmet1, zmet2, logZZsol, f1, f2)
    if isinstance(mag_sigma, float):
        mag_sigma = np.ones(len(bands)) * mag_sigma
    mock_sigma = (mock_mjy * mag_sigma) / 1.0875
    if apply_errors:
        nbands = len(bands)
        mock_mjy += mock_sigma * np.random.normal(loc=0.0, scale=1.0,
                size=nbands)
    return mock_mjy, mock_sigma


def burnin_flatchain(sampler, n_burn):
    """Create a 'flatchain' of emcee walkers, removing the burn-in steps.
    
    Parameters
    ----------
    sampler : obj
        An `emcee` sampler.
    n_burn : int
        Number of burn-in steps.

    Returns
    -------
    flatchain : ndarray (nsteps, ndim)
        The flattened chain with burn-in steps removed.
    """
    nwalkers, nsteps, ndim = sampler.chain.shape
    return sampler.chain[:, n_burn:, :].reshape((-1, ndim))


def reset_seed_limits(start_points, lower, upper):
    """In-place replacement of starting point seed values that are above/below
    specified limits.
    
    This function is useful in :func:`init_chain` functions. If initial points
    are generated from a Gaussian distribution, then they may randomly scatter
    below or above hard physical limits. This fnction resets those low/high
    values to be precisely the limiting value.

    Parameters
    ----------
    start_points : ndarray
        Array view of starting points. Values are replaced in-place (there is
        no return value).
    lower : float
        Low limit.
    upper : float
        High limit.
    """
    start_points[start_points < lower] = lower
    start_points[start_points > upper] = upper


class EmceeTimer(object):
    """Timer for emcee runs. Computes total run time and mean time of each
    likelihood call.

    Example::

    >>> with EmceeTimer(n_steps, n_walkers) as emceet:
    >>>     sampler.run_mcmc(p0, n_steps)
    >>> print emceet

    Parameters
    ----------
    nsteps : int
        Number of emcee steps.
    nwalkers : int
        Number of emcee walkers.
    """
    def __init__(self, nsteps, nwalkers):
        super(EmceeTimer, self).__init__()
        self._nsteps = nsteps
        self._nwalkers = nwalkers
        self._start = None
        self._stop = None
        self._endtime = None
        self._interval = None

    def __enter__(self):
        self._start = time.clock()
        return self

    def __exit__(self, *args):
        self._end = time.clock()
        self._endtime = time.localtime()
        self._interval = self._end - self._start

    def __str__(self):
        dt, unit = self._human_time(self._interval)
        ct, cunit = self._human_time(self.seconds_per_call)
        enddate = time.strftime('%Y-%m-%d %H:%M:%S', self._endtime)
        l1 = "Finished emcee run at {enddate}"
        l2 = "\tRun time: {dt} {unit}"
        l3 = "\t{ct} {cunit} per call"
        return "\n".join((l1, l2, l3)).format(dt=dt, unit=unit,
                ct=ct, cunit=cunit, enddate=enddate)

    def _human_time(self, interval):
        """Return a time interval scaled to human-usable units."""
        if interval < 60.:
            dt, unit = interval, "seconds"
        if interval >= 60.:
            dt, unit = interval / 60., "minutes"
        elif interval >= 3600.:
            dt, unit = interval / 3600., "hours"
        elif interval >= 86400:
            dt, unit = interval / 86400., "days"
        return dt, unit

    @property
    def seconds_per_call(self):
        """The mean number of seconds elapsed per likelihood call.
        
        Returns
        -------
        interval : float
            Description
        """
        return self._interval / float(self._nsteps * self._nwalkers)
