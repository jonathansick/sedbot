#!/usr/bin/env python
# encoding: utf-8
"""
Utilities for setting up models.
"""

import time


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
