#!/usr/bin/env python
# encoding: utf-8
"""
Emcee-based sampler for singlepix SED MCMC.
"""

import time
import numpy as np
from collections import OrderedDict

import emcee
import fsps
from astropy.table import Table, hstack, Column

from sedbot.photconv import micro_jy_to_luminosity
from sedbot.chain import SinglePixelChain


class SinglePixelSampler(object):
    """Emcee-based sampler for single pixel SEDs.

    Parameters
    ----------
    model : object
        A model instance (see :mod:`sedbot.models`).
    n_walkers : int
        Number of emcee walkers.
    """
    def __init__(self, model, n_walkers=100):
        super(SinglePixelSampler, self).__init__()
        self.model = model
        self.n_walkers = n_walkers
        self._sampler = None
        self._run_time = 0.
        self._call_time = np.nan

    def generate_initial_point(self, x0, x_sigma):
        """Generate a chain starting point given an array of Gaussian
        dispersions to build balls from for each parameter.

        Parameters
        ----------
        x0 : ndarray
            Initial points in parameter space. Shape ``(n_params,)``.
        x_sigma : ndarray
            Gaussian standard deviations to disperse starting points for each
            walkers. Shape ``(n_params,)``.
        """
        ndim = len(x0)
        assert len(x0) == len(x_sigma), "Lengths do not match"
        assert ndim == self.model.n_params
        p0 = np.random.randn(ndim * self.n_walkers).reshape((self.n_walkers,
                                                             ndim))
        for i, (x, sigma) in enumerate(zip(x0, x_sigma)):
            p0[:, i] = sigma * p0[:, i] + x
            priorf = self.model.priors[self.model.param_names[i]]
            p0[:, i][p0[:, i] < priorf.lower_limit] = priorf.lower_limit
            p0[:, i][p0[:, i] > priorf.upper_limit] = priorf.upper_limit
        return p0

    def sample(self, n_steps, theta0=None, chain=None):
        """Sample for `n_steps` Emcee steps.

        Parameters
        ----------
        n_steps : int
            Number of emcee steps. The total number of samples will be
            ``n_steps x n_walkers``.
        theta0 : ndarray
            Initial locations of the walkers; shape
        """
        self.sampler = emcee.EnsembleSampler(
            self.n_walkers,
            self.model.n_params, self.model)

        # Do burn-in + run
        with EmceeTimer(n_steps, self.n_walkers) as emcee_timer:
            self.sampler.run_mcmc(theta0, n_steps)
        print(emcee_timer)

        self._run_time += emcee_timer.interval
        self._call_time = emcee_timer.seconds_per_call

    @property
    def table(self):
        """An :class:`astropy.table.Table` with the chain."""
        if self.sampler is None:
            return None
        msuns = np.array([fsps.get_filter(n).msun_ab
                          for n in self.model.computed_bands])
        meta = OrderedDict((
            ('observed_bands', self.model.observed_bands),
            ('instruments', self.model.instruments),
            ('computed_bands', self.model.computed_bands),
            ('msun_ab', msuns),
            ('d', self.model.d),  # expected distance in parsecs
            ('band_indices', self.model.band_indices),
            ('theta_params', self.model.param_names),
            ('compute_time', self._run_time),
            ('step_time', self._call_time),
            ('sed', self.model._sed),
            ('sed_err', self.model._err),
            ('pixels', self.model.pixel_metadata),
            ('area', self.model._area),
            ('n_walkers', self.n_walkers),
            ("f_accept", self.sampler.acceptance_fraction),
            ("acor", self.sampler.acor)))

        # Convert flatchain into a structured array
        nwalkers, nsteps, ndim = self.sampler.chain.shape
        flatchain_arr = self.sampler.chain[:, :, :].reshape((-1, ndim))
        dt = [(n, np.float) for n in self.model.param_names]
        flatchain = np.empty(flatchain_arr.shape[0], dtype=np.dtype(dt))
        for i, n in enumerate(self.model.param_names):
            flatchain[n][:] = flatchain_arr[:, i]

        # Flatten the blob list and make a structured array
        blobs = self.sampler.blobs
        blobchain = np.empty(nwalkers * nsteps, self.model.blob_dtype)
        blobchain.fill(np.nan)
        for i in xrange(nsteps):
            for j in xrange(nwalkers):
                for k in self.model.blob_dtype.names:
                    blobchain[k][i * self.n_walkers + j] = blobs[i][j][k]

        chain_table = Table(flatchain, meta=meta)
        blob_table = Table(blobchain)
        tbl = SinglePixelChain(hstack((chain_table, blob_table),
                                      join_type='exact'))

        # Add M/L computations for each computed band.
        for i, (band_name, msun) in enumerate(zip(self.model.computed_bands,
                                                  msuns)):
            # Either use expected distance or the chain distance
            # FIXME fragile code
            if 'd' in self.model.param_names:
                d = np.array(tbl['d'])
            else:
                d = self.model.d
            logLsol = micro_jy_to_luminosity(tbl['model_sed'][:, i], msun, d)
            ml = tbl['logMstar'] - logLsol
            colname = "logML_{0}".format(band_name)
            tbl.add_column(Column(name=colname, data=ml))

        return tbl


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
                                              ct=ct, cunit=cunit,
                                              enddate=enddate)

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

    @property
    def interval(self):
        """The timer interval in secods."""
        return self._interval
