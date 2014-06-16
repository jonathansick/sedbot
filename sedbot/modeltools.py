#!/usr/bin/env python
# encoding: utf-8
"""
Utilities for setting up models and working with emcee samplers.
"""

import time
import numpy as np

from astropy.table import Table


from sedbot.photconv import abs_ab_mag_to_mjy
from sedbot.zinterp import bracket_logz, interp_logz


def init_chain(n_walkers, x0, x_sigma, limits):
    """Init chain position given the number of walkers.
    
    Parameters
    ----------
    n_walkers : int
        Number of `emcee` walkers.
    x0 : length-ndim iterable
        List of initial guesses for each parameter
    x_sigma : length-ndim iterable
        List of dispersions around initial guesses to start chains from.
    limits : length-ndim iterable
        List of length-2 tuples giving the lower and upper bounds of each
        parameter. This ensures that each chain is initialized in a valid
        point of parameter space.

    Returns
    -------
    p0 : (n_walkers, n_dim) ndarray
        Initial position for the emcee chain.
    """
    ndim = len(x0)
    assert len(x0) == len(x_sigma) == len(limits), "Lengths do not match"
    p0 = np.random.randn(ndim * n_walkers).reshape((n_walkers, ndim))
    for i, (x, sigma, lim) in enumerate(zip(x0, x_sigma, limits)):
        p0[:, i] = sigma * p0[:, i] + x
        reset_seed_limits(p0[:, i], *lim)
    return p0


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
    f1 = abs_ab_mag_to_mjy(sp.get_mags(tage=13.8, bands=bands), d0)
    sp.params['zmet'] = zmet2
    f2 = abs_ab_mag_to_mjy(sp.get_mags(tage=13.8, bands=bands), d0)
    mock_mjy = interp_logz(zmet1, zmet2, logZZsol, f1, f2)
    if isinstance(mag_sigma, float):
        mag_sigma = np.ones(len(bands)) * mag_sigma
    mock_sigma = (mock_mjy * mag_sigma) / 1.0875
    if apply_errors:
        nbands = len(bands)
        mock_mjy += mock_sigma * np.random.normal(loc=0.0, scale=1.0,
                                                  size=nbands)
    return mock_mjy, mock_sigma


def make_flatchain(sampler, param_names, bands, metadata=None,
                   n_burn=0, append_mstar=False, append_mdust=False,
                   append_lbol=False, append_sfr=False, append_age=False,
                   append_model_sed=False, append_lnpost=False):
    """Create an Astropy Table of 'flatchain' of emcee walkers, removing any
    burn-in steps, and appending blob metadata.

    Parameters
    ----------
    sampler : obj
        An `emcee` sampler.
    param_names : list
        List of strings identifying each parameter, and thus columns in
        the table.
    bands : list
        List of names of bandpasses (as defined for python-fsps).
    metadata : dict
        Optional dictonary of metadata to persist with the table.
    n_burn : int
        Number of burn-in steps.
    param_names : str
        Names of the parameters being sampled.
    append_mstar : bool
        Append stellar mass to the chain.
    append_mdust : bool
        Append dust mass to the chain.
    append_lbol : bool
        Append bolometric luminosity to the chain.
    append_sfr : bool
        Append the star formation rate to the chain.
    append_model_sed : bool
        Append the model SED (units of µJy); an array with modelled
        fluxes in each bandpass.

    Returns
    -------
    tbl : `astropy.table.Table`
        The flattened chain as an astropy Table with burn-in steps removed and
        (if applicable) stellar population metadata appended.
    """
    # Construction of the data type
    n_bands = len(bands)
    dt = [(n, float) for n in param_names]

    nwalkers, nsteps, ndim = sampler.chain.shape
    flatchain_arr = sampler.chain[:, n_burn:, :].reshape((-1, ndim))

    # Add blob columns to the data type
    blob_names = []
    blob_index = []
    dt.append(('lnpost', float))
    blob_names.append('lnpost')
    blob_index.append(0)
    if append_mstar:
        dt.append(('logMstar', float))
        blob_names.append('logMstar')
        blob_index.append(1)
    if append_mdust:
        dt.append(('logMdust', float))
        blob_names.append('logMdust')
        blob_index.append(2)
    if append_lbol:
        dt.append('logLbol', float)
        blob_names.append('logLbol')
        blob_index.append(3)
    if append_sfr:
        dt.append('logsfr', float)
        blob_names.append('logsfr')
        blob_index.append(4)
    if append_age:
        dt.append('logage', float)
        blob_names.append('logage')
        blob_index.append(5)
    if append_model_sed:
        dt.append(('model_sed', float, n_bands))
        blob_names.append('model_sed')
        blob_index.append(6)

    # Make an empty flatchain and fill
    flatchain = np.empty(flatchain_arr.shape[0], dtype=np.dtype(dt))
    for i, n in enumerate(param_names):
        flatchain[n][:] = flatchain_arr[:, i]
    if len(blob_names) > 0:
        blobs = sampler.blobs

        # Flatten the blob list and append it too
        n_steps = nsteps - n_burn
        for i in xrange(n_steps):
            for j in xrange(nwalkers):
                for k, n in zip(blob_index, blob_names):
                    flatchain[n][i * j] = blobs[i + n_burn][j][k]

    if metadata is None:
        metadata = {}
    metadata.update({"bandpasses": bands})
    tbl = Table(flatchain, meta=metadata)
    # Bad posterior samples are given values of 0 by emcee; so filter them
    bad = np.where((flatchain['lnpost'] >= 0.))[0]
    for n in tbl.keys():
        tbl[n][bad] = np.nan
    return tbl


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
