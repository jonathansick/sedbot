#!/usr/bin/env python
# encoding: utf-8
"""
Marginalize over a library to estimate the stellar population of an SED.
"""

import numpy as np
import multiprocessing

from scipy.optimize import leastsq


class LibraryMarginalizer(object):
    """Estimate a stellar population by marginalizing the observed SED
    over a library of models.

    Parameters
    ----------
    h5_file : :class:`h5py.File`
        File object for an HDF5 file. To create an in-memory file, use:
        `h5py.File('in_memory.hdf5', driver='core', backing_store=False)`.
    group : str
        (optional) Name of group within the HDF5 table to store the library's
        tables. By default tables are stored in the HDF5 file's root group.
    """
    def __init__(self, h5_file, group='/'):
        super(LibraryMarginalizer, self).__init__()
        self.h5_file = h5_file
        self.group_name = group

    def model_ln_likelihood(self, obs_flux, obs_err, bands, d, ncpu=1):
        """Compute ln likelihood of model SEDs given observations.

        Parameters
        ----------
        obs_flux : ndarray
            Observed flux, in apparent micro Janskies.
        obs_err : ndarray
            Observed flux error, in apparent micro Janskies.
        bands : list
            FSPS bandpass names.
        d : float
            Distance, in parsecs.
        ncpu : int
            Number of processors to use for computations.

        Returns
        -------
        result : todo
            Table of ln likelihood for each model
        """
        if ncpu > 1:
            pool = multiprocessing.Pool(ncpu)
            _map = pool.map
        else:
            _map = map

        # FIXME this pattern loads everything into memory; maybe not good
        args = []
        bands = tuple(bands)  # to slice by bandpass (columns)
        x = self.h5_file[self.group_name]['seds'][bands]
        model_flux = x.view(np.float64).reshape(x.shape + (-1,))
        for i in xrange(model_flux.shape[0]):
            args.append((obs_flux, obs_err, model_flux[i, :].flatten()))

        results = _map(_compute_lnp, args)
        print results


def _compute_lnp(args):
    obs_flux, obs_err, model_flux = args
    residuals = lambda x: (x * model_flux - obs_flux) / obs_err
    info = leastsq(residuals, 1.)
    mass = float(info[0])
    lnL = - np.sum(residuals(mass) ** 2.)
    return lnL, mass
