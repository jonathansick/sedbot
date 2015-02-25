#!/usr/bin/env python
# encoding: utf-8
"""
Marginalize over a library to estimate the stellar population of an SED.
"""

import numpy as np
import multiprocessing


class LibraryEstimator(object):
    """Estimate a stellar population by marginalizing the observed SED
    over a library of models to estimate stellar population parameters.

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
    library_h5_file : :class:`h5py.File`
        File object for an HDF5 file. To create an in-memory file, use:
        `h5py.File('in_memory.hdf5', driver='core', backing_store=False)`.
    library_group : str
        (optional) Name of group within the HDF5 table to store the library's
        tables. By default tables are stored in the HDF5 file's root group.
    ncpu : int
        Number of processors to use for computations.
    """
    def __init__(self, obs_flux, obs_err, bands, d,
                 library_h5_file, library_group='/', ncpu=1):
        super(LibraryEstimator, self).__init__()
        self.library_h5_file = library_h5_file
        self.library_group_name = library_group
        self.chisq_data = self._model_ln_likelihood(obs_flux, obs_err,
                                                    bands, d, ncpu)

    def _model_ln_likelihood(self, obs_flux, obs_err, bands, d, ncpu):
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
        result : ndarray
            Structured numpy array with column ``'lnp'`` and ``'mass'``
            giving the un-normalized posterior probability and mass scaling
            for each model template.
        """
        if ncpu > 1:
            pool = multiprocessing.Pool(ncpu)
            _map = pool.map
        else:
            _map = map

        # FIXME this pattern loads everything into memory; maybe not good
        args = []
        bands = tuple(bands)  # to slice by bandpass (columns)
        x = self.library_h5_file[self.library_group_name]['seds'][bands]
        model_flux = x.view(np.float64).reshape(x.shape + (-1,))
        for i in xrange(model_flux.shape[0]):
            args.append((obs_flux, obs_err, model_flux[i, :].flatten()))

        results = _map(LibraryEstimator._compute_lnp, args)
        d = np.dtype([('lnp', np.float), ('mass', np.float)])
        output_data = np.empty(model_flux.shape[0], dtype=d)
        for i, (lnp, mass) in enumerate(results):
            output_data['lnp'][i] = lnp
            output_data['mass'][i] = mass
        return output_data

    def estimate(self, name, p=(20., 50., 80.)):
        """Perform marginalization to generate PDF estimates at the given
        probability thresholds.
        """
        # Get the library parameter values
        # TODO think about how to special-case the mass scaling parameter
        if name == 'mass':
            model_values = self.chisq_data['mass']
        else:
            model_values = \
                self.library_h5_file[self.library_group_name]['params'][name]
        # print('model values', model_values)
        sort = np.argsort(model_values)
        # Build the empirical cumulative distribution function (cdf)
        model_prob = np.exp(self.chisq_data['lnp'][sort])
        # print('model_prob', model_prob)
        p_norm = np.sum(model_prob)
        print('p_norm', p_norm)
        model_prob /= p_norm
        sorted_values = model_values[sort]
        cdf = np.cumsum(model_prob)
        # print('cdf.shape', cdf.shape)
        # print('cdf', cdf)
        # Linearly Interpolate the CDF to get value at each percentile
        percentile_values = np.interp(p, cdf, sorted_values)
        print('percentile_values', percentile_values)
        return percentile_values

    @staticmethod
    def _compute_lnp(args):
        """See da Cunha, Charlot and Elbaz (2008) eq 33 for info."""
        obs_flux, obs_err, model_flux = args
        # This math minimizes chi-sq in the residuals equation; and gives mass
        _a = np.sum(obs_flux * model_flux / obs_err ** 2.)
        _b = np.sum((model_flux / obs_err) ** 2.)
        mass = _a / _b
        residuals = lambda x: (x * model_flux - obs_flux) / obs_err
        lnL = -0.5 * np.sum(residuals(mass) ** 2.)
        return lnL, mass
