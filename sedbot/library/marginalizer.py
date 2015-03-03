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

    def estimate(self, name, p=(0.2, 0.5, 0.8)):
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
        grid = self._build_histogram_grid(model_values)
        pdf = self._build_pdf(model_values, self.chisq_data['lnp'], grid)
        # convert the PDF to a CDF
        delta = grid[1] - grid[0]
        cdf = np.cumsum(pdf * delta)
        grid_center = 0.5 * (grid[0:-1] + grid[1:])
        percentile_values = np.interp(p, cdf, grid_center)
        return percentile_values

    def _build_histogram_grid(self, model_values, n_elements=1500):
        grid = np.linspace(model_values.min(), model_values.max(),
                           num=n_elements)
        return grid

    def _build_pdf(self, model_values, lnp, grid):
        pdf, _ = np.histogram(model_values, bins=grid, weights=np.exp(lnp),
                              density=True)
        return pdf

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
