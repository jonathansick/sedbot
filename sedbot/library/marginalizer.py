#!/usr/bin/env python
# encoding: utf-8
"""
Marginalize over a library to estimate the stellar population of an SED.
"""

import numpy as np
import multiprocessing

from sedbot.utils.timer import Timer


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
        self.d = d
        self._bands = bands
        self._band_indices = None
        self.chisq_data = self._model_ln_likelihood(obs_flux, obs_err,
                                                    bands, d, ncpu)

    @property
    def group(self):
        return self.library_h5_file[self.library_group_name]

    @property
    def bands(self):
        return self.group['seds'].dtype.names

    @property
    def params(self):
        return self.group['params'].dtype.names

    @property
    def meta_params(self):
        return self.group['meta'].dtype.names

    @property
    def band_indices(self):
        """List of indices of observed bands in the library's bands."""
        # caching
        # TODO throw exception if more observed bands than in library
        if self._band_indices is None:
            indices = np.zeros(len(self._bands), dtype=np.int)
            for i, band in enumerate(self._bands):
                idx = self.bands.index(band)
                indices[i] = idx
            self._band_indices = indices
        return self._band_indices

    def estimate(self, name, p=(0.2, 0.5, 0.8)):
        """Perform marginalization to generate PDF estimates at the given
        probability thresholds.
        """
        assert name in self.params
        model_values = self.group['params'][name]
        return self._estimate(model_values, p=p)

    def estimate_mass_scale(self, p=(0.2, 0.5, 0.8)):
        """Estimate of the mass scaling parameter."""
        model_values = self.chisq_data['mass']
        return self._estimate(model_values, p=p)

    def estimate_flux(self, band, p=(0.2, 0.5, 0.8)):
        """Estimate of marginalized model flux in a given band."""
        assert band in self.bands
        # Flux is scaled by mass
        model_values = self.group['seds'][band] * self.chisq_data['mass']
        return self._estimate(model_values, p=p)

    def estimate_ml(self, band, p=(0.2, 0.5, 0.8)):
        """Estimate of the M/L parameter in a given band."""
        assert band in self.bands
        model_values = self.group['mass_light'][band]
        return self._estimate(model_values, p=p)

    def estimate_meta(self, name, p=(0.2, 0.5, 0.8)):
        """Estimate of a metadata parameter (computed stellar
        population values that aren't input parameters or M/L.
        """
        assert name in self.meta_params
        model_values = self.group['meta'][name]
        # Whitelist of metadata that must be scaled by mass;
        # FIXME a better way to do this more reliably?
        if name in ('logMstar', 'logMdust', 'logLbol', 'logSFR'):
            model_values += np.log10(self.chisq_data['mass'])
        return self._estimate(model_values, p=p)

    def _estimate(self, model_values, p=(0.2, 0.5, 0.8)):
        """Perform marginalization to generate PDF estimates at the given
        probability thresholds.

        Parameters
        ----------
        model_values : ndarray
            A 1D ndarray of model values, corresponding to the table of models.
        """
        # TODO can this be trivially extended to 2D model value arrays?
        grid = self._build_histogram_grid(model_values)
        pdf = self._build_pdf(model_values, self.chisq_data['lnp'], grid)
        # convert the PDF to a CDF
        delta = grid[1] - grid[0]
        cdf = np.cumsum(pdf * delta)
        grid_center = 0.5 * (grid[0:-1] + grid[1:])
        # TODO is this the hard part for using 2D model_values inputs?
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
        # this slices only the bands used for observations
        x = self.library_h5_file[self.library_group_name]['seds'][bands]
        model_flux = x.view(np.float64).reshape(x.shape + (-1,))
        for i in xrange(model_flux.shape[0]):
            m_flux = model_flux[i, :].flatten()
            args.append((obs_flux,
                         obs_err,
                         m_flux))

        with Timer() as timer:
            results = _map(_compute_lnp, args)
        print "Marginalization took", timer
        d = np.dtype([('lnp', np.float), ('mass', np.float)])
        output_data = np.empty(model_flux.shape[0], dtype=d)
        for i, (lnp, mass) in enumerate(results):
            output_data['lnp'][i] = lnp
            output_data['mass'][i] = mass
        return output_data


def _compute_lnp(args):
    """See da Cunha, Charlot and Elbaz (2008) eq 33 for info."""
    obs_flux, obs_err, model_flux = args
    # This math minimizes chi-sq in the residuals equation; and gives mass
    _a = np.sum(obs_flux * model_flux / obs_err ** 2.)
    _b = np.sum((model_flux / obs_err) ** 2.)
    mass = _a / _b
    residuals = (mass * model_flux - obs_flux) / obs_err
    lnL = -0.5 * np.sum(residuals ** 2.)
    return lnL, mass
