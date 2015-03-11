#!/usr/bin/env python
# encoding: utf-8
"""
Marginalize over a library to estimate the stellar population of an SED.
"""

import numpy as np

# from sedbot.utils.timer import Timer


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
        self._p = np.exp(self.chisq_data['lnp'])

        self._param_sort_cache = {}
        self._flux_sort_cache = {}
        self._ml_sort_cache = {}
        self._meta_sort_cache = {}
        self._mass_sort_cache = None
        self._param_cdf_cache = {}
        self._flux_cdf_cache = {}
        self._ml_cdf_cache = {}
        self._meta_cdf_cache = {}
        self._mass_cdf_cache = None

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
        if name not in self._param_sort_cache:
            srt, cdf = self._build_cdf(model_values)
            self._param_sort_cache[name] = srt
            self._param_cdf_cache[name] = cdf
        return self._estimate(model_values,
                              srt=self._param_sort_cache[name],
                              cdf=self._param_cdf_cache[name],
                              p=p)

    def estimate_mass_scale(self, p=(0.2, 0.5, 0.8)):
        """Estimate of the mass scaling parameter."""
        model_values = self.chisq_data['mass']
        if self._mass_sort_cache is None or self._mass_cdf_cache is None:
            srt, cdf = self._build_cdf(model_values)
            self._mass_sort_cache = srt
            self._mass_cdf_cache = cdf
        return self._estimate(model_values,
                              srt=self._mass_sort_cache,
                              cdf=self._param_cdf_cache,
                              p=p)

    def estimate_flux(self, band, p=(0.2, 0.5, 0.8)):
        """Estimate of marginalized model flux in a given band."""
        assert band in self.bands
        # Flux is scaled by mass
        model_values = self.group['seds'][band] * self.chisq_data['mass']
        if band not in self._flux_sort_cache:
            srt, cdf = self._build_cdf(model_values)
            self._flux_sort_cache[band] = srt
            self._flux_cdf_cache[band] = cdf
        return self._estimate(model_values,
                              srt=self._flux_sort_cache[band],
                              cdf=self._flux_cdf_cache[band],
                              p=p)

    def estimate_ml(self, band, p=(0.2, 0.5, 0.8)):
        """Estimate of the M/L parameter in a given band."""
        assert band in self.bands
        model_values = self.group['mass_light'][band]
        if band not in self._ml_sort_cache:
            srt, cdf = self._build_cdf(model_values)
            self._ml_sort_cache[band] = srt
            self._ml_cdf_cache[band] = cdf
        return self._estimate(model_values,
                              srt=self._ml_sort_cache[band],
                              cdf=self._ml_cdf_cache[band],
                              p=p)

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
        if name not in self._meta_sort_cache:
            srt, cdf = self._build_cdf(model_values)
            self._meta_sort_cache[name] = srt
            self._meta_cdf_cache[name] = cdf
        return self._estimate(model_values,
                              srt=self._meta_sort_cache[name],
                              cdf=self._meta_cdf_cache[name],
                              p=p)

    def _estimate(self, model_values, srt=None, cdf=None, p=(0.2, 0.5, 0.8)):
        """Perform marginalization to generate PDF estimates at the given
        probability thresholds.

        Parameters
        ----------
        model_values : ndarray
            A 1D ndarray of model values, corresponding to the table of models.
        """
        if srt is None or cdf is None:
            srt, cdf = self._build_cdf(model_values)
        # Find the values at each probability
        # values_at_p = np.interp(p, cdf, model_values[srt])
        values_at_p = np.empty(len(p), dtype=np.float)
        # trying argmin instead of interpolation
        for i in xrange(len(p)):
            values_at_p[i] = model_values[np.argmin(np.power(cdf - p[i], 2.))]
        return values_at_p

    def _build_cdf(self, model_values):
        srt = np.argsort(model_values)
        # use float32 to preserve memory
        cdf = np.cumsum(self._p[srt], dtype=np.float32)
        # Normalize the cdf to a total probability of 1.
        cdf /= cdf[-1]
        # assert np.all(np.diff(cdf) >= 0.)
        return srt, cdf

    def _estimate_from_pdf(self, model_values, p=(0.2, 0.5, 0.8)):
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
        values_at_p = np.interp(p, cdf, grid_center)
        return values_at_p

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
        # FIXME this pattern loads everything into memory; maybe not good
        # args = []
        bands = tuple(bands)  # to slice by bandpass (columns)
        # this slices only the bands used for observations
        x = self.library_h5_file[self.library_group_name]['seds'][bands]
        model_flux = x.view(np.float64).reshape(x.shape + (-1,))

        # with Timer() as timer:
        # See da Cunha, Charlot and Elbaz (2008) eq 33 for info.
        a = np.sum(obs_flux * model_flux / obs_err ** 2., axis=1)
        b = np.sum((model_flux / obs_err) ** 2., axis=1)
        mass = a / b
        # assert np.all(mass > 0.)
        residuals = (np.atleast_2d(mass).T * model_flux - obs_flux) \
            / obs_err
        # assert np.all(np.power(residuals, 2) >= 0.)
        lnp = -0.5 * np.sum(np.power(residuals, 2), axis=1)
        # Re-scale the probabilities
        lnp -= lnp.max() + 1.
        # assert np.all(lnp < 0.)
        # print "Marginalization took", timer
        d = np.dtype([('lnp', np.float), ('mass', np.float)])
        output_data = np.empty(model_flux.shape[0], dtype=d)
        output_data['lnp'] = lnp.flatten()
        output_data['mass'] = mass.flatten()
        return output_data
