#!/usr/bin/env python
# encoding: utf-8
"""
Models to use with a library-based Gibbs multipixel sampler.

2015-03-04 - Created by Jonathan Sick
"""

import numpy as np

from sedbot.library.marginalizer import LibraryEstimator


class LibraryModel(object):
    """Docstring for LibraryModel.

    Parameters
    ----------
    h5_file : :class:`h5py.File`
        The HDF5 file containing the model library.
    group_name : str
        Name of the group in the HDF5 file containing the library.
    seds : ndarray
        SEDs in µJy, shape ``(npix, nbands)``.
    seds_errs : ndarray
        SEDs uncertainties in µJy, shape ``(npix, nbands)``.
    areas : ndarray
        Area of each SED's pixel in arcsec^2. Shape ``(npix,)``.
    d : float
        The distance, in parsecs.
    pixel_metadata : ndarray
        Arbitrary structured array with metadata about the pixels. This
        will be appended to the chain metadata under the `'pixels'` field.
    bands : list
        List of bandpass names corresponding to the ``seds``
    instruments : list
        List of instrument names corresponding to the ``seds``
    fixed_bg : dict
        Dictionary of ``(instrument, bandpass name): background level``.
        e.g. ``{('megacam', 'megacam_i'): 0.0}``
    """
    def __init__(self, h5_file, group_name,
                 seds, sed_errs, sed_bands, instruments, areas, d,
                 pixel_metadata=None,
                 fixed_bg=None):
        super(LibraryModel, self).__init__()
        self._h5_file = h5_file
        self._group_name = group_name
        self._seds = seds
        self._errs = sed_errs
        self._areas = areas
        self._d
        self._obs_bands = sed_bands
        self._instruments = instruments
        self.pixel_metadata = pixel_metadata

        self._group = h5_file[group_name]

        # Set indices of bands where background should always be reset to 0.
        if fixed_bg is not None:
            self._fixed_bg = {}
            for (instr, b), level in fixed_bg.iteritems():
                for i, (instrument, band) in enumerate(zip(instruments,
                                                           sed_bands)):
                    if (instr == instrument) and (b == band):
                        self._fixed_bg[i] = level

    @property
    def observed_bands(self):
        """Names of bandpasses in the *observed* SED."""
        return self._obs_bands

    @property
    def instruments(self):
        """Names of instruments in the *observed* SED."""
        return self._instruments

    @property
    def band_indices(self):
        """List of indices of observed bands in the library's bands."""
        if self._band_indices is None:
            indices = np.zeros(len(self.observed_bands), dtype=np.int)
            for i, band in enumerate(self.observed_bands):
                idx = self.library_bands.index(band)
                indices[i] = idx
            self._band_indices = indices
        return self._band_indices

    @property
    def n_pix(self):
        return self._seds.shape[0]

    @property
    def n_bands(self):
        return self._seds.shape[1]

    @property
    def n_theta(self):
        """Number of theta parameters."""
        return len(self.theta_params)

    @property
    def theta_params(self):
        """Ordered list of theta-level parameter names."""
        return self._group['params'].dtype.names

    @property
    def meta_params(self):
        """Ordered list of all computed parameters associated with a model."""
        return self._group['meta'].dtype.names

    @property
    def library_bands(self):
        """Ordered list of all bands in the library SEDs
        (c.f. `observed_bands`)"""
        return self._group.attrs['bands']

    @property
    def msun_ab(self):
        """AB solar magnitudes for all library bands."""
        return self._group.attrs['msun_ab']

    @property
    def d(self):
        """The distance, parsecs"""
        return self._d

    @property
    def library_file(self):
        return self._library_h5_file

    @property
    def library_group(self):
        return self._group_name

    def estimator_for_pixel(self, pix_id, B, ncpu=1):
        """Make a LibraryEstimator instance for this pixel."""
        le = LibraryEstimator(
            self._seds[pix_id, :] - B, self._errs[pix_id, :],
            self._obs_bands, self.d,
            self.library_file, self.library_group,
            ncpu=ncpu)
        return le

    def estimate_background(self, model_seds):
        diff_mean = np.empty(self.n_bands, dtype=np.float)
        diff_var = np.empty(self.n_bands, dtype=np.float)
        A = self._areas
        # Do it band-at-a-time so we can filter out NaNs in each band
        for i in xrange(self.n_bands):
            obs_var = (self._errs[:, i] / A) ** 2.
            g = np.where(np.isfinite(self._seds[:, i]))[0]
            residuals = ((self._seds[:, i] - model_seds[:, i]) / A)[g]
            diff_mean[i] = np.average(residuals,
                                      weights=1. / obs_var[g])
            # Establish the variance of the sampled Gaussian from
            # error propagation, or from the sample variance of residuals???
            diff_var[i] = 1. / np.sum(1. / obs_var[g])
        B_new = np.sqrt(diff_var) * np.random.randn(self.n_bands) \
            + diff_mean

        # reset B for any images with fixed background
        for band_index, level in self._fixed_bg.iteritems():
            B_new[band_index] = level

        return B_new
