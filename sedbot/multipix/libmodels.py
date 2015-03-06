#!/usr/bin/env python
# encoding: utf-8
"""
Models to use with a library-based Gibbs multipixel sampler.

2015-03-04 - Created by Jonathan Sick
"""

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

        # Set indices of bands where background should always be reset to 0.
        if fixed_bg is not None:
            self._fixed_bg = {}
            for (instr, b), level in fixed_bg.iteritems():
                for i, (instrument, band) in enumerate(zip(instruments,
                                                           sed_bands)):
                    if (instr == instrument) and (b == band):
                        self._fixed_bg[i] = level

        self._estimator = LibraryEstimator(self._h5_file, self._group_name)

    @property
    def observed_bands(self):
        """Names of bandpasses in the *observed* SED."""
        return self._obs_bands

    @property
    def instruments(self):
        """Names of instruments in the *observed* SED."""
        return self._instruments

    @property
    def n_pix(self):
        return self._seds.shape[0]

    @property
    def n_bands(self):
        return self._seds.shape[1]

    @property
    def n_computed_bands(self):
        return len(self._compute_bands)

    @property
    def n_theta(self):
        """Number of theta parameters."""
        return len(self._theta_params)

    @property
    def theta_params(self):
        """Ordered list of theta-level parameter names."""
        return self._theta_params

    @property
    def meta_params(self):
        """Ordered list of all computed parameters associated with a model."""
        return self._estimator.meta_params

    @property
    def library_bands(self):
        """Ordered list of all bands in the library SEDs
        (c.f. `observed_bands`)"""
        return self._estimator.bands

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
