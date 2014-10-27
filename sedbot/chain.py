#!/usr/bin/env python
# encoding: utf-8
"""
Data structure and persistence for emcee chains with HDF5.
"""

import os

import numpy as np
from astropy.table import Table, vstack, hstack
import h5py


class SinglePixelChain(Table):
    """Table for single-pixel-class samplers (ie, ``sedbot.singlepixel``)."""
    def __add__(self, other_table):
        """Concatenate tables.

        >>> tbl = tbl1 + tbl2

        Note this is a left-add, so `tbl1` in the example gets called.
        """
        other_table.meta = {}
        # Throw out last value of this table because it's probably going
        # to be the first value of the next one
        return MultiPixelChain(vstack((self[:-1], other_table),
                                      metadata_conflicts='silent',
                                      join_type='exact'))

    @property
    def sed_residuals_chain(self):
        """Chain of :math:`F_\mathrm{obs} - F_\mathrm{model}` SED residuals
        for a single pixel.

        Parameters
        ----------
        ipix : int
            Pixel ID.
        add_background : bool
            If ``True`` then the background is added to the modelled SED.

        Returns
        -------
        residuals : ndarray
            A ``(n_sample, n_band)`` numpy array. The order of bandpasses
            corresponds to ``self.meta['compute_bands']``.
        """
        idx = self.meta['band_indices']
        model = self['model_sed'][:, idx]
        obs = self.meta['sed']
        return obs - model

    @property
    def sed_ratios_chain(self):
        """Chain of :math:`F_\mathrm{obs} / F_\mathrm{model}` SED ratios
        for a single pixel.

        Returns
        -------
        ratios : ndarray
            A ``(n_sample, n_band)`` numpy array. The order of bandpasses
            corresponds to ``self.meta['compute_bands']``.
        """
        idx = self.meta['band_indices']
        model = self['model_sed'][:, idx]
        obs = self.meta['sed']
        return obs / model


class MultiPixelChain(Table):
    """Table for the MCMC chain generated by a ``sedbot.multipixel``
    Gibbs sampler.
    """

    @classmethod
    def join_single_pixel_chains(cls, tables):
        """Join a list of :class:`SinglePixelChain` instances, creating a
        :class:`MultiPixelChain`.

        Parameters
        ----------
        tables : list
            List of :class:`SinglePixelChain` instances. The order defines
            the order of the pixels in the combined chain.

        Returns
        -------
        chain : :class:`MultiPixelChain` instance
            The :class:`MultiPixelChain` with single pixel chains
            combined.
        """
        chain_colnames = tables[0].colnames
        n_chains = len(tables)
        n_sed_bands = len(tables[0].meta['compute_bands'])
        n_params = len(tables[0].meta['theta_params'])
        n_steps = len(tables[0])

        # Copy chain data
        dt = [(n, np.float, n_chains) for n in chain_colnames
              if n not in ['model_sed', 'lnpriors']] \
            + [('model_sed', np.float, (n_chains, n_sed_bands))] \
            + [('lnpriors', np.float, (n_chains, n_params))]
        data = np.empty(n_steps, dtype=np.dtype(dt))
        data.fill(np.nan)
        for i, tbl in enumerate(tables):
            for colname in chain_colnames:
                if colname == 'model_sed':
                    data['model_sed'][:, i, :] = tbl['model_sed']
                elif colname == 'lnpriors':
                    data['lnpriors'][:, i, :] = tbl['lnpriors']
                else:
                    data[colname][:, i] = tbl[colname]

        # Copy metadata
        meta = {}
        # Copy scalar constant quantites for all pixels
        copy_keys = ['d', 'obs_bands', 'compute_bands', 'msun_ab',
                     'band_indices', 'theta_params', 'n_walkers']
        for k in copy_keys:
            meta[k] = tables[0].meta[k]
        # Array metadata
        meta['sed'] = np.array([t.meta['sed'] for t in tables])
        meta['sed_err'] = np.array([t.meta['sed_err'] for t in tables])
        meta['area'] = np.array([t.meta['area'] for t in tables])
        meta['pixels'] = np.concatenate([t.meta['pixels'] for t in tables])
        meta['acor'] = np.concatenate([t.meta['acor'] for t in tables])
        meta['f_accept'] = np.concatenate([t.meta['f_accept'] for t in tables])

        tbl = cls(data, meta=meta)
        return tbl

    def __add__(self, other_table):
        """Concatenate tables.

        >>> tbl = tbl1 + tbl2

        Note this is a left-add, so `tbl1` in the example gets called.
        """
        other_table.meta = {}
        # Throw out last value of this table because it's probably going
        # to be the first value of the next one
        return MultiPixelChain(vstack((self[:-1], other_table),
                                      metadata_conflicts='silent',
                                      join_type='exact'))

    @property
    def n_pixels(self):
        return self['model_sed'].shape[1]

    def chain_for_pixel(self, ipix, use_cols=None):
        """Return the chain for a single pixel as a structured array.

        Parameters
        ----------
        ipix : int
            Pixel ID.
        use_cols : list
            Names of columns to include in output, or ``None`` to include all.

        Returns
        -------
        chain : ndarray
            A structured numpy array.
        """
        if use_cols is None:
            colnames = list(self.dtype.names)
        else:
            colnames = list(use_cols)

        # NOTE temporarily forcing removal of sed_model from chain output
        if 'model_sed' in colnames:
            colnames.remove('model_sed')

        dtype = [(n, np.float) for n in colnames]
        arr = np.empty(len(self), dtype=np.dtype(dtype))
        single_value_fields = list(self.meta['phi_params']) \
            + ["B_{0}".format(n) for n in self.meta['obs_bands']]
        for i, n in enumerate(colnames):
            if n in single_value_fields:
                # single value for all pixels
                arr[n] = self[n]
            else:
                arr[:][n] = self[n][:, ipix]
        return arr

    def chain_array_for_pixel(self, ipix,
                              use_cols=None,
                              return_colnames=False):
        """Same as :meth:`chain_for_pixel`, except the returned array is
        2D

        Parameters
        ----------
        ipix : int
            Pixel ID.
        use_cols : list
            Names of columns to include in output, or ``None`` to include all.
        return_colnames : bool
            If ``True``, then a list of column names will be provided.

        Returns
        -------
        chain : ndarray
            A ``(n_sample, n_param)`` numpy array.
        colnames : list
            List of column names, if ``return_colnames == True``.
        """
        chain_tbl = self.chain_for_pixel(ipix, use_cols=use_cols)
        colnames = chain_tbl.dtype.names
        chain = chain_tbl.view(np.float).reshape(chain_tbl.shape + (-1,))
        # Expand columns as necessary
        # e.g. the sed_model array should be converted into a flat array
        if return_colnames:
            return chain, colnames
        else:
            return chain

    def sed_chain_for_pixel(self, ipix):
        """Provides a chain array for the modelled SED of a single pixel.

        .. note:: The modelled background *is not added* because the modelled
           SED may be a superset of the observed SED.

        Parameters
        ----------
        ipix : int
            Pixel ID.

        Returns
        -------
        chain : ndarray
            A ``(n_sample, n_band)`` numpy array. The order of bandpasses
            corresponds to ``self.meta['compute_bands']``.
        """
        arr = self['model_sed'][:, ipix, :]
        return arr

    def sed_residuals_chain_for_pixel(self, ipix, add_background=True):
        """Chain of :math:`F_\mathrm{obs} - F_\mathrm{model}` SED residuals
        for a single pixel.

        Parameters
        ----------
        ipix : int
            Pixel ID.
        add_background : bool
            If ``True`` then the background is added to the modelled SED.

        Returns
        -------
        residuals : ndarray
            A ``(n_sample, n_band)`` numpy array. The order of bandpasses
            corresponds to ``self.meta['compute_bands']``.
        """
        idx = self.meta['band_indices']
        model = self['model_sed'][:, ipix, idx]
        if add_background:
            model += self.background_array * self.meta['area'][ipix]
        obs = self.meta['sed'][ipix, :]
        return obs - model

    def sed_ratios_chain_for_pixel(self, ipix, add_background=True):
        """Chain of :math:`F_\mathrm{obs} - F_\mathrm{model}`
        SED ratios for a single pixel.

        Parameters
        ----------
        ipix : int
            Pixel ID.
        add_background : bool
            If ``True`` then the background is added to the modelled SED.

        Returns
        -------
        ratios : ndarray
            A ``(n_sample, n_band)`` numpy array. The order of bandpasses
            corresponds to ``self.meta['compute_bands']``.
        """
        idx = self.meta['band_indices']
        model = self['model_sed'][:, ipix, idx]
        if add_background:
            model += self.background_array * self.meta['area'][ipix]
        obs = self.meta['sed'][ipix, :]
        return obs / model

    @property
    def background_array(self):
        """A ``(n_sample, n_band)`` ndarray of background samples."""
        fields = ["B_{0}".format(n) for n in self.meta['obs_bands']]
        B = np.empty((len(self), len(fields)), dtype=np.float)
        for i, f in enumerate(fields):
            B[:, i] = self[f]
        return B


# DEPRECATED FUNCTIONS from early versions of sedbot ========================

def write_flatchain(flatchain_table, filepath, tabledir,
                    overwrite=False):
    """Persist the flattened chain into an HDF5 file (uses `astropy.table`).

    .. note:: This function is deprecated. Use the ``singlepix`` subpackage.

    Note that by using the :func:`sedbot.modeltools.burnin_flatchain` function
    FSPS blobs (such as stellar mass, dust mass, etc) can be automatically
    appended to the flatchain.

    Parameters
    ----------
    flatchain_table : `astropy.table.Table`
        The flatchain, likely created by
        :func:`sedbot.modeltools.burnin_flatchain_table`
    filepath : str
        Path on disk for the HDF5 file
    tabledir : str
        Directory *inside* the HDF5 file where the flatchain will be stored.
        The chain is stored in `tabledir/flatchain`
    overwrite : bool
        If `False`, the flatchain will be appended to an existing file rather
        than overwriting it.
    """
    chain_path = os.path.join(tabledir, "flatchain")
    if overwrite:
        _append = False
    else:
        _append = True
    flatchain_table.write(filepath, path=chain_path, append=_append,
                          format='hdf5', overwrite=overwrite)


def read_flatchain(filepath, tabledir):
    """Read a flatchain table from an HDF5 file.

    .. note:: This function is deprecated. Use the ``singlepix`` subpackage.

    Parameters
    ----------
    filepath : str
        Path on disk for the HDF5 file
    tabledir : str
        Directory *inside* the HDF5 file where the flatchain will be stored.
        The chain is stored in `tabledir/flatchain`

    Returns
    -------
    tbl : :class:`astropy.table.Table`
        A `Table` instance with the flatchain.
    """
    chain_path = os.path.join(tabledir, "flatchain")
    tbl = Table.read(filepath, path=chain_path, format='hdf5')
    return tbl


def make_flatchain(sampler, param_names, bands, metadata=None,
                   n_burn=0, append_mstar=False, append_mdust=False,
                   append_lbol=False, append_sfr=False, append_age=False,
                   append_model_sed=False, append_lnpost=False,
                   save_acor=False):
    """Create an Astropy Table of 'flatchain' of emcee walkers, removing any
    burn-in steps, and appending blob metadata.

    .. note:: This function is deprecated. Use the ``singlepix`` subpackage.

    Default metadata is

    - ``bandpasses``: the FSPS names of modelled bandpasses.
    - ``f_accept``: acceptance fraction.

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
    save_acor : bool
        If ``True`` append autocorrelation time for sampler in metadata under
        the ``accor`` field.

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
                if blobs[i + n_burn][j] is not np.nan:
                    for k, n in zip(blob_index, blob_names):
                        flatchain[n][i * j] = blobs[i + n_burn][j][k]
                else:
                    for k, n in zip(blob_index, blob_names):
                        flatchain[n][i * j] = np.nan

    if metadata is None:
        metadata = {}
    metadata.update({"bandpasses": bands,
                     "f_accept": sampler.acceptance_fraction})
    if save_acor:
        metadata['acor'] = sampler.acor
    tbl = Table(flatchain, meta=metadata)
    # Bad posterior samples are given values of 0 by emcee; so filter them
    bad = np.where((flatchain['lnpost'] >= 0.))[0]
    for n in tbl.keys():
        tbl[n][bad] = np.nan
    return tbl


class MultiPixelDataset(object):
    """Representation of multiple SinglePixelChains in a multi-table HDF5
    file, as well as parameter estimate tables.

    This class is better suited to handling large ensembles than joining
    SinglePixelChains in a MultiPixelChain.
    """
    def __init__(self, hdf5_path):
        super(MultiPixelDataset, self).__init__()
        self._filepath = hdf5_path

    @classmethod
    def build_dataset(cls, hdf5_path, chains):
        for i, chain in enumerate(chains):
            chain_path = "chains/{0:d}".format(i)
            chain.write(hdf5_path, format='hdf5', path=chain_path,
                        append=True, overwrite=True)
        instance = cls(hdf5_path)
        instance.build_pixels_table()
        return instance

    def read_chain(self, pixel_id):
        """Read an individual :class:`SinglePixelChain`."""
        pixel_id = int(pixel_id)
        tbl = SinglePixelChain.read(self._filepath,
                                    path="chains/{0:d}".format(pixel_id))
        return tbl

    @property
    def pixels(self):
        """The pixels table, containing information on each pixel."""
        return Table.read(self._filepath, path='pixels')

    def build_pixels_table(self):
        """Build the ``pixels`` table, which is built from the `'pixels'`
        metadata of individual chains.
        """
        with h5py.File(self._filepath, 'r+') as f:
            if 'pixels' in f:
                del f['pixels']
            pixel_ids = [int(k) for k in f['chains'].keys()]
        pixel_ids.sort()

        pixel_rows = []
        chain0 = self.read_chain(0)
        n_pixels = len(pixel_ids)
        n_bands = len(chain0.meta['sed'])
        sed_data = np.empty(
            n_pixels,
            dtype=np.dtype([('sed', np.float, n_bands),
                            ('sed_err', np.float, n_bands)]))
        sed_data.fill(np.nan)
        for pixel_id in pixel_ids:
            chain = self.read_chain(pixel_id)
            pixel_rows.append(chain.meta['pixels'])
            obs_bands = chain.meta['obs_bands']
            sed_data['sed'][pixel_id] = chain.meta['sed']
            sed_data['sed_err'][pixel_id] = chain.meta['sed_err']
        pixel_rows = tuple(pixel_rows)
        pixel_data = np.concatenate(pixel_rows)
        meta = {"obs_bands": obs_bands}
        pixel_table = Table(pixel_data, meta=meta)
        sed_table = Table(sed_data)
        pixel_table = hstack([pixel_table, sed_table], join_type='exact')
        pixel_table.write(self._filepath, path="pixels", format="hdf5",
                          append=True, overwrite=True)

    def build_estimates_table(self):
        pass
