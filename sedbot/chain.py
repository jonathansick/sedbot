#!/usr/bin/env python
# encoding: utf-8
"""
Data structure and persistence for emcee chains with HDF5.
"""

import os

import numpy as np
from astropy.table import Table


def write_flatchain(flatchain_table, filepath, tabledir,
                    overwrite=False):
    """Persist the flattened chain into an HDF5 file (uses `astropy.table`).

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
    metadata.update({"bandpasses": bands,
                     "f_accept": sampler.acceptance_fraction})
    tbl = Table(flatchain, meta=metadata)
    # Bad posterior samples are given values of 0 by emcee; so filter them
    bad = np.where((flatchain['lnpost'] >= 0.))[0]
    for n in tbl.keys():
        tbl[n][bad] = np.nan
    return tbl
