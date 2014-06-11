#!/usr/bin/env python
# encoding: utf-8
"""
Persistence for emcee chains to disk with HDF5.
"""

import os
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
