#!/usr/bin/env python
# encoding: utf-8
"""
This module contains code for constructing stellar population realization
libraries.

These procedures are based on Kauffmann 2003, Zibetti 2009,
da Cunha & Charlot 2010 and Taylor 2011.

The storage backend for these libraries is an HDF5 table.

2015-01-25 - Created by Jonathan Sick
"""

from fsps import StellarPopulation


class LibraryBuilder(object):
    """Baseclass for building a stellar population realization library.

    Parameters
    ----------
    hdf5_path : str
        File path for the HDF5 table.
    group : str
        (optional) group within the HDF5 table to store the library's tables.
        If `None` the tables are stored in the HDF5 file's root group.
    default_pset : dict
        Default Python-FSPS parameters.
    """
    def __init__(self, hdf5_path,
                 default_pset=None,
                 bands=None,
                 group=None):
        super(LibraryBuilder, self).__init__()
        pass

    def add_parameter(self, param_generator):
        """Add a parameter to our library.

        Parameters
        ----------
        param_generator : :class:`FSPSParamGenerator` subclass
            A subclass of :class:`FSPSParamGenerator` that generates random
            values of a parameter.
        """
        pass

    def make(self, n_models, hdf5_path, group=None):
        """Baseclass for building a stellar population realization library.

        Parameters
        ----------
        n_models : int
            Number of models to generate.
        hdf5_path : str
            File path for the HDF5 table.
        group : str
            (optional) group within the HDF5 table to store the library's
            tables. If `None` the tables are stored in the HDF5 file's root
            group.
        """
        pass


class FSPSParamGenerator(object):
    """Baseclass for a library stellar population parameter generator."""
    def __init__(self, pname, low_limit=None, high_limit=None):
        super(FSPSParamGenerator, self).__init__()

    def sample(self):
        """Build a single sample."""
        pass
