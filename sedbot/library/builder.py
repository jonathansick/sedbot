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

import abc
from collections import namedtuple, OrderedDict

import numpy as np

# from fsps import StellarPopulation


Limits = namedtuple('Limits', ['low', 'high'])


class LibraryBuilder(object):
    """Baseclass for building a stellar population realization library.

    Parameters
    ----------
    h5_file : :class:`h5py.File`
        File object for an HDF5 file.
    group : str
        (optional) group within the HDF5 table to store the library's tables.
        If `None` the tables are stored in the HDF5 file's root group.
    default_pset : dict
        Default Python-FSPS parameters.
    """
    def __init__(self, h5_file,
                 default_pset=None,
                 bands=None,
                 group=None):
        super(LibraryBuilder, self).__init__()
        self.generators = OrderedDict()

    def add_parameter(self, param_generator):
        """Add a parameter to our library.

        Parameters
        ----------
        param_generator : :class:`FSPSParamGenerator` subclass
            A subclass of :class:`FSPSParamGenerator` that generates random
            values of a parameter.
        """
        self.generators[param_generator.name] = param_generator

    def make(self, n_models):
        """Baseclass for building a stellar population realization library.

        Parameters
        ----------
        n_models : int
            Number of models to generate.
        """
        pass


class FSPSParamGenerator(object):
    """Baseclass for a library stellar population parameter generator."""

    __metaclass__ = abc.ABCMeta

    def __init__(self, name, low_limit=None, high_limit=None):
        super(FSPSParamGenerator, self).__init__()
        self.name = name
        if low_limit is None:
            low_limit = -np.inf
        if high_limit is None:
            high_limit = np.inf
        self.limits = Limits(low=low_limit, high=high_limit)

    @abc.abstractmethod
    def sample(self):
        """Build a single sample."""
        pass

    def check_value(self, x):
        if x > self.limits.high or x < self.limits.low:
            raise ValueError


class UniformParamGenerator(object):

    def __init__(self, pname, low_limit=None, high_limit=None):
        super(UniformParamGenerator, self).__init__(pname,
                                                    low_limit=low_limit,
                                                    high_limit=high_limit)

    def sample(self):
        # No need to do error checking on this
        return np.random.uniform(self.limits.low, high=self.limits.high)
