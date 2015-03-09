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

import fsps

from sedbot.photconv import abs_ab_mag_to_micro_jy, micro_jy_to_luminosity


Limits = namedtuple('Limits', ['low', 'high'])


class LibraryBuilder(object):
    """Baseclass for building a stellar population realization library.

    Parameters
    ----------
    h5_file : :class:`h5py.File`
        File object for an HDF5 file. To create an in-memory file, use:
        `h5py.File('in_memory.hdf5', driver='core', backing_store=False)`.
    group : str
        (optional) Name of group within the HDF5 table to store the library's
        tables. By default tables are stored in the HDF5 file's root group.
    """
    def __init__(self, h5_file,
                 group='/'):
        super(LibraryBuilder, self).__init__()
        self.h5_file = h5_file
        self.group_name = group
        if group not in self.h5_file:
            self.h5_file.create_group(self.group_name)

        self.generators = OrderedDict()

    @property
    def group(self):
        """The HDF5 group with the model parameters and realizations."""
        return self.h5_file[self.group_name]

    def add_parameter(self, param_generator):
        """Add a parameter to our library.

        Parameters
        ----------
        param_generator : :class:`FSPSParamGenerator` subclass
            A subclass of :class:`FSPSParamGenerator` that generates random
            values of a parameter.
        """
        self.generators[param_generator.name] = param_generator

    def define_library(self, n_models):
        """Define a table of FSPS stellar population parameters to derive
        a table from given the parameter generators.

        Parameters
        ----------
        n_models : int
            Number of models to generate.
        """
        dt = self._create_parameter_dtype()
        data = np.empty(n_models, dtype=dt)
        data.fill(np.nan)
        i = 0
        while i < n_models:
            for name, generator in self.generators.iteritems():
                data[name][i] = generator.sample()
            i += 1
            # TODO create a mechanism for reviewing all parameters for sanity
            # checking (i.e., might want all tburst after tstart, etc)

        # Persist the parameter catalog to HDF5; delete existing
        if "params" in self.group:
            del self.group["params"]
        self.group.create_dataset("params", data=data)

    def compute_library_seds(self, bands, age=13.7, default_pset=None):
        """Compute an SED for each library model instance.

        Model SEDs are stored as absolute fluxes (µJy at d=10pc), normalized
        to a 1 Solar mass stellar population.

        .. todo:: Support parallel computation.

        Parameters
        ----------
        bands : list
            List of `FSPS bandpass names
            <http://dan.iel.fm/python-fsps/current/filters/>`_.
        default_pset : dict
            Default Python-FSPS parameters.
        """
        if default_pset is None:
            default_pset = {}

        # ALWAYS compute AB mags
        default_pset['compute_vega_mags'] = False

        # Solar magnitude in each bandpass
        solar_mags = [fsps.get_filter(n).msun_ab for n in bands]

        # Add bands and AB solar mags to the group attr metadata
        self.group.attrs['bands'] = bands
        self.group.attrs['msun_ab'] = solar_mags

        # Build the SED table
        table_names = ['seds', 'mass_light', 'meta']
        for name in table_names:
            if name in self.group:
                del self.group[name]

        # Table for SEDs
        n_models = len(self.group["params"])
        dtype = np.dtype([(n, np.float) for n in bands])
        sed_table = self.group.create_dataset("seds",
                                              (n_models,),
                                              dtype=dtype)

        # Table for M/L ratios in each bandpass
        dtype = np.dtype([(n, np.float) for n in bands])
        ml_table = self.group.create_dataset("mass_light",
                                             (n_models,),
                                             dtype=dtype)

        # Table for metadata (stellar mass, dust mass, etc..)
        meta_cols = ('logMstar', 'logMdust', 'logLbol', 'logSFR', 'logAge')
        dtype = np.dtype([(n, np.float) for n in meta_cols])
        meta_table = self.group.create_dataset("meta",
                                               (n_models,),
                                               dtype=dtype)

        # Iterate on each model
        # TODO eventually split this work between processors
        sp_param_names = self.group['params'].dtype.names
        sp = fsps.StellarPopulation(**default_pset)
        for i, row in enumerate(self.group["params"]):
            for n, p in zip(sp_param_names, row):
                sp.params[n] = float(p)
            mags = sp.get_mags(tage=age, bands=bands)
            fluxes = abs_ab_mag_to_micro_jy(mags, 10.)

            # Fill in SED and ML tables
            for n, msun, flux in zip(bands, solar_mags, fluxes):
                # interesting slicing syntax for structured array assignment
                sed_table[n, i] = flux

                logL = micro_jy_to_luminosity(flux, msun, 10.)
                log_ml = np.log10(sp.stellar_mass) - logL
                ml_table[n, i] = log_ml

            # Fill in meta data table
            meta_table['logMstar', i] = np.log10(sp.stellar_mass)
            meta_table['logMdust', i] = np.log10(sp.dust_mass)
            meta_table['logLbol', i] = np.log10(sp.log_lbol)
            meta_table['logSFR', i] = np.log10(sp.sfr)
            meta_table['logAge', i] = sp.log_age

    def _create_parameter_dtype(self):
        dt = [g.type_def for name, g in self.generators.iteritems()]
        return np.dtype(dt)


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
        # while True:
        #     try:
        #         x = generator.sample()
        #     except ValueError:
        #         # Repeat sampling to get a finite value
        #         pass
        #     else:
        #         break

    @property
    @abc.abstractmethod
    def type_def(self):
        return (self.name, np.float)

    def check_value(self, x):
        if x > self.limits.high or x < self.limits.low:
            raise ValueError


class UniformParamGenerator(FSPSParamGenerator):

    def __init__(self, name, low_limit=None, high_limit=None):
        super(UniformParamGenerator, self).__init__(name,
                                                    low_limit=low_limit,
                                                    high_limit=high_limit)

    def sample(self):
        # No need to do error checking on this
        return np.random.uniform(self.limits.low, high=self.limits.high)

    @property
    def type_def(self):
        return (self.name, np.float)


class TauUniformGammaGenerator(FSPSParamGenerator):
    """Generate a tau (SFR e-folding time) parameter that is uniformly
    distributed in as 1/tau (gamma).

    Why is this useful? Most Library-based SED works generate libraries
    uniformly distributed in gamma space, but FSPS operates on tau.
    """

    def __init__(self, name, low_limit=None, high_limit=None):
        super(TauUniformGammaGenerator, self).__init__(name,
                                                       low_limit=low_limit,
                                                       high_limit=high_limit)

    def sample(self):
        # This should generate a result within the bounds of tau
        gamma = np.random.uniform(1. / self.limits.high,
                                  high=1. / self.limits.low)
        return 1. / gamma

    @property
    def type_def(self):
        return (self.name, np.float)
