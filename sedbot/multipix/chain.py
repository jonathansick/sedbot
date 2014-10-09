#!/usr/bin/env python
# encoding: utf-8
"""
Utilities for handing the chain tables made by the multipixel Gibbs sampler.
"""

import numpy as np
from astropy.table import Table, vstack


class MultiPixelChain(Table):
    """Table for the MCMC chain generated by a multipixel Gibbs sampler."""

    def __add__(self, other_table):
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
            print n
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
        # nbands = len(self.meta['compute_bands'])
        # nsamples = len(self)
        # arr = np.empty()
        arr = self['model_sed'][:, ipix, :]
        return arr

    def sed_residuals_chain_for_pixel(self, ipix):
        """Chain of model - observed SED residuals for a single pixel."""
        idx = self.meta['band_indices']
        model = self['model_sed'][:, ipix, idx]
        obs = self.meta['sed'][ipix, :]
        return model - obs
