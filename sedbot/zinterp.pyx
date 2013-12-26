#!/usr/bin/env python
# encoding: utf-8
"""
This module provides facilities for logarithmically interpolating metallicity.

Currently this module assumes that FSPS is being used with the Padova+BaSeL
isochrone set of 22 metallicities.
"""

import numpy as np
cimport numpy as np

# Metallicity grid for Padova+BaSeL isochrones, log(Z/Z_sol)
_PADOVA_BASEL = np.log10(np.array([0.0002, 0.0003, 0.0004,
    0.0005, 0.0006, 0.0008, 0.0010, 0.0012, 0.0016, 0.0020, 0.0025, 0.0031,
    0.0039, 0.0049, 0.0061, 0.0077, 0.0096, 0.0120, 0.0150, 0.0190, 0.0240,
    0.0300], dtype=np.float) / 0.019)
# Build a Cython view
cdef double [:] PADOVA_BASEL = _PADOVA_BASEL


cpdef bracket_logz(double logZZsol):
    """Find the pair of `zmet` indices that bracket the given metallicity.
    
    That is, to generate a model spectrum of the given metallicity, FSPS
    an be run with bracketing isochrones. This function finds those
    metallicities.

    Parameters
    ----------
    logZZsol : float
        Desired value of log(Z/Z_sol).

    Returns
    -------
    zmet1 : int
        Lower metallicity index, zmet.
    zmet2 : int
        Upper metallicity index, zmet.
    """
    if logZZsol < PADOVA_BASEL[0]:
        return 1, 2
    elif logZZsol > PADOVA_BASEL[21]:
        return 21, 22
    cdef int i
    for i in range(22):  # number of elements in PADOVA_BASEL
        if PADOVA_BASEL[i] >= logZZsol:
            return i, i + 1


cpdef interp_logz(int zmet1, int zmet2, float logZZsol,
        double[:] f1, double[:] f2):
    """Interpolate between two flux densities, in logZ/Z_sol space.

    Fluxes must be passed as 1D arrays (such as ``numpy.ndarray``). To pass
    a scalar flux, use ``np.atleast_1d(flux)``.
    
    Parameters
    ----------
    zmet1 : int
        zmet index of the low-metallicity flux array, ``f1``.
    zmet2 : int
        zmet index of the high-metallicity flux array, ``f2``.
    logZZsol : float
        Desired value of log(Z/Z_sol).
    f1 : 1D array
        Flux density from the low metallicity isochrone.
    f2 : 1D array
        Flux density from the high metallicity isochrone.

    Returns
    -------
    f : 1D array
        Interpolated flux density.
    """
    cdef double c
    cdef int j, N
    cdef int i1 = zmet1 - 1
    cdef int i2 = zmet2 - 1
    N = f1.shape[0]
    f = np.empty(N, dtype=np.float)
    cdef double [:] f_view = f
    c = (logZZsol - PADOVA_BASEL[i1]) / (PADOVA_BASEL[i2] - PADOVA_BASEL[i1])
    if c < 0.:
        c = 0.
    if c > 1.:
        c = 1.
    for j in range(N):
        f_view[j] = (1. - c) * f1[j] + c * f2[j]
    return f
