#!/usr/bin/env python
# encoding: utf-8
"""
This module provides facilities for logarithmically interpolating metallicity.
"""

import numpy as np


# Metallicity grid for Padova+BaSeL isochrones, log(Z/Z_sol)
PADOVA_BASEL = np.log10(np.array([0.0002, 0.0003, 0.0004, 0.0005, 0.0006,
    0.0008, 0.0010, 0.0012, 0.0016, 0.0020, 0.0025, 0.0031, 0.0039, 0.0049,
    0.0061, 0.0077, 0.0096, 0.0120, 0.0150, 0.0190, 0.0240, 0.0300]) / 0.019)


def bracket_logz(logZZsol):
    """Find the pair of `zmet` indices that bracket the given metallicity."""
    if logZZsol < PADOVA_BASEL[0]:
        return 1, 2
    elif logZZsol > PADOVA_BASEL[-1]:
        return 21, 22
    for i, lz in enumerate(PADOVA_BASEL):
        if lz >= logZZsol:
            return i, i + 1


def interp_logz(zmet1, zmet2, logZZsol, f1, f2):
    """Interpolate between two flux densities, in logZ/Z_sol space."""
    i1 = zmet1 - 1
    i2 = zmet2 - 1
    c = (logZZsol - PADOVA_BASEL[i1]) / (PADOVA_BASEL[i2] - PADOVA_BASEL[i1])
    if c < 0.:
        c = 0.
    if c > 1.:
        c = 1.
    return (1. - c) * f1 + c * f2
