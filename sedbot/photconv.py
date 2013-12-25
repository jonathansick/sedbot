#!/usr/bin/env python
# encoding: utf-8
"""
Photometric conversion utilities.

FSPS generates quantities in magnitudes, while sedbot's likelihood functions
operate on flux densities in micro Janksies (µJy). FSPS also provides
spectra in units of solar-luminosity per Hz (:math:`f_\\nu`). This module
provides utilties for converting between these quantities.
"""

import numpy as np


# Zeropoint factor for converting mags to µJy
# µJy = 10.^6 * 10.^23 * 10^(-48.6/2.5) = MICROJY_ZP * 10.^(-AB/2.5)
# see http://en.wikipedia.org/wiki/Jansky
MICROJY_ZP = 10. ** 6. * 10. ** 23. * 10. ** (-48.6/ 2.5)


def ab_to_mjy(mags, parsecs):
    r"""Convert scalar (or array) AB magnitudes to µJy given a distance in
    parsecs.

    That is, we apply the transformation:

    .. math::
       m &= M - 5 (1 - \log_{10}d) \\
       f_\nu & = 10^6 10^{23} 10^{-48.6/2.5} 10^{-m/2.5}~\mathrm{[µJy]}

    Parameters
    ----------
    mags : ndarray
        AB magnitudes, either a scalar or an ``ndarray``. We assume these
        are *absolute magnitudes*.
    parsecs : float
        Distance (or equivalently the luminosity distance) in parsecs.

    Returns
    -------
    mjy : ndarray
        Flux densities, in micro Janskies (µJy). Same shape as input ``mags``
        array.
    """
    m = mags - 5. * (1. - np.log10(parsecs))  # apparent magnitudes
    mjy = MICROJY_ZP * np.power(10., -m / 2.5)
    return mjy
