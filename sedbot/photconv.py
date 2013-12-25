#!/usr/bin/env python
# encoding: utf-8
"""
Prototype code for transforming FSPS magnitudes into µJy.

Ultimately this will be implemented in the likelihood call.
"""

import numpy as np


# Zeropoint factor for converting mags to µJy
# µJy = 10.^6 * 10.^23 * 10^(-48.6/2.5) = MICROJY_ZP * 10.^(-AB/2.5)
MICROJY_ZP = 10. ** 6. * 10. ** 23. * 10. ** (-48.6/ 2.5)


def mags_to_mjy(mags, parsecs):
    """Convert scalar (or array) magnitudes to µJy given a distance in
    parsecs.
    """
    m = mags - 5. * (1. - np.log10(parsecs))  # apparent magnitudes
    mjy = MICROJY_ZP * np.power(10., -m / 2.5)
    return mjy
