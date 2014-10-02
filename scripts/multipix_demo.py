#!/usr/bin/env python
# encoding: utf-8
"""
Demonstration for the multipix package.
"""

import numpy as np

from sedbot.multipix.models import ThreeParamSFH
from sedbot.multipix.sampler import MultiPixelGibbsBgSampler


def main():
    npix = 2
    sed_bands = "ugri"
    nbands = len(sed_bands)
    seds = np.ones((npix, nbands))
    sed_errs = np.ones((npix, nbands))
    sed_bands = None

    def model_initializer():
        m = ThreeParamSFH(seds, sed_errs, sed_bands)
        return m

    sampler = MultiPixelGibbsBgSampler(model_initializer)
    print sampler


if __name__ == '__main__':
    main()
