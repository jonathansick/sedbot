#!/usr/bin/env python
# encoding: utf-8
"""
Demonstration for the multipix package.
"""

import numpy as np
from fsps import StellarPopulation

from sedbot.multipix.models import ThreeParamSFH
from sedbot.multipix.sampler import MultiPixelGibbsBgSampler

from sedbot.modeltools import mock_dataset


def main():
    npix = 2
    sed_bands = ['sdss_u', 'sdss_g', 'sdss_r', 'sdss_i']
    seds, sed_errs = make_dataset(npix, sed_bands)
    print seds

    def model_initializer():
        m = ThreeParamSFH(seds, sed_errs, sed_bands)
        return m

    sampler = MultiPixelGibbsBgSampler(model_initializer)
    print sampler
    print sampler._model


def make_dataset(npix, bands):
    PSET_INIT = {'add_dust_emission': True,
                 'add_stellar_remnants': True,
                 'compute_vega_mags': False,  # use AB mags
                 'dust_type': 0,  # Power law dust
                 'sf_start': 3.,
                 'sfh': 1,
                 'tburst': 10.,
                 'fburst': 0.,
                 'dust1': 0.2,
                 'dust2': 0.2,
                 'zmet': 20}
    sp = StellarPopulation(**PSET_INIT)

    nbands = len(bands)
    seds = np.empty((npix, nbands), dtype=np.float)
    sed_errs = np.empty((npix, nbands), dtype=np.float)
    for i in xrange(npix):
        pix_sed, pix_sigma = mock_dataset(sp,
                                          bands,
                                          785 * 1000.,
                                          1e8,
                                          0.0,
                                          0.05,
                                          apply_errors=True)
        seds[i, :] = pix_sed.T
        sed_errs[i, :] = pix_sigma.T
    return seds, sed_errs


if __name__ == '__main__':
    main()
