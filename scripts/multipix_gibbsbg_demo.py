#!/usr/bin/env python
# encoding: utf-8
"""
Demonstration/test script for hierarchical multipixel SED modelling with a
scalar background parameter.

2014-09-17 - Created by Jonathan Sick
"""

import numpy as np

import fsps
from sedbot.photconv import sb_to_mass, ab_mag_to_mjy, mjy_to_ab_sb

from sedbot.probf import LnUniform, LnNormal

N_PIXELS = 5
BANDS = ['sdss_u', 'sdss_g', 'sdss_r', 'sdss_i', '2mass_J', '2mass_Ks']
D0 = 785. * 1000.  # distance in parsecs (McConnachie 2005)
D0_SIGMA = 25. * 1000.
PIX_AREA = 1.  # arcsec-sq
MSUN_I = fsps.get_filter('sdss_i').msun_ab
BKG_AMPLITUDE = 1e-10  # units of janksy per sq arcsec

# Hard limits on parameters
LIMITS = {'logmass': (sb_to_mass(30., MSUN_I, PIX_AREA, 0.4, D0),
                      sb_to_mass(12., MSUN_I, PIX_AREA, 0.4, D0)),
          'logtau': (-1., 2.),
          'const': (0., 1.0),
          'sf_start': (0.5, 10.),
          'tburst': (10.0, 13.8),
          'fburst': (0., 1.0),
          'logZZsol': (-1.98, 0.2),
          'd': (D0 - 3. * D0_SIGMA, D0 + 3. * D0_SIGMA),
          'dust1': (0., 5.),
          'dust2': (0., 3.),
          'ml': (-0.5, 1.5)}


# Prior probability functions
PRIOR_FUNCS = {"logtau": LnUniform(*LIMITS['logtau']),
               "dust2": LnUniform(*LIMITS['dust2']),
               "sf_start": LnUniform(*LIMITS['sf_start']),
               "logZZsol": LnUniform(*LIMITS['logZZsol']),
               "logmass": LnUniform(*LIMITS['logmass']),
               "d": LnNormal(D0, D0_SIGMA, limits=LIMITS['d'])}


# Truth dictionary
# Note that all pixels share the same background bias, which is different
# in each band.
TRUTHS = {"B": np.array([1e-12, -5e-10, -7e-10, 1.2e-10, -1.5e-10, 1.8e-9]),
          "d": D0,
          'logmass': sb_to_mass(np.linspace(15., 22., N_PIXELS),
                                MSUN_I, PIX_AREA, 0.4, D0),
          "logtau": np.random.uniform(low=0.5, high=1.5, size=N_PIXELS),
          "sf_start": 3.,
          "dust1": np.random.uniform(low=1., high=1.1, size=N_PIXELS),
          "dust2": np.random.uniform(low=0.1, high=0.5, size=N_PIXELS),
          "const": 0.1,
          "logZZsol": 0.}  # zmet=20


# Default pset to initialize fsps StellarPopulation with
PSET_INIT = {'add_dust_emission': True,
             'add_stellar_remnants': True,
             'compute_vega_mags': False,  # use AB mags
             'dust_type': 0,  # Power law dust
             'sf_start': TRUTHS['sf_start'],
             'sfh': 1,
             'tburst': 10.,
             'fburst': 0.,
             'dust1': 0.2,
             'dust2': 0.2,
             'zmet': 20}


# Default chain starting points for each parameter
CHAIN0 = {'logtau': 0.9,
          'const': 0.1,
          'sf_start': 2.,
          'tburst': 13.,
          'fburst': 0.1,
          'logZZsol': -0.1,
          'logmass': 8.0,
          'd': D0,
          'dust2': 0.3,
          'dust1': 0.3}


# Default chain dispersion for each parameter
SIGMA0 = {'logtau': 0.1,
          'const': 0.1,
          'sf_start': 0.1,
          'tburst': 0.1,
          'fburst': 0.1,
          'logZZsol': 0.1,
          'logmass': 0.1,
          'd': 0.5 * D0_SIGMA,
          'dust1': 0.1,
          'dust2': 0.1}


PARAM_NAMES = ['logmass', 'logZZsol', 'sf_start', 'logtau', 'const', 'dust1',
               'dust2']


def main():
    sed, sed_err = build_test_data(N_PIXELS, BANDS)
    print "Model SB", mjy_to_ab_sb(sed, PIX_AREA)


def build_test_data(n_pixels, bands):
    n_bands = len(bands)

    seds = np.empty((n_pixels, n_bands), dtype=np.float)
    sederrs = np.empty((n_pixels, n_bands), dtype=np.float)

    sp = fsps.StellarPopulation(**PSET_INIT)
    for i in xrange(n_pixels):
        sp.params['tau'] = 10. ** TRUTHS['logtau'][i]
        sp.params['sf_start'] = TRUTHS['sf_start']
        sp.params['const'] = TRUTHS['const']
        sp.params['dust1'] = TRUTHS['dust1'][i]
        sp.params['dust2'] = TRUTHS['dust2'][i]
        # Get apparent mags at M31
        # Note these are normalized to 1Msun, so scale by the logmass truth
        mags = sp.get_mags(tage=13.8, bands=bands) + 5 * np.log10(D0) - 5.
        # Convert to µJy
        sed_i, sed_err_i = ab_mag_to_mjy(mags, err=0.05)
        seds[i, :] = sed_i * 10. ** TRUTHS['logmass'][i]
        sederrs[i, :] = sed_err_i * 10. ** TRUTHS['logmass'][i]

    # Apply uniform background bias to all pixels
    seds += TRUTHS['B']

    return seds, sederrs


if __name__ == '__main__':
    main()
