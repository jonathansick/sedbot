#!/usr/bin/env python
# encoding: utf-8
"""
Simple test of FSPS.
"""

import fsps


def main():
    sp = fsps.StellarPopulation(sfh=1, imf_type=2, dust_type=1, mwr=3.1,
            dust2=0.3)
    bands = ['2MASS_J', '2MASS_Ks']
    mags = sp.get_mags(zmet=20, tage=0, bands=bands)
    print zip(bands, mags)
    print "mass", sp.stellar_mass
    print "mdust", sp.dust_mass
    print "log lbol", sp.log_lbol
    print "log age", sp.log_age
    print "sfr", sp.sfr


if __name__ == '__main__':
    main()
