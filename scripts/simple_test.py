#!/usr/bin/env python
# encoding: utf-8
"""
Simple test of FSPS.
"""

import fsps


def main():
    sp = fsps.StellarPopulation(imf_type=2, dust_type=1, mwr=3.1, dust2=0.3)
    bands = ['2MASS_J', '2MASS_Ks']
    tage = 10. ** (8.04 - 9)
    mags = sp.get_mags(zmet=20, tage=tage, bands=bands)
    print zip(bands, mags)


if __name__ == '__main__':
    main()
