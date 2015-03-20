#!/usr/bin/env python
# encoding: utf-8
"""
SED corrections for MW extinction (for extragalactic sources).
"""

from pkg_resources import resource_stream, resource_exists

import numpy as np
from astropy.table import Table


# Mapping of fsps band names to SF 2011 extinction table
SF_BAND_NAMES = {'sdss_u': 'SDSS u',
                 'sdss_g': 'SDSS g',
                 'sdss_r': 'SDSS r',
                 'sdss_i': 'SDSS i',
                 '2mass_J': 'UKIRT J',
                 '2mass_Ks': 'UKIRT K',
                 'wfc3_ir_f110w': 'WFC3 F110W',
                 'wfc3_ir_f160w': 'WFC3 F160W',
                 'wfc3_uvis_f275w': 'WFC3 F275W',
                 'wfc3_uvis_f336w': 'WFC3 F336W',
                 'wfc_acs_f475w': 'ACS F475W',
                 'wfc_acs_f606w': 'ACS F606W',
                 'wfc_acs_f814w': 'ACS F814W'}


def get_fsps_extinction_coefficient(fsps_band, Rv='3.1'):
    """Get the extinction coefficient for this bandpass, in FSPS band naming,
    based on SF2011.

    Parameters
    ----------
    bandname : str
        Name of the FSPS bandpass.
    Rv : str
        The value of :math:`R_V`, as a string. Can be ``'2.1'``, ``'3.1'``
        ``'4.1'``, and ``'5.1'``. Default is ``'3.1'``.
    """
    sf_band = SF_BAND_NAMES[fsps_band]
    return get_extinction_coefficient(sf_band, Rv=Rv)


def get_extinction_coefficient(bandname, Rv='3.1'):
    """Get the extinction coefficient for this bandpass, based on SF2011.

    Parameters
    ----------
    bandname : str
        Name of the bandpass. Use :func:`sedbot.mwextinction.sf2001_bandpasses`
        to get a list of valid bandpass names.
    Rv : str
        The value of :math:`R_V`, as a string. Can be ``'2.1'``, ``'3.1'``
        ``'4.1'``, and ``'5.1'``. Default is ``'3.1'``.
    """
    assert Rv in ('2.1', '3.1', '4.1', '5.1')
    tbl = load_sf2011()
    i = np.where(tbl['bandpass'] == bandname)[0][0]
    X = float(tbl[i]["R_V_{0}".format(Rv)])
    return X


def correct_mw_extinction(mag, eBV, bandname, Rv='3.1'):
    """Correct a magnitude (or surface brightness) for MW extinction.

    Using Schlafley & Finkbeiner 2011's conversions of E(B-V) to attenuation in
    an set of bandpasses.

    Parameters
    ----------
    mag : float or `ndarray`
        A magnitude or surface brightness.
    eBV : float
        Colour excess, :math:`E(B-V)`.
    bandname : str
        Name of the bandpass. Use :func:`sedbot.mwextinction.sf2001_bandpasses`
        to get a list of valid bandpass names.
    Rv : str
        The value of :math:`R_V`, as a string. Can be ``'2.1'``, ``'3.1'``
        ``'4.1'``, and ``'5.1'``. Default is ``'3.1'``.
    """
    assert Rv in ('2.1', '3.1', '4.1', '5.1')
    X = get_extinction_coefficient(bandname, Rv=Rv)
    return mag - X * eBV


def sf2001_bandpasses():
    tbl = load_sf2011()
    bands = list(tbl['bandpass'])
    print bands


SF2011_CACHE = None  # cache for load_sf2011()


def load_sf2011():
    """Load Table 6 from Schlafly & Finkbeiner 2011 for ratios of
    :math:`A_X / E(B-V)`.

    The table is cached for subsequent calls.

    Returns
    -------
    tbl : :class:`astropy.table.Table`
        SF2011 Table 6 as an astropy table.
    """
    global SF2011_CACHE
    if SF2011_CACHE is None:
        path = "data/schlafly_finkbeiner_2011_table6.txt"
        assert resource_exists(__name__, path)
        tbl = Table.read(resource_stream(__name__, path),
                         delimiter='\t',
                         guess=False,
                         quotechar="'",
                         format="ascii.commented_header")
        SF2011_CACHE = tbl
    return SF2011_CACHE
