#!/usr/bin/env python
# encoding: utf-8
"""
SED corrections for MW extinction (for extragalactic sources).
"""

from pkg_resources import resource_stream, resource_exists

import numpy as np
from astropy.table import Table


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
        Name of the bandpass. Use :func:`sedbot.mwextinction.sd2001_bandpasses`
        to get a list of valid bandpass names.
    Rv : str
        The value of :math:`R_V`, as a string. Can be ``'2.1'``, ``'3.1'``
        ``'4.1'``, and ``'5.1'``. Default is ``'3.1'``.
    """
    assert Rv in ('2.1', '3.1', '4.1', '5.1')
    tbl = load_sd2011()
    i = np.where(tbl['bandpass'] == bandname)[0][0]
    X = float(tbl[i]["R_V_{0}".format(Rv)])
    return mag - X * eBV


def sd2001_bandpasses():
    tbl = load_sd2011()
    bands = list(tbl['bandpass'])
    print bands


SD2011_CACHE = None  # cache for load_sd2011()


def load_sd2011():
    """Load Table 6 from Schlafly & Finkbeiner 2011 for ratios of
    :math:`A_X / E(B-V)`.

    The table is cached for subsequent calls.

    Returns
    -------
    tbl : :class:`astropy.table.Table`
        SD2011 Table 6 as an astropy table.
    """
    global SD2011_CACHE
    if SD2011_CACHE is None:
        path = "data/schlafly_finkbeiner_2011_table6.txt"
        assert resource_exists(__name__, path)
        tbl = Table.read(resource_stream(__name__, path),
                         delimiter='\t',
                         guess=False,
                         quotechar="'",
                         format="ascii.commented_header")
        SD2011_CACHE = tbl
    return SD2011_CACHE
