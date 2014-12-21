#!/usr/bin/env python
# encoding: utf-8
"""
Photometric conversion utilities.

FSPS generates quantities in magnitudes, while sedbot's likelihood functions
operate on flux densities in micro Janksies (µJy). FSPS also provides
spectra in units of solar-luminosity per Hz (:math:`f_\\nu`). This module
provides utilties for converting between these quantities.

See http://en.wikipedia.org/wiki/Jansky and
http://en.wikipedia.org/wiki/AB_magnitude for background on these systems.
"""

import numpy as np


# Zeropoint factor for converting mags to µJy
# µJy = 10.^6 * 10.^23 * 10^(-48.6/2.5) = MICROJY_ZP * 10.^(-AB/2.5)
# see http://en.wikipedia.org/wiki/Jansky
MICROJY_ZP = 10. ** 6. * 10. ** 23. * 10. ** (-48.6 / 2.5)


def ab_mag_to_micro_jy(mags, err=None):
    r"""Convert scalar (or array) AB magnitudes to µJy.

    That is, we apply the transformation:

    .. math::
       f_\nu & = 10^6 10^{23} 10^{-48.6/2.5} 10^{-m/2.5}~\mathrm{[µJy]}

    Parameters
    ----------
    mags : ndarray
        AB magnitudes, either a scalar or an ``ndarray``.
    err : ndarray
        Optional array of magnitude uncertainties (1-sigma).

    Returns
    -------
    mjy : ndarray
        Flux densities, in micro Janskies (µJy). Same shape as input ``mags``
        array.
    mjy_err : ndarray
        Flux uncertainty, in micro Janskies (µJy) if `err` is specified.
    """
    mjy = MICROJY_ZP * np.power(10., -mags / 2.5)
    if err is not None:
        mjy_err = (mjy * err) / 1.0875
        return mjy, mjy_err
    else:
        return mjy


def ab_sb_to_micro_jy(mu, area, err=None):
    r"""Convert scalar (or array) AB surface brightness to µJy.

    That is, we apply the transformation:

    .. math::
       f_\nu & = A 10^6 10^{23} 10^{-48.6/2.5} 10^{-\mu/2.5}~\mathrm{[µJy]}

    Parameters
    ----------
    mu : ndarray
        AB surface brightnesses (mag AB per arcsec^2), either a scalar or an
        ``ndarray``.
    area : ndarray
        Area, in square arcseconds. Must be the same shape as `sb`.
    err : ndarray
        Optional array of magnitude uncertainties (1-sigma).

    Returns
    -------
    mjy : ndarray
        Flux densities, in micro Janskies (µJy). Same shape as input ``mags``
        array.
    mjy_err : ndarray
        Flux uncertainty, in micro Janskies (µJy) if `err` is specified.
    """
    mjy = area * MICROJY_ZP * np.power(10., -mu / 2.5)
    if err is not None:
        mjy_err = (mjy * err) / 1.0875
        return mjy, mjy_err
    else:
        return mjy


def micro_jy_to_ab_sb(mjy, area, err=None):
    """Convert a flux in µJy to a surface brightness.

    Parameters
    ----------
    mjy : ndarray
        Flux densities, in micro Janskie (µJy), either a scalar or an
        ``ndarray``.
    area : ndarray
        Area, in square arcseconds.
    err : ndarray
        Optional array of flux density uncertainties (1-sigma).

    Returns
    -------
    mjy : ndarray
        Surface brightness.
        array.
    mjy_err : ndarray
        Surface brightness uncertainty, if ``err`` was specified.
    """
    sb = - 2.5 * np.log10(mjy / area / MICROJY_ZP)
    if err is not None:
        sb_err = 1.0875 * err / mjy
        return sb, sb_err
    else:
        return sb


def abs_ab_mag_to_micro_jy(mags, parsecs):
    r"""Convert scalar (or array) absolute AB magnitudes to µJy assuming a
    distance in parsecs.

    That is, we apply the transformation:

    .. math::
       m &= M - 5 (1 - \log_{10}d) \\
       f_\nu & = 10^6 10^{23} 10^{-48.6/2.5} 10^{-m/2.5}~\mathrm{[µJy]}

    This function is useful for converting magnitudes generated by FSPS into
    the same system as observations.

    Parameters
    ----------
    mags : ndarray
        AB magnitudes, either a scalar or an ``ndarray``.
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


def sb_to_mass(sb, msun, logml, area, d):
    """Compute :math:`\log \mathcal{M}_*` given a surface brightness
    and mass to light ratio.

    This can be useful for establishing a prior on the total stellar mass
    in a pixel given prior knowledge of M/L ratio.

    Parameters
    ----------
    sb : ndarray
        Surface brightness (mag/arcsec^2)
    msun : float
        Absolute magnitude of the sun. This can be obtained from
    logml : ndarray
        Log of M/L (stellar mass to light ratio).
    area : ndarray
        Area of region/pixel in square arcseconds.
    d : ndarray
        Distance in parsecs.
    """
    return logml + sb_to_luminosity(sb, msun, area, d)


def sb_to_luminosity(sb, msun, A, d):
    """Convert a surface brightness to
    :math:`\log \mathcal{L}/\mathcal{L}_\odot`, the luminosity in solar units
    of a region of projected area `A`.

    Parameters
    ----------
    sb : ndarray
        Surface brightness (mag/arcsec^2)
    msun : float
        Absolute magnitude of the Sun.
    area : ndarray
        Area of region in square arcseconds.
    d : ndarray
        Distance in parsecs.

    Returns
    -------
    logLsun : ndarray
        Log of luminosity in solar units.
    """
    m = sb - 5. * np.log10(d) + 5. - msun
    return np.log10(A * 10. ** (-0.4 * m))


def micro_jy_to_luminosity(mjy, msun, d):
    """Convert an SED in µJy to log solar luminosities.

    Parameters
    ----------
    mjy : ndarray
        Flux in microjankies.
    msun : float
        Absolute magnitude of the Sun.
    d : ndarray
        Distance in parsecs.

    Returns
    -------
    logLsun : ndarray
        Log of luminosity in solar units.
    """
    dmod = 5. * np.log10(d) - 5.
    AB = 23.9 - 2.5 * np.log10(mjy)
    logLsol = -0.4 * (AB - dmod - msun)
    return logLsol
