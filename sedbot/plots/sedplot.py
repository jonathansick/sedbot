#!/usr/bin/env python
# encoding: utf-8
"""
Tools for plotting observed SEDs.
"""

import numpy as np
import fsps

import astropy.units as u
from astropy import constants as const


# Speed of light in cgs
c = 2.997924 * 10. ** 10.  # cm/sec


def plot_sed_points(ax, flux, bands, fluxerr=None, **kwargs):
    """Plot SED as points, possibly with errorbars.

    X-axis is log(Lambda/µm) and y-axis in log(lambda f_lambda) in cgs.

    Parameters
    ----------
    ax : `axes`
        Matplotlib axes.
    flux : ndarray
        SED in micro-Janskies.
    bands : list
        List of `python-fsps` bandpass names, corresponding to flux array.
    fluxerr : ndarray
        Uncertainty in SED flux, micro-Janskies. Interpret as a symmetric
        standard deviation.
    """
    y = np.log10(microJy_to_lambdaFlambda(flux, bands))
    x = np.log10(wavelength_microns(bands))
    if fluxerr is not None:
        settings = {'ecolor': 'k',
                    'elinewidth': None,
                    'capsize': 3,
                    'fmt': '-',
                    'barsabove': False,
                    'errorevery': 1,
                    'capthick': None,
                    'ls': 'None',
                    'color': 'k',
                    'marker': 'o',
                    'ms': 2}
        settings.update(kwargs)
        yhi = np.log10(microJy_to_lambdaFlambda(flux + fluxerr, bands))
        ylo = np.log10(microJy_to_lambdaFlambda(flux - fluxerr, bands))
        yerrhi = yhi - y
        yerrlo = y - ylo
        yerr = np.vstack([yerrhi, yerrlo])
        ax.errorbar(x, y, yerr=yerr, **settings)
    else:
        settings = {'color': 'k',
                    'marker': 'o',
                    's': 4}
        settings.update(kwargs)
        ax.scatter(x, y, **settings)


def plot_sed_error_band(ax, lower_flux, upper_flux, bands, **kwargs):
    """Plot an SED confidence interface as a continuous band. Useful for
    plotting the distribution of model SEDs in the MCMC chain.

    Parameters
    ----------
    ax : `axes`
        Matplotlib axes.
    lower_flux : ndarray
        Lower confidence limit of SED in micro-Janskies.
    upper_flux : ndarray
        Upper confidence limit of SED in micro-Janskies.
    bands : list
        List of `python-fsps` bandpass names, corresponding to flux array.
    """
    settings = {'edgecolor': 'None',
                'facecolor': 'y',
                'alpha': 0.5,
                'interpolate': True}
    settings.update(kwargs)
    x = np.log10(wavelength_microns(bands))
    s = np.argsort(x)
    upper = np.log10(microJy_to_lambdaFlambda(upper_flux, bands))
    lower = np.log10(microJy_to_lambdaFlambda(lower_flux, bands))
    ax.fill_between(x[s], lower[s], y2=upper[s], **settings)


def label_filters(ax, flux, bands, **kwargs):
    """Plot filter labels above each SED plot."""
    y = np.log10(microJy_to_lambdaFlambda(flux, bands))
    x = np.log10(wavelength_microns(bands))
    s = np.argsort(x)
    x = x[s]
    y = y[s]
    labels = [bands[i] for i in s]
    for xi, yi, label in zip(x, y, labels):
        label = label.replace("_", "\_")
        # ax.text(xi, yi, label, rotation='vertical', fontsize=9.)
        ax.annotate(label, (xi, yi),
                    textcoords='offset points',
                    xytext=(0., 10.),
                    fontsize=5.,
                    rotation='vertical',
                    ha='center',
                    va='bottom',
                    zorder=-5)


def microJy_to_lambdaFlambda(flux, bands):
    # see http://coolwiki.ipac.caltech.edu/index.php/Units#Notes_on_plotting
    Fnu = flux * u.microJansky
    lmbda = wavelength_microns(bands) * u.micron
    # lmbda_cm = lmbda / 10000.
    Fnu_cgs = Fnu.to(u.erg / u.cm**2 / u.s / u.Hz,
                     equivalencies=u.spectral_density(lmbda))
    # Fcgs = flux * 10. ** (6. - 23)
    # lambdaFlambda = lmbda_cm * Fcgs
    Flambda = Fnu_cgs * const.c / lmbda ** 2.
    # print 'F_lambda', Flambda
    # print Flambda.decompose(bases=[u.erg, u.cm, u.s])
    lambdaFlambda = (lmbda * Flambda).decompose(bases=[u.erg, u.cm, u.s])
    print 'lambda F_lambda', lambdaFlambda

    f_sun = u.L_sun.cgs / (4. * np.pi * const.au.cgs ** 2)

    return lambdaFlambda / f_sun


def wavelength_microns(bands):
    lambdas = []
    for band in bands:
        filtr = fsps.fsps.FILTERS[band.lower()]
        lambdas.append(filtr.lambda_eff / 10000.)
    return np.array(lambdas)
