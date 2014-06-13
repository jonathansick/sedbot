#!/usr/bin/env python
# encoding: utf-8
"""
Tools for plotting observed SEDs.
"""

import numpy as np
import fsps


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
        List of bandpass names, corresponding to flux array.
    fluxerr : ndarray
        Uncertainty in SED flux, micro-Janskies. Interpret as a symmetric
        standard deviation.
    """
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
    y = np.log10(microJy_to_lambdaFlambda(flux, bands))
    x = np.log10(wavelength_microns(bands))
    if fluxerr is not None:
        yhi = np.log10(microJy_to_lambdaFlambda(flux + fluxerr, bands))
        ylo = np.log10(microJy_to_lambdaFlambda(flux - fluxerr, bands))
        yerrhi = yhi - y
        yerrlo = y - ylo
        yerr = np.vstack([yerrhi, yerrlo])
        ax.errorbar(x, y, yerr=yerr, **settings)
    else:
        ax.scatter(x, y, **settings)


def plot_sed_error_band(ax, lower_sed, upper_sed, bands, **kwargs):
    settings = {'edgecolor': 'None',
                'facecolor': 'y',
                'alpha': 0.5}
    settings.update(kwargs)
    x = np.log10(wavelength_microns(bands))
    ax.fill_between(x, lower_sed, y2=upper_sed, interpolate=True, **settings)


def microJy_to_lambdaFlambda(flux, bands):
    # see http://coolwiki.ipac.caltech.edu/index.php/Units#Notes_on_plotting
    lmbda = wavelength_microns(bands)
    lmbda_cm = lmbda / 10000.
    Fcgs = flux * 10. ** (6. - 23)
    lambdaFlambda = lmbda_cm * Fcgs
    return lambdaFlambda


def wavelength_microns(bands):
    lambdas = []
    for band in bands:
        filtr = fsps.fsps.FILTERS[band.lower()]
        lambdas.append(filtr.lambda_eff / 10000.)
    return np.array(lambdas)
