#!/usr/bin/env python
# encoding: utf-8
"""
Plots for the library models.

2015-03-18 - Created by Jonathan Sick
"""

import numpy as np

import matplotlib as mpl

from astroML.stats import binned_statistic_2d

from androphotsys import latex_name


def plot_cc_density(*args, **kwargs):
    kwargs.update(dict(statistic='count',
                       meta_property=None,
                       theta_property=None))
    return plot_cc(*args, **kwargs)


def plot_cc(h5file, group_name, ax, x_bands, y_bands,
            meta_property=None, theta_property=None,
            ml_band=None, values=None,
            xlim=None, ylim=None, statistic='median', bins=100,
            cmap=mpl.cm.cubehelix, value_func=None):
    group = h5file[group_name]
    x = get_colour(group, x_bands)
    y = get_colour(group, y_bands)

    if xlim is not None and ylim is not None:
        # [[xmin, xmax], [ymin, ymax]]
        rng = [[xlim[0], xlim[-1]], [ylim[0], ylim[-1]]]
    else:
        rng = None

    # Get the property to use for the binned statistic, if any
    if meta_property is not None:
        values = group['meta'][meta_property]
    elif theta_property is not None:
        values = group['params'][theta_property]
    elif ml_band is not None:
        values = group['mass_light'][ml_band]
    elif values is not None:
        assert len(values) == len(x)
    else:
        values = None

    H, x_edges, y_edges = binned_statistic_2d(x, y, values,
                                              statistic=statistic,
                                              bins=bins,
                                              range=rng)
    if value_func is not None:
        H = value_func(H)

    im = ax.imshow(H, origin='lower',
                   cmap=cmap,
                   interpolation='nearest',
                   extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]])
    ax.set_xlim(x_edges[0], x_edges[-1])
    ax.set_ylim(y_edges[0], y_edges[-1])

    xlabel = r"${0} - {1}$".format(latex_name(x_bands[0], mathmode=False),
                                   latex_name(x_bands[-1], mathmode=False))
    ax.set_xlabel(xlabel)
    ylabel = r"${0} - {1}$".format(latex_name(y_bands[0], mathmode=False),
                                   latex_name(y_bands[-1], mathmode=False))
    ax.set_ylabel(ylabel)

    return im


def get_colour(group, bands):
    sed_1 = group['seds'][bands[0]]
    sed_2 = group['seds'][bands[1]]
    return -2.5 * np.log10(sed_1 / sed_2)
