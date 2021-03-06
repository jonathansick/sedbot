#!/usr/bin/env python
# encoding: utf-8
"""
Plot the timeseries of emcee samples.
"""

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator

from .tools import prep_plot_dir


def chain_plot(path, flatchain, param_names, limits=None,
               truths=None, param_labels=None, figsize=(5, 10)):
    """Diagnostic plot of walker chains.

    The chain plot shows lineplots of each walker, for each parameter. This
    plot can be useful for assessing convergence, and establishing an
    appropriate burn-in limit.

    Parameters
    ----------
    path : str
        Path where the corner plot will be saved (as a PDF file).
    flatchain : :class:`astropy.table.Table`
        The flattened chain table.
        A flattened chain of emcee samples. To obtain a flat chain with
        burn-in steps removed, use
        `samples = sampler.chain[:, nburn:, :].reshape((-1, ndim))`
    param_names : list (ndim,)
        Sequence of strings identifying parameters (columns) to plot
    limits : list (ndim,)
        Sequence of `(lower, upper)` tuples defining the extent of each
        parameter. Must be the same length and order as `param_names` and
        parameters in the sampler's chain.
    truths : list (ndim,)
        True values for each parameter.
    param_labels : list
        Optional list of names for each parameter to be used for the plot
        itself.
    """
    prep_plot_dir(path)

    fig = Figure(figsize=figsize)
    canvas = FigureCanvas(fig)
    gs = gridspec.GridSpec(len(param_names), 1, left=0.25, right=0.92,
                           bottom=0.05, top=0.95,
                           wspace=None, hspace=0.2,
                           width_ratios=None, height_ratios=None)
    axes = {}
    steps = np.arange(len(flatchain))
    for i, name in enumerate(param_names):
        ax = fig.add_subplot(gs[i])
        if name == "d":
            vals = flatchain['d'] / 1000.
        else:
            vals = flatchain[name]
        ax.scatter(steps, vals,
                   s=1, marker='o',
                   facecolors='k', edgecolors='None',
                   rasterized=True)
        ax.set_xlim(steps.min(), steps.max())
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        # if limits is not None and name in limits:
        #     ax.set_ylim(*limit)
        if param_labels:
            ax.set_ylabel(param_labels[i])
        else:
            ax.set_ylabel(name)
        axes[name] = ax
        if i < len(param_names) - 1:
            for tl in ax.get_xmajorticklabels():
                tl.set_visible(False)
    axes[param_names[-1]].set_xlabel('steps')

    gs.tight_layout(fig, pad=1.08, h_pad=None, w_pad=None, rect=None)
    canvas.print_figure(path + ".pdf", format="pdf")
