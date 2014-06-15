#!/usr/bin/env python
# encoding: utf-8
"""
Plotting tools to analyze MCMC runs.
"""
import os
import numpy as np
import matplotlib as mpl
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.gridspec as gridspec


def chain_plot(path, flatchain, param_names, limits=None,
               truths=None, param_labels=None, figsize=(3.5, 10)):
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
    _prep_plot_dir(path)

    fig = Figure(figsize=figsize)
    canvas = FigureCanvas(fig)
    gs = gridspec.GridSpec(len(param_names), 1, left=0.2, right=0.95,
                           bottom=0.05, top=0.99,
                           wspace=None, hspace=0.1,
                           width_ratios=None, height_ratios=None)
    axes = {}
    steps = np.arange(len(flatchain))
    for i, name in enumerate(param_names):
        ax = fig.add_subplot(gs[i])
        ax.scatter(steps, flatchain[name],
                   s=1, marker='o',
                   facecolors='k', edgecolors='None',
                   rasterized=True)
        ax.set_xlim(steps.min(), steps.max())
        # if limits is not None and name in limits:
        #     ax.set_ylim(*limit)
        ax.set_ylabel(name)
        axes[name] = ax
        if i < len(param_names) - 1:
            for tl in ax.get_xmajorticklabels():
                tl.set_visible(False)
    axes[param_names[-1]].set_xlabel('steps')

    gs.tight_layout(fig, pad=1.08, h_pad=None, w_pad=None, rect=None)
    canvas.print_figure(path + ".pdf", format="pdf")


def triangle_plot(path, flatchain, param_names, limits,
                  figsize=(5, 5), truths=None, param_labels=None):
    """Make a corner plot using the triangle.py package.
    
    Triangle/corner plots are convenient visualizations of posterior
    distributions and covariances between parameters.

    samples = sampler.chain[:, nburn:, :].reshape((-1, ndim))

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
    from triangle import corner

    n_steps = len(flatchain)
    n_cols = len(param_names)
    samples = np.empty((n_steps, n_cols), dtype=float)
    for i, n in enumerate(param_names):
        samples[:, i] = flatchain[n]

    if param_labels:
        _labels = param_labels
    else:
        _labels = [escape_latex(n) for n in param_names]

    _prep_plot_dir(path)
    fig = corner(samples, labels=_labels, extents=limits, truths=truths,
                 truth_color="#4682b4", scale_hist=False, quantiles=[],
                 verbose=True, plot_contours=True, plot_datapoints=True,
                 fig=None)
    fig.savefig(path)
    mpl.pyplot.close(fig)


def _prep_plot_dir(path):
    plotdir = os.path.dirname(path)
    if len(plotdir) > 0 and not os.path.exists(plotdir):
        os.makedirs(plotdir)


def escape_latex(text):
    """Escape a string for matplotlib latex processing.
    
    Run this function on any FSPS parameter names before passing to the
    plotting functions in this module as axis labels.
    
    Parameters
    ----------
    text : str
        String to escape.
    """
    return text.replace("_", "\_")
