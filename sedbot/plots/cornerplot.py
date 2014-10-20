#!/usr/bin/env python
# encoding: utf-8
"""
Triangle plot.
"""

import numpy as np

import matplotlib as mpl
import triangle

from .tools import prep_plot_dir, escape_latex


def triangle_plot(path, flatchain, param_names, limits=None,
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
    n_steps = len(flatchain)
    n_cols = len(param_names)
    samples = np.empty((n_steps, n_cols), dtype=float)
    for i, n in enumerate(param_names):
        samples[:, i] = flatchain[n]

    if param_labels:
        _labels = param_labels
    else:
        _labels = [escape_latex(n) for n in param_names]

    prep_plot_dir(path)
    fig = triangle.corner(samples, labels=_labels, extents=limits,
                          truths=truths,
                          truth_color="#4682b4", scale_hist=False,
                          quantiles=[],
                          verbose=True, plot_contours=True,
                          plot_datapoints=True,
                          fig=None)
    fig.savefig(path)
    mpl.pyplot.close(fig)
