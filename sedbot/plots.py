#!/usr/bin/env python
# encoding: utf-8
"""
Plotting tools to analyze MCMC runs.
"""
import os
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.gridspec as gridspec


def chain_plot(path, sampler, param_names, limits):
    """docstring for chain_plot"""
    _prep_plot_dir(path)

    nwalkers, nsteps, ndim = sampler.chain.shape
    steps = np.arange(nsteps)

    fig = Figure(figsize=(3.5, 10))
    canvas = FigureCanvas(fig)
    gs = gridspec.GridSpec(ndim, 1, left=0.2, right=0.95,
            bottom=0.05, top=0.99,
            wspace=None, hspace=0.1, width_ratios=None, height_ratios=None)
    axes = {}
    for i, (name, limit) in enumerate(zip(param_names, limits)):
        ax = fig.add_subplot(gs[i])
        for j in xrange(nwalkers):
            ax.plot(steps, sampler.chain[j, :, i], '-', lw=0.5, alpha=0.5)
        ax.set_xlim(steps.min(), steps.max())
        if limits is not None and name in limits:
            ax.set_ylim(*limit)
        name_clean = name.replace("_", "\_")
        ax.set_ylabel(name_clean)
        axes[name] = ax
        if i < ndim - 1:
            for tl in ax.get_xmajorticklabels():
                tl.set_visible(False)
    axes[param_names[-1]].set_xlabel('steps')

    gs.tight_layout(fig, pad=1.08, h_pad=None, w_pad=None, rect=None)
    canvas.print_figure(path + ".pdf", format="pdf")


def triangle_plot(path, samples, param_names, limits,
        figsize=(5, 5), truths=None):
    """Make a corner plot using the triangle.py package.
    
    Triangle/corner plots are convenient visualizations of posterior
    distributions and covariances between parameters.

    samples = sampler.chain[:, nburn:, :].reshape((-1, ndim))
    """
    from triangle import corner
    _prep_plot_dir(path)
    clean_labels = [l.replace("_", "\_") for l in param_names]
    fig = corner(samples, labels=clean_labels, extents=limits, truths=truths,
            truth_color="#4682b4", scale_hist=False, quantiles=[],
            verbose=True, plot_contours=True, plot_datapoints=True,
            fig=None)
    fig.savefig(path)


def _prep_plot_dir(path):
    plotdir = os.path.dirname(path)
    if len(plotdir) > 0 and not os.path.exists(plotdir):
        os.makedirs(plotdir)
