#!/usr/bin/env python
# encoding: utf-8
"""
Plot a SED and spectrum for a stellar population.
"""

import fsps
import numpy as np

# import matplotlib as mpl
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.gridspec as gridspec

from sedbot.photconv import abs_ab_mag_to_mjy
from sedbot.plots.sedplot import plot_sed_points, label_filters


def main():
    d = 10.
    bands = ['2mass_j', '2mass_h', '2mass_ks',
             'sdss_u', 'sdss_g', 'sdss_r', 'sdss_i', 'sdss_z',
             'galex_nuv', 'galex_fuv',
             'irac_1', 'irac_2', 'irac_3', 'irac_4',
             'wise_w1', 'wise_w2', 'wise_w3', 'wise_w4',
             'mips_24', 'mips_70', 'mips_160',
             'scuba_450wb', 'scuba_850wb',
             'pacs_70', 'pacs_100', 'pacs_160',
             'spire_250', 'spire_350', 'spire_500']
    sp = fsps.StellarPopulation(compute_vega_mags=False,
                                add_dust_emission=True,
                                sfh=1,
                                tau=10.,
                                sf_start=3.,
                                const=0.1,
                                fburst=0.1,
                                tburst=13.3,
                                zmet=20,
                                dust_type=2,
                                dust2=0.3)
    sed_mjy = abs_ab_mag_to_mjy(sp.get_mags(tage=13.8, bands=bands), d)
    wave, spec = sp.get_spectrum(tage=13.8, peraa=True)
    wave = wave / 10000.  # µm

    fig = Figure(figsize=(6, 6))
    canvas = FigureCanvas(fig)
    gs = gridspec.GridSpec(
        2, 1, left=0.15, right=0.95, bottom=0.15, top=0.95,
        wspace=None, hspace=None, width_ratios=None, height_ratios=None)
    ax = fig.add_subplot(gs[0])
    axspec = fig.add_subplot(gs[1])

    ax.set_ylabel(r"$\log(\lambda F_\lambda$)")
    plot_sed_points(ax, sed_mjy, bands)
    label_filters(ax, sed_mjy, bands)
    for tl in ax.get_xmajorticklabels():
        tl.set_visible(False)

    axspec.plot(np.log10(wave), np.log10(spec), c='k', ls='-')
    axspec.set_xlabel(r"$\log(\lambda/\mu \mathrm{m})$")
    axspec.set_ylabel(r"$\log(L/L_\odot/$\AA)")
    axspec.set_ylim(-10., -2.)

    ax.set_xlim(-1, 3)
    axspec.set_xlim(-1, 3)

    gs.tight_layout(fig, pad=1.08, h_pad=None, w_pad=None, rect=None)
    canvas.print_figure("demo_sed.pdf", format="pdf")


if __name__ == '__main__':
    main()
