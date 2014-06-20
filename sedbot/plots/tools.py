#!/usr/bin/env python
# encoding: utf-8
"""
Plotting utilities.
"""

import os


def prep_plot_dir(path):
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
