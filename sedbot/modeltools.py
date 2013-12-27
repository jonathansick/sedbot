#!/usr/bin/env python
# encoding: utf-8
"""
Utilities for setting up models.
"""

def reset_seed_limits(start_points, lower, upper):
    """In-place replacement of starting point seed values that are above/below
    specified limits.
    
    This function is useful in :func:`init_chain` functions. If initial points
    are generated from a Gaussian distribution, then they may randomly scatter
    below or above hard physical limits. This fnction resets those low/high
    values to be precisely the limiting value.

    Parameters
    ----------
    start_points : ndarray
        Array view of starting points. Values are replaced in-place (there is
        no return value).
    lower : float
        Low limit.
    upper : float
        High limit.
    """
    start_points[start_points < lower] = lower
    start_points[start_points > upper] = upper
