#!/usr/bin/env python
# encoding: utf-8
"""
Tests for the :mod:`sedbot.probf` module.
"""

import numpy as np
from sedbot.probf import ln_normal_factory, ln_uniform_factory


def test_ln_uniform_factory():
    limits = (0., 1.)
    f = ln_uniform_factory(*limits)
    assert np.isfinite(f(-1.)) == False
    assert np.isfinite(f(2.)) == False
    assert f(0.) == 0.
    assert f(0.5) == 0.
    assert f(1.) == 0.


def test_ln_normal_factory_limits():
    f = ln_normal_factory(0., 1., limits=(-3., 3.))
    assert np.isfinite(f(-4.)) == False
    assert np.isfinite(f(4.)) == False
    assert np.isfinite(f(0.)) == True
    assert np.isfinite(f(-3.)) == True
    assert np.isfinite(f(3.)) == True
