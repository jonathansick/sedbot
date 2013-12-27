#!/usr/bin/env python
# encoding: utf-8
"""
Tests for the :mod:`sedbot.probf` module.
"""

import numpy as np
from sedbot.probf import ln_uniform_factory


def test_ln_uniform_factory():
    limits = (0., 1.)
    f = ln_uniform_factory(*limits)
    assert np.isfinite(f(-1.)) == False
    assert np.isfinite(f(2.)) == False
    assert f(0.) == 0.
    assert f(0.5) == 0.
    assert f(1.) == 0.
