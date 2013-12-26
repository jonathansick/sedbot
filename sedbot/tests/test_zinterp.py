#!/usr/bin/env python
# encoding: utf-8
"""
Tests for zinterp module.
"""

from sedbot.zinterp import bracket_logz, interp_logz


def test_bottom_bracket_logz():
    """Test for zinterp.bracket_logz()"""
    ZZsol = -1.98
    zmet1, zmet2 = bracket_logz(ZZsol)
    assert zmet1 == 1 and zmet2 == 2

    f1 = 2.
    f2 = 4.
    f = interp_logz(zmet1, zmet2, ZZsol, f1, f2)
    assert f == f1


def test_low_bracket_logz():
    """Test Z smaller than grid"""
    zmet1, zmet2 = bracket_logz(-2.0)
    assert zmet1 == 1 and zmet2 == 2

def test_top_bracket_logz():
    """Test for Z in top bracket"""
    zmet1, zmet2 = bracket_logz(0.20 - 0.01)
    assert zmet1 == 21 and zmet2 == 22


def test_high_bracket_logz():
    """Test for Z above grid."""
    # Test too high Z
    logZZsol = 0.2 + 0.01
    zmet1, zmet2 = bracket_logz(logZZsol)
    assert zmet1 == 21 and zmet2 == 22

    f1 = 2.
    f2 = 4.
    f = interp_logz(zmet1, zmet2, logZZsol, f1, f2)
    assert f == f2


def test_super_solar_bracket():
    """Test for bracket slightly above solar metallicity."""
    zmet1, zmet2 = bracket_logz(0.05)
    assert zmet1 == 20 and zmet2 == 21
