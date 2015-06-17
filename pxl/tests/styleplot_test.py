# -*- coding: utf-8 -*-
"""
Tests for the pxl.styleplot module.
"""
from pxl.styleplot import *

def test_set_sns(**args):
    """
    Test the Seaborn plot styling.
    """
    print("Testing styleplot.set_sns")
    set_sns(**args)
    plt.plot([1, 2, 3], label="Test")
    plt.xlabel(r"$T^{est}$")
    plt.ylabel("Test")
    plt.legend()
    plt.show()