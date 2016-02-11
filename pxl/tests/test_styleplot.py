# -*- coding: utf-8 -*-
"""Tests for the pxl.styleplot module."""

from pxl.styleplot import *


def test_set_sns(**args):
    """Test the Seaborn plot styling."""
    set_sns(**args)
    plt.plot(range(10), marker="o", mfc="none", mec="b", label="Unfilled")
    plt.plot(range(1, 6), marker="s", color="r", label="Seaborn red")
    plt.xlabel(r"$T^{est}$")
    plt.ylabel("Test")
    plt.text(x=5, y=1.5, s="Here's some text")
    plt.legend()
    plt.tight_layout()
    plt.show()
