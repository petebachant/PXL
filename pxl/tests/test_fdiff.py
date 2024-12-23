from __future__ import division, print_function

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from uncertainties import unumpy

from .. import fdiff
from ..fdiff import *

plot = False


def test_second_order_diff():
    """Test `second_order_diff`."""
    # Create a non-equally spaced x vector
    x = np.append(
        np.linspace(0, np.pi, 100), np.linspace(np.pi + 0.01, 2 * np.pi, 400)
    )
    u = np.sin(x)
    dudx = second_order_diff(u, x)
    assert dudx.shape == u.shape
    # Assert that this function is almost identical to cos(x)
    np.testing.assert_allclose(dudx, np.cos(x), rtol=1e-3)
    if plot:
        plt.plot(x, dudx, "-o", lw=2, alpha=0.5)
        plt.plot(x, np.cos(x), "--^", lw=2, alpha=0.5)
        plt.show()


def test_second_order_diff_uncertainties():
    """Test that `second_order_diff` works with uncertainties."""
    # Create a non-equally spaced x vector
    x = np.append(
        np.linspace(0, np.pi, 50), np.linspace(np.pi + 0.01, 2 * np.pi, 100)
    )
    x_unc = unumpy.uarray(x, np.ones(len(x)) * 1e-3)
    u = unumpy.uarray(np.sin(x), np.ones(len(x)) * 1e-2)
    dudx = second_order_diff(u, x)
    print(dudx[:5])
    print(dudx[-5:])
    if plot:
        plt.errorbar(
            x,
            unumpy.nominal_values(dudx),
            yerr=unumpy.std_devs(dudx),
            fmt="-o",
            lw=2,
            alpha=0.5,
        )
        plt.plot(x, np.cos(x), "--^", lw=2, alpha=0.5)
        plt.show()
