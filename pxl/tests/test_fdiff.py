from __future__ import division, print_function
from .. import fdiff
from ..fdiff import *
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
from uncertainties import unumpy


def test_second_order_diff(plot=False):
    # Create a non-equally spaced x vector
    x = np.append(np.linspace(0, np.pi, 100),
                  np.linspace(np.pi + 0.01, 2*np.pi, 400))
    dx = x[1] - x[0]
    u = np.sin(x)
    dudx = second_order_diff(u, x)
    # Assert that this function is almost identical to cos(x)
    np.testing.assert_allclose(dudx, np.cos(x), rtol=1e-3)
    if plot:
        plt.plot(x, u)
        plt.plot(x, dudx)
        plt.plot(x, np.cos(x), "--")
        plt.show()
