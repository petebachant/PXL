# -*- coding: utf-8 -*-
"""
Finite difference utilities
"""

import numpy as np


def second_order_diff(arr, x):
    """Compute second order difference of an array.

    A 2nd order forward difference is used for the first point, 2nd order
    central difference for interior, and 2nd order backward difference for last
    point, returning an array the same length as the input array.
    """
    # Convert to array, so this will work with pandas Series
    arr = np.array(arr)
    # Calculate dx for forward diff point
    dxf = (x[2] - x[0])/2
    # Calculate dx for backward diff point
    dxb = (x[-1] - x[-3])/2
    # Calculate dx array for central difference
    dx = (x[2:] - x[:-2])/2
    # For first data point, use 2nd order forward difference
    first = (-3*arr[0] + 4*arr[1] - arr[2])/(2*dxf)
    # For last point, use 2nd order backward difference
    last = (3*arr[-1] - 4*arr[-2] + arr[-3])/(2*dxb)
    # For all interior points, use 2nd order central difference
    interior = (arr[2:] - arr[:-2])/(2*dx)
    # Create entire array
    darr = np.concatenate(([first], interior, [last]))
    return darr


if __name__ == "__main__":
    pass
