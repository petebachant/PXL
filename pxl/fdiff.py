# -*- coding: utf-8 -*-
"""
fdiff
=====
Finite difference utilities

Created on Wed Jan 01 21:18:17 2014
@author: Pete

To-do
-----
  * Add multi-dimensional array support
  
"""
import numpy as np

def second_order_diff(arr, x):
    """Computes second order difference of an array, using 2nd order forward
    difference for first point, 2nd order central difference for interior, 
    and 2nd order backward difference for last point, returning an array the
    same length as the input array.
    """
    # Convert to array, so this will work with pandas Series
    arr = np.array(arr)
    # Calculate dx for forward diff point
    dxf = (x[2] - x[0])/2
    # Calculate dx for backward diff point
    dxb = (x[-1] - x[-3])/2
    # Calculate dx array for central difference
    dx = (x[2:] - x[:-2])/2
    # Pre-allocate zeros array for difference array
    darr = np.zeros(len(arr))
    # For first data point, use 2nd order forward difference
    darr[0] = (-3*arr[0] + 4*arr[1] - arr[2])/(2*dxf)
    # For last point, use 2nd order backward difference
    darr[-1] = (3*arr[-1] - 4*arr[-2] + arr[-3])/(2*dxb)
    # For all interior points, use 2nd order central difference
    darr[1:-1] = (arr[2:] - arr[:-2])/(2*dx)
    return darr
    
if __name__ == "__main__":
    x = np.append(np.linspace(0, np.pi,50), np.linspace(np.pi+0.01, 2*np.pi, 200))
    dx = x[1] - x[0]
    u = np.sin(x)
    dudx = second_order_diff(u, x)
    import matplotlib.pyplot as plt
    plt.close("all")
    plt.plot(x, u)
    plt.hold(True)
    plt.plot(x, dudx)
    plt.plot(x, np.cos(x), "--")
    plt.show()
