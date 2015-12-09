# -*- coding: utf-8 -*-
"""
A collection of useful time series analysis functions.
"""

from __future__ import division, print_function
import numpy as np
from scipy.signal import lfilter
import scipy.stats
import pandas as _pd
from .io import *


def sigmafilter(data, sigmas, passes):
    """Remove datapoints outside of a specified standard deviation range."""
    for n in range(passes):
        meandata = np.mean(data[~np.isnan(data)])
        sigma = np.std(data[~np.isnan(data)])
        data[data > meandata+sigmas*sigma] = np.nan
        data[data < meandata-sigmas*sigma] = np.nan
    return data


def psd(t, data, window=None, n_band_average=1):
    """Compute one-sided power spectral density, subtracting mean automatically.

    Parameters
    ----------
    t : Time array
    data : Time series data
    window : {None, "Hanning"}
    n_band_average : Number of samples over which to band average

    Returns
    -------
    f : Frequency array
    psd : Spectral density array
    """
    dt = t[1] - t[0]
    N = len(data)
    data = data - np.mean(data)
    if window == "Hanning":
        data = data*np.hanning(N)
    f = np.fft.fftfreq(N, dt)
    y = np.fft.fft(data)
    f = f[0:N/2]
    psd = (2*dt/N)*abs(y)**2
    psd = np.real(psd[0:N/2])
    if n_band_average > 1:
        f_raw, s_raw = f*1, psd*1
        f = np.zeros(len(f_raw)//n_band_average)
        psd = np.zeros(len(f_raw)//n_band_average)
        for n in range(len(f_raw)//n_band_average):
            f[n] = np.mean(f_raw[n*n_band_average:(n+1)*n_band_average])
            psd[n] = np.mean(s_raw[n*n_band_average:(n+1)*n_band_average])
    return f, psd


def runningstd(t, data, width):
    """Compute the running standard deviation of a time series.

    Returns `t_new`, `std_r`.
    """
    ne = len(t) - width
    t_new = np.zeros(ne)
    std_r = np.zeros(ne)
    for i in range(ne):
        t_new[i] = np.mean(t[i:i+width+1])
        std_r[i] = scipy.stats.nanstd(data[i:i+width+1])
    return t_new, std_r


def smooth(data, fw):
    """Smooth data with a moving average."""
    if fw == 0:
        fdata = data
    else:
        fdata = lfilter(np.ones(fw)/fw, 1, data)
    return fdata


def calcstats(data, t1, t2, sr):
    """Calculate the mean and standard deviation of some array between
    t1 and t2 provided the sample rate sr.
    """
    dataseg = data[sr*t1:sr*t2]
    meandata = np.mean(dataseg[~np.isnan(dataseg)])
    stddata = np.std(dataseg[~np.isnan(dataseg)])
    return meandata, stddata


def average_over_area(q, x, y):
    """Averages a quantity `q` over a rectangular area given a 2D array and
    the x and y vectors for sample locations, using the trapezoidal rule"""
    area = (np.max(x) - np.min(x))*(np.max(y) - np.min(y))
    integral = np.trapz(np.trapz(q, y, axis=0), x)
    return integral/area


def build_plane_arrays(x, y, qlist):
    """Build a 2-D array out of data taken in the same plane, for contour
    plotting.
    """
    if type(qlist) is not list:
        return_list = False
        qlist = [qlist]
    else:
        return_list = True
    xv = x[np.where(y==y[0])[0]]
    yv = y[np.where(x==x[0])[0]]
    qlistp = []
    for n in range(len(qlist)):
        qlistp.append(np.zeros((len(yv), len(xv))))
    for j in range(len(qlist)):
        for n in range(len(yv)):
            i = np.where(y==yv[n])[0]
            qlistp[j][n,:] = qlist[j][i]
    if not return_list:
        qlistp = qlistp[0]
    return xv, yv, qlistp


def corr_coeff(x1, x2, t, tau1, tau2):
    """Compute lagged correlation coefficient for two time series."""
    dt = t[1] - t[0]
    tau = np.arange(tau1, tau2+dt, dt)
    rho = np.zeros(len(tau))
    for n in range(len(tau)):
        i = np.abs(int(tau[n]/dt))
        if tau[n] >= 0: # Positive lag, push x2 forward in time
            seg2 = x2[0:-1-i]
            seg1 = x1[i:-1]
        elif tau[n] < 0: # Negative lag, push x2 back in time
            seg1 = x1[0:-i-1]
            seg2 = x2[i:-1]
        seg1 = seg1 - seg1.mean()
        seg2 = seg2 - seg2.mean()
        rho[n] = np.mean(seg1*seg2)/seg1.std()/seg2.std()
    return tau, rho


def autocorr_coeff(x, t, tau1, tau2):
    """Calculate the autocorrelation coefficient."""
    return corr_coeff(x, x, t, tau1, tau2)


def integral_scale(u, t, tau1=0.0, tau2=1.0):
    """Calculate the integral scale of a time series by integrating up to
    the first zero crossing.
    """
    tau, rho = autocorr_coeff(u, t, tau1, tau2)
    zero_cross_ind = np.where(np.diff(np.sign(rho)))[0][0]
    int_scale = np.trapz(rho[:zero_cross_ind], tau[:zero_cross_ind])
    return int_scale
