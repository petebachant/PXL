# -*- coding: utf-8 -*-
"""
This is a collection of useful time series analysis functions.

"""
from __future__ import division, print_function
import numpy as np
from scipy.signal import lfilter
import scipy.stats
import json
import pandas as _pd
import h5py as _h5py

def sigmafilter(data, sigmas, passes):
    """Removes datapoints outside of a specified standard deviation range."""
    for n in range(passes):
        meandata = np.mean(data[~np.isnan(data)])
        sigma = np.std(data[~np.isnan(data)])
        data[data > meandata+sigmas*sigma] = np.nan
        data[data < meandata-sigmas*sigma] = np.nan
    return data

def psd(t, data, window=None, n_band_average=1):
    """
    Computes one-sided power spectral density. Subtracts mean. Returns f, psd.
       
    Windows
    -------
      * "Hanning" 
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
    """
    This function computes the running standard deviation of 
    a time series. Returns `t_new`, `std_r`.
    """
    ne = len(t) - width
    t_new = np.zeros(ne)
    std_r = np.zeros(ne)
    for i in range(ne):
        t_new[i] = np.mean(t[i:i+width+1])
        std_r[i] = scipy.stats.nanstd(data[i:i+width+1])
    return t_new, std_r
    
def smooth(data, fw):
    """Smooths data with a moving average."""
    if fw == 0:
        fdata = data
    else:
        fdata = lfilter(np.ones(fw)/fw, 1, data)
    return fdata

def calcstats(data, t1, t2, sr):
    """Calculates the mean and standard deviation of some array between
    t1 and t2 provided the sample rate sr."""
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
    
def savejson(filename, datadict):
    """Saves data from a dictionary in JSON format. Note that this only
    works to the second level of the dictionary with Numpy arrays."""
    for key, value in datadict.items():
        if type(value) == np.ndarray:
            datadict[key] = value.tolist()
        if type(value) == dict:
            for key2, value2 in value.items():
                if type(value2) == np.ndarray:
                    datadict[key][key2] = value2.tolist()
    with open(filename, "w") as f:
        f.write(json.dumps(datadict, indent=4))
    
def loadjson(filename, asnparrays=False):
    """Loads data from text file in JSON format. Numpy arrays are converted
    if specified with the `asnparrays` keyword argument. Note that this only
    works to the second level of the dictionary. Returns a single dict."""
    with open(filename) as f:
        data = json.load(f)
    if asnparrays:
        for key, value in data.items():
            if type(value) is list:
                data[key] = np.asarray(value)
            if type(value) is dict:
                for key2, value2 in value.items():
                    if type(value2) is list:
                        data[key][key2] = np.asarray(value2)
    return data
    
def savecsv(filename, datadict, mode="w"):
    """Save a dictionary of data to CSV."""
    if mode == "a" :
        header = False
    else:
        header = True
    with open(filename, mode) as f:
        _pd.DataFrame(datadict).to_csv(f, index=False, header=header)
    
def loadcsv(filename):
    """Loads data from CSV file. Returns a single dict with column names as
    keys."""
    dataframe = _pd.read_csv(filename)
    data = {}
    for key, value in dataframe.items():
        data[key] = value.values
    return data
    
def savehdf(filename, datadict, groupname="data", mode="a", metadata=None,
            as_dataframe=False, append=False):
    """
    Saves a dictionary of arrays to file--similar to how scipy.io.savemat 
    works. If `datadict` is a DataFrame, it will be converted automatically.
    """
    if as_dataframe:
        df = _pd.DataFrame(datadict)
        df.to_hdf(filename, groupname)
    else:
        if isinstance(datadict, _pd.DataFrame):
            datadict = datadict.to_dict("list")
        with _h5py.File(filename, mode) as f:
            for key, value in datadict.items():
                if append:
                    try:
                        f[groupname + "/" + key] = np.append(f[groupname + "/" + key], value)
                    except KeyError:
                        f[groupname + "/" + key] = value
                else:
                    f[groupname + "/" + key] = value
            if metadata:
                for key, value in metadata.items():
                    f[groupname].attrs[key] = value
        
def loadhdf(filename, groupname="data", to_dataframe=False):
    """Loads all data from top level of HDF5 file--similar to how scipy.io.loadmat 
    works."""
    data = {}
    with _h5py.File(filename, "r") as f:
        for key, value in f[groupname].items():
            data[key] = np.array(value)
    if to_dataframe:
        return _pd.DataFrame(data)
    else:
        return data

def save_hdf_metadata(filename, metadata, groupname="data", mode="a"):
    """"Saves a dictionary of metadata to a group's attrs."""
    with _h5py.File(filename, mode) as f:
        for key, val in metadata.items():
            f[groupname].attrs[key] = val
        
def load_hdf_metadata(filename, groupname="data"):
    """"Loads attrs of the desired group into a dictionary."""
    with _h5py.File(filename, "r") as f:
        data = dict(f[groupname].attrs)
    return data

def build_plane_arrays(x, y, qlist):
    """Builds a 2-D array out of data taken in the same plane, for contour
    plotting."""
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
    """Computes lagged correlation coefficient for two time series."""
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
    return corr_coeff(x, x, t, tau1, tau2)
    
def integral_scale(u, t, tau1=0.0, tau2=1.0):
    """Calculates the integral scale of a time series by integrating up to
    the first zero crossing."""
    tau, rho = autocorr_coeff(u, t, tau1, tau2)
    zero_cross_ind = np.where(np.diff(np.sign(rho)))[0][0]
    int_scale = np.trapz(rho[:zero_cross_ind], tau[:zero_cross_ind])
    return int_scale
    
if __name__ == "__main__":
    pass
