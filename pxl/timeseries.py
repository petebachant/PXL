# -*- coding: utf-8 -*-
"""
This is a collection of useful time series analysis functions.

"""
from __future__ import division, print_function
import numpy as np
from scipy.signal import lfilter
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

def psd(t, data, window=None):
    """Computes one-sided power spectral density. Subtracts mean.
       Returns f, psd.
       
       Windows
       -------
         * "Hanning" """
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
    return f, psd
    
def runningstd(t, data, samples, overlap=None):
    """This function computes the running standard deviation of 
    a time series"""
    ne = int(np.floor(len(t)/samples))
    t_new = np.zeros(ne)
    std_r = np.zeros(ne)
    for i in range(ne):
        t_new[i] = np.mean(t[i*samples:(i+1)*samples])
        datasegment = data[i*samples:(i+1)*samples]
        datasegment = datasegment[~np.isnan(datasegment)]
        std_r[i] = np.std(datasegment)
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
    for key, value in datadict.iteritems():
        if type(value) == np.ndarray:
            datadict[key] = value.tolist()
        if type(value) == dict:
            for key2, value2 in value.iteritems():
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
        for key, value in data.iteritems():
            if type(value) is list:
                data[key] = np.asarray(value)
            if type(value) is dict:
                for key2, value2 in value.iteritems():
                    if type(value2) is list:
                        data[key][key2] = np.asarray(value2)
    return data
    
def savecsv(filename, datadict):
    """Save a dictionary of data to CSV."""
    _pd.DataFrame(datadict).to_csv(filename, index=False)
    
def loadcsv(filename):
    """Loads data from CSV file. Returns a single dict with column names as
    keys."""
    dataframe = _pd.read_csv(filename)
    data = {}
    for key, value in dataframe.iteritems():
        data[key] = value.values
    return data
    
def savehdf(filename, datadict, groupname="data", mode="a", metadata=None):
    """Saves a dictionary of arrays to file--similar to how scipy.io.savemat 
    works."""
    with _h5py.File(filename, mode) as f:
        for key, value in datadict.iteritems():
            f[groupname + "/" + key] = value
        if metadata:
            for key, value in metadata.iteritems():
                f[groupname].attrs[key] = value
        
def loadhdf(filename, groupname="data", to_dataframe=False):
    """Loads all data from top level of HDF5 file--similar to how scipy.io.loadmat 
    works."""
    data = {}
    with _h5py.File(filename, "r") as f:
        for key, value in f[groupname].iteritems():
            data[key] = np.array(value)
    if to_dataframe:
        return _pd.DataFrame(data)
    else:
        return data

def save_hdf_metadata(filename, metadata, groupname="data", mode="a"):
    """"Saves a dictionary of metadata to a group's attrs."""
    with _h5py.File(filename, mode) as f:
        for key, val in metadata.iteritems():
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
    
def corr(x1, x2, t, tau1, tau2):
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

def autocorr(x, t, tau1, tau2):
    return corr(x, x, t, tau1, tau2)
    
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    t = np.linspace(0, 1, num=1000)
    fs = 1.0/(t[1] - t[0])
    y = np.sin(2*np.pi*t) #+ 0.5*np.sin(5*t)
    y = np.random.rand(len(t))
    y2 = np.random.rand(len(t))
    tau, rho = corr(y, y, t, -0.5, 0.5)
    print(rho.max())
    plt.close("all")
    plt.figure()
    plt.plot(tau, rho)
