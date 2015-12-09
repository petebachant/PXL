from __future__ import division, print_function
from .. import timeseries
from ..timeseries import *
import matplotlib.pyplot as plt
import pandas as pd
import os

def test_autocorrelation():
    print("Testing autocorrelation function")
    t = np.linspace(0, 1, num=1000)
    fs = 1.0/(t[1] - t[0])
    y = np.sin(2*np.pi*t) #+ 0.5*np.sin(5*t)
    y = np.random.rand(len(t))
    y2 = np.random.rand(len(t))
    tau, rho = autocorr_coeff(y, t, -0.5, 0.5)
    if np.abs(rho.max() - 1.0) < 1e-12:
        print("PASSED: Autocorrelation test 1")
    else:
        print("FAILED: Autocorrelation test 1")
    print("Mean autocorrelation coefficient for random time series =", rho.mean())

def test_spectrum_band_averaging():
    """Test band averaging with `psd` function."""
    from scipy.signal import firwin, lfilter
    n_band_average = 20
    n = 10000
    ts = np.random.randn(n)
    t = np.linspace(0.0, 1.0, num=n)
    N = 5
    Fc = 50
    Fs = 1500
    h = firwin(numtaps=N, cutoff=40, nyq=Fs/2)
    ts = lfilter(h, 1.0, ts) # 'x' is the time-series data you are filtering
    f, spec = timeseries.psd(t, ts, n_band_average=1)
    f2, spec2 = timeseries.psd(t, ts, n_band_average=n_band_average)
    plt.plot(f, spec, label="Raw")
    plt.hold(True)
    plt.plot(f2, spec2, label="Band-averaged over {} bands".format(n_band_average))
    plt.xlabel("$f$")
    plt.ylabel("$S$")
    plt.legend()
    plt.show()
    
def test_average_over_area():
    """Should result in a value of 1"""
    print("Testing average_over_area")
    x = np.array([-6.0, -2.0, -1.0, 0.0, 1.0, 2.0, 6.0])
    y = np.array([-6.0, -2.0, -1.0, 0.0, 1.0, 2.0, 6.0])
    q = np.ones((x.size, y.size))
    ave = average_over_area(q, x, y)
    print("Computed average is", ave)
    if np.abs(ave - 1) < 1e-12: 
        print("PASSED: First test of average_over_area")
    else:
        print("FAILED: First test of average_over_area")
    # Now test something a bit harder
    y = np.array([0., 1.])
    q = np.array([[1.0, 0., 0., 0., 0., 0., 1.0],
                  [1.0, 0., 0., 0., 0., 0., 1.0]])
    ave = average_over_area(q, x, y)
    print("Computed average is", ave)
    if np.abs(ave - 0.33333333333333) < 1e-12:
        print("PASSED: Second test of average_over_area")
    else:
        print("FAILED: Second test of average_over_area")
        
def test_integral_scale(plot=False):
    print("Testing integral scale calculation function")
    t = np.linspace(0, 1, num=1000)
    u = 0.1*np.random.randn(len(t))
    u += 0.1*np.sin(2*np.pi*t)
    print(integral_scale(u, t, tau2=1.0))
    if plot:
        tau, rho = autocorr_coeff(u, t, 0.0, 0.5)
        plt.figure()
        plt.plot(tau, rho)
        plt.xlabel(r"$\tau$")
        plt.ylabel(r"$\rho$")    
        
def test_append_hdf():
    data = {"zeros(5)" : np.zeros(5),
            "arange(5)" : np.arange(5)}
    savehdf("test.h5", data, append=True)
    print(loadhdf("test.h5"))
    
def test_save_hdf_df_to_dict():
    print("Testing savehdf with a pandas DataFrame")
    data = pd.DataFrame()
    data["test"] = np.zeros(10)
    savehdf("test.h5", data)
    data1 = loadhdf("test.h5")
    os.remove("test.h5")
    assert((data1["test"] == data["test"]).all())
    print("PASS")
    
def test_runningstd():
    """
    Tests the running standard deviation calculation function in the 
    `timeseries` module.
    """
    print("Testing timeseries.runningstd")
    t = np.linspace(0, 10, 10000)
    a = np.sin(2*np.pi*t)
    a += np.random.randn(len(t))
    t_std, a_std = runningstd(t, a, 1000)
    print("Mean of running standard deviation:", a_std.mean())
    print("Standard deviation of test time series:", a.std())
    assert(np.abs(a_std.mean() - a.std()) < 0.05)
    plt.figure()
    plt.plot(t_std, a_std)
    plt.ylim((-4, 4))
    plt.figure()
    plt.plot(t, a)
    plt.show()
    print("PASS")