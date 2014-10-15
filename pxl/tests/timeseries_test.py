from __future__ import division, print_function
from .. import timeseries
from ..timeseries import *
import matplotlib.pyplot as plt

def test_autocorrelation():
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
    plt.show()

def test_spectrum_band_averaging():
    """Test band averaging with `psd` function."""
    from scipy.signal import firwin, lfilter
    n = 10000
    ts = np.random.randn(n)
    t = np.linspace(0.0, 1.0, num=n)
    N = 5
    Fc = 50
    Fs = 1500
    h = firwin(numtaps=N, cutoff=40, nyq=Fs/2)
    ts = lfilter(h, 1.0, ts) # 'x' is the time-series data you are filtering
    f, spec = timeseries.psd(t, ts, n_band_average=1)
    f2, spec2 = timeseries.psd(t, ts, n_band_average=20)
    plt.plot(f, spec)
    plt.hold(True)
    plt.plot(f2, spec2)
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
        print("First test of average_over_area passed")
    else:
        print("First test of average_over_area failed")
    # Now test something a bit harder
    y = np.array([0., 1.])
    q = np.array([[1.0, 0., 0., 0., 0., 0., 1.0],
                  [1.0, 0., 0., 0., 0., 0., 1.0]])
    ave = average_over_area(q, x, y)
    print("Computed average is", ave)
    if np.abs(ave - 0.33333333333333) < 1e-12:
        print("Second test of average_over_area passed")
    else:
        print("Second test of average_over_area failed")
    