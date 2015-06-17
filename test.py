from pxl.tests.timeseries_test import *
from pxl.tests.styleplot_test import *

def test_all():
    test_average_over_area()
    test_spectrum_band_averaging()
    test_autocorrelation()
    test_integral_scale(plot=True)
    test_save_hdf_df_to_dict()
    test_runningstd()

if __name__ == "__main__":
    test_set_sns(scale=2)
    test_runningstd()