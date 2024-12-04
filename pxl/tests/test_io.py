from __future__ import division, print_function

import os

import pandas as pd

from .. import io
from ..io import *


def test_append_hdf():
    data = {"zeros(5)": np.zeros(5), "arange(5)": np.arange(5)}
    savehdf("test.h5", data, append=True)
    print(loadhdf("test.h5"))


def test_save_hdf_df_to_dict():
    data = pd.DataFrame()
    data["test"] = np.zeros(10)
    savehdf("test.h5", data)
    data1 = loadhdf("test.h5")
    os.remove("test.h5")
    assert (data1["test"] == data["test"]).all()
