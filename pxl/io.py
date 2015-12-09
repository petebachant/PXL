# -*- coding: utf-8 -*-
"""
This is a collection of useful I/O functions.
"""

from __future__ import division, print_function
import numpy as np
import json
import pandas as _pd
import h5py as _h5py


def savejson(filename, datadict):
    """Save data from a dictionary in JSON format. Note that this only
    works to the second level of the dictionary with Numpy arrays.
    """
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
    """Load data from text file in JSON format.

    Numpy arrays are converted if specified with the `asnparrays` keyword
    argument. Note that this only works to the second level of the dictionary.
    Returns a single dict.
    """
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
    """Load data from CSV file.

    Returns a single dict with column names as keys.
    """
    dataframe = _pd.read_csv(filename)
    data = {}
    for key, value in dataframe.items():
        data[key] = value.values
    return data


def savehdf(filename, datadict, groupname="data", mode="a", metadata=None,
            as_dataframe=False, append=False):
    """Save a dictionary of arrays to file--similar to how `scipy.io.savemat`
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
    """Load all data from top level of HDF5 file--similar to how
    `scipy.io.loadmat` works.
    """
    data = {}
    with _h5py.File(filename, "r") as f:
        for key, value in f[groupname].items():
            data[key] = np.array(value)
    if to_dataframe:
        return _pd.DataFrame(data)
    else:
        return data


def save_hdf_metadata(filename, metadata, groupname="data", mode="a"):
    """"Save a dictionary of metadata to a group's attrs."""
    with _h5py.File(filename, mode) as f:
        for key, val in metadata.items():
            f[groupname].attrs[key] = val


def load_hdf_metadata(filename, groupname="data"):
    """"Load attrs of the desired group into a dictionary."""
    with _h5py.File(filename, "r") as f:
        data = dict(f[groupname].attrs)
    return data
