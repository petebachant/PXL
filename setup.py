#!/usr/bin/env python
# coding=utf-8

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup
from pxl import __version__ as version


def check_dependencies():
    """Emulate seaborn dependency checking."""
    install_requires = []
    try:
        import numpy
    except ImportError:
        install_requires.append("numpy")
    try:
        import scipy
    except ImportError:
        install_requires.append("scipy")
    try:
        import matplotlib
    except ImportError:
        install_requires.append("matplotlib")
    try:
        import pandas
    except ImportError:
        install_requires.append("pandas")
    try:
        import seaborn
    except ImportError:
        install_requires.append("seaborn")
    try:
        import h5py
    except ImportError:
        install_requires.append("h5py")

    return install_requires


setup(
    name="PXL",
    version=version,
    author="Pete Bachant",
    author_email="petebachant@gmail.com",
    packages=["pxl", "pxl.tests"],
    scripts=[],
    url="https://github.com/petebachant/pxl.git",
    license="GPL v3",
    description="Extra functions built on NumPy, SciPy, pandas, matplotlib, etc.",
    long_description=open("README.md").read(),
    install_requires=check_dependencies(),
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 2.6",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5"]
)
