#!/usr/bin/env python
# coding=utf-8

from distutils.core import setup

setup(
    name='PXL',
    version='0.0.1',
    author='Pete Bachant',
    author_email='petebachant@gmail.com',
    packages=['pxl'],
    scripts=[],
    url='https://github.com/petebachant/pxl.git',
    license='LICENSE',
    description='Extra functions built on NumPy, SciPy, pandas, matplotlib, etc.',
    long_description=open('README.md').read()
)