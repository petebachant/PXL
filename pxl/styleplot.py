# -*- coding: utf-8 -*-
"""
Created on Sun Jun 02 13:07:41 2013

@author: Pete
"""
import matplotlib
import matplotlib.pyplot as plt

def styleplot():
    font = {'family':'serif','serif':'cmr10','size':23}
    lines = {'markersize':9, 'markeredgewidth':0.9}
    legend = {'numpoints':1, 'fontsize': 'small'}
    matplotlib.rc('text', usetex = True)
    matplotlib.rc('font', **font)
    matplotlib.rc('lines', **lines)
    matplotlib.rc('legend', **legend)
    matplotlib.rc('xtick', **{'major.pad':12})
    plt.tight_layout()

def setpltparams(fontsize=23):
    font = {'family' : 'serif', 'serif' : 'cmr10', 'size' : fontsize}
    lines = {'markersize' : 9, 'markeredgewidth' : 0.9}
    legend = {'numpoints' : 1, 'fontsize' : 'small'}
    matplotlib.rc('text', usetex = True)
    matplotlib.rc('font', **font)
    matplotlib.rc('lines', **lines)
    matplotlib.rc('legend', **legend)
    matplotlib.rc('xtick', **{'major.pad':12})

def set_default_fontsize(size=23):
    matplotlib.rc("font", size=size)
