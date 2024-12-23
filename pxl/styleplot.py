"""This module contains helper functions for working with matplotlib."""

from __future__ import division, print_function

import matplotlib
import matplotlib.pyplot as plt


def styleplot():
    font = {"family": "serif", "serif": "cmr10", "size": 23}
    lines = {"markersize": 9, "markeredgewidth": 0.9}
    legend = {"numpoints": 1, "fontsize": "small"}
    matplotlib.rc("text", usetex=True)
    matplotlib.rc("font", **font)
    matplotlib.rc("lines", **lines)
    matplotlib.rc("legend", **legend)
    matplotlib.rc("xtick", **{"major.pad": 12})
    plt.tight_layout()


def setpltparams(fontsize=16, latex=True):
    if latex:
        font = {"family": "serif", "serif": "cmr10", "size": fontsize}
    else:
        font = {"size": fontsize}
    lines = {"markersize": 9, "markeredgewidth": 1, "linewidth": 1.2}
    legend = {"numpoints": 1, "fontsize": "small"}
    matplotlib.rc("text", usetex=latex)
    matplotlib.rc("font", **font)
    matplotlib.rc("lines", **lines)
    matplotlib.rc("legend", **legend)
    matplotlib.rc("xtick", **{"major.pad": 12})


def set_default_fontsize(size=23):
    matplotlib.rc("font", size=size)


def set_sns(
    style="white", context="paper", font_scale=1.5, color_codes=True, rc={}
):
    """Set default plot style using seaborn.

    Font size is set to match the size of the tick labels, rather than the axes
    labels.
    """
    rcd = {
        "lines.markersize": 8,
        "lines.markeredgewidth": 1.25,
        "legend.fontsize": "small",
        "font.size": 12 / 1.5 * font_scale,
        "legend.frameon": True,
        "axes.formatter.limits": (-5, 5),
        "axes.grid": True,
    }
    rcd.update(rc)
    import seaborn as sns

    sns.set(
        style=style,
        context=context,
        font_scale=font_scale,
        color_codes=color_codes,
        rc=rcd,
    )


def label_subplot(ax=None, x=0.5, y=-0.25, text="(a)", **kwargs):
    """Create a subplot label."""
    if ax is None:
        ax = plt.gca()
    ax.text(
        x=x,
        y=y,
        s=text,
        transform=ax.transAxes,
        horizontalalignment="center",
        verticalalignment="top",
        **kwargs
    )
