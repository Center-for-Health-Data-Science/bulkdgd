#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-

#    plotting.py
#
#    Plotting utilities.
#
#    Copyright (C) 2023 Valentina Sora 
#                       <sora.valentina1@gmail.com>
#
#    This program is free software: you can redistribute it and/or
#    modify it under the terms of the GNU General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public
#    License along with this program. 
#    If not, see <http://www.gnu.org/licenses/>.


# Description of the module
__doc__ = "Plotting utilities."
# Package name
pkg_name = "bulkDGD"


# Standard library
import logging as log
from pkg_resources import resource_filename, Requirement
# Third-party packages
import matplotlib.pyplot as plt
import seaborn as sns
# bulkDGD
from . import _util


# Get the module's logger
logger = log.getLogger(__name__)


#------------------------- Public constants --------------------------#


# The default configuration file for plotting the results of the PCA
CONFIG_PLOT_PCA = \
    resource_filename(Requirement(pkg_name),
                      "configs/plot/config_pca_scatter.yaml")


#------------------------------- Plots -------------------------------#


def plot_r_values_hist(r_values,
                       output_file,
                       config):
    """Plot a histogram of the r-values.

    Parameters
    ----------
    r_values : ``torch.Tensor``
        The r-values. This is a 1D tensor whose length
        is equal to the size of the gene space.

    output_file : ``str``
        The file where the plot will be saved.

    config : ``dict``
        The configuration of the plot's aesthetics.
    """
  
    # Close any figure that may be open
    plt.close()

    # Take all r-values
    r_values = r_values.numpy().flatten()

    # Get the configuration for the histogram
    config_hist = config["plot"]["histogram"]


    #----------------------- Generate the plot -----------------------#


    # Generate the figure and axes
    fig, ax = plt.subplots()

    # Generate the histogram
    n, bins, patches = ax.hist(x = r_values,
                               **config_hist)


    #------------------------ Set the spines -------------------------#


    # Hide the top and right spine
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    # Set the position of the bottom and left spine
    for spine in ["bottom", "left"]:
        ax.spines[spine].set_position(("outward", 5))


    #------------------------ Set the x-axis -------------------------#


    # Get the configuration of the axis
    config_x_axis = config["plot"]["xaxis"]
    
    # Get the positions of the ticks on the axis
    x_ticks = _util.get_ticks_positions(values = bins,
                                        item = "x-axis",
                                        config = config_x_axis)

    # Set the axis
    _util.set_axis(ax = ax,
                   axis = "x",
                   config = config_x_axis,
                   ticks = x_ticks)


    #------------------------ Set the y-axis -------------------------#


    # Get the configuration of the axis
    config_y_axis = config["plot"]["yaxis"]
    
    # Get the positions of the ticks on the axis
    y_ticks = _util.get_ticks_positions(values = n,
                                        item = "y-axis",
                                        config = config_y_axis)

    # Set the axis
    _util.set_axis(ax = ax,
                   axis = "y",
                   config = config_y_axis,
                   ticks = y_ticks)

    
    #------------------------- Save the plot -------------------------#


    # Write the plot in the output file
    plt.savefig(fname = output_file,
                **config["output"])


def plot_2d_pca(df_pca,
                output_file,
                config,
                pc_columns = ["PC1", "PC2"],
                groups_column = None):
    """Plot the results of a principal component analysis (PCA).

    Parameters
    ----------
    df_pca : ``pandas.DataFrame``
        A data frame containing the results of the PCA. The rows
        should contain the representations, while the columns
        should contain the values of each representation's
        projection along the principal components.

    output_file : ``str``
        The file where the plot will be saved.

    config : ``dict``
        A dictionary containing the configuration of the plot's
        aesthetics.

    pc_columns : ``list``, ``["PC1", "PC2"]``
        A list with the names of the two columns that contain
        the values of the first two principal components.

    groups_column : ``str``, optional
        The name of the column containing the labels of different
        samples' groups, if any. If not provided, the samples
        will be assumed to belong to the sample group. If
        provided, the samples will be colored according to the
        group they belong.
    """

    # Close any figure that may be open
    plt.close()

    # Get the names of the columns containg the values of the
    # projections along the first and second principal
    # component
    pc1_col, pc2_col = pc_columns


    #----------------------- Generate the plot -----------------------#


    # Generate the figure and axes
    fig, ax = plt.subplots()

    # Get the configuration for the scatterplot
    config_hist = config["plot"]["scatterplot"]

    # Generate the scatterplot
    sns.scatterplot(x = pc1_col,
                    y = pc2_col,
                    data = df_pca,
                    ax = ax,
                    hue = groups_column)


    #------------------------ Set the spines -------------------------#


    # Hide the top and right spine
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    # Set the position of the bottom and left spine
    for spine in ["bottom", "left"]:
        ax.spines[spine].set_position(("outward", 5))


    #------------------------ Set the x-axis -------------------------#


    # Get the configuration of the axis
    config_x_axis = config["plot"]["xaxis"]
    
    # Get the positions of the ticks on the axis
    x_ticks = _util.get_ticks_positions(values = df_pca[pc1_col],
                                        item = "x-axis",
                                        config = config_x_axis)

    # Set the axis
    _util.set_axis(ax = ax,
                   axis = "x",
                   config = config_x_axis,
                   ticks = x_ticks)


    #------------------------ Set the y-axis -------------------------#


    # Get the configuration of the axis
    config_y_axis = config["plot"]["yaxis"]
    
    # Get the positions of the ticks on the axis
    y_ticks = _util.get_ticks_positions(values = df_pca[pc2_col],
                                        item = "y-axis",
                                        config = config_y_axis)

    # Set the axis
    _util.set_axis(ax = ax,
                   axis = "y",
                   config = config_y_axis,
                   ticks = y_ticks)


    #------------------------ Set the legend -------------------------#


    # Get the configuration for the legend
    config_legend = config["plot"]["legend"]

    # Set the legend
    _util.set_legend(ax = ax,
                     config = config_legend)


    #------------------------- Save the plot -------------------------#


    # Write the plot in the output file
    plt.savefig(fname = output_file,
                **config["output"])