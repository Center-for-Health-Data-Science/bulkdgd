#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-

#    plots.py
#
#    Utilities for plotting.
#
#    Copyright (C) 2024 Valentina Sora 
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


#######################################################################


# Set the module's description.
__doc__ = "Utilities for plotting."


#######################################################################


# Import from the standard library.
import copy
import logging as log
import warnings
# Import from third-party packages.
import matplotlib.pyplot as plt
import seaborn as sns
# Import from 'bulkDGD'.
from . import _util


#######################################################################


# Ignore warnings (matplotlib's 'UserWarnings').
warnings.filterwarnings("ignore", category = UserWarning)


#######################################################################


# Get the module's logger.
logger = log.getLogger(__name__)


#######################################################################


def plot_r_values_hist(r_values,
                       output_file,
                       config):
    """Plot a histogram of the r-values.

    Parameters
    ----------
    r_values : ``torch.Tensor``
        The r-values. This is a 1D tensor whose length is equal to
        the number of genes included in the DGD model.

    output_file : ``str``
        The file where the plot will be saved.

    config : ``dict``
        The configuration of the plot's aesthetics.
    """
  
    # Close any figure that may be open.
    plt.close()

    #-----------------------------------------------------------------#

    # Get the configuration for the histogram.
    config_hist = config["plot"]["histogram"]

    #-----------------------------------------------------------------#

    # Take all r-values.
    r_values = r_values.numpy().flatten()

    #-----------------------------------------------------------------#

    # Generate the figure and axes.
    fig, ax = plt.subplots()

    #-----------------------------------------------------------------#

    # Generate the histogram.
    n, bins, patches = ax.hist(x = r_values,
                               **config_hist)

    #-----------------------------------------------------------------#

    # Hide the top and right spine.
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    # Set the position of the bottom and left spine.
    for spine in ["bottom", "left"]:
        ax.spines[spine].set_position(("outward", 5))

    #-----------------------------------------------------------------#

    # Get the configuration of the x-axis.
    config_x_axis = config["plot"]["xaxis"]
    
    # Get the positions of the ticks on the x-axis.
    x_ticks = \
        _util.get_ticks_positions(values = bins,
                                  item = "x-axis",
                                  config = config_x_axis)

    # Set the x-axis.
    _util.set_axis(ax = ax,
                   axis = "x",
                   config = config_x_axis,
                   ticks = x_ticks)

    #-----------------------------------------------------------------#

    # Get the configuration of the y-axis.
    config_y_axis = config["plot"]["yaxis"]
    
    # Get the positions of the ticks on the y-axis.
    y_ticks = \
        _util.get_ticks_positions(values = n,
                                  item = "y-axis",
                                  config = config_y_axis)

    # Set the y-axis
    _util.set_axis(ax = ax,
                   axis = "y",
                   config = config_y_axis,
                   ticks = y_ticks)

    #-----------------------------------------------------------------#

    # Save the plot in the output file.
    plt.savefig(fname = output_file,
                **config["output"])


def plot_get_representations_time(df_time,
                                  output_file,
                                  config):
    """Plot the CPU/wall clock time spent in each epoch of each
    round of optimization when finding the representations for a
    set of samples (both for the full epoch and for the
    backward step performed in each epoch).

    Parameters
    ----------
    df_time : ``pandas.DataFrame``
        A data frame containing the time data. This data frame is
        produced as an output by the
        ``core.model.DGDModel.get_representations`` method.

    output_file : ``str``
        The file where the plot will be saved.

    config : ``dict``
        A dictionary containing the configuration of the plot's
        aesthetics.
    """

    # Close any figure that may be open.
    plt.close()

    #-----------------------------------------------------------------#

    # Create a copy of the original data frame to modify before
    # generating the plot.
    df_to_plot = copy.deepcopy(df_time)

    # 'Unpack' the columns representing the different 'types' of
    # time reported (CPU/wall clock) into only one column.
    df_to_plot = \
        df_to_plot.melt(["platform", "processor", "num_threads",
                         "opt_round", "epoch"],
                         var_name = "Time (CPU/Wall clock)",
                         value_name = "Time (s)")

    #-----------------------------------------------------------------#

    # Get the optimization round(s) that was (were) performed.
    opt_rounds = df_to_plot["opt_round"].unique()

    # Get all the time values reported in the data frame.
    time_values = df_to_plot["Time (s)"].values

    #-----------------------------------------------------------------#

    # Generate the figure and axes. The plots will be arranged into
    # one row and as many columns as the number of optimization rounds
    # run when finding the representations.
    fig, axes = plt.subplots(nrows = 1,
                             ncols = len(opt_rounds))

    #-----------------------------------------------------------------#

    # For each optimization round and the axis where the corresponding
    # data will be plotted.
    for opt_round, ax in zip(opt_rounds, axes):

        # Get the slice of the data frame with the data corresponding
        # to the current optimization round.
        sub_df = \
            df_to_plot.loc[\
                (df_to_plot["opt_round"] == opt_round)]

        #-------------------------------------------------------------#

        # Get the number of epochs in the current optimization round.
        epochs = sub_df["epoch"].values

        #-------------------------------------------------------------#

        # Generate the plot.
        sns.lineplot(data = sub_df,
                     x = "epoch",
                     y = "Time (s)",
                     hue = "Time (CPU/Wall clock)",
                     ax = ax,
                     **config["plot"]["lineplot"])

        #-------------------------------------------------------------#

        # Create a copy of the title's configuration.
        config_title = copy.deepcopy(config["plot"]["title"])

        # If there is a label
        if config_title.get("label") is not None:

            # Get the raw label.
            label_raw = config_title.pop("label")

            # Substitute the '[opt_round]' string with the actual
            # number/name of the current optimization round.
            label = label_raw.replace("[opt_round]",
                                      str(opt_round),
                                      1)

        # Set the title for the sub-plot.
        ax.set_title(label = label,
                     **config_title)

        #-------------------------------------------------------------#

        # Hide the top and right spine.
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

        # Set the position of the bottom and left spine.
        for spine in ["bottom", "left"]:
            ax.spines[spine].set_position(("outward", 5))

        #-------------------------------------------------------------#

        # Get the configuration of the x-axis.
        config_x_axis = config["plot"]["xaxis"]
        
        # Get the positions of the ticks on the x-axis.
        x_ticks = \
            _util.get_ticks_positions(values = epochs,
                                      item = "x-axis",
                                      config = config_x_axis)

        # Set the x-axis.
        _util.set_axis(ax = ax,
                       axis = "x",
                       config = config_x_axis,
                       ticks = x_ticks)

        #-------------------------------------------------------------#

        # Get the configuration of the y-axis.
        config_y_axis = config["plot"]["yaxis"]
        
        # Get the positions of the ticks on the y-axis.
        y_ticks = \
            _util.get_ticks_positions(values = time_values,
                                      item = "y-axis",
                                      config = config_y_axis)

        # Set the y-axis.
        _util.set_axis(ax = ax,
                       axis = "y",
                       config = config_y_axis,
                       ticks = y_ticks)

        #-------------------------------------------------------------#

        # Get the configuration for the legend.
        config_legend = config["plot"]["legend"]

        # Set the legend.
        _util.set_legend(ax = ax,
                         config = config_legend)

    #-----------------------------------------------------------------#

    # Save the plot in the output file.
    plt.savefig(fname = output_file,
                **config["output"])


def plot_2d_pca(df_pca,
                output_file,
                config,
                pc_columns = ["PC1", "PC2"],
                groups_column = None):
    """Plot the results of a two-dimensional principal component
    analysis (PCA).

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
        samples' groups, if any.

        If not provided, the samples will be assumed to belong to
        one sample group.

        If provided, the samples will be colored according to the
        group they belong.
    """

    # Close any figure that may be open.
    plt.close()

    #-----------------------------------------------------------------#

    # Get the names of the columns containg the values of the
    # projections along the first and second principal component.
    pc1_col, pc2_col = pc_columns

    #-----------------------------------------------------------------#

    # Generate the figure and axes.
    fig, ax = plt.subplots()

    #-----------------------------------------------------------------#

    # Get the configuration for the scatterplot.
    config_hist = config["plot"]["scatterplot"]

    #-----------------------------------------------------------------#

    # Generate the scatterplot.
    sns.scatterplot(x = pc1_col,
                    y = pc2_col,
                    data = df_pca,
                    ax = ax,
                    hue = groups_column)

    #-----------------------------------------------------------------#

    # Hide the top and right spine.
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    # Set the position of the bottom and left spine.
    for spine in ["bottom", "left"]:
        ax.spines[spine].set_position(("outward", 5))

    #-----------------------------------------------------------------#

    # Get the configuration of the x-axis.
    config_x_axis = config["plot"]["xaxis"]
    
    # Get the positions of the ticks on the x-axis.
    x_ticks = \
        _util.get_ticks_positions(values = df_pca[pc1_col],
                                  item = "x-axis",
                                  config = config_x_axis)

    # Set the x-axis
    _util.set_axis(ax = ax,
                   axis = "x",
                   config = config_x_axis,
                   ticks = x_ticks)

    #-----------------------------------------------------------------#

    # Get the configuration of the y-axis.
    config_y_axis = config["plot"]["yaxis"]
    
    # Get the positions of the ticks on the y-axis.
    y_ticks = \
        _util.get_ticks_positions(values = df_pca[pc2_col],
                                  item = "y-axis",
                                  config = config_y_axis)

    # Set the y-axis.
    _util.set_axis(ax = ax,
                   axis = "y",
                   config = config_y_axis,
                   ticks = y_ticks)

    #-----------------------------------------------------------------#

    # Get the configuration for the legend.
    config_legend = config["plot"]["legend"]

    # Set the legend.
    _util.set_legend(ax = ax,
                     config = config_legend)

    #-----------------------------------------------------------------#

    # Save the plot in the output file.
    plt.savefig(fname = output_file,
                **config["output"])
