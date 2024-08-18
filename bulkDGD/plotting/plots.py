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
import itertools
import logging as log
import warnings
# Import from third-party packages.
import matplotlib.pyplot as plt
import seaborn as sns
# Import from 'bulkDGD'.
from . import _util
from bulkDGD import util


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
    r_values : :class:`numpy.ndarray`
        The r-values. This is a 1D array whose length is equal to
        the number of genes included in the DGD model.

    output_file : :class:`str`
        The file where the plot will be saved.

    config : :class:`dict`
        The configuration for the plot's aesthetics.
    """
  
    # Close any figure that may be open.
    plt.close()

    #-----------------------------------------------------------------#

    # Get the sections that need to be in the configuration.
    sections_needed = ["plot", "plot:histogram", "output"]

    # Get the sections that may be in the configuration.
    sections = ["plot:xaxis", "plot:yaxis"]

    # Check the configuration.
    _util.check_custom_configs(config = config,
                               sections = sections,
                               sections_needed = sections_needed)

    #-----------------------------------------------------------------#

    # Get the configuration for the histogram.
    config_hist = config["plot"]["histogram"]

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

    #-----------------------------------------------------------------#

    # Get the configuration of the x-axis.
    config_x_axis = config["plot"].get("xaxis", {})
    
    # Get the positions of the ticks on the x-axis.
    x_ticks = \
        _util.get_ticks_positions(values = bins,
                                  item = "x-axis",
                                  config = config_x_axis)

    # Set the x-axis.
    ax = _util.set_axis(ax = ax,
                        axis = "x",
                        config = config_x_axis,
                        ticks = x_ticks)

    #-----------------------------------------------------------------#

    # Get the configuration of the y-axis.
    config_y_axis = config["plot"].get("yaxis", {})
    
    # Get the positions of the ticks on the y-axis.
    y_ticks = \
        _util.get_ticks_positions(values = n,
                                  item = "y-axis",
                                  config = config_y_axis)

    # Set the y-axis.
    ax = _util.set_axis(ax = ax,
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
    df_time : :class:`pandas.DataFrame`
        A data frame containing the time data. This data frame is
        produced as an output by the
        :class:`bulkDGD.core.model.DGDModel.get_representations`
        method.

    output_file : :class:`str`
        The file where the plot will be saved.

    config : :class:`dict`
        A dictionary containing the configuration for the plot's
        aesthetics.
    """

    # Close any figure that may be open.
    plt.close()

    #-----------------------------------------------------------------#

    # Get the sections that need to be in the configuration.
    sections_needed = ["plot", "plot:lineplot", "output"]

    # Get the sections that may be in the configuration.
    sections = \
        ["plot:xaxis", "plot:yaxis", "plot:title", "plot:legend"]

    # Check the configuration.
    _util.check_custom_configs(config = config,
                               sections = sections,
                               sections_needed = sections_needed)

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
    # data will be plotted
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
        config_title = copy.deepcopy(config["plot"].get("title"))

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

        #-------------------------------------------------------------#

        # Get the configuration of the x-axis.
        config_x_axis = config["plot"].get("xaxis", {})
        
        # Get the positions of the ticks on the x-axis.
        x_ticks = \
            _util.get_ticks_positions(values = epochs,
                                      item = "x-axis",
                                      config = config_x_axis)

        # Set the x-axis.
        ax = _util.set_axis(ax = ax,
                            axis = "x",
                            config = config_x_axis,
                            ticks = x_ticks)

        #-------------------------------------------------------------#

        # Get the configuration of the y-axis.
        config_y_axis = config["plot"].get("yaxis", {})
        
        # Get the positions of the ticks on the y-axis.
        y_ticks = \
            _util.get_ticks_positions(values = time_values,
                                      item = "y-axis",
                                      config = config_y_axis)

        # Set the y-axis.
        ax = _util.set_axis(ax = ax,
                            axis = "y",
                            config = config_y_axis,
                            ticks = y_ticks)

        #-------------------------------------------------------------#

        # Get the configuration for the legend.
        config_legend = config["plot"].get("legend", {})

        # Set the legend.
        ax = _util.set_legend(ax = ax,
                              config = config_legend)

    #-----------------------------------------------------------------#

    # Save the plot in the output file.
    plt.savefig(fname = output_file,
                **config["output"])


def plot_2d_dim_red(df_dim_red,
                    output_file,
                    config,
                    columns = ["C1", "C2"],
                    groups_column = None,
                    groups = None,
                    plot_other_groups = False):
    """Plot the results of a two-dimensional dimensionality reduction.

    Parameters
    ----------
    df_dim_red : :class:`pandas.DataFrame`
        A data frame containing the results of the dimensionality
        reduction.

        The rows should contain the data points, while the columns
        should contain the values of each data point's projection
        along the principal components.

    output_file : :class:`str`
        The file where the plot will be saved.

    config : :class:`dict`
        A dictionary containing the configuration for the plot's
        aesthetics.

    columns : :class:`list`, ``["PC1", "PC2"]``
        A list with the names of the two columns that contain
        the values of the two dimensions of the projection's space
        to be considered when plotting.

    groups_column : :class:`str`, optional
        The name of the column containing the labels of different
        groups, if any.

        If not provided, the data points will be assumed to belong to
        one group.

        If provided, the data points will be colored according to the
        group they belong.

    groups : :class:`list`, optional
        A list of groups of interest. If a list of groups is provided
        and ``plot_other_groups`` is ``False``, only data points
        belonging to the groups of interest will be plotted. If
        ``plot_other_groups`` is ``True``, the other groups will
        be plotted according to the aesthetic specifications provided
        in the configuration.

    plot_other_groups : :class:`bool`, :obj:`False`
        If a list of ``groups`` of interest if provided, set whether
        to plot data points belonging to the other groups according to
        the aesthetic specifications provided in the configuration
        (``True``) or not to plot the data points belonging to the
        other groups at all (``False``).
    """

    # Close any figure that may be open.
    plt.close()

    #-----------------------------------------------------------------#

    # Generate the figure and axes.
    fig, ax = plt.subplots()

    #-----------------------------------------------------------------#

    # Get the sections that need to be in the configuration.
    sections_needed = ["plot", "plot:scatterplot", "output"]

    # Get the sections that may be in the configuration.
    sections = \
        ["plot:xaxis", "plot:yaxis", "plot:legend", "plot:text"]

    # Check the configuration.
    _util.check_custom_configs(config = config,
                               sections = sections,
                               sections_needed = sections_needed)

    #-----------------------------------------------------------------#

    # Get the configuration for the plot area.
    config_plot = config["plot"]

    #-----------------------------------------------------------------#

    # If more or less than two columns were selected
    if len(columns) != 2:

        # Raise an error.
        errstr = \
            "Exactly two 'columns' must be provided, but " \
            f"{len(columns)} were passed."
        raise ValueError(errstr)

    # Get the names of the columns containing the values of the
    # projections along the first and second dimension.
    c1_col, c2_col = columns

    #-----------------------------------------------------------------#

    # If specific groups were defined
    if groups is not None:

        # Get the data points belonging to other groups.
        df_dim_red_others = \
            df_dim_red[~df_dim_red[groups_column].isin(groups)]

        # Get the data points belonging to the groups of interest.
        df_dim_red = \
            df_dim_red[df_dim_red[groups_column].isin(groups)]

    # Otherwise
    else:

        # There will be no data points belonging to other groups.
        df_dim_red_others = None

    #-----------------------------------------------------------------#

    # If we have to color the data points belonging to the other
    # groups differently.
    if df_dim_red_others is not None and plot_other_groups:

        # Plot them (we plot them fist so that the points of the
        # groups of interest are plotted on top of them).
        ax = sns.scatterplot(x = c1_col,
                             y = c2_col,
                             data = df_dim_red_others,
                             ax = ax,
                             **config_plot.get("other_groups", {}))

    #-----------------------------------------------------------------#

    # Generate the scatter plot.
    ax = sns.scatterplot(x = c1_col,
                         y = c2_col,
                         data = df_dim_red,
                         ax = ax,
                         hue = groups_column,
                         **config_plot.get("scatterplot", {}))

    #-----------------------------------------------------------------#

    # Hide the top and right spine.
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    #-----------------------------------------------------------------#

    # Get the configuration of the x-axis.
    config_x_axis = config_plot.get("xaxis", {})
    
    # Get the positions of the ticks on the x-axis.
    x_ticks = \
        _util.get_ticks_positions(values = df_dim_red[c1_col],
                                  item = "x-axis",
                                  config = config_x_axis)

    # Set the x-axis.
    ax = _util.set_axis(ax = ax,
                        axis = "x",
                        config = config_x_axis,
                        ticks = x_ticks)

    #-----------------------------------------------------------------#

    # Get the configuration of the y-axis.
    config_y_axis = config_plot.get("yaxis", {})
    
    # Get the positions of the ticks on the y-axis.
    y_ticks = \
        _util.get_ticks_positions(values = df_dim_red[c2_col],
                                  item = "y-axis",
                                  config = config_y_axis)

    # Set the y-axis.
    ax = _util.set_axis(ax = ax,
                        axis = "y",
                        config = config_y_axis,
                        ticks = y_ticks)

    #-----------------------------------------------------------------#

    # Get the configuration for the legend.
    config_legend = config_plot.get("legend", {})

    # Set the legend.
    ax = _util.set_legend(ax = ax,
                          config = config_legend)

    #-----------------------------------------------------------------#

    # Get the configuration for the text.
    config_text = config_plot.get("text", {})

    # If there is a configuration for the number of data points
    if "n_data_points" in config_text:

        # Get the configuration.
        config_n_data_points = config_text["n_data_points"]

        # Replace the '{n_data_points}' placeholder with the
        # actual number of data points.
        config_n_data_points["s"].replace(\
            r"{n_data_points}", str(len(df_dim_red)))

        # Add the text to the figure.
        fig = _util.set_text(fig = fig,
                             config = config_n_data_points)

    #-----------------------------------------------------------------#

    # Save the plot in the output file.
    plt.savefig(fname = output_file,
                **config["output"])


def plot_multiple_2d_dim_red(dfs_dim_red,
                             output_prefix,
                             output_fmt,
                             config,
                             plots_per_output = 9,
                             columns = ["C1", "C2"],
                             groups_column = None,
                             groups = None,
                             plot_other_groups = False,
                             dfs_names = None):
    """Plot the results of a series of dimensionality reduction
    analyses on a single figure (which may be split on multiple
    pages).

    Parameters
    ----------
    dfs_dim_red : :class:`pandas.DataFrame`
        A list of data frames containing the results of the
        dimensionality reduction analyses.

        The rows of each data frame should contain the data points,
        while the columns should contain the values of each data
        point's projection along the principal components.

    output_prefix : :class:`str`
        The prefix of the output file(s) that will be written.

        The number of output files depends on the number of data
        frames passed and on the number of ``plots_per_output``.

    output_fmt : :class`str`
        The format of the output file(s) that will be written.

    config : :class:`dict`
        A dictionary containing the configuration for the plots'
        aesthetics.
    
    plots_per_output : :class:`int`, ``9``
        The maximum number of plots to draw on each output file.

    columns : :class:`list`, ``["PC1", "PC2"]``
        A list with the names of the two columns in each data frame
        that contain the values of the two dimensions of the
        projection's space to be considered when plotting.

    groups_column : :class:`str`, optional
        The name of the column containing the labels of different
        groups in the data frames, if any.

        If not provided, the data points will be assumed to belong to
        one group.

        If provided, the data points will be colored according to the
        group they belong.

    groups : :class:`list`, optional
        A list of groups of interest. If a list of groups is provided
        and ``plot_other_groups`` is ``False``, only data points
        belonging to the groups of interest will be plotted. If
        ``plot_other_groups`` is ``True``, the other groups will
        be plotted according to the aesthetic specifications provided
        in the configuration.

    plot_other_groups : :class:`bool`, :obj:`False`
        If a list of ``groups`` of interest if provided, set whether
        to plot data points belonging to the other groups according to
        the aesthetic specifications provided in the configuration
        (``True``) or not to plot the data points belonging to the
        other groups at all (``False``).

    dfs_names : :class:`list`, optional
        A list of names for the data frames passed. These names, if
        passed, will be used as the titles of the corresponding plots.
    """

    # Get the sections that need to be in the configuration.
    sections_needed = ["output"]

    # Get the sections that may be in the configuration.
    sections = \
        ["font_properties", "colors", "figure", "general", "subplots"]

    # Get the sections that may be in the general or sub-plots'
    # configuration.
    sections_sub = \
        ["scatterplot", "other_groups", "xaxis", "yaxis", "title"]

    # Check the configuration.
    _util.check_custom_configs(config = config,
                               sections = sections,
                               sections_needed = sections_needed)

    #-----------------------------------------------------------------#

    # Get the configuration for the output.
    config_out = config["output"]

    #-----------------------------------------------------------------#

    # Get the configuration for the all sub-plots.
    config_subs = config.get("subplots", {})

    # Get the general configuration for all the sub-plots.
    config_subs_general = config_subs.get("general", {})

    #-----------------------------------------------------------------#

    # Get the configuration for the colors to be used for the sub-
    # plots.
    config_colors = config.get("colors", {})

    # Get the configuration for the figure.
    config_fig = config.get("figure", {})

    #-----------------------------------------------------------------#

    # If the user specified the plot's format in the configuration
    if "format" in config_out:

        # Warn the user that the format's specification will be
        # ignored.
        warnstr = \
            "The output's format was defined in the configuration " \
            "('output' section). It will be ignored, and the " \
            "value specified by the 'output_fmt' option will be "\
            "used instead."
        logger.warning(warnstr)

        # Remove the format's specification from the configuration.
        config_out.pop("format")

    #-----------------------------------------------------------------#

    # If there is a single color specified in the configuration.
    if "color" in config_colors:

        # Each sub-plot will have the defined color.
        colors_plot = [config_colors["color"]]

    # If there is a list of colors defined
    if "colors" in config_colors:

        # Get now many colors were provided.
        num_colors = len(config_colors["colors"])

        # If there are fewer colors than the number of sub-plots.
        if num_colors < len(dfs_dim_red):

            # Warn the user that the colors will be re-used.
            warnstr = \
                f"{num_colors} were provided to plot the " \
                f"data contained in {len(dfs_dim_red)} data " \
                "frames. The colors will be re-used."
            logger.warning(warnstr)

            # Set the cycle from which we are going to extract the
            # colors.
            cycle_colors = itertools.cycle(config_colors["colors"])

            # Set the list of colors cycling through the available
            # ones.
            colors_plot = \
                [next(cycle_colors) for _ in range(len(dfs_dim_red))]

    # If there is a color map defined
    if "cmap" in config_colors:
        
        # Get the colors that will be used for the sub-plots from the
        # color map.
        colors_plot = \
            _util.get_colormap(cmap = config_colors["cmap"],
                               n_colors = len(dfs_dim_red))

    #-----------------------------------------------------------------#

    # If the user passed names for the data frames
    if dfs_names is not None:

        # Make sure they are as many as the data frames passed.
        if len(dfs_dim_red) != len(dfs_names):

            # Raise an error.
            errstr = \
                "'dfs_names' must be the same length as 'dfs_dim_red'."
            raise ValueError(errstr)

        # Get the configuration to be used for each of them. If a data
        # frame does not have a specific configuration, the general
        # one will be used.
        configs_spec = \
            {df_name if config_subs.get(df_name) else "general" : \
                util.recursive_merge_dicts(\
                    config_subs_general, 
                    config_subs.get(df_name, {})) \
             for df_name in dfs_names}

    # Otherwise
    else:

        # All sub-plots will use the general configuration.
        configs_spec = {"general" : config_subs_general}

    #-----------------------------------------------------------------#

    # Split the list of data frames into equally-sized chunks with
    # 'plots_per_output' data frames per chunk.
    dfs_chunks = \
        [dfs_dim_red[i:i + plots_per_output] \
         for i in range(0, len(dfs_dim_red), plots_per_output)]

    # Get the maximum number of plots in a chunk.
    max_chunk = max([len(chunk) for chunk in dfs_chunks])

    # Inform the user about how many output files will be written.
    infostr = f"{len(dfs_chunks)} output files will be generated."
    logger.info(infostr)

    #-----------------------------------------------------------------#
    
    # Initialize the counter to keep track of the current plot's
    # number to 0.
    plot_num = 0
    
    # For each chunk and associated output number
    for num_output, dfs_chunk in enumerate(dfs_chunks):

        # Get the number of plots that will be in the current output
        # file from the number of data frames in the chunk.
        num_plots = max_chunk

        # Get the best layout (the rectangle with the smallest
        # difference between dimensions) for the sub-plots from the
        # number of data frames passed.
        nrows, ncols = \
            _util.find_rectangular_grid(n = num_plots)

        # Generate the figure and subplots.
        fig, axes = \
            plt.subplots(\
                nrows = nrows,
                ncols = ncols,
                figsize = config_fig.get("size_inches", None))

        #-------------------------------------------------------------#

        # If there is a configuration to adjust the figure
        if config_fig:

            # Adjust the sub-plots.
            plt.subplots_adjust(**config_fig.get("subplots", {}))

        #-------------------------------------------------------------#
            
        # For each data frame
        for df_dim_red, ax in \
            zip(dfs_chunk, axes.flat[:len(dfs_chunk)]):

            # If the user provided the names of the data frames
            if dfs_names is not None:

                # Get the name of the current plot.
                plot_name = dfs_names[plot_num]

            # Otherwise
            else:

                # The name of the current plot will be just the plot's
                # number.
                plot_name = plot_num

            #---------------------------------------------------------#

            # If there are specific groups to be plotted for the
            # current plot, and the plot is defined by its name
            if plot_name in groups:

                # Get them.
                plot_groups = groups[plot_name]

            # If there are specific groups to be plotted for the
            # current plot, and the plot is defined by its number
            elif plot_num in groups:
                
                # Get them.
                plot_groups = groups[plot_num]

            # Otherwise
            else:

                # All groups will be treated the same for the current
                # plot.
                plot_groups = None

            #---------------------------------------------------------#

            # Get the configuration for the current plot.
            config_sub = \
                configs_spec.get(plot_name, 
                                 configs_spec.get("general", {}))

            #---------------------------------------------------------#

            # Check the configuration.
            _util.check_custom_configs(config = config_sub,
                                       sections = sections_sub,
                                       multiple_plots = True)

            #---------------------------------------------------------#

            # If more or less than two columns were selected
            if len(columns) != 2:

                # Raise an error.
                errstr = \
                    "Exactly two 'columns' must be provided, but " \
                    f"{len(columns)} were passed."
                raise ValueError(errstr)

            # Get the names of the columns containing the values of the
            # projections along the first and second dimension.
            c1_col, c2_col = columns

            #---------------------------------------------------------#

            # If specific groups were defined
            if plot_groups is not None:

                # Get the data points belonging to other groups,
                df_dim_red_others = \
                    df_dim_red[~df_dim_red[groups_column].isin(\
                        plot_groups)]

                # Get the data points belonging to the groups of
                # interest.
                df_dim_red_selected = \
                    df_dim_red[df_dim_red[groups_column].isin(\
                        plot_groups)]

            # Otherwise
            else:

                # There will be no data points belonging to other
                # groups.
                df_dim_red_others = None

                # The selected groups will be all the groups.
                df_dim_red_selected = df_dim_red

            #---------------------------------------------------------#

            # If we have to color the data points belonging to the
            # other groups differently
            if df_dim_red_others is not None and plot_other_groups:

                # Plot them (we plot them fist so that the points of
                # the groups of interest are plotted on top of them).
                ax = sns.scatterplot(\
                        x = c1_col,
                        y = c2_col,
                        data = df_dim_red_others,
                        ax = ax,
                        legend = False,
                        **config_sub.get("other_groups", {}))

            #---------------------------------------------------------#

            # Generate the scatter plot.
            ax = sns.scatterplot(\
                    x = c1_col,
                    y = c2_col,
                    data = df_dim_red_selected,
                    ax = ax,
                    legend = False,
                    color = colors_plot[plot_num],
                    **config_sub.get("scatterplot", {}))

            #---------------------------------------------------------#

            # Hide the top and right spine.
            for spine in ["top", "right"]:
                ax.spines[spine].set_visible(False)

            #---------------------------------------------------------#

            # If the current plot has a name
            if plot_name != "general":

                # Get the configuration for the plot's title.
                config_sub_title = config_sub.get("title", {})
                
                # Set the current plot's title based on the data frame
                # that is being plotted.
                ax.set_title(label = plot_name,
                             **config_sub_title)

            #---------------------------------------------------------#

            # Get the configuration for each x-axis.
            config_sub_x_axis = config_sub.get("xaxis", {})

            # Get the positions of the ticks on the x-axis.
            x_ticks = \
                _util.get_ticks_positions(values = df_dim_red[c1_col],
                                          item = "x-axis",
                                          config = config_sub_x_axis)

            # Set the x-axis.
            ax = _util.set_axis(ax = ax,
                                axis = "x",
                                config = config_sub_x_axis,
                                ticks = x_ticks)

            #---------------------------------------------------------#
            
            # Get the configuration for each y-axis.
            config_sub_y_axis = config_sub.get("yaxis", {})

            # Get the positions of the ticks on the y-axis.
            y_ticks = \
                _util.get_ticks_positions(values = df_dim_red[c2_col],
                                          item = "y-axis",
                                          config = config_sub_y_axis)

            # Set the y-axis.
            ax = _util.set_axis(ax = ax,
                                axis = "y",
                                config = config_sub_y_axis,
                                ticks = y_ticks)

            #---------------------------------------------------------#

            # Update the plot's number.
            plot_num += 1

        #-------------------------------------------------------------#

        # For each extra axis in the figure
        for ax in axes.flat[len(dfs_chunk):]:

            # Set it to mimic the last one in the figure.

            # Set the x-axis.
            ax = _util.set_axis(ax = ax,
                                axis = "x",
                                config = config_sub_x_axis,
                                ticks = x_ticks)

            #---------------------------------------------------------#

            # Set the y-axis.
            ax = _util.set_axis(ax = ax,
                                axis = "y",
                                config = config_sub_y_axis,
                                ticks = y_ticks)

            #---------------------------------------------------------#

            # Remove it.
            ax.axis("off")

        #-------------------------------------------------------------#

        # Set the name of the output file.
        output_file = output_prefix + f"{num_output+1}.{output_fmt}"

        # Write the plot in the output file.
        plt.savefig(fname = output_file,
                    **config_out)

        #-------------------------------------------------------------#

        # Close any figure that may be open.
        plt.close()