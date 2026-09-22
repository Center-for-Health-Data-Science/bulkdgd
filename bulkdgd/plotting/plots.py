#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-

#    plots.py
#
#    Utilities for plotting.
#
#    Copyright (C) 2026 Valentina Sora
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
from typing import Optional
import warnings

# Import from third-party libraries.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

# Import from 'bulkdgd'.
from . import _util
from bulkdgd import _internals, defaults


#######################################################################


# Ignore warnings (matplotlib's 'UserWarnings').
warnings.filterwarnings("ignore", category = UserWarning)


#######################################################################


# Get the module's logger.
logger = log.getLogger(__name__)


#######################################################################


def plot_representations_time(
        df: pd.DataFrame,
        output_file: Optional[str] = None,
        config: Optional[dict[str, object]] = None,
        **kwargs: Optional[dict[str, object]]) -> None:
    """Plot the CPU/wall clock time spent in each epoch (and in its
    backward step) of each optimization round when finding the
    representations for a set of samples.

    Parameters
    ----------
    df : :class:`pandas.DataFrame`
        A data frame containing the time data, as produced by
        :meth:`bulkdgd.core.model.BulkDGD.get_representations`.

    output_file : :class:`str`, optional
        The file where the plot will be saved. If not provided, the
        plot is shown instead.

    config : :class:`dict`, optional
        The configuration for the plot's aesthetics. The available
        options are listed in the :doc:`documentation <plotting>`.
        It is merged with ``configs/plotting/lineplot.yaml``.

    **kwargs
        Additional options for the plot's aesthetics.
    """

    # Close any figure that may be open.
    plt.close()

    #-----------------------------------------------------------------#

    # Remove the keyword arguments that are not needed.
    kwargs = \
        _internals.kwargs_to_dict(\
            kwargs = {k : v for k, v in kwargs.items() \
                      if k not in ["dfs", "config", "output_file"]})

    #-----------------------------------------------------------------#

    # Merge the configuration provided (if any) with the keyword
    # arguments (if any).
    config = \
        _internals.recursive_merge_dicts(\
            config if config is not None else {},
            kwargs)

    #-----------------------------------------------------------------#

    # Get the default configuration for the plot.
    config_default = \
        yaml.safe_load(open(defaults.CONFIG_FILES_PLOT["lineplot"],
                            "r"))

    # Add the default label for the x-axis to the configuration.
    config_default = \
        _internals.recursive_add_items(\
            d = config_default,
            paths2values = {("xaxis", "label", "xlabel"): "Epochs"})

    # Add the default label for the y-axis to the configuration.
    config_default = \
        _internals.recursive_add_items(\
            d = config_default,
            paths2values = {("yaxis", "label", "ylabel"): "Time (s)"})

    #-----------------------------------------------------------------#

    # Merge the default configuration with the configuration provided
    # by the user.
    config = \
        _internals.recursive_merge_dicts(\
            config_default,
            config)

    #-----------------------------------------------------------------#

    # Check the configuration.
    config, errors = _util.check_config_plot(config = config)

    # If there are errors in the configuration
    if errors:

        # Raise an exception.
        errstr = \
            "The configuration is not valid. Errors: " + \
            " ".join(errors)
        raise ValueError(errstr)

    #-----------------------------------------------------------------#

    # Create a copy of the original data frame.
    df_to_plot = copy.deepcopy(df)

    # Melt the CPU and wall clock time columns into one column.
    df_to_plot = \
        df_to_plot.melt(\
            ["platform", "processor", "num_threads",
             "opt_round", "epoch"],
            var_name = "Time (CPU/Wall clock)",
            value_name = "Time (s)")

    #-----------------------------------------------------------------#

    # Get the optimization round(s) that was (were) performed.
    opt_rounds = df_to_plot["opt_round"].unique()

    #-----------------------------------------------------------------#

    # Generate the figure and axes (one column per optimization
    # round).
    _, axes = plt.subplots(nrows = 1,
                           ncols = len(opt_rounds))

    # Make the axes iterable if there is a single subplot.
    axes = np.atleast_1d(axes)

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

        # Create a copy of the configuration.
        config_copy = copy.deepcopy(config)

        #-------------------------------------------------------------#

        # Get the title's configuration.
        config_title = config_copy.get("title", {})

        # If there is a label
        if config_title.get("label") is not None:

            # Get the raw label.
            label_raw = config_title.pop("label")

            # Substitute the '[opt_round]' string with the actual
            # number/name of the current optimization round.
            label = label_raw.replace("[opt_round]",
                                      str(opt_round),
                                      1)

            # Replace it in the configuration.
            config_copy["title"]["label"] = label

        #-------------------------------------------------------------#

        # Generate the line plot.
        _util.plot_lineplot(data = sub_df,
                            x = "epoch",
                            y = "Time (s)",
                            hue = "Time (CPU/Wall clock)",
                            ax = ax,
                            config = config_copy)

    #-----------------------------------------------------------------#

    # If an output file was provided
    if output_file is not None:

        # Save the plot in the output file.
        plt.savefig(fname = output_file,
                    **config.get("output", {}))

    # Otherwise
    else:

        # Show the plot.
        plt.show()


def plot_dimensionality_reduction(
        dfs: list[pd.DataFrame],
        output_file: Optional[str] = None,
        config: Optional[dict[str, object]] = None,
        max_plots_per_output: int = 9,
        columns: Optional[list[str]] = None,
        groups_column: Optional[str] = None,
        groups: Optional[list[str]] = None,
        plot_other_groups: bool = False,
        dfs_names: Optional[list[str]] = None,
        **kwargs: Optional[dict[str, object]]) -> None:
    """Plot the results of one or several dimensionality reduction
    analyses on a single figure (which may be split in multiple
    output files).

    Parameters
    ----------
    dfs : :class:`list` or :class:`pandas.DataFrame`
        A list of data frames containing the results of the
        dimensionality reduction analyses (data points as rows,
        projections along the components as columns).

    output_file : :class:`str`, optional
        The file where the plot will be saved. If not provided, the
        plot is shown instead.

    config : :class:`dict`, optional
        The configuration for the plot's aesthetics. The available
        options are listed in the :doc:`documentation <plotting>`.
        It is merged with ``configs/plotting/scatterplot.yaml``.

    max_plots_per_output : :class:`int`, ``9``
        The maximum number of plots for each output file.

    columns : :class:`list`, ``["C1", "C2"]``
        The names of the two columns to plot on the x- and y-axis.

    groups_column : :class:`str`, optional
        The name of the column containing the groups of the data
        points, if any.

    groups : :class:`list`, optional
        A list of groups of interest. If provided, only the data
        points belonging to these groups are plotted, unless
        ``plot_other_groups`` is ``True``.

    plot_other_groups : :class:`bool`, :obj:`False`
        If ``groups`` is provided, whether to also plot the data
        points belonging to the other groups.

    dfs_names : :class:`list` or :class:`str`, optional
        The names of the data frames, used as the titles of the
        corresponding plots.

    **kwargs
        Additional options for the plot's aesthetics.
    """

    # Close any figure that may be open.
    plt.close()

    #-----------------------------------------------------------------#

    # If no columns were passed
    if columns is None:

        # Use the default columns.
        columns = ["C1", "C2"]

    #-----------------------------------------------------------------#

    # If the data is a single data frame
    if isinstance(dfs, pd.DataFrame):

        # Put it in a list.
        dfs = [dfs]

        # If the data frame's name was provided but is not a string
        if dfs_names is not None and not isinstance(dfs_names, str):

            # Raise an error.
            errstr = \
                "'dfs_names' must be a string if 'dfs' is a single " \
                "data frame."
            raise ValueError(errstr)

    # If the data is a list of data frames
    elif isinstance(dfs, list):

        # If the names were provided but are not a list
        if dfs_names is not None and not isinstance(dfs_names, list):

            # Raise an error.
            errstr = \
                "'dfs_names' must be a list if 'dfs' is a list of " \
                "data frames."
            raise ValueError(errstr)

    # Otherwise
    else:

        # Raise an error.
        errstr = \
            "'dfs' must be a single data frame or a list of data " \
            "frames."
        raise ValueError(errstr)

    #-----------------------------------------------------------------#

    # Get the default configuration for the scatter plots.
    config_default = \
        yaml.safe_load(\
            open(defaults.CONFIG_FILES_PLOT["scatterplot"], "r"))

    # Add the default label for the x-axis to the configuration.
    config_default = \
        _internals.recursive_add_items(\
            d = config_default,
            paths2values = {("xaxis", "label", "xlabel"): "C1"})

    # Add the default label for the y-axis to the configuration.
    config_default = \
        _internals.recursive_add_items(\
            d = config_default,
            paths2values = {("yaxis", "label", "ylabel"): "C2"})

    #-----------------------------------------------------------------#

    # Set the keyword arguments for the plotting function.
    plot_func_kwargs = \
       {"columns" : columns,
        "groups_column" : groups_column,
        "groups" : groups,
        "plot_other_groups" : plot_other_groups}

    #-----------------------------------------------------------------#

    # Generate the plots.
    _util.generate_plots(dfs = dfs,
                         output_file = output_file,
                         plot_type = "scatterplot",
                         plot_func_kwargs = plot_func_kwargs,
                         max_plots_per_output = max_plots_per_output,
                         config = config,
                         config_default = config_default,
                         dfs_names = dfs_names,
                         kwargs = kwargs)


def plot_enrichment_scores(
        df: pd.DataFrame,
        groups_column: str,
        gene_set: str,
        gene_set_column: str = "gene_set",
        num_genes_in_set_column: str = "num_genes_in_set",
        num_genes_significant_column: str = "num_genes_significant",
        e_score_column: str = "e_score",
        groups: Optional[list[str]] = None,
        config: Optional[dict[str, object]] = None,
        output_file: Optional[str] = None,
        **kwargs: Optional[dict[str, object]]) -> None:
    """Plot the enrichment scores for sets of samples belonging to
    different groups.

    Parameters
    ----------
    df : :class:`pandas.DataFrame`
        A data frame containing the enrichment scores.

    groups_column : :class:`str`
        The name of the column containing the labels of different
        groups in the data frame.

    gene_set : :class:`str`
        The name of the gene set for which the enrichment scores will
        be plotted.

    gene_set_column : :class:`str`, ``"gene_set"``
        The name of the column containing the labels of the gene sets
        in the data frame.

    num_genes_in_set_column : :class:`str`, ``"num_genes_in_set"``
        The name of the column containing the number of genes in each
        gene set in the data frame.

    num_genes_significant_column : :class:`str`, \
        ``"num_genes_significant"``
        The name of the column containing the number of significant
        genes in the data frame.

    e_score_column : :class:`str`, ``"e_score"``
        The name of the column containing the enrichment scores in the
        data frame.

    groups : :class:`list`, optional
        A list of groups of interest. If provided, only the enrichment
        scores of these groups are plotted.

    config : :class:`dict`, optional
        The configuration for the plot's aesthetics. The available
        options are listed in the :doc:`documentation <plotting>`.

    output_file : :class:`str`, optional
        The file where the plot will be saved. If not provided, the
        plot is shown instead.

    **kwargs
        Additional options for the plot's aesthetics.
    """

    # Close any figure that may be open.
    plt.close()

    #-----------------------------------------------------------------#

    # Remove the keyword arguments that are not needed to update
    # the configuration.
    kwargs = \
        _internals.kwargs_to_dict(\
            kwargs = {k : v for k, v in kwargs.items() if k not in \
                      ["df", "groups_column", "gene_set",
                       "gene_set_column", "num_genes_in_set_column",
                       "num_genes_significant_column",
                       "e_score_column", "groups", "config",
                       "output_file"]})

    #-----------------------------------------------------------------#

    # Get the default configuration for the enrich plot.
    config_default = \
        yaml.safe_load(\
            open(defaults.CONFIG_FILES_PLOT["enrichplot"], "r"))

    #-----------------------------------------------------------------#

    # Merge the default configuration, the configuration provided
    # (if any), and the keyword arguments (if any).
    config = \
        _internals.recursive_merge_dicts(\
            config_default,
            config if config is not None else {},
            kwargs)

    #-----------------------------------------------------------------#

    # Check the configuration.
    config, errors = _util.check_config_plot(config = config)

    # If there are errors in the configuration
    if errors:

        # Raise an exception.
        errstr = \
            "The configuration is not valid. Errors: " + \
            " ".join(errors)
        raise ValueError(errstr)

    #-----------------------------------------------------------------#

    # If the user passed a list of groups of interest
    if groups is not None:

        # Take only the rows of interest.
        df = df.loc[df[groups_column].isin(groups)]

    #-----------------------------------------------------------------#

    # Take only the rows for the selected gene set.
    df = df.loc[df[gene_set_column] == gene_set]

    #-----------------------------------------------------------------#

    # Generate the plot.
    _util.plot_enrichplot(data = df,
                          x = groups_column,
                          y_1 = num_genes_in_set_column,
                          y_2 = num_genes_significant_column,
                          y_3 = e_score_column,
                          hue = gene_set_column,
                          config = config)

    #-----------------------------------------------------------------#

    # If an output file was provided
    if output_file is not None:

        # Save the plot in the output file.
        plt.savefig(fname = output_file,
                    **config.get("output", {}))

    # Otherwise
    else:

        # Show the plot.
        plt.show()


def plot_rvalues(dfs: list[pd.DataFrame] | pd.DataFrame,
                 genes: list[str],
                 output_file: Optional[str] = None,
                 plot_type: str = "histogram",
                 config: Optional[dict[str, object]] = None,
                 max_plots_per_output: int = 9,
                 categories: Optional[list[str]] = None,
                 **kwargs: Optional[dict[str, object]]) -> None:
    """Plot the distribution of the r-values of specific genes in
    one set of samples or in two paired sets of samples.

    Parameters
    ----------
    dfs : :class:`list` or :class:`pandas.DataFrame`
        One or two data frames containing the r-values for one or two
        sets of samples (samples as rows, genes as columns).

    genes : :class:`list`
        The names of the genes whose r-values will be plotted.

    output_file : :class:`str`, optional
        The file where the plot(s) will be saved. Multiple files get
        a number appended to the name. If not provided, the plot(s)
        are shown instead.

    plot_type : :class:`str`, {``"histogram"``, \
        ``"histogram_bihist"``, ``"histogram_overlap"``, \
        ``"boxplot"``, ``"violinplot"``}, ``"histogram"``
        The type of plot to generate:

        * ``"histogram"``: histograms for one set of samples.

        * ``"histogram_bihist"``: bi-histograms for two paired sets
          of samples.

        * ``"histogram_overlap"``: overlapping histograms for two
          paired sets of samples.

        * ``"boxplot"``: box plots for one set of samples or two
          paired sets of samples.

        * ``"violinplot"``: violin plots for one set of samples or two
          paired sets of samples.

    config : :class:`dict`, optional
        The configuration for the plot's aesthetics. The available
        options are listed in the :doc:`documentation <plotting>`.

    max_plots_per_output : :class:`int`, ``9``
        The maximum number of plots for each output file.

    categories : :class:`list`, optional
        The names of the two paired sets of samples, used in the
        legend.

    **kwargs
        Additional options for the plot's aesthetics.
    """

    # Close any figure that may be open.
    plt.close()

    #-----------------------------------------------------------------#

    # Keep a reference to the original input.
    dfs_input = dfs

    # If the data is a single data frame
    if isinstance(dfs_input, pd.DataFrame):

        # Put it in a list.
        dfs_input = [dfs_input]

    #-----------------------------------------------------------------#

    # If only one data frame was passed
    if len(dfs_input) == 1:

        # Take only the columns of interest.
        dfs = [dfs_input[0][gene] for gene in genes]

        # Substitute infinite values with NaN.
        dfs = [df.replace([np.inf, -np.inf], np.nan) for df in dfs]

        # Set no second set of data frames.
        dfs_2 = None

    # If two data frames were passed
    elif len(dfs_input) == 2:

        # Take only the columns of interest.
        dfs = [dfs_input[0][gene] for gene in genes]

        # Substitute infinite values with NaN.
        dfs = [df.replace([np.inf, -np.inf], np.nan) for df in dfs]

        # Take only the columns of interest for the second set.
        dfs_2 = [dfs_input[1][gene] for gene in genes]

        # Substitute infinite values with NaN.
        dfs_2 = [df.replace([np.inf, -np.inf], np.nan) for df in dfs_2]

        # If no categories were passed
        if categories is None:

            # Set them to default values.
            categories = ["Category 1", "Category 2"]

    # Otherwise
    else:

        # Raise an error.
        errstr = \
            "'dfs' must be a single data frame or a list of two " \
            "data frames."
        raise ValueError(errstr)

    #-----------------------------------------------------------------#

    # Merge the configuration provided by the user with the keyword
    # arguments.
    config = \
        _internals.recursive_merge_dicts(\
            config if config is not None else {},
            kwargs)

    #-----------------------------------------------------------------#

    # Initialize an empty dictionary to store the default
    # configuration.
    config_default = {}

    #-----------------------------------------------------------------#

    # If the plot type is a histogram
    if plot_type in \
        ["histogram", "histogram_bihist", "histogram_overlap"]:

        # Get the default configuration for the histogram.
        config_hist = \
            yaml.safe_load(\
                open(defaults.CONFIG_FILES_PLOT[plot_type],
                     "r")).get("histogram", {})

        # Update it with the configuration provided.
        config_hist.update(config.get("histogram", {}))

        # Get whether the plot is a density plot or not.
        is_density = config_hist.get("density", False)

        # Set the default label for the x-axis.
        x_label = "Magnitude of the r-values"

        # Set the default label for the y-axis.
        y_label = "Density" if is_density else "Counts"

        # If the plot type is a simple histogram
        if plot_type == "histogram":

            # Add the default colorbar label to the configuration.
            config_default = \
                _internals.recursive_add_items(\
                    d = config_default,
                    paths2values = \
                        {("colorbar", "label", "label"): y_label})

    #-----------------------------------------------------------------#

    # If the plot type is a boxplot or a violin plot
    elif plot_type in ["boxplot", "violinplot"]:

        # Set the default label for the x-axis.
        x_label = "Categories"

        # Set the default label for the y-axis.
        y_label = "Magnitude of the r-values"

    #-----------------------------------------------------------------#

    # Otherwise
    else:

        # Raise an error.
        errstr = \
            f"Unsupported plot type '{plot_type}'. Supported plot " \
            "types are: 'histogram', 'histogram_bihist', " \
            "'histogram_overlap', 'boxplot', and 'violinplot'."
        raise ValueError(errstr)

    #-----------------------------------------------------------------#

    # Add the default label for the x-axis to the configuration.
    config_default = \
        _internals.recursive_add_items(\
            d = config_default,
            paths2values = {("xaxis", "label", "xlabel"): x_label})

    # Add the default label for the y-axis to the configuration.
    config_default = \
        _internals.recursive_add_items(\
            d = config_default,
            paths2values = {("yaxis", "label", "ylabel"): y_label})

    #-----------------------------------------------------------------#

    # Get the final default configuration.
    config_default = \
        _internals.recursive_merge_dicts(\
            config_default,
            yaml.safe_load(\
                open(defaults.CONFIG_FILES_PLOT[plot_type], "r")))

    #-----------------------------------------------------------------#

    # Generate the plots.
    _util.generate_plots(dfs = dfs,
                         dfs_2 = dfs_2,
                         output_file = output_file,
                         plot_type = plot_type,
                         max_plots_per_output = max_plots_per_output,
                         config = config,
                         config_default = config_default,
                         dfs_names = genes,
                         categories = categories,
                         kwargs = kwargs)
