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
    """Plot the CPU/wall clock time spent in each epoch of each
    round of optimization when finding the representations for a
    set of samples (both for the full epoch and for the
    backward step performed in each epoch).

    Parameters
    ----------
    df : :class:`pandas.DataFrame`
        A data frame containing the time data. This data frame is
        produced as an output by the
        :class:`bulkdgd.core.model.BulkDGD.get_representations`
        method.

    output_file : :class:`str`, optional
        The file where the plot will be saved. If not provided, the
        plot will be generated but not saved.

    config : :class:`dict`, optional
        A dictionary containing the configuration for the plot's
        aesthetics.

        Alternatively, the options for the plot's aesthetics can be
        provided using keyword arguments.

        The available options can be found in the :doc:`documentation
        <plotting>`.

        If no configuration is provided, the default configuration
        (taken from the ``configs/plotting/lineplot.yaml`` file) will
        be used.

    **kwargs
        Additional keyword arguments representing options for the
        plot's aesthetics.

        The available options can be found in the :doc:`documentation
        <plotting>`.
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

    # Get the configuration for the plot's aesthetics by merging the
    # configuration provided (if any) with the keyword arguments (it
    # any).
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
            "The configiration is not valid. Errors: " + \
            " ".join(errors)
        raise ValueError(errstr)

    #-----------------------------------------------------------------#

    # Create a copy of the original data frame to modify before
    # generating the plot.
    df_to_plot = copy.deepcopy(df)

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

    #-----------------------------------------------------------------#

    # Generate the figure and axes. The plots will be arranged into
    # one row and as many columns as the number of optimization rounds
    # run when finding the representations.
    _, axes = plt.subplots(nrows = 1,
                           ncols = len(opt_rounds))

    # Ensure iterable axes also when there is a single subplot.
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
        config_copy = dict(config)

        #-------------------------------------------------------------#

        # Get title's configuration.
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
        columns: list[str] = ["C1", "C2"],
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
    dfs : :class:`pandas.DataFrame`
        A list of data frames containing the results of the
        dimensionality reduction analyses.

        The rows of each data frame should contain the data points,
        while the columns should contain the values of each data
        point's projection along the principal components.
    
    output_file : :class:`str`, optional
        The file where the plot will be saved. If not provided, the
        plot will be generated but not saved.

    config : :class:`dict`, optional
        A dictionary containing the configuration for the plot's
        aesthetics.

        Alternatively, the options for the plot's aesthetics can be
        provided using keyword arguments.

        The available options can be found in the :doc:`documentation
        <plotting>`.

        If no configuration is provided, the default configuration
        (taken from the ``configs/plotting/scatterplot.yaml`` file)
        will be used.
    
    max_plots_per_output : :class:`int`, ``9``
        The maximum number of plots for each output file.

    columns : :class:`list`, ``["PC1", "PC2"]``
        A list with the names of the two columns in each data frame
        that contain the values of the two dimensions of the
        projection's space to be considered when plotting.

    groups_column : :class:`str`, optional
        The name of the column containing the labels of different
        groups of data points in the data frames, if any.

        If provided, the data points will be colored according to the
        group they belong.

        If not provided, the data points will be assumed to belong to
        one group.

    groups : :class:`list`, optional
        A list of groups of interest.
        
        If a list of groups is provided and ``plot_other_groups``
        is ``False``, only data points belonging to the groups of
        interest will be plotted.
        
        If ``plot_other_groups`` is ``True``, the other groups will
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

    **kwargs
        Additional keyword arguments representing options for the
        plot's aesthetics.

        The available options can be found in the :doc:`documentation
        <plotting>`.
    """

    # Close any figure that may be open.
    plt.close()

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

        # If the names were provided but is not a list
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
        A list of groups of interest.
        
        If a list of groups is provided, only the enrichment scores of
        the samples belonging to the groups of interest will be
        plotted.
        
        If not provided, the enrichment scores of all the samples
        will be plotted.
    
    config : :class:`dict`, optional
        A dictionary containing the configuration for the plot's
        aesthetics.

        Alternatively, the options for the plot's aesthetics can be
        provided using keyword arguments.

        The available options can be found in the :doc:`documentation
        <plotting>`.
    
    output_file : :class:`str`, optional
        The file where the plot will be saved.
    
    **kwargs
        Additional keyword arguments representing options for the
        plot's aesthetics.

        The available options can be found in the :doc:`documentation
        <plotting>`.
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

    # Get the configuration for the plot's aesthetics by merging the
    # configuration provided (if any) with the keyword arguments (it
    # any) and the default configuration.
    config = \
        _internals.recursive_merge_dicts(\
            config if config is not None else {},
            kwargs,
            config_default)
    
    #-----------------------------------------------------------------#

    # Check the configuration.
    config, errors = _util.check_config_plot(config = config)

    # If there are errors in the configuration
    if errors:

        # Raise an exception.
        errstr = \
            "The configiration is not valid. Errors: " + \
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
    """Plot the distribution of r-values for specific genes for one
    set of samples or two paired sets of samples (for instance, normal
    samples and cancer samples for the same tissue).

    Parameters
    ----------
    dfs : :class:`list` or :class:`pandas.DataFrame`
        One or two data frames containing the r-values values for one
        or two sets of samples.

        The rows should contain the samples, while the columns should
        contain the r-values of the negative binomial distributions
        modeling the genes.
    
    genes : :class:`list`
        A list of the names of the genes for which the r-values will be
        plotted.

    output_file : :class:`str`, optional
        The file(s) where the plot(s) will be saved.
        
        If multiple files need to be generated, the output file name
        will be constructed by appending a number to the name provided.
        
        The format of the output file is inferred from its extension. 
        
        If not provided, the plot(s) will be generated but not saved.
    
    plot_type : :class:`str`, {``"histogram"``, ``"histogram_dual``, \
        ``"histogram_overlap"``, ``"boxplot"``, ``"violinplot"``}, \
        ``"histogram"``
        The type of plot to generate. The available options are:

        * ``"histogram"``: histograms of the r-values of the genes of
          interest for one set of samples.

        * ``"histogram_bihist"``: bi-histograms showing the
          distributions of r-values of the genes of interest for two
          paired sets of samples.
        
        * ``"histogram_overlap"``: two overlapping histograms showing
          the distribution of r-values of the genes of interest for two
          paired sets of samples.

        * ``"boxplot"``: box plots showing the distributions of
          r-values of the genes of interest in either a set of samples
          or in two paired sets of samples (paired box plots).

        * ``"violinplot"``: violin plots showing the distributions of
          r-values of the genes of interest in either a set of samples
          or in two paired sets of samples (paired violin plots).

    config : :class:`dict`, optional
        A dictionary containing the configuration for the plot's
        aesthetics.

        Alternatively, the options for the plot's aesthetics can be
        provided using keyword arguments.

        The available options can be found in the :doc:`documentation
        <plotting>`.
    
    max_plots_per_output : :class:`int`, ``9``
        The maximum number of plots for each output file.
    
    categories : :class:`list`, optional
        A list of two categories used when generating dual histograms
        or paired box plots or violin plots.

        If provided, they are used to generate the legend of the plot.

    **kwargs
        Additional keyword arguments representing options for the
        plot's aesthetics.

        The available options can be found in the :doc:`documentation
        <plotting>`.
    """

    # Close any figure that may be open.
    plt.close()

    #-----------------------------------------------------------------#

    # Keep a stable reference to the original input before reshaping
    # per-gene data.
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

        # The second set of data frames will be None.
        dfs_2 = None

    # If two data frames were passed
    elif len(dfs_input) == 2:

        # Take only the columns of interest.
        dfs = [dfs_input[0][gene] for gene in genes]

        # Substitute infinite values with NaN.
        dfs = [df.replace([np.inf, -np.inf], np.nan) for df in dfs]

        # Take only the columns of interest.
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
    if plot_type.startswith("histogram"):

        # Get whether the plot is a density plot or not.
        is_density = config.get(plot_type, {}).get("density", False)

        # Set the default label for the x-axis.
        x_label = "Magnitude of the r-values"

        # Set the default label for the y-axis.
        y_label = "Density" if is_density else "Counts"

        # If the plot type is a simple histogram
        if plot_type == "histogram":

            # Add the default label to the configuration.
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
