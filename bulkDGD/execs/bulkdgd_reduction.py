#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-

#    bulkdgd_reduction.py
#
#    Perform dimensionality reduction analyses.
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
__doc__ = "Perform dimensionality reduction analyses."


#######################################################################


# Import from the standard library.
import logging as log
import os
import pickle as pk
import sys
# Import from third-party packages.
import numpy as np
import pandas as pd
# Import from 'bulkDGD'.
from bulkDGD import defaults, ioutil, plotting
from bulkDGD.analysis import reduction
from . import util


#######################################################################


# Get the module's logger.
logger = log.getLogger(__name__)


#######################################################################


# Set a mapping between the names of the dimensionality reduction
# analyses used in the program and the names to display in the help
# messages.
NAME2HELPNAME = \
    {# PCA
     "pca" : "PCA",
     # KPCA
     "kpca" : "KPCA",
     # MDS
     "mds" : "MDS",
     # t-SNE
     "tsne" : "t-SNE"}

# Set a mapping between the name each dimensionality reduction analysis
# and the function used to perform it.
DIMREDNAME2DIMREDFUNC = \
    {# PCA
     "pca" : reduction.perform_pca,
     # KPCA
     "kpca" : reduction.perform_kpca,
     # MDS
     "mds" : reduction.perform_mds,
     # t-SNE
     "tsne" : reduction.perform_tsne}


#######################################################################


# Define a function to set up the parser.
def set_parser(sub_parsers):

    # Create the argument parser.
    parser = \
        sub_parsers.add_parser(\
            name = "reduction",
            description = __doc__,
            help = __doc__,
            formatter_class = util.CustomHelpFormatter)

    #-----------------------------------------------------------------#

    # Return the parser.
    return parser


#######################################################################


# Define a function to set up the sub-parser.
def set_sub_parser(sub_parsers,
                   dim_red_name):

    # Set the name of the dimensionality reduction analysis that will
    # be used in the help messages.
    dim_red_help = NAME2HELPNAME[dim_red_name]

    #-----------------------------------------------------------------#

    # Create the argument parser.
    parser = \
        sub_parsers.add_parser(\
            name = dim_red_name,
            description = f"Perform {dim_red_help}.",
            help = f"Perform {dim_red_help}.",
            formatter_class = util.CustomHelpFormatter)

    #-----------------------------------------------------------------#

    # Create a group of arguments for the input files.
    input_group = \
        parser.add_argument_group(title = "Input options")

    # Create a group of arguments for the output files.
    output_group = \
        parser.add_argument_group(title = "Output files")

    # Create a group of arguments for the configuration files.
    config_group = \
        parser.add_argument_group(title = "Configuration files")

    # Create a group of arguments for the pre-processing.
    preproc_group = \
        parser.add_argument_group(title = "Pre-processing options")

    # Create a group of arguments for the plotting options.
    plot_group = \
        parser.add_argument_group(title = "Plotting options")

    #-----------------------------------------------------------------#
        
    # Set a help message.
    id_help = \
        "The input CSV file containing the data frame with the " \
        "data points."

    # Add the argument to the group.
    input_group.add_argument("-id", "--input-data",
                             type = str,
                             required = True,
                             help = id_help)

    #-----------------------------------------------------------------#

    # Set a help message.
    im_help = \
        "The input PKL file containing the fitted model on which " \
        "to project the new data points."

    # Add the argument to the group.
    input_group.add_argument("-im", "--input-model",
                             type = str,
                             help = im_help)

    #-----------------------------------------------------------------#

    # Set the help message.
    ic_help = \
        "A comma-separated list of columns or a string representing " \
        "a pattern matching the columns of interest. These will be " \
        "the columns considered when performing the " \
        f"{dim_red_help}. By default, all columns are considered."

    # Add the argument to the group.
    input_group.add_argument("-ic", "--input-columns",
                             type = util.process_arg_input_columns,
                             help = ic_help)

    #-----------------------------------------------------------------#
    
    # Set the default value for the argument.
    oa_default = f"{dim_red_name}.csv"

    # Set a help message.
    oa_help = \
        "The name of the output CSV file containing the results " \
        f"of the {dim_red_help}. The file will be written in " \
        "the working directory. The default file name is " \
        f"'{oa_default}'."
    
    # Add the argument to the group.
    output_group.add_argument("-oa", "--output-analysis",
                              type = str,
                              default = oa_default,
                              help = oa_help)

    #-----------------------------------------------------------------#

    # Set the default value for the argument.
    om_default = f"{dim_red_name}.pkl"

    # Set a help message.
    om_help = \
        "The name of the output pickle file containing the " \
        f"fitted model used to perform the {dim_red_help}. The " \
        "file will be written in the working directory. The " \
        f"default file name is '{om_default}'."

    # Add the argument to the group.
    output_group.add_argument("-om", "--output-model",
                              type = str,
                              default = om_default,
                              help = om_help)

    #-----------------------------------------------------------------#

    # Set the default value for the argument.
    op_default = f"{dim_red_name}"

    # Set a help message.
    op_help = \
        "The name of the output file containing the plot " \
        f"displaying the results of the {dim_red_help}. This " \
        "file will be written in the working directory. The " \
        f"default file name is '{op_default}'. The file format and, " \
        "therefore, extension are inferred from the 'output' " \
        "section of the configuration file for plotting."

    # Add the argument to the group.
    output_group.add_argument("-op", "--output-plot",
                              type = str,
                              default = op_default,
                              help = op_help)

    #-----------------------------------------------------------------#

    # Set the default directory where the configuration files are
    # located.
    cd_dir = defaults.CONFIG_DIRS["dimensionality_reduction"]

    # Set the default value for the argument.
    cd_default = defaults.CONFIG_FILES_DIM_RED["pca"]

    # Set a help message.
    cd_help = \
        "The YAML configuration file specifying the options " \
        f"for the {dim_red_help}. If it is a name without an " \
        "extension, it is assumed to be the name of a configuration " \
        f"file in '{cd_dir}'. If not provided, the default " \
        f"configuration file ('{cd_default}') will be used."
    
    # Add the argument to the group.
    config_group.add_argument("-cd", "--config-file-dim-red",
                              type = str,
                              default = cd_default,
                              help = cd_help)

    #-----------------------------------------------------------------#

    # Set the default directory where the configuration files are
    # located.
    cp_dir = defaults.CONFIG_DIRS["plotting"]

    # Set the default value for the argument.
    cp_default = defaults.CONFIG_FILES_PLOT["pca"]

    # Set a help message.
    cp_help = \
        "The YAML configuration file specifying the plot's " \
        "aesthetics and output format. If it is a name without an " \
        "extension, it is assumed to be the name of a configuration " \
        f"file in '{cp_dir}'. If not provided, the " \
        f"default configuration file ('{cp_default}') will be used."

    # Add the argument to the group.
    config_group.add_argument("-cp", "--config-file-plot",
                              type = str,
                              default = cp_default,
                              help = cp_help)

    #-----------------------------------------------------------------#

    # Set a help message.
    fp_help = \
        "Replace all positive infinite values with the given value " \
        f"before performing the {dim_red_help}."

    # Add the argument to the group.
    preproc_group.add_argument("-fp", "--fill-pos-inf",
                               type = float,
                               help = fp_help)

    #-----------------------------------------------------------------#

    # Set a help message.
    fn_help = \
        "Replace all negative infinite values with the given value " \
        f"before performing the {dim_red_help}."

    # Add the argument to the group.
    preproc_group.add_argument("-fn", "--fill-neg-inf",
                               type = float,
                               help = fn_help)

    #-----------------------------------------------------------------#

    # Set a help message.
    gc_help = \
        "The name/index of the column in the input data frame " \
        "containing the groups by which the data points will be " \
        "colored in the output plot. By default, the program " \
        "assumes that no such column is present."

    # Add the argument to the group.
    plot_group.add_argument("-gc", "--groups-column",
                            type = util.process_arg_groups_column,
                            help = gc_help)

    #-----------------------------------------------------------------#

    # Set a help message.
    gr_help = \
        "A comma-separated list of groups whose data points " \
        "should be plotted. By default, all groups found in the " \
        "'-gc' '--groups-column' column, if passed, will be " \
        "included in the plot. Data points not belonging to " \
        "these groups will not be included. However, you can " \
        "use the '-pg', '--plot-other-groups' option to plot " \
        "them using different aesthetics compared to the groups " \
        "of interest."

    # Add the argument to the group.
    plot_group.add_argument("-gr", "--groups",
                            type = util.process_arg_groups,
                            help = gr_help)

    #-----------------------------------------------------------------#

    # Set a help message.
    pg_help = \
        "Whether to plot data points from the groups not " \
        "included in the '-gr', '--groups' list. The  " \
        "aesthetics to plot these data points should also be " \
        "defined in the configuration file for plotting."

    # Add the argument to the group.
    plot_group.add_argument("-pg", "--plot-other-groups",
                            action = "store_true",
                            help = pg_help)

    #-----------------------------------------------------------------#

    # Return the parser.
    return parser


#######################################################################


# Define the 'main' function.
def main(args,
         dim_red_name):

    # Get the argument corresponding to the working directory.
    wd = args.work_dir

    # Get the arguments corresponding to the input files.
    input_data = args.input_data
    input_model = args.input_model
    input_columns = args.input_columns

    # Get the arguments corresponding to the output files.
    output_analysis = os.path.join(wd, args.output_analysis)
    output_model = os.path.join(wd, args.output_model)
    output_plot = os.path.join(wd, args.output_plot)

    # Get the arguments corresponding to the configuration files.
    config_file_dim_red = args.config_file_dim_red
    config_file_plot = args.config_file_plot

    # Get the arguments corresponding to the pre-processing options.
    fill_pos_inf = args.fill_pos_inf
    fill_neg_inf = args.fill_neg_inf

    # Get the arguments corresponding to the plotting options.
    groups_column = args.groups_column
    groups = args.groups
    plot_other_groups = args.plot_other_groups

    #-----------------------------------------------------------------#

    # Set the name of the dimensionality reduction analysis that will
    # be used in the help messages.
    dim_red_help = NAME2HELPNAME[dim_red_name]

    #-----------------------------------------------------------------#

    # Try to load the data points.
    try:

        df_data = pd.read_csv(input_data,
                              sep = ",",
                              header = 0,
                              index_col = 0)

    # If something went wrong
    except Exception as e:

        # Warn the user and exit.
        errstr = \
            "It was not possible to load the data points from " \
            f"'{input_data}'. Error: {e}"
        logger.exception(errstr)
        sys.exit(errstr)

    # Inform the user that the data points were successfully loaded.
    infostr = \
        "The data points were successfully loaded from " \
        f"'{input_data}'."
    logger.info(infostr)

    #-----------------------------------------------------------------#

    # Try to load the configuration.
    try:

        config_dim_red = \
            ioutil.load_config_dim_red(config_file_dim_red)

    # If something went wrong
    except Exception as e:

        # Warn the user and exit.
        errstr = \
            "It was not possible to load the configuration from " \
            f"'{config_file_dim_red}'. Error: {e}"
        logger.exception(errstr)
        sys.exit(errstr)

    # Inform the user that the configuration was successfully loaded.
    infostr = \
        "The configuration was successfully loaded from " \
        f"'{config_file_dim_red}'."
    logger.info(infostr)

    #-----------------------------------------------------------------#

    # Try to load the configuration.
    try:

        config_plot = \
            ioutil.load_config_plot(config_file_plot)

    # If something went wrong
    except Exception as e:

        # Warn the user and exit.
        errstr = \
            "It was not possible to load the configuration from " \
            f"'{config_file_plot}'. Error: {e}"
        logger.exception(errstr)
        sys.exit(errstr)

    # Inform the user that the configuration was successfully loaded.
    infostr = \
        "The configuration was successfully loaded from " \
        f"'{config_file_plot}'."
    logger.info(infostr)

    #-----------------------------------------------------------------#

    # If an input fitted model was passed
    if input_model is not None:

        # Try to load the input model
        try:

            fitted_model = pk.load(open(input_model, "rb"))

        # If something went wrong
        except Exception as e:

            # Warn the user and exit.
            errstr = \
                "It was not possible to load the fitted model from " \
                f"'{input_model}'. Error: {e}"
            logger.exception(errstr)
            sys.exit(errstr)

        # Inform the user that the fitted model was successfully
        # loaded.
        infostr = \
            "The fitted model was successfully loaded from " \
            f"'{input_model}'."
        logger.info(infostr)

    # Otherwise
    else:

        # The fitted model will be None.
        fitted_model = None

    #-----------------------------------------------------------------#

    # If positive infinite values need to be replaced
    if fill_pos_inf is not None:

        # Replace the positive infinite values in the data frame
        # with the given value.
        df_data = df_data.replace(np.inf, fill_pos_inf)

    # Inform the user about the replacement.
    infostr = \
        "All positive infinite values in the input data points " \
        f"were replaced with {fill_pos_inf}."
    logger.info(infostr)

    #-----------------------------------------------------------------#

    # If negative infinite values need to be replaced
    if fill_neg_inf is not None:

        # Replace the negative infinite values in the data frame
        # with the given value.
        df_data = df_data.replace(-np.inf, fill_neg_inf)

    # Inform the user about the replacement.
    infostr = \
        "All negative infinite values in the input data points " \
        f"were replaced with {fill_neg_inf}."
    logger.info(infostr)

    #-----------------------------------------------------------------#

    # Get the function to perform dimensionality reduction analysis.
    dim_red_func = DIMREDNAME2DIMREDFUNC[dim_red_name]

    # Set a string representing the options used for the analysis.
    dim_red_opts_str = \
        ", ".join([f"{opt} = '{val}'" if isinstance(val, str) \
                   else f"{opt} = {val}" \
                   for opt, val in config_dim_red.items()])

    # Log the options that are used for the analysis.
    infostr = \
        f"Performing the {dim_red_help} with the following " \
        f"options: {dim_red_opts_str}."
    logger.info(infostr)

    # Try to perform the dimensionality reduction analysis.
    try:

        df_dim_red, dim_red = \
            dim_red_func(df = df_data,
                         fitted_model = fitted_model,
                         options = config_dim_red,
                         input_columns = input_columns)

    # If something went wrong
    except Exception as e:

        # Warn the user and exit.
        errstr = \
            f"It was not possible to perform the {dim_red_help}. " \
            f"Error: {e}"
        logger.exception(errstr)
        sys.exit(errstr)

    # Inform the user that the analysis was successfully performed.
    infostr = f"The {dim_red_help} was successfully performed."
    logger.info(infostr)

    #-----------------------------------------------------------------#

    # Try to save the results of the dimensionality reduction.
    try:

        df_dim_red.to_csv(output_analysis,
                          sep = ",",
                          index = True,
                          header = True)

    # If something went wrong
    except Exception as e:

        # Warn the user and exit.
        errstr = \
            "It was not possible to write the results of the " \
            f"{dim_red_help} in '{output_analysis}'. Error: {e}"
        logger.exception(errstr)
        sys.exit(errstr)

    # Inform the user that the results were successfully written in the
    # output file.
    infostr = \
        f"The results of the {dim_red_help} were successfully " \
        f"written in '{output_analysis}'."
    logger.info(infostr)

    #-----------------------------------------------------------------#

    # Try to save the fitted model.
    try:

        pk.dump(dim_red, open(output_model, "wb"))

    # If something went wrong
    except Exception as e:

        # Warn the user and exit.
        errstr = \
            "It was not possible to write the fitted model used " \
            f"for the {dim_red_help} in '{output_model}'. " \
            f"Error: {e}"
        logger.exception(errstr)
        sys.exit(errstr)

    # Inform the user that the fitted model was successfully written
    # in the output file.
    infostr = \
        f"The fitted model used for the {dim_red_help} was " \
        f"successfully written in '{output_model}'."
    logger.info(infostr)

    #-----------------------------------------------------------------#

    # Get the format of the output file.
    output_fmt = config_plot.get("output", {}).get("format", "pdf")

    # Try to plot the results and save them.
    try:

        plotting.plot_2d_dim_red(\
            df_dim_red = df_dim_red,
            output_file = f"{output_plot}.{output_fmt}",
            config = config_plot,
            columns = ["C1", "C2"],
            groups_column = groups_column,
            groups = groups,
            plot_other_groups = plot_other_groups)

    # If something went wrong
    except Exception as e:

        # Warn the user and exit.
        errstr = \
            "It was not possible to plot the results of the " \
            f"{dim_red_help} and save them in '{output_plot}'. " \
            f"Error: {e}"
        logger.exception(errstr)
        sys.exit(errstr)

    # Inform the user that the results were successfully plotted and
    # saved.
    infostr = \
        f"The results of the {dim_red_help} were successfully " \
        f"plotted, and the plot was saved in '{output_plot}'."
    logger.info(infostr)
