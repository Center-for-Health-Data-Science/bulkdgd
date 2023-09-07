#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-

#    dgd_perform_pca.py
#
#    Do a two-dimensional principal component analysis on a set of
#    input representations.
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
__doc__ = \
    "Do a two-dimensional principal component analysis on a set of " \
    "input representations."


# Standard library
import argparse
import logging as log
import os
import sys
# Third-party packages
import pandas as pd
# bulkDGD
from bulkDGD import defaults, ioutil, plotting
from bulkDGD.analysis import reduction


def main():


    #-------------------- Command-line arguments ---------------------#


    # Create the argument parser
    parser = argparse.ArgumentParser(description = __doc__)

    i_help = \
        "The input CSV file containing the data frame " \
        "with the representations."
    parser.add_argument("-i", "--input-csv",
                        type = str,
                        required = True,
                        help = i_help)

    oc_default = "pca.csv"
    oc_help = \
        "The name of the output CSV file containing the results of " \
        "the PCA. The file will be written in the working " \
        f"directory. The default file name is '{oc_default}'."
    parser.add_argument("-oc", "--output-csv-pca",
                        type = str,
                        default = oc_default,
                        help = oc_help)

    op_default = "pca.pdf"
    op_help = \
        "The name of the output file containing the plot displaying " \
        "the results of the PCA. This file will be written in the " \
        f"working directory. The default file name is '{op_default}'."
    parser.add_argument("-op", "--output-plot-pca",
                        type = str,
                        default = op_default,
                        help = op_help)

    cp_help = \
        "The YAML configuration file specifying the aesthetics " \
        "of the plot and the plot's output format. If not " \
        "provided, the default configuration " \
        f"file ('{defaults.CONFIG_PLOT_PCA}') will be used."
    parser.add_argument("-cp", "--config-file-plot",
                        type = str,
                        default = defaults.CONFIG_PLOT_PCA,
                        help = cp_help)

    gc_help = \
        "The name/index of the column in the input data frame " \
        "containing the groups by which the samples will be " \
        "colored in the output plot. " \
        "By default, the program assumes that no such column is " \
        "present."
    parser.add_argument("-gc", "--groups-column",
                        type = str,
                        default = None,
                        help = gc_help)

    d_help = \
        "The working directory. The default is the current " \
        "working directory."
    parser.add_argument("-d", "--work-dir",
                        type = str,
                        default = os.getcwd(),
                        help = d_help)

    lf_default = "dgd_perform_pca.log"
    lf_help = \
        "The name of the log file. The file will be written " \
        "in the working directory. The default file name is " \
        f"'{lf_default}'."
    parser.add_argument("-lf", "--log-file",
                        type = str,
                        default = lf_default,
                        help = lf_help)

    lc_help = "Show log messages also on the console."
    parser.add_argument("-lc", "--log-console",
                        action = "store_true",
                        help = lc_help)

    v_help = "Enable verbose logging (INFO level)."
    parser.add_argument("-v", "--log-verbose",
                        action = "store_true",
                        help = v_help)

    vv_help = \
        "Enable maximally verbose logging for debugging " \
        "purposes (DEBUG level)."
    parser.add_argument("-vv", "--log-debug",
                        action = "store_true",
                        help = vv_help)

    # Parse the arguments
    args = parser.parse_args()
    input_csv = args.input_csv
    output_csv_pca = args.output_csv_pca
    output_plot_pca = args.output_plot_pca
    config_file_plot = args.config_file_plot
    groups_column = \
        args.groups_column \
        if args.groups_column is None \
        or not args.groups_column.isdigit() \
        else int(args.groups_column)
    wd = args.work_dir
    log_file = args.log_file
    log_console = args.log_console
    v = args.log_verbose
    vv = args.log_debug


    #---------------------------- Logging ----------------------------#


    # Get the module's logger
    logger = log.getLogger(__name__)

    # Set WARNING logging level by default
    level = log.WARNING

    # If the user requested verbose logging
    if v:

        # The minimal logging level will be INFO
        level = log.INFO

    # If the user requested logging for debug purposes
    # (-vv overrides -v if both are provided)
    if vv:

        # The minimal logging level will be DEBUG
        level = log.DEBUG

    # Initialize the logging handlers to a list containing only
    # the FileHandler (to log to the log file)
    handlers = [log.FileHandler(# The log file
                                filename = log_file,
                                # How to open the log file ('w' means
                                # re-create it every time the
                                # executable is called)
                                mode = "w")]

    # If the user requested logging to the console, too
    if log_console:

        # Append a StreamHandler to the list
        handlers.append(log.StreamHandler())

    # Set the logging level
    log.basicConfig(# The level below which log messages are silenced
                    level = level,
                    # The format of the log strings
                    format = "{asctime}:{levelname}:{name}:{message}",
                    # The format for dates/time
                    datefmt="%Y-%m-%d,%H:%M",
                    # The format style
                    style = "{",
                    # The handlers
                    handlers = handlers)


    #--------------------- Configuration - plot ----------------------#


    # Try to load the configuration
    try:

        config_plot = ioutil.load_config_plot(config_file_plot)

    # If something went wrong
    except Exception as e:

        # Warn the user and exit
        errstr = \
            "It was not possible to load the configuration from " \
            f"'{config_file_plot}'. Error: {e}"
        logger.exception(errstr)
        sys.exit(errstr)

    # Inform the user that the configuration was successfully loaded
    infostr = \
        "The configuration was successfully loaded from " \
        f"'{config_file_plot}'."
    logger.info(infostr)


    #------------------- Load the representations --------------------#


    # Try to load the representations
    try:

        df_rep_data, df_other_data = \
            ioutil.load_representations(csv_file = input_csv,
                                        sep = ",",
                                        split = True)

    # If something went wrong
    except Exception as e:

        # Warn the user and exit
        errstr = \
            "It was not possible to load the representations from " \
            f"'{input_csv}'. Error: {e}"
        logger.exception(errstr)
        sys.exit(errstr)

    # Inform the user that the representations were successfully
    # loaded
    infostr = \
        "The representations were successfully loaded from " \
        f"'{input_csv}'."
    logger.info(infostr)


    #-------------------------- Do the PCA ---------------------------#


    # Try to perform the pca
    try:

        df_pca = \
            reduction.perform_2d_pca(df_rep = df_rep_data,
                                     pc_columns = ["PC1", "PC2"])

    # If something went wrong
    except Exception as e:

        # Warn the user and exit
        errstr = \
            f"It was not possible to perform the PCA. Error: {e}"
        logger.exception(errstr)
        sys.exit(errstr)

    # Inform the user that the PCA was successfully performed
    infostr = "The PCA was performed successfully."
    logger.info(infostr)


    #--------------------- Save the PCA results ----------------------#


    # Set the path to the output file
    output_csv_pca_path = os.path.join(wd, output_csv_pca)

    # Try to save the results of the PCA
    try:

        df_pca.to_csv(output_csv_pca_path,
                      sep = ",",
                      index = True,
                      header = True)

    # If something went wrong
    except Exception as e:

        # Warn the user and exit
        errstr = \
            "It was not possible to save the results of the PCA " \
            f"in '{output_csv_pca_path}'. Error: {e}"
        logger.exception(errstr)
        sys.exit(errstr)

    # Inform the user that the results of the PCA were successfully
    # written to the output file
    infostr = \
        "The results of the PCA were successfully written in " \
        f"'{output_csv_pca_path}'."
    logger.info(infostr)


    #--------------------- Plot the PCA results ----------------------#


    # Merge the data frame containing the PCA results with the
    # one containing the extra information about the
    # representations
    df_pca = df_pca.join(df_other_data)

    # Set the path to the output file
    output_plot_pca_path = os.path.join(wd, output_plot_pca)
    
    # Try to plot the results of the PCA and save them
    try:

        # Plot the PCA results
        plotting.plot_2d_pca(df_pca = df_pca,
                             output_file = output_plot_pca_path,
                             config = config_plot,
                             pc_columns = ["PC1", "PC2"],
                             groups_column = groups_column)

    # If something went wrong
    except Exception as e:

        # Warn the user and exit
        errstr = \
            "It was not possible to plot the results of the " \
            f"PCA and save them in '{output_plot_pca_path}'. " \
            f"Error: {e}"
        logger.exception(errstr)
        sys.exit(errstr)

    # Inform the user that the results of the PCA were successfully
    # plotted and saved
    infostr = \
        "The results of the PCA were successfully plotted and " \
        f"the plot saved to '{output_plot_pca_path}'."
    logger.info(infostr)