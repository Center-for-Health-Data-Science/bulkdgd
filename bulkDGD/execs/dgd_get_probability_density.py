#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-

#    dgd_get_probability_density.py
#
#    Given a CSV file containing the representations of one or 
#    multiple samples and the Gaussian mixture model (GMM)
#    modeling the representation space, find the probability 
#    density of each representation for each GMM component.
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
__doc__ = \
    "Given a CSV file containing the representations of one or " \
    "multiple samples and the Gaussian mixture model (GMM) " \
    "modeling the representation space, find the probability " \
    "density of each representation for each GMM component."


#######################################################################


# Import from the standard library.
import argparse
import logging as log
import os
import sys
# Import from third-party packages.
import pandas as pd
import torch
# Import from 'bulkDGD'.
from bulkDGD.core import model
from bulkDGD import defaults, ioutil, util


def main():


    # Create the argument parser.
    parser = argparse.ArgumentParser(description = __doc__)

    #-----------------------------------------------------------------#

    # Add the arguments.
    i_help = \
        "The input CSV file containing the data frame with " \
        "the representations."
    parser.add_argument("-i", "--input-csv",
                        type = str,
                        required = True,
                        help = i_help)

    #-----------------------------------------------------------------#

    or_default = "probability_density_representations.csv"
    or_help = \
        "The name of the output CSV file containing, for each " \
        "representation, its probability density for each of the " \
        "Gaussian mixture model's components, the maximum " \
        "probability density found, and the component the maximum " \
        "probability density comes from. The file will be " \
        "written in the working directory. The default file " \
        f"name is '{or_default}'."
    parser.add_argument("-or", "--output-csv-prob-rep",
                        type = str,
                        default = or_default,
                        help = or_help)

    #-----------------------------------------------------------------#

    oc_default = "probability_density_components.csv"
    oc_help = \
        "The name of the output CSV file containing, for each " \
        "component of the Gaussian mixture model, the " \
        "representation(s) having the maximum probability " \
        "density with respect to it. The file will be written " \
        "in the working directory. The default file name is " \
        f"'{oc_default}'."
    parser.add_argument("-oc", "--output-csv-prob-comp",
                        type = str,
                        default = oc_default,
                        help = oc_help)

    #-----------------------------------------------------------------#

    cm_help = \
        "The YAML configuration file specifying the " \
        "DGD model's parameters and files containing " \
        "the trained model. If it is a name without an " \
        "extension, it is assumed to be the name of a " \
        f"configuration file in '{ioutil.CONFIG_MODEL_DIR}'."
    parser.add_argument("-cm", "--config-file-model",
                        type = str,
                        required = True,
                        help = cm_help)

    #-----------------------------------------------------------------#

    d_help = \
        "The working directory. The default is the current " \
        "working directory."
    parser.add_argument("-d", "--work-dir",
                        type = str,
                        default = os.getcwd(),
                        help = d_help)

    #-----------------------------------------------------------------#

    lf_default = "dgd_get_probability_density.log"
    lf_help = \
        "The name of the log file. The file will be written " \
        "in the working directory. The default file name is " \
        f"'{lf_default}'."
    parser.add_argument("-lf", "--log-file",
                        type = str,
                        default = lf_default,
                        help = lf_help)

    #-----------------------------------------------------------------#

    lc_help = "Show log messages also on the console."
    parser.add_argument("-lc", "--log-console",
                        action = "store_true",
                        help = lc_help)

    #-----------------------------------------------------------------#

    v_help = "Enable verbose logging (INFO level)."
    parser.add_argument("-v", "--log-verbose",
                        action = "store_true",
                        help = v_help)

    #-----------------------------------------------------------------#

    vv_help = \
        "Enable maximally verbose logging for debugging " \
        "purposes (DEBUG level)."
    parser.add_argument("-vv", "--log-debug",
                        action = "store_true",
                        help = vv_help)

    #-----------------------------------------------------------------#

    # Parse the arguments.
    args = parser.parse_args()
    input_csv = args.input_csv
    output_csv_prob_rep = args.output_csv_prob_rep
    output_csv_prob_comp = args.output_csv_prob_comp
    config_file_model = args.config_file_model
    wd = os.path.abspath(args.work_dir)
    log_file = os.path.join(wd, args.log_file)
    log_console = args.log_console
    v = args.log_verbose
    vv = args.log_debug

    #-----------------------------------------------------------------#

    # Set WARNING logging level by default.
    log_level = log.WARNING

    # If the user requested verbose logging
    if v:

        # The minimal logging level will be INFO.
        log_level = log.INFO

    # If the user requested logging for debug purposes
    # (-vv overrides -v if both are provided)
    if vv:

        # The minimal logging level will be DEBUG.
        log_level = log.DEBUG

    # Configure the logging.
    handlers = \
        util.get_handlers(\
            log_console = log_console,
            log_file_class = log.FileHandler,
            log_file_options = {"filename" : log_file,
                                "mode" : "w"},
            log_level = log_level)

    # Set the logging configuration.
    log.basicConfig(level = log_level,
                    format = defaults.LOG_FMT,
                    datefmt = defaults.LOG_DATEFMT,
                    style = defaults.LOG_STYLE,
                    handlers = handlers)

    #-----------------------------------------------------------------#
    
    # Try to load the configuration.
    try:

        config_model = ioutil.load_config_model(config_file_model)

    # If something went wrong
    except Exception as e:

        # Warn the user and exit.
        errstr = \
            "It was not possible to load the configuration from " \
            f"'{config_file_model}'. Error: {e}"
        log.exception(errstr)
        sys.exit(errstr)

    # Inform the user that the configuration was successfully loaded.
    infostr = \
        "The configuration was successfully loaded from " \
        f"'{config_file_model}'."
    log.info(infostr)

    #-----------------------------------------------------------------#

    # Try to load the data.
    try:

        df_rep  = \
            ioutil.load_representations(csv_file = input_csv,
                                        sep = ",",
                                        split = False)

    # If something went wrong
    except Exception as e:

        # Warn the user and exit.
        errstr = \
            "It was not possible to load the representations from " \
            f"'{input_csv}'. Error: {e}"
        log.exception(errstr)
        sys.exit(errstr)

    # Inform the user that the data were successfully loaded.
    infostr = \
        "The representations were successfully loaded " \
        f"from '{input_csv}'."
    log.info(infostr)

    #-----------------------------------------------------------------#

    # Try to get the GMM.
    try:
        
        dgd_model = model.DGDModel(**config_model)

    # If something went wrong
    except Exception as e:

        # Warn the user and exit.
        errstr = \
            f"It was not possible to set the DGD model. Error: {e}"
        log.exception(errstr)
        sys.exit(errstr)

    # Inform the user that the model was successfully set.
    infostr = "The DGD model was successfully set."
    log.info(infostr)

    #-----------------------------------------------------------------#

    # Try to calculate the probability densities.
    try:
        
        df_prob_rep, df_prob_comp = \
            dgd_model.get_probability_density(df_rep = df_rep)

    # If something went wrong
    except Exception as e:

        # Warn the user and exit.
        errstr = \
            "It was not possible to calculate the probability " \
            f"densities for the representations. Error: {e}"
        log.exception(errstr)
        sys.exit(errstr)

    # Inform the user that the probability densities were successfully
    # calculated.
    infostr = \
        "The probability densities for the representations were " \
        "successfully calculated."
    log.info(infostr)

    #-----------------------------------------------------------------#

    # Set the path to the output file.
    output_csv_prob_rep_path = os.path.join(wd, output_csv_prob_rep)

    # Try to write the probability densities for the representations
    # to the output CSV file.
    try:

        df_prob_rep.to_csv(output_csv_prob_rep_path,
                           sep = ",",
                           header = True,
                           index = True)

    # If something went wrong
    except Exception as e:

        # Warn the user and exit.
        errstr = \
            "It was not possible to write the probability " \
            "densities for each representation in " \
            f"'{output_csv_prob_rep_path}'. Error: {e}"
        log.exception(errstr)
        sys.exit(errstr)

    # Inform the user that the probability densities for the
    # representations were successfully written in the output file.
    infostr = \
        "The probability densities for each representation " \
        "were successfully written in " \
        f"'{output_csv_prob_rep_path}'."
    log.info(infostr)

    #-----------------------------------------------------------------#

    # Set the path to the output.
    output_csv_prob_comp_path = os.path.join(wd, output_csv_prob_comp)

    # Try to write the probability densities for the representations
    # having the highest probability density for each component
    # to the output CSV file.
    try:

        df_prob_comp.to_csv(output_csv_prob_comp_path,
                            sep = ",",
                            header = True,
                            index = True)

    # If something went wrong
    except Exception as e:

        # Warn the user and exit.
        errstr = \
            "It was not possible to write the probability " \
            "densities for the representations having the " \
            "highest probability density for each component " \
            f"in '{output_csv_prob_comp_path}'. Error: {e}"
        log.exception(errstr)
        sys.exit(errstr)

    # Inform the user that the probability densities for the
    # representations were successfully written in the output file.
    infostr = \
        "The probability densities for the representations having " \
        "the highest probability density for each component " \
        "were successfully written in " \
        f"'{output_csv_prob_comp_path}'."
    log.info(infostr)
