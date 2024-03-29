#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-

#    dgd_get_representations.py
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
    "Find representations in the latent space defined by the " \
    "DGD model for a set of samples."


# Standard library
import argparse
import logging as log
import os
import sys
# bulkDGD
from bulkDGD.core import model
from bulkDGD import defaults, ioutil, util


def main():


    # Create the argument parser
    parser = argparse.ArgumentParser(description = __doc__)

    #-----------------------------------------------------------------#

    # Add the arguments
    i_help = \
        "The input CSV file containing a data frame with " \
        "the gene expression data for the samples for which a " \
        "representation in latent space should be found."
    parser.add_argument("-i", "--input-csv",
                        type = str,
                        required = True,
                        help = i_help)

    #-----------------------------------------------------------------#

    or_default = "representations.csv"
    or_help = \
        "The name of the output CSV file containing the data frame " \
        "with the representation of each input sample in latent " \
        "space. The file will be written in the working directory. " \
        f"The default file name is '{or_default}'."
    parser.add_argument("-or", "--output-csv-rep",
                        type = str,
                        default = or_default,
                        help = or_help)

    #-----------------------------------------------------------------#

    od_default = "decoder_outputs.csv"
    od_help = \
        "The name of the output CSV file containing the data frame " \
        "with the decoder output for each input sample. " \
        "The file will be written in the working directory. " \
        f"The default file name is '{od_default}'."
    parser.add_argument("-od", "--output-csv-dec",
                        type = str,
                        default = od_default,
                        help = od_help)

    #-----------------------------------------------------------------#

    ot_default = "opt_time.csv"
    ot_help = \
        "The name of the output CSV file containing the data frame " \
        "with information about the CPU and wall clock time " \
        "spent for each optimization epoch and each backpropagation " \
        "step through the decoder. The file will be written in the " \
        f"working directory. The default file name is '{ot_default}'."
    parser.add_argument("-ot", "--output-csv-time",
                        type = str,
                        default = ot_default,
                        help = ot_help)

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

    cr_help = \
        "The YAML configuration file containing the " \
        "options for the optimization step(s) when " \
        "finding the best representations. If it is a name " \
        "without an extension, it is assumed to be the name of " \
        f"a configuration file in '{ioutil.CONFIG_REP_DIR}'."
    parser.add_argument("-cr", "--config-file-rep",
                        type = str,
                        required = True,
                        help = cr_help)

    #-----------------------------------------------------------------#

    m_choices = ["one_opt", "two_opt"]
    m_help = \
        "The method for optimizing the representations. " \
        "The file specified with the '-cr', '--config-file-rep' " \
        "option must contain options compatible with the " \
        "chosen method."
    parser.add_argument("-m", "--method-optimization",
                        type = str,
                        required = True,
                        choices = m_choices,
                        help = m_help)

    #-----------------------------------------------------------------#

    d_help = \
        "The working directory. The default is the current " \
        "working directory."
    parser.add_argument("-d", "--work-dir",
                        type = str,
                        default = os.getcwd(),
                        help = d_help)

    #-----------------------------------------------------------------#

    lf_default = "dgd_get_representations.log"
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

    # Parse the arguments
    args = parser.parse_args()
    input_csv = args.input_csv
    output_csv_rep = args.output_csv_rep
    output_csv_dec = args.output_csv_dec
    output_csv_time = args.output_csv_time
    config_file_model = args.config_file_model
    config_file_rep = args.config_file_rep
    method_optimization = args.method_optimization
    wd = args.work_dir
    log_file = args.log_file
    log_console = args.log_console
    v = args.log_verbose
    vv = args.log_debug

    #-----------------------------------------------------------------#

    # Set WARNING logging level by default
    log_level = log.WARNING

    # If the user requested verbose logging
    if v:

        # The minimal logging level will be INFO
        log_level = log.INFO

    # If the user requested logging for debug purposes
    # (-vv overrides -v if both are provided)
    if vv:

        # The minimal logging level will be DEBUG
        log_level = log.DEBUG

    # Configure the logging (for non-Dask operations)
    handlers = \
        util.get_handlers(\
            log_console = log_console,
            log_file_class = log.FileHandler,
            log_file_options = {"filename" : log_file,
                                "mode" : "w"},
            log_level = log_level)

    # Set the logging configuration
    log.basicConfig(# The level below which log messages are silenced
                    level = log_level,
                    # The format of the log strings
                    format = defaults.LOG_FMT,
                    # The format for dates/time
                    datefmt = defaults.LOG_DATEFMT,
                    # The format style
                    style = defaults.LOG_STYLE,
                    # The handlers
                    handlers = handlers)

    #-----------------------------------------------------------------#

    # Try to load the configuration
    try:

        config_model = ioutil.load_config_model(config_file_model)

    # If something went wrong
    except Exception as e:

        # Warn the user and exit
        errstr = \
            "It was not possible to load the configuration from " \
            f"'{config_file_model}'. Error: {e}"
        log.exception(errstr)
        sys.exit(errstr)

    # Inform the user that the configuration was successfully loaded
    infostr = \
        "The configuration was successfully loaded from " \
        f"'{config_file_model}'."
    log.info(infostr)

    #-----------------------------------------------------------------#

    # Try to load the configuration
    try:

        config_rep = ioutil.load_config_rep(config_file_rep)

    # If something went wrong
    except Exception as e:

        # Warn the user and exit
        errstr = \
            "It was not possible to load the configuration from " \
            f"'{config_file_rep}'. Error: {e}"
        log.exception(errstr)
        sys.exit(errstr)

    # Inform the user that the configuration was successfully loaded
    infostr = \
        "The configuration was successfully loaded from " \
        f"'{config_file_rep}'."
    log.info(infostr)

    #-----------------------------------------------------------------#

    # Try to load the samples' data
    try:

        df_samples = \
            ioutil.load_samples(csv_file = input_csv,
                                sep = ",",
                                keep_samples_names = True,
                                split = False)

    # If something went wrong
    except Exception as e:

        # Warn the user and exit
        errstr = \
            "It was not possible to load the samples from " \
            f"'{input_csv}'. Error: {e}"
        log.exception(errstr)
        sys.exit(errstr)

    # Inform the user that the data were successfully loaded
    infostr = \
        f"The samples were successfully loaded from '{input_csv}'."
    log.info(infostr)

    #-----------------------------------------------------------------#

    # Try to set the model
    try:
        
        dgd_model = model.DGDModel(**config_model)

    # If something went wrong
    except Exception as e:

        # Warn the user and exit
        errstr = \
            f"It was not possible to set the DGD model. Error: {e}"
        log.exception(errstr)
        sys.exit(errstr)

    # Inform the user that the model was successfully set
    infostr = "The DGD model was successfully set."
    log.info(infostr)

    #-----------------------------------------------------------------#

    # Try to get the representations
    try:
        
        df_rep, df_dec_out, df_time = \
            dgd_model.get_representations(\
                # The data frame with the samples
                df_samples = df_samples,
                # The method to use to get the representation
                method = method_optimization,
                # The configuration for the optimization                         
                config_opt = config_rep["optimization"],
                # The number of new representations per component
                # per sample                         
                n_rep_per_comp = config_rep["n_rep_per_comp"])

    # If something went wrong
    except Exception as e:

        # Warn the user and exit
        errstr = \
            f"It was not possible to find the representations. " \
            f"Error: {e}"
        log.exception(errstr)
        sys.exit(errstr)

    # Inform the user that the representations were successfully
    # optimized
    infostr = "The representations were successfully found."
    log.info(infostr)

    #-----------------------------------------------------------------#

    # Set the path to the output file
    output_csv_rep_path = os.path.join(wd, output_csv_rep)

    # Try to write the representations in the output CSV file
    try:

        ioutil.save_representations(\
            df = df_rep,
            csv_file = output_csv_rep_path,
            sep = ",")

    # If something went wrong
    except Exception as e:

        # Warn the user and exit
        errstr = \
            "It was not possible to write the representations " \
            f"in '{output_csv_rep_path}'. Error: {e}"
        log.exception(errstr)
        sys.exit(errstr)

    # Inform the user that the representations were successfully
    # written in the output file
    infostr = \
        "The representations were successfully written in " \
        f"'{output_csv_rep_path}'."
    log.info(infostr)

    #-----------------------------------------------------------------#

    # Set the path to the output file
    output_csv_dec_path = os.path.join(wd, output_csv_dec)

    # Try to write the decoder outputs in the dedicated CSV file
    try:

        ioutil.save_decoder_outputs(\
            df = df_dec_out,
            csv_file = output_csv_dec_path,
            sep = ",")

    # If something went wrong
    except Exception as e:

        # Warn the user and exit
        errstr = \
            "It was not possible to write the decoder outputs " \
            f"in '{output_csv_dec_path}'. Error: {e}"
        log.exception(errstr)
        sys.exit(errstr)

    # Inform the user that the decoder outputs were successfully
    # written in the output file
    infostr = \
        "The decoder outputs were successfully written in " \
        f"'{output_csv_dec_path}'."
    log.info(infostr)

    #-----------------------------------------------------------------#

    # Set the path to the output file
    output_csv_time_path = os.path.join(wd, output_csv_time)

    # Try to write the time data in the dedicated CSV file
    try:

        ioutil.save_time(\
            df = df_time,
            csv_file = output_csv_time_path,
            sep = ",")

    # If something went wrong
    except Exception as e:

        # Warn the user and exit
        errstr = \
            "It was not possible to write the time data " \
            f"in '{output_csv_time_path}'. Error: {e}"
        log.exception(errstr)
        sys.exit(errstr)

    # Inform the user that the time data was successfully
    # written in the output file
    infostr = \
        "The time data were successfully written in " \
        f"'{output_csv_time_path}'."
    log.info(infostr)
