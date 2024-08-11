#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-

#    dgd_get_representations.py
#
#    Find representations in the latent space defined by the DGD
#    model for a set of samples.
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
    "Find representations in the latent space defined by the " \
    "DGD model for a set of samples."


#######################################################################


# Import from the standard library.
import argparse
import logging as log
import os
import sys
# Import from 'bulkDGD'.
from bulkDGD.core import model
from bulkDGD import defaults, ioutil, util


#######################################################################


def main():


    # Create the argument parser.
    parser = argparse.ArgumentParser(description = __doc__)

    #-----------------------------------------------------------------#

    # Add the arguments.
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

    om_default = "pred_means.csv"
    om_help = \
        "The name of the output CSV file containing the data frame " \
        "with the predicted scaled means of the negative  " \
        "binomials for the in silico samples obtained from the best " \
        "representations found. The file will be written in the " \
        f"working directory. The default file name is '{om_default}'."
    parser.add_argument("-om", "--output-csv-means",
                        type = str,
                        default = om_default,
                        help = om_help)

    #-----------------------------------------------------------------#

    ov_default = "pred_r_values.csv"
    ov_help = \
        "The name of the output CSV file containing the data frame " \
        "with the predicted r-values of the negative binomials for " \
        "the in silico samples obtained from the best " \
        "representations found. The file will be written in the " \
        "working directory. The default file name is " \
        f"'{ov_default}'. The file is produced only if negative' " \
        "binomial distributions are used to model the genes' counts."
    parser.add_argument("-ov", "--output-csv-rvalues",
                        type = str,
                        default = ov_default,
                        help = ov_help)

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
        f"configuration file in '{defaults.CONFIG_MODEL_DIR}'."
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
        f"a configuration file in '{defaults.CONFIG_REP_DIR}'."
    parser.add_argument("-cr", "--config-file-rep",
                        type = str,
                        required = True,
                        help = cr_help)

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

    # Parse the arguments.
    args = parser.parse_args()
    input_csv = args.input_csv
    output_csv_rep = args.output_csv_rep
    output_csv_means = args.output_csv_means
    output_csv_rvalues = args.output_csv_rvalues
    output_csv_time = args.output_csv_time
    config_file_model = args.config_file_model
    config_file_rep = args.config_file_rep
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
            log_console_level = log_level,
            log_file_class = log.FileHandler,
            log_file_options = {"filename" : log_file,
                                "mode" : "w"},
            log_file_level = log_level)

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

    # Try to load the configuration.
    try:

        config_rep = ioutil.load_config_rep(config_file_rep)

    # If something went wrong
    except Exception as e:

        # Warn the user and exit.
        errstr = \
            "It was not possible to load the configuration from " \
            f"'{config_file_rep}'. Error: {e}"
        log.exception(errstr)
        sys.exit(errstr)

    # Inform the user that the configuration was successfully loaded.
    infostr = \
        "The configuration was successfully loaded from " \
        f"'{config_file_rep}'."
    log.info(infostr)

    #-----------------------------------------------------------------#

    # Try to load the samples' data.
    try:

        df_samples = \
            ioutil.load_samples(csv_file = input_csv,
                                sep = ",",
                                keep_samples_names = True,
                                split = False)

    # If something went wrong
    except Exception as e:

        # Warn the user and exit.
        errstr = \
            "It was not possible to load the samples from " \
            f"'{input_csv}'. Error: {e}"
        log.exception(errstr)
        sys.exit(errstr)

    # Inform the user that the data were successfully loaded.
    infostr = \
        f"The samples were successfully loaded from '{input_csv}'."
    log.info(infostr)

    #-----------------------------------------------------------------#

    # Try to set the model.
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

    # Try to get the representations.
    try:
        
        df_rep, df_pred_means, df_pred_r_values, df_time = \
            dgd_model.get_representations(\
                # The data frame with the samples
                df_samples = df_samples,
                # The configuration to find the representations                        
                config_rep = config_rep)

    # If something went wrong
    except Exception as e:

        # Warn the user and exit.
        errstr = \
            f"It was not possible to find the representations. " \
            f"Error: {e}"
        log.exception(errstr)
        sys.exit(errstr)

    # Inform the user that the representations were successfully
    # optimized.
    infostr = "The representations were successfully found."
    log.info(infostr)

    #-----------------------------------------------------------------#

    # Set the path to the output file.
    output_csv_rep_path = os.path.join(wd, output_csv_rep)

    # Try to write the representations in the output CSV file.
    try:

        ioutil.save_representations(\
            df = df_rep,
            csv_file = output_csv_rep_path,
            sep = ",")

    # If something went wrong
    except Exception as e:

        # Warn the user and exit.
        errstr = \
            "It was not possible to write the representations " \
            f"in '{output_csv_rep_path}'. Error: {e}"
        log.exception(errstr)
        sys.exit(errstr)

    # Inform the user that the representations were successfully
    # written in the output file.
    infostr = \
        "The representations were successfully written in " \
        f"'{output_csv_rep_path}'."
    log.info(infostr)

    #-----------------------------------------------------------------#

    # Set the path to the output file.
    output_csv_means_path = os.path.join(wd, output_csv_means)

    # Try to write the predicted means in the dedicated CSV file.
    try:

        ioutil.save_decoder_outputs(\
            df = df_pred_means,
            csv_file = output_csv_means_path,
            sep = ",")

    # If something went wrong
    except Exception as e:

        # Warn the user and exit.
        errstr = \
            "It was not possible to write the predicted means " \
            f"in '{output_csv_means_path}'. Error: {e}"
        log.exception(errstr)
        sys.exit(errstr)

    # Inform the user that the predicted means were successfully
    # written in the output file.
    infostr = \
        "The predicted means were successfully written in " \
        f"'{output_csv_means_path}'."
    log.info(infostr)

    #-----------------------------------------------------------------#

    # If the r-values were returned
    if df_pred_r_values is not None:

        # Set the path to the output file.
        output_csv_rvalues_path = os.path.join(wd, output_csv_rvalues)

        # Try to write the predicted r-values in the dedicated CSV
        # file.
        try:

            ioutil.save_decoder_outputs(\
                df = df_pred_r_values,
                csv_file = output_csv_rvalues_path,
                sep = ",")

        # If something went wrong
        except Exception as e:

            # Warn the user and exit.
            errstr = \
                "It was not possible to write the predicted " \
                f"r-values in '{output_csv_rvalues_path}'. Error: {e}"
            log.exception(errstr)
            sys.exit(errstr)

        # Inform the user that the predicted r-values were successfully
        # written in the output file.
        infostr = \
            "The predicted r-values were successfully written in " \
            f"'{output_csv_rvalues_path}'."
        log.info(infostr)

    #-----------------------------------------------------------------#

    # Set the path to the output file.
    output_csv_time_path = os.path.join(wd, output_csv_time)

    # Try to write the time data in the dedicated CSV file.
    try:

        ioutil.save_time(\
            df = df_time,
            csv_file = output_csv_time_path,
            sep = ",")

    # If something went wrong
    except Exception as e:

        # Warn the user and exit.
        errstr = \
            "It was not possible to write the time data " \
            f"in '{output_csv_time_path}'. Error: {e}"
        log.exception(errstr)
        sys.exit(errstr)

    # Inform the user that the time data was successfully
    # written in the output file.
    infostr = \
        "The time data were successfully written in " \
        f"'{output_csv_time_path}'."
    log.info(infostr)


#######################################################################


if __name__ == "__main__":
    main()
