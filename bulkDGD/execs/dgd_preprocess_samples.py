#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-

#    dgd_preprocess_samples.py
#
#    Preprocess new samples to use them with the DGD model.
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
    "Preprocess new samples to use them with the DGD model."

# Standard library
import argparse
import logging as log
import os
import sys
# Third-party packages
import pandas as pd
# bulkDGD
from bulkDGD.utils import dgd

def main():


    #-------------------- Command-line arguments ---------------------#


    # Create the argument parser
    parser = argparse.ArgumentParser(description = __doc__)

    # Add the arguments
    i_help = \
        "The input CSV file containing a data frame with the " \
        "samples to be preprocessed."
    parser.add_argument("-i", "--input-csv",
                        type = str,
                        required = True,
                        help = i_help)

    os_default = "samples_preprocessed.csv"
    os_help = \
        "The name of the output CSV file containing the data frame " \
        "with the preprocessed samples. The file will be written " \
        "in the working directory. The default file name is " \
        f"'{os_default}'."
    parser.add_argument("-os", "--output-csv-samples",
                        type = str,
                        default = os_default,
                        help = os_help)

    oge_default = "genes_excluded.csv"
    oge_help = \
        "The name of the output plain text file containing the " \
        "list of genes whose expression data are excluded from the " \
        "data frame with the preprocessed samples. The " \
        "file will be written in the working directory. " \
        f"The default file name is '{oge_default}'."
    parser.add_argument("-oe", "--output-txt-genes-excluded",
                        type = str,
                        default = oge_default,
                        help = oge_help)

    ogm_default = "genes_missing.csv"
    ogm_help = \
        "The name of the output plain text file containing the " \
        "list of genes for which no available expression data " \
        "are found in the input data frame. A default count of " \
        "0 is assigned to these genes in the output data frame " \
        "containing the preprocessed samples. The file will " \
        "be written in the working directory. The " \
        f"default file name is '{ogm_default}'."
    parser.add_argument("-om", "--output-txt-genes-missing",
                        type = str,
                        default = ogm_default,
                        help = ogm_help)

    d_help = \
        "The working directory. The default is the current " \
        "working directory."
    parser.add_argument("-d", "--work-dir",
                        type = str,
                        default = os.getcwd(),
                        help = d_help)

    lf_default = "dgd_preprocess_samples.log"
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
    output_csv_samples = args.output_csv_samples
    output_txt_genes_excluded = args.output_txt_genes_excluded
    output_txt_genes_missing = args.output_txt_genes_missing
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


    #------------------------- Data loading --------------------------#


    # Try to load the input samples
    try:

        df_expr_data, df_other_data = \
            dgd.load_samples(csv_file = input_csv,
                             sep = ",",
                             keep_samples_names = True)

        df_samples = pd.concat([df_expr_data, df_other_data],
                               axis = 1)
        print(df_samples)
    # If something went wrong
    except Exception as e:

        # Warn the user and exit
        errstr = \
            "It was not possible to load the samples from " \
            f"'{input_csv}'. Error: {e}"
        logger.exception(errstr)
        sys.exit(errstr)


    #-------------------- Samples' preprocessing ---------------------#


    # Try to preprocess the samples
    try:

        preproc_df, genes_excluded, genes_missing = \
            dgd.preprocess_samples(df_samples = df_samples)

    # If something went wrong
    except Exception as e:

        # Warn the user and exit
        errstr = \
            "It was not possible to preprocess the samples. " \
            f"Error: {e}"
        logger.exception(errstr)
        sys.exit(errstr)


    #----------------- Output - preprocessed samples -----------------#


    # Set the path to the output file
    output_csv_samples_path = os.path.join(wd, output_csv_samples)

    # Try to write out the preprocessed samples
    try:

        dgd.save_samples(df = preproc_df,
                         csv_file = output_csv_samples_path,
                         sep = ",")

    # If something went wrong
    except Exception as e:

        # Warn the user and exit
        errstr = \
            "It was not possible to write the preprocessed " \
            f"samples to '{output_csv_samples_path}'. Error: {e}"
        logger.exception(errstr)
        sys.exit(errstr)

    # Inform the user that the preprocessed samples were
    # successfilly written in the output file
    infostr = \
        "The preprocessed samples were successfully written in " \
        f"'{output_csv_samples_path}'."
    logger.info(infostr)


    #-------------------- Output - excluded genes --------------------#


    # If some genes were excluded
    if genes_excluded:

        # Set the path to the output file
        output_txt_genes_excluded_path = \
            os.path.join(wd, output_txt_genes_excluded)

        # Try to write the list of excluded genes
        try:

            with open(output_txt_genes_excluded_path, "w") as out:
                out.write("\n".join(gene for gene in genes_excluded))

        # If something went wrong
        except Exception as e:

            # Warn the user and exit
            errstr = \
                "It was not possible to write the list of genes " \
                "that are present in the input samples but are " \
                "not among the genes used to train the DGD model " \
                f"to '{output_txt_genes_excluded_path}'. Error: {e}"
            logger.exception(errstr)
            sys.exit(errstr)

        # Inform the user that the genes were successfully written
        # to the output file
        infostr = \
            "The list of genes that are present in the input " \
            "samples but are not among the genes used to train " \
            "the DGD model was successfully written in " \
            f"'{output_txt_genes_excluded_path}'."
        logger.info(infostr)


    #-------------------- Output - missing genes ---------------------#


    # If some genes were missing
    if genes_missing:

        # Set the path to the output file
        output_txt_genes_missing_path = \
            os.path.join(wd, output_txt_genes_missing)

        # Try to write the list of missing genes
        try:

            with open(output_txt_genes_missing_path, "w") as out:
                out.write("\n".join(gene for gene in genes_missing))

        # If something went wrong
        except Exception as e:

            # Warn the user and exit
            errstr = \
                "It was not possible to write the list of genes " \
                "that were used to train the DGD model but are " \
                "not present in the input samples to " \
                f"'{output_txt_genes_missing_path}'."
            logger.exception(errstr)
            sys.exit(errstr)

        # Inform the user that the genes were successfully written in the
        # output file
        infostr = \
            "The list of genes that were used to train the DGD " \
            "model but are not present in the input samples " \
            "was successfully written in " \
            f"'{output_txt_genes_missing_path}'."
        logger.info(infostr)