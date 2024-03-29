#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-

#    dgd_preprocess_samples.py
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
# bulkDGD
from bulkDGD import defaults, ioutil, util


def main():


    # Create the argument parser
    parser = argparse.ArgumentParser(description = __doc__)

    #-----------------------------------------------------------------#

    # Add the arguments
    i_help = \
        "The input CSV file containing a data frame with the " \
        "samples to be preprocessed."
    parser.add_argument("-i", "--input-csv",
                        type = str,
                        required = True,
                        help = i_help)

    #-----------------------------------------------------------------#

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

    #-----------------------------------------------------------------#

    oge_default = "genes_excluded.txt"
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

    #-----------------------------------------------------------------#

    ogm_default = "genes_missing.txt"
    ogm_help = \
        "The name of the output plain text file containing the list " \
        "of genes for which no expression data are found in the " \
        "input data frame. A default count of 0 is assigned to " \
        "these genes in the output data frame containing the " \
        "preprocessed samples. The file will be written in the " \
        f"working directory. The default file name is '{ogm_default}'."
    parser.add_argument("-om", "--output-txt-genes-missing",
                        type = str,
                        default = ogm_default,
                        help = ogm_help)

    #-----------------------------------------------------------------#

    d_help = \
        "The working directory. The default is the current " \
        "working directory."
    parser.add_argument("-d", "--work-dir",
                        type = str,
                        default = os.getcwd(),
                        help = d_help)

    #-----------------------------------------------------------------#

    lf_default = "dgd_preprocess_samples.log"
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
    output_csv_samples = args.output_csv_samples
    output_txt_genes_excluded = args.output_txt_genes_excluded
    output_txt_genes_missing = args.output_txt_genes_missing
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

    # Try to load the input samples
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

    #-----------------------------------------------------------------#

    # Try to preprocess the samples
    try:

        df_preproc, genes_excluded, genes_missing = \
            ioutil.preprocess_samples(df_samples = df_samples)

    # If something went wrong
    except Exception as e:

        # Warn the user and exit
        errstr = \
            "It was not possible to preprocess the samples. " \
            f"Error: {e}"
        log.exception(errstr)
        sys.exit(errstr)

    #-----------------------------------------------------------------#

    # Set the path to the output file
    output_csv_samples_path = os.path.join(wd, output_csv_samples)

    # Try to write out the preprocessed samples
    try:

        ioutil.save_samples(df = df_preproc,
                            csv_file = output_csv_samples_path,
                            sep = ",")

    # If something went wrong
    except Exception as e:

        # Warn the user and exit
        errstr = \
            "It was not possible to write the preprocessed " \
            f"samples to '{output_csv_samples_path}'. Error: {e}"
        log.exception(errstr)
        sys.exit(errstr)

    # Inform the user that the preprocessed samples were
    # successfilly written in the output file
    infostr = \
        "The preprocessed samples were successfully written in " \
        f"'{output_csv_samples_path}'."
    log.info(infostr)

    #-----------------------------------------------------------------#

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
                "not among the genes included in the DGD model " \
                f"in '{output_txt_genes_excluded_path}'. Error: {e}"
            log.exception(errstr)
            sys.exit(errstr)

        # Inform the user that the genes were successfully written
        # to the output file
        infostr = \
            "The list of genes that are present in the input " \
            "samples but are not among the genes included in " \
            "the DGD model was successfully written in " \
            f"'{output_txt_genes_excluded_path}'."
        log.info(infostr)

    #-----------------------------------------------------------------#

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
                "that are included in the DGD model but are " \
                "not present in the input samples in " \
                f"'{output_txt_genes_missing_path}'."
            log.exception(errstr)
            sys.exit(errstr)

        # Inform the user that the genes were successfully written in
        # the output file
        infostr = \
            "The list of genes that are included in the DGD model " \
            "model but are not present in the input samples " \
            "was successfully written in " \
            f"'{output_txt_genes_missing_path}'."
        log.info(infostr)
