#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-

#    dgd_perform_dea.py
#
#    Perform differential expression analysis comparing experimental
#    samples to their "closest normal" sample found in latent space
#    by the DGD model.
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
    "Perform differential expression analysis comparing " \
    "experimental samples to their 'closest normal' sample " \
    "found in latent space by the DGD model."


# Standard library
import argparse
from distributed import (
    as_completed,
    Client,
    LocalCluster)
import logging as log
import os
import sys
# Third-party packages
import pandas as pd
import torch
# bulkDGD
from bulkDGD.core import model
from bulkDGD.analysis import dea
from bulkDGD import ioutil


def main():


    #-------------------- Command-line arguments ---------------------#


    # Create the argument parser
    parser = argparse.ArgumentParser(description = __doc__)

    # Add the arguments
    is_help = \
        "The input CSV file containing a data frame with " \
        "the gene expression data for the samples."
    parser.add_argument("-is", "--input-csv-samples",
                        type = str,
                        required = True,
                        help = is_help)

    id_help = \
        "The input CSV file containing the data frame " \
        "with the decoder output for each samples' best " \
        "representation."
    parser.add_argument("-id", "--input-csv-dec",
                        type = str,
                        required = True,
                        help = id_help)

    op_default = "dea_"
    op_help = \
        "The prefix of the output CSV file(s) that will contain " \
        "the results of the differential expression analysys. " \
        "Since the analysis will be performed for each sample, " \
        "one file per sample will be created. The files' names " \
        "will have the form {output_csv_prefix}{sample_name}.csv. " \
        f"The default prefix is '{op_default}'."
    parser.add_argument("-op", "--output-csv-prefix",
                        type = str,
                        default = op_default,
                        help = op_help)

    cm_help = \
        "The YAML configuration file specifying the " \
        "DGD model's parameters and files containing " \
        "the trained model. If it is a name without " \
        "extension, it is assumed to be the name of a " \
        f"configuration file in '{ioutil.CONFIG_MODEL_DIR}'."
    parser.add_argument("-cm", "--config-file-model",
                        type = str,
                        required = True,
                        help = cm_help)

    pr_help = \
        "The resolution at which to sum over the probability " \
        "mass function to compute the p-values. The higher the " \
        "resolution, the more accurate the calculation. A " \
        "The default is an exact calculation."
    parser.add_argument("-pr", "--p-values-resolution",
                        type = lambda x: int(float(x)),
                        default = None,
                        help = pr_help)

    qa_default = 0.05
    qa_help = \
        "The alpha value used to calculate the q-values (adjusted " \
        f"p-values). The default is {qa_default}."
    parser.add_argument("-qa", "--q-values-alpha",
                        type = float,
                        default = qa_default,
                        help = qa_help)

    qm_default = "fdr_bh"
    qm_help = \
        "The method used to calculate the q-values (i.e., to " \
        f"adjust the p-values). The default is '{qm_default}'. " \
        "The available methods can be found in the documentation " \
        "of 'statsmodels.stats.multitest.multipletests', " \
        "which is used to perform the calculation."
    parser.add_argument("-qm", "--q-values-method",
                        type = str,
                        default = qm_default,
                        help = qm_help)

    d_help = \
        "The working directory. The default is the current " \
        "working directory."
    parser.add_argument("-d", "--work-dir",
                        type = str,
                        default = os.getcwd(),
                        help = d_help)

    n_default = 1
    n_help = \
        "The number of processes to start (the higher " \
        "the number, the faster the execution of the program). " \
        "The default number of processes started is " \
        f"{n_default}."
    parser.add_argument("-n", "--n-proc",
                        type = int,
                        default = n_default,
                        help = n_help)
    
    lf_default = "dgd_perform_dea.log"
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
    input_csv_samples = args.input_csv_samples
    input_csv_dec = args.input_csv_dec
    output_csv_prefix = args.output_csv_prefix
    config_file_model = args.config_file_model
    p_values_resolution = args.p_values_resolution
    q_values_alpha = args.q_values_alpha
    q_values_method = args.q_values_method
    wd = args.work_dir
    n_proc = args.n_proc
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

    # Suppress all Dask logs below WARNING
    log.getLogger("distributed").setLevel(log.WARNING)


    #------------------------- Configuration -------------------------#


    # Try to load the configuration
    try:

        config_model = ioutil.load_config_model(config_file_model)

    # If something went wrong
    except Exception as e:

        # Warn the user and exit
        errstr = \
            "It was not possible to load the configuration from " \
            f"'{config_file_model}'. Error: {e}"
        logger.exception(errstr)
        sys.exit(errstr)

    # Inform the user that the configuration was successfully loaded
    infostr = \
        "The configuration was successfully loaded from " \
        f"'{config_file_model}'."
    logger.info(infostr)


    #----------------------- Input - samples -------------------------#


    # Try to load the samples' data
    try:

        # Get the observed counts
        obs_counts = \
            ioutil.load_samples(\
                csv_file = input_csv_samples,
                sep = ",",
                keep_samples_names = True,
                split = False)

        # Get the sample's names
        obs_counts_names = obs_counts.index.tolist()

    # If something went wrong
    except Exception as e:

        # Warn the user and exit
        errstr = \
            "It was not possible to load the samples from " \
            f"'{input_csv_samples}'. Error: {e}"
        logger.exception(errstr)
        sys.exit(errstr)


    #-------------------- Input - decoder outputs --------------------#


    # Try to load the decoder outputs
    try:

        # Get the predicted means
        pred_means = \
            ioutil.load_decoder_outputs(\
                csv_file = input_csv_dec,
                sep = ",",
                split = False)

    # If something went wrong
    except Exception as e:

        # Warn the user and exit
        errstr = \
            "It was not possible to load the decoder outputs from " \
            f"'{input_csv_dec}'. Error: {e}"
        logger.exception(errstr)
        sys.exit(errstr)


    #------------------------ Load the model -------------------------#
    

    # Try to get the model
    try:
        
        dgd_model = model.DGDModel(**config_model)

    # If something went wrong
    except Exception as e:

        # Warn the user and exit
        errstr = \
            f"It was not possible to set the DGD model. Error: {e}"
        logger.exception(errstr)
        sys.exit(errstr)

    # Inform the user that the model was successfully set
    infostr = "The DGD model was successfully set."
    logger.info(infostr)


    #----------------------- Get the r-values ------------------------#


    # Try to get the r-values
    try:

        # Get the r-values of the negative binomials
        r_values = dgd_model.get_r_values()

    # If something went wrong
    except Exception as e:

        # Warn the user and exit
        errstr = \
            "It was not possible to retrieve the r-values from the " \
            f"models's paramenters. Error: {e}"
        logger.exception(errstr)
        sys.exit(errstr)

    # Inform the user that the r-values were successfully retrieved
    infostr = \
        "The r-values were successfully retrieved from the " \
        "model's paramenters."
    logger.info(infostr)


    #--------------- Set the loocal cluster and client ---------------#


    # Create the local cluster
    cluster = LocalCluster(# Number of workers
                           n_workers = n_proc,
                           # Below which level log messages will
                           # be silenced
                           silence_logs = "WARNING",
                           # Whether to use processes, single-core
                           # or threads
                           processes = True,
                           # How many threads for each worker should
                           # be used
                           threads_per_worker = 1)
    
    # Open the client from the cluster
    client = Client(cluster)


    #---------------------- Get the statistics -----------------------#


    # Set the statistics to be calculated
    statistics = ["p_values", "q_values", "log2_fold_changes"]

    # Create a list to store the futures
    futures = []
    
    # For each samples' observed counts, predicted means, 
    # and name
    for obs_counts_sample, pred_means_sample, sample_name \
    in zip(obs_counts, pred_means, obs_counts_names):
            
        # Submit the calculation to the cluster
        futures.append(\
            client.submit(dea.perform_dea,
                          obs_counts_sample = obs_counts_sample,
                          pred_means_sample = pred_means_sample,
                          sample_name = sample_name,
                          statistics = statistics,
                          genes_names = obs_counts_genes,
                          r_values = r_values,
                          resolution = p_values_resolution,
                          alpha = q_values_alpha,
                          method = q_values_method))


    #-------------------- Write the output files ---------------------#


    # For each future (which is retrieved as it completes)
    for future in as_completed(futures):
        
        # Get the data frame containing the DEA results for the
        # current sample and the ame of the sample
        df_stats, sample_name = future.result()

        # Set the path to the output file
        output_csv_path = \
            os.path.join(wd,
                         f"{output_csv_prefix}{sample_name}.csv")

        # Try to write the data frame to the output file
        try:

            df_stats.to_csv(output_csv_path,
                            sep = ",",
                            index = True,
                            header = True)

        # If something went wrong
        except Exception as e:

            # Warn the user and exit
            errstr = \
                "It was not possible to write the DEA results " \
                f"for sample '{sample_name}' to " \
                f"'{output_csv_path}'. Error: {e}"
            logger.exception(errstr)
            sys.exit(errstr)

        # Inform the user that the file was successfully written
        infostr = \
            f"The DEA results for sample '{sample_name}' were " \
            f"successfully written to '{output_csv_path}'."
        logger.info(infostr)