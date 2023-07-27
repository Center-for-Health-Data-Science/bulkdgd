#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-

#    dgd_get_representations.py
#
#    Find representations in latent space for a set of samples.
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
    "Find representations in latent space for a set of samples."


# Standard library
import argparse
import logging as log
import os
import sys
# Third-party packages
import pandas as pd
import torch
from torch.utils.data import DataLoader
# bulkDGD
from bulkDGD.utils import dgd, misc


def main():


    #-------------------- Command-line arguments ---------------------#


    # Create the argument parser
    parser = argparse.ArgumentParser(description = __doc__)

    # Add the arguments
    i_help = \
        "The input CSV file containing a data frame with " \
        "the gene expression data for the samples for which a " \
        "representation in latent space should be found. " \
        "The rows must represent the samples. The first column " \
        "must contain the unique names of the samples, while " \
        "the other columns must represent the genes (identified " \
        "by their Ensembl IDs). The genes should be the ones " \
        "the DGD model was trained on, whose Ensembl IDs can " \
        f"be found in '{dgd.DGD_GENES_FILE}'."
    parser.add_argument("-i", "--input-csv",
                        type = str,
                        required = True,
                        help = i_help)

    or_default = "representations.csv"
    or_help = \
        "The name of the output CSV file containing the data frame " \
        "with the representation of each input sample in latent " \
        "space. The rows represent the samples. The first column " \
        "contains the unique names of the samples, while the other " \
        "columns represent the values of the representations along " \
        "the latent space's dimensions. The file will be written in " 
        "the working directory. The default file name is " \
        f"'{or_default}'."
    parser.add_argument("-or", "--output-csv-rep",
                        type = str,
                        default = or_default,
                        help = or_help)

    od_default = "decoder_outputs.csv"
    od_help = \
        "The name of the output CSV file containing the data frame " \
        "with the decoder output for each input sample. The " \
        "rows represent the samples. The first column contains the " \
        "unique names of the samples, while the other columns " \
        "represent the genes (identified by their Ensembl IDs). " \
        "The file will be written in the working directory. " \
        f"The default file name is '{od_default}'."
    parser.add_argument("-od", "--output-csv-dec",
                        type = str,
                        default = od_default,
                        help = od_help)

    ol_default = "loss.csv"
    ol_help = \
        "The name of the output CSV file containing the data frame " \
        "with the per-sample loss associated with the best " \
        "representation found for each sample of interest. " \
        "The rows represent the samples, while the only column " \
        "contains the loss. The file will be written in the " \
        "working directory. The default file name is " \
        f"'{ol_default}'."
    parser.add_argument("-ol", "--output-csv-loss",
                        type = str,
                        default = ol_default,
                        help = ol_help)


    ot_default = "tissues.csv"
    ot_help = \
        "The name of the output CSV file containing the data  " \
        "frame with the labels of the tissues the amples belong to. " \
        "The rows represent the samples, while the only column " \
        "contains the labels of the tissues. "
        "The file will be written in the working directory. The " \
        f"default file name is '{ot_default}'. This file will not " \
        "be generated unless the input CSV file has a " \
        f"'{dgd._TISSUE_COL}' column containing the labels."
    parser.add_argument("-ot", "--output-csv-tissues",
                        type = str,
                        default = ot_default,
                        help = ot_help)

    cm_help = \
        "The YAML configuration file specifying the " \
        "DGD model parameters and files containing " \
        "the trained model. If it is a name without " \
        "extension, it is assumed to be the name of a " \
        f"configuration file in '{dgd.CONFIG_MODEL_DIR}'."
    parser.add_argument("-cm", "--config-file-model",
                        type = str,
                        required = True,
                        help = cm_help)


    cr_help = \
        "The YAMl configuration file containing the " \
        "options for data loading and optimization when " \
        "finding the best representations. If it is a name " \
        "without extension, it is assumed to be the name of " \
        f"a configuration file in '{dgd.CONFIG_REP_DIR}'."
    parser.add_argument("-cr", "--config-file-rep",
                        type = str,
                        required = True,
                        help = cr_help)

    d_help = \
        "The working directory. The default is the current " \
        "working directory."
    parser.add_argument("-d", "--work-dir",
                        type = str,
                        default = os.getcwd(),
                        help = d_help)

    v_help = "Verbose logging (INFO level)."
    parser.add_argument("-v", "--logging-verbose",
                        action = "store_true",
                        help = v_help)

    vv_help = \
        "Maximally verbose logging for debugging " \
        "purposes (DEBUG level)."
    parser.add_argument("-vv", "--logging-debug",
                        action = "store_true",
                        help = vv_help)

    # Parse the arguments
    args = parser.parse_args()
    input_csv = args.input_csv
    output_csv_rep = args.output_csv_rep
    output_csv_dec = args.output_csv_dec
    output_csv_loss = args.output_csv_loss
    output_csv_tissues = args.output_csv_tissues
    config_file_model = args.config_file_model
    config_file_rep = args.config_file_rep
    wd = args.work_dir
    v = args.logging_verbose
    vv = args.logging_debug


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

    # Set the logging level
    log.basicConfig(level = level)


    #--------------------- Configuration - model ---------------------#


    # Try to load the configuration
    try:

        config_model = misc.get_config_model(config_file_model)

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


    #---------------- Configuration - representations ----------------#


    # Try to load the configuration
    try:

        config_rep = misc.get_config_rep(config_file_rep)

    # If something went wrong
    except Exception as e:

        # Warn the user and exit
        errstr = \
            "It was not possible to load the configuration from " \
            f"'{config_file_rep}'. Error: {e}"
        logger.exception(errstr)
        sys.exit(errstr)

    # Inform the user that the configuration was successfully loaded
    infostr = \
        "The configuration was successfully loaded from " \
        f"'{config_file_rep}'."
    logger.info(infostr)


    #------------------------ Load the data --------------------------#


    # Try to load the samples' data
    try:

        # Create the dataset
        dataset, indexes, genes, tissues = \
            dgd.load_samples_data(csv_file = input_csv,
                                  keep_samples_names = True)

        # Create the data loader
        dataloader = DataLoader(dataset, **config_rep["data"])
    
    # If something went wrong
    except Exception as e:

        # Warn the user and exit
        errstr = \
            "It was not possible to load the samples from " \
            f"'{input_csv}'. Error: {e}"
        logger.exception(errstr)
        sys.exit(errstr)

    # Inform the user that the data were successfully loaded
    infostr = \
        f"The samples were successfully loaded from '{input_csv}'."
    logger.info(infostr)


    #-------------------- Gaussian mixture model ---------------------#


    # Get the dimensionality of the latent space
    dim_latent = config_model["dim_latent"]

    # Get the configuration for the GMM
    config_gmm = config_model["gmm"]

    # Try to get the GMN
    try:
        
        gmm = dgd.get_gmm(dim = dim_latent,
                          config = config_gmm)

    # If something went wrong
    except Exception as e:

        # Warn the user and exit
        errstr = \
            "It was not possible to set the Gaussian mixture " \
            f"model. Error: {e}"
        logger.exception(errstr)
        sys.exit(errstr)

    # Inform the user that the GMM was successfully set
    infostr = "The Gaussian mixture model was successfully set."
    logger.info(infostr)


    #---------------------------- Decoder ----------------------------#


    # Get the configuration for the decoder
    config_dec = config_model["dec"]

    # Try to get the decoder
    try:

        dec = dgd.get_decoder(dim = dim_latent,
                              config = config_dec)

    # If something went wrong
    except Exception as e:

        # Warn the user and exit
        errstr = \
            f"It was not possible to set the decoder. Error: {e}"
        logger.exception(errstr)
        sys.exit(errstr)

    # Inform the user that the decoder was successfully set
    infostr = "The decoder was successfully set."
    logger.info(infostr)


    #---------------- Representation layer (training) ----------------#


    # Get the configuration for the representation layer
    config_rep_layer = config_model["rep_layer"]

    # Try to get the representation layer
    try:
        
        rep_layer = dgd.get_rep_layer(dim = dim_latent,
                                      config = config_rep_layer)

    # If something went wrong
    except Exception as e:

        # Warn the user and exit
        errstr = \
            "It was not possible to set the representation layer. " \
            f"Error: {e}"
        logger.exception(errstr)
        sys.exit(errstr)

    # Inform the user that the representation layer was successfully
    # set
    infostr = "The representation layer was successfully set."
    logger.info(infostr)


    #------------------------- Optimization --------------------------#


    # Set how many representations per sample to start from
    n_new = 1

    # Get the configuration for the initial optimization
    config_opt1 = config_rep["optimization"]["opt1"]

    # Get the configuration for further optimization
    config_opt2 = config_rep["optimization"]["opt2"]
    
    # Try to get the representations
    try:
        
        df_loss, df_rep, df_dec_out = \
            dgd.get_representations(\
                # DataLoader object
                dataloader = dataloader,
                # Samples' unique indexes
                indexes = indexes,
                # Gaussian mixture model
                gmm = gmm,
                # Decoder
                dec = dec,
                # Number of samples in the dataset
                n_samples = len(indexes),
                # Number of genes in the dataset                         
                n_genes = len(genes),
                # Number of new representations per component
                # per sample                         
                n_samples_per_comp = 1,
                # Dimensionality of the latent space
                dim = dim_latent,
                # Configuration for the first round of optimization                         
                config_opt1 = config_opt1,
                # Configuration for the second round of optimization                         
                config_opt2 = config_opt2)

    # If something went wrong
    except Exception as e:

        # Warn the user and exit
        errstr = \
            "It was not possible to find the representations. " \
            f"Error: {e}"
        logger.exception(errstr)
        sys.exit(errstr)

    # Inform the user that the representations were successfully
    # optimized
    infostr = "The representations were successfully found."
    logger.info(infostr)


    #------------------- Output - representations --------------------#


    # Set the path to the output file
    output_csv_rep_path = os.path.join(wd, output_csv_rep)

    # Try to write the representations in the output CSV file
    try:

        df_rep.to_csv(output_csv_rep_path,
                      sep = ",",
                      header = False)

    # If something went wrong
    except Exception as e:

        # Warn the user and exit
        errstr = \
            "It was not possible to write the representations " \
            f"in '{output_csv_rep_path}'. Error: {e}"
        logger.exception(errstr)
        sys.exit(errstr)

    # Inform the user that the representations were successfully
    # written in the output file
    infostr = \
        "The representations were successfully written in " \
        f"'{output_csv_rep_path}'."
    logger.info(infostr)


    #-------------------- Output - decoder output --------------------#


    # Set the path to the output file
    output_csv_dec_path = os.path.join(wd, output_csv_dec)

    # Set the names of the columns of the data frame to be the
    # names of the genes
    df_dec_out.columns = genes

    # Try to write the decoder outputs in the dedicated CSV file
    try:

        df_dec_out.to_csv(output_csv_dec_path,
                          sep = ",",
                          header = True)

    # If something went wrong
    except Exception as e:

        # Warn the user and exit
        errstr = \
            "It was not possible to write the decoder outputs " \
            f"in '{output_csv_dec_path}'. Error: {e}"
        logger.exception(errstr)
        sys.exit(errstr)

    # Inform the user that the decoder outputs were successfully
    # written in the output file
    infostr = \
        "The decoder outputs were successfully written in " \
        f"'{output_csv_dec_path}'."
    logger.info(infostr)


    #------------------------- Output - loss -------------------------#


    # Set the path to the output file
    output_csv_loss_path = os.path.join(wd, output_csv_loss)

    # Try to save the per-sample loss to the output file
    try:

        df_loss.to_csv(output_csv_loss_path,
                       sep = ",",
                       header = False)

    # If something went wrong
    except Exception as e:

        # Warn the user and exit
        errstr = \
            "It was not possible to write the per-sample loss " \
            f"in '{output_csv_loss_path}'. Error: {e}"
        logger.exception(errstr)
        sys.exit(errstr)

    # Inform the user that the per-sample loss was successfully
    # written in the output file
    infostr = \
        "The per-sample loss wwas successfully written in " \
        f"'{output_csv_loss_path}'."
    logger.info(infostr)


    #----------------------- Output - tissues ------------------------#


    # If there were tissue labels in the input CSV file
    if not tissues.empty:

        # Set the path to the output file
        output_csv_tissues_path = os.path.join(wd, output_csv_tissues)

        # Try to write the tissue labels in the output CSV file
        try:

            tissues.to_csv(output_csv_tissues_path,
                           sep = ",",
                           header = False)

        # If something went wrong
        except Exception as e:

            # Warn the user and exit
            errstr = \
                "It was not possible to write the tissues' labels " \
                f"in '{output_csv_tissues_path}'. Error: {e}"
            logger.exception(errstr)
            sys.exit(errstr)

        # Inform the user that the tissues' labels were successfully
        # written in the output file
        infostr = \
            "The tissues' labels were successfully written in " \
            f"'{output_csv_tissues_path}'."
        logger.info(infostr)