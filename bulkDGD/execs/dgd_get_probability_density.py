#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-

#    dgd_get_probability_density.py
#
#    Given a CSV file containing the representations of one or 
#    multiple samples and the Gaussian mixture model (GMM)
#    modeling the representation space, find the probability 
#    density of each representation for each GMM component.
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
    "Given a CSV file containing the representations of one or " \
    "multiple samples and the Gaussian mixture model (GMM) " \
    "modeling the representation space, find the probability " \
    "density of each representation for each GMM component."


# Standard library
import argparse
import logging as log
import os
import sys
# Third-party packages
import pandas as pd
import torch
# bulkDGD
from bulkDGD.utils import dgd, misc


def main():


    #-------------------- Command-line arguments ---------------------#


    # Create the argument parser
    parser = argparse.ArgumentParser(description = __doc__)

    # Add the arguments
    ir_help = \
        "The input CSV file containing the data frame with " \
        "the samples' representations in latent space. " \
        "The columns represent the values of the " \
        "representations along the latent space's dimensions, " \
        "while the rows represent the samples."
    parser.add_argument("-ir", "--input-csv-rep",
                        type = str,
                        required = True,
                        help = ir_help)

    it_help = \
        f"The input CSV file containing one column with the labels " \
        f"of the tissues the input samples belong to. This " \
        f"input file is optional."
    parser.add_argument("-it", "--input-csv-tissues",
                        type = str,
                        default = None,
                        help = it_help)

    or_default = "probability_density_representations.csv"
    or_help = \
        f"The name of the output CSV file containing, for each " \
        f"representation, its probability density for each of the " \
        f"Gaussian mixture model's components, the maximum " \
        f"probability density found, the component the maximum " \
        f"probability density comes from, and the label of the " \
        f"tissue the input sample belongs to. The file will be " \
        f"written in the working directory. The default file " \
        f"name is '{or_default}'."
    parser.add_argument("-or", "--output-csv-prob-rep",
                        type = str,
                        default = or_default,
                        help = or_help)

    oc_default = "probability_density_components.csv"
    oc_help = \
        f"The name of the output CSV file containing, for each " \
        f"component of the Gaussian mixture model, the " \
        f"representation(s) having the maximum probability " \
        f"density with respect to it. The file will be written " \
        f"in the working directory. The default file name is " \
        f"'{oc_default}'."
    parser.add_argument("-oc", "--output-csv-prob-comp",
                        type = str,
                        default = oc_default,
                        help = oc_help)

    c_help = \
        f"The YAML configuration file specifying the " \
        f"DGD model parameters and files containing " \
        f"the trained model. If it is a name without " \
        f"extension, it is assumed to be the name of a " \
        f"configuration file in {dgd.CONFIG_REP_DIR}."
    parser.add_argument("-c", "--config-file",
                        type = str,
                        required = True,
                        help = c_help)

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
    input_csv_rep = args.input_csv_rep
    input_csv_tissues = args.input_csv_tissues
    output_csv_prob_rep = args.output_csv_prob_rep
    output_csv_prob_comp = args.output_csv_prob_comp
    config_file = args.config_file
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


    #------------------------- Configuration -------------------------#


    # Try to load the configuration
    try:

        config = misc.get_config_rep(config_file)

    # If something went wrong
    except Exception as e:

        # Warn the user and exit
        errstr = \
            f"It was not possible to load the configuration from " \
            f"'{config_file}'. Error: {e}"
        logger.exception(errstr)
        sys.exit(errstr)

    # Inform the user that the configuration was successfully loaded
    infostr = \
        f"The configuration was successfully loaded from " \
        f"'{config_file}'."
    logger.info(infostr)


    #---------------- Data loading (representations) -----------------#


    # Try to load the data
    try:

        df_rep = pd.read_csv(input_csv_rep,
                             sep = ",",
                             index_col = 0,
                             header = None)

    # If something went wrong
    except Exception as e:

        # Warn the user and exit
        errstr = \
            f"It was not possible to load the data from " \
            f"'{input_csv_rep}'. Error: {e}"
        logger.exception(errstr)
        sys.exit(errstr)

    # Get the number of samples and the dimensionality
    # of the latent space from the dimensions of the
    # input data frame
    n_samples, dim_latent = df_rep.shape

    # Inform the user that the data were successfully loaded
    infostr = \
        f"The data were successfully loaded from '{input_csv_rep}'."
    logger.info(infostr)


    #-------------------- Data loading (tissues) ---------------------#


    # If the user passed a CSV file containing the label of the
    # tissues the original samples belonged to
    if input_csv_tissues is not None:

        # Try to load the data
        try:

            df_tissues = pd.read_csv(input_csv_tissues,
                                     sep = ",",
                                     index_col = 0,
                                     header = None)

        # If something went wrong
        except Exception as e:

            # Warn the user and exit
            errstr = \
                f"It was not possible to load the data from " \
                f"'{input_csv_tissues}'. Error: {e}"
            logger.exception(errstr)
            sys.exit(errstr)

        # Inform the user that the data were successfully loaded
        infostr = \
            f"The data were successfully loaded from " \
            f"'{input_csv_tissues}'."
        logger.info(infostr)


    #-------------------- Gaussian mixture model ---------------------#


    # If the dimensionality of the latent space provided in the
    # configuration file does not match the one found in the
    # input file
    if dim_latent != config["dim_latent"]:

        # Warn the user and exit
        errstr = \
            f"The representations found in the input file " \
            f"'{input_csv_rep}' have {dim_latent} dimensions, " \
            f"while the configuration file defines a Gaussian " \
            f"mixture model with {config['dim_latent']} dimensions."
        logger.error(errstr)
        sys.exit(errstr)

    # Get the configuration for the GMM
    config_gmm = config["gmm"]

    # Try to get the GMN
    try:
        
        gmm = dgd.get_gmm(dim = dim_latent,
                          config = config_gmm)

    # If something went wrong
    except Exception as e:

        # Warn the user and exit
        errstr = \
            f"It was not possible to get the Gaussian mixture " \
            f"model. Error: {e}"
        logger.exception(errstr)
        sys.exit(errstr)

    # Inform the user that the GMM was successfully set
    infostr = "The Gaussian mixture model was successfully set."
    logger.info(infostr)


    #------------------- Components' probabilities -------------------#


    df_prob_rep, df_prob_comp = \
        dgd.get_probability_density(gmm = gmm,
                                    df_rep = df_rep,
                                    df_tissues = df_tissues)


    #------------- Output - p.d. for all representations --------------#


    # Set the path to the output file
    output_csv_prob_rep_path = os.path.join(wd, output_csv_prob_rep)

    # Try to write the probability densities for the representations
    # to the output CSV file
    try:

        df_prob_rep.to_csv(output_csv_prob_rep_path,
                           sep = ",",
                           header = True,
                           index = True)

    # If something went wrong
    except Exception as e:

        # Warn the user and exit
        errstr = \
            f"It was not possible to write the probability " \
            f"densities for each representation to " \
            f"'{output_csv_prob_rep_path}'. Error: {e}"
        logger.exception(errstr)
        sys.exit(errstr)

    # Inform the user that the probability densities for the
    # representations were successfully written in the output file
    infostr = \
        f"The probability densities for each representation " \
        f"were successfully written in " \
        f"'{output_csv_prob_rep_path}'."
    logger.info(infostr)


    #---------- Output - representations with highest p.d. -----------#


    # Set the path to the output
    output_csv_prob_comp_path = os.path.join(wd, output_csv_prob_comp)

    # Try to write the probability densities for the representations
    # having the highest probability density for each component
    # to the output CSV file
    try:

        df_prob_comp.to_csv(output_csv_prob_comp_path,
                            sep = ",",
                            header = True,
                            index = True)

    # If something went wrong
    except Exception as e:

        # Warn the user and exit
        errstr = \
            f"It was not possible to write the probability " \
            f"densities for the representations having the " \
            f"highest probability density for each component " \
            f"to '{output_csv_prob_comp_path}'. Error: {e}"
        logger.exception(errstr)
        sys.exit(errstr)

    # Inform the user that the probability densities for the
    # representations were successfully written in the output file
    infostr = \
        f"The probability densities for the representations having " \
        f"the highest probability density for each component " \
        f"were successfully written in " \
        f"'{output_csv_prob_comp_path}'."
    logger.info(infostr)