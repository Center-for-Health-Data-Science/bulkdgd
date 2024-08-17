#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-

#    bulkdgd_find_representations.py
#
#    Find representations in the latent space defined by the
#    :class:`core.model.BulkDGDModel` for a set of samples.
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
    ":class:`core.model.BulkDGDModel` for a set of samples."


#######################################################################


# Import from the standard library.
import logging as log
import os
import sys
# Import from 'bulkDGD'.
from bulkDGD.core import model
from bulkDGD import defaults, ioutil
from . import util


#######################################################################


# Get the module's logger.
logger = log.getLogger(__name__)


#######################################################################


# Define a function to set up the parser.
def set_sub_parser(sub_parsers):

    # Create the argument parser.
    parser = \
        sub_parsers.add_parser(\
            name = "representations",
            description = __doc__,
            help = __doc__,
            formatter_class = util.CustomHelpFormatter)

    #-----------------------------------------------------------------#

    # Create a group of arguments for the input files.
    input_group = \
        parser.add_argument_group(title = "Input files")

    # Create a group of arguments for the output files.
    output_group = \
        parser.add_argument_group(title = "Output files")

    # Create a group of arguments for the configuration files.
    config_group = \
        parser.add_argument_group(title = "Configuration files")

    #-----------------------------------------------------------------#

    # Set a help message.
    is_help = \
        "The input CSV file containing the data frame with " \
        "the gene expression data for the samples for which a " \
        "representation in latent space should be found."

    # Add the argument to the group.
    input_group.add_argument("-is", "--input-samples",
                             type = str,
                             required = True,
                             help = is_help)

    #-----------------------------------------------------------------#

    # Set the default value for the argument.
    or_default = "representations.csv"

    # Set a help message.
    or_help = \
        "The name of the output CSV file containing the data frame " \
        "with the representation of each input sample in latent " \
        f"space. The default file name is '{or_default}'."

    # Add the argument to the group.
    output_group.add_argument("-or", "--output-rep",
                              type = str,
                              default = or_default,
                              help = or_help)

    #-----------------------------------------------------------------#

    # Set the default value for the argument.
    om_default = "pred_means.csv"

    # Set a help message.
    om_help = \
        "The name of the output CSV file containing the data frame " \
        "with the predicted scaled means of the negative  " \
        "binomials for the in silico samples obtained from the best " \
        "representations found. The default file name is " \
        f"'{om_default}'."

    # Add the argument to the group.
    output_group.add_argument("-om", "--output-means",
                              type = str,
                              default = om_default,
                              help = om_help)

    #-----------------------------------------------------------------#

    # Set the default value for the argument.
    ov_default = "pred_r_values.csv"

    # Set a help message.
    ov_help = \
        "The name of the output CSV file containing the data frame " \
        "with the predicted r-values of the negative binomials for " \
        "the in silico samples obtained from the best " \
        "representations found. The default file name is " \
        f"'{ov_default}'. The file is produced only if negative' " \
        "binomial distributions are used to model the genes' counts."

    # Add the argument to the group.
    output_group.add_argument("-ov", "--output-rvalues",
                              type = str,
                              default = ov_default,
                              help = ov_help)

    #-----------------------------------------------------------------#

    # Set the default value for the argument.
    ot_default = "opt_time.csv"

    # Set a help message.
    ot_help = \
        "The name of the output CSV file containing the data frame " \
        "with information about the CPU and wall clock time " \
        "spent for each optimization epoch and each backpropagation " \
        "step through the decoder. The default file name is " \
        f"'{ot_default}'."

    # Add the argument to the group.
    output_group.add_argument("-ot", "--output-time",
                              type = str,
                              default = ot_default,
                              help = ot_help)

    #-----------------------------------------------------------------#

    # Set the default directory where the configuration files are
    # located.
    cm_dir = defaults.CONFIG_DIRS["model"]

    # Set a help message.
    cm_help = \
        "The YAML configuration file specifying the bulkDGD model's " \
        "parameters and files containing the trained model. If it " \
        "is a name without an extension, it is assumed to be the " \
        f"name of a configuration file in '{cm_dir}'."

    # Add the argument to the group.
    config_group.add_argument("-cm", "--config-file-model",
                              type = str,
                              required = True,
                              help = cm_help)

    #-----------------------------------------------------------------#

    # Set the default directory where the configuration files are
    # located.
    cr_dir = defaults.CONFIG_DIRS["representations"]

    # Set a help message.
    cr_help = \
        "The YAML configuration file specifying the options for the " \
        "optimization step(s) when finding the best " \
        "representations. If it is a name without an extension, it " \
        "is assumed to be the name of a configuration file in " \
        f"'{cr_dir}'."

    # Add the argument to the group.
    config_group.add_argument("-cr", "--config-file-rep",
                              type = str,
                              required = True,
                              help = cr_help)

    #-----------------------------------------------------------------#

    # Return the parser.
    return parser


#######################################################################


# Define the 'main' function.
def main(args):

    # Get the argument corresponding to the working directory.
    wd = args.work_dir

    # Get the argument corresponding to the input file.
    input_samples = args.input_samples

    # Get the arguments corresponding to the configuration files.
    config_file_model = args.config_file_model            
    config_file_rep = args.config_file_rep

    # Get the arguments corresponding to the output files.
    output_rep = os.path.join(wd, args.output_rep)
    output_means = os.path.join(wd, args.output_means)
    output_rvalues = os.path.join(args.output_rvalues)
    output_time = os.path.join(wd, args.output_time)

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
        logger.exception(errstr)
        sys.exit(errstr)

    # Inform the user that the configuration was successfully loaded.
    infostr = \
        "The configuration was successfully loaded from " \
        f"'{config_file_model}'."
    logger.info(infostr)

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
        logger.exception(errstr)
        sys.exit(errstr)

    # Inform the user that the configuration was successfully loaded.
    infostr = \
        "The configuration was successfully loaded from " \
        f"'{config_file_rep}'."
    logger.info(infostr)

    #-----------------------------------------------------------------#

    # Try to load the samples' data.
    try:

        df_samples = \
            ioutil.load_samples(csv_file = input_samples,
                                sep = ",",
                                keep_samples_names = True,
                                split = False)

    # If something went wrong
    except Exception as e:

        # Warn the user and exit.
        errstr = \
            "It was not possible to load the samples from " \
            f"'{input_samples}'. Error: {e}"
        logger.exception(errstr)
        sys.exit(errstr)

    # Inform the user that the data were successfully loaded.
    infostr = \
        f"The samples were successfully loaded from '{input_samples}'."
    logger.info(infostr)

    #-----------------------------------------------------------------#

    # Try to set the model.
    try:
        
        dgd_model = model.BulkDGDModel(**config_model)

    # If something went wrong
    except Exception as e:

        # Warn the user and exit.
        errstr = \
            f"It was not possible to set the bulkDGD model. Error: {e}"
        logger.exception(errstr)
        sys.exit(errstr)

    # Inform the user that the model was successfully set.
    infostr = "The bulkDGD model was successfully set."
    logger.info(infostr)

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
        logger.exception(errstr)
        sys.exit(errstr)

    # Inform the user that the representations were successfully
    # optimized.
    infostr = "The representations were successfully found."
    logger.info(infostr)

    #-----------------------------------------------------------------#

    # Try to write the representations in the output CSV file.
    try:

        ioutil.save_representations(\
            df = df_rep,
            csv_file = output_rep,
            sep = ",")

    # If something went wrong
    except Exception as e:

        # Warn the user and exit.
        errstr = \
            "It was not possible to write the representations " \
            f"in '{output_rep}'. Error: {e}"
        logger.exception(errstr)
        sys.exit(errstr)

    # Inform the user that the representations were successfully
    # written in the output file.
    infostr = \
        "The representations were successfully written in " \
        f"'{output_rep}'."
    logger.info(infostr)

    #-----------------------------------------------------------------#

    # Try to write the predicted means in the dedicated CSV file.
    try:

        ioutil.save_decoder_outputs(\
            df = df_pred_means,
            csv_file = output_means,
            sep = ",")

    # If something went wrong
    except Exception as e:

        # Warn the user and exit.
        errstr = \
            "It was not possible to write the predicted means " \
            f"in '{output_means}'. Error: {e}"
        logger.exception(errstr)
        sys.exit(errstr)

    # Inform the user that the predicted means were successfully
    # written in the output file.
    infostr = \
        "The predicted means were successfully written in " \
        f"'{output_means}'."
    logger.info(infostr)

    #-----------------------------------------------------------------#

    # If the r-values were returned
    if df_pred_r_values is not None:

        # Try to write the predicted r-values in the dedicated CSV
        # file.
        try:

            ioutil.save_decoder_outputs(\
                df = df_pred_r_values,
                csv_file = output_rvalues,
                sep = ",")

        # If something went wrong
        except Exception as e:

            # Warn the user and exit.
            errstr = \
                "It was not possible to write the predicted " \
                f"r-values in '{output_rvalues}'. Error: {e}"
            logger.exception(errstr)
            sys.exit(errstr)

        # Inform the user that the predicted r-values were successfully
        # written in the output file.
        infostr = \
            "The predicted r-values were successfully written in " \
            f"'{output_rvalues}'."
        logger.info(infostr)

    #-----------------------------------------------------------------#

    # Try to write the time data in the dedicated CSV file.
    try:

        ioutil.save_time(\
            df = df_time,
            csv_file = output_time,
            sep = ",")

    # If something went wrong
    except Exception as e:

        # Warn the user and exit.
        errstr = \
            "It was not possible to write the time data " \
            f"in '{output_time}'. Error: {e}"
        logger.exception(errstr)
        sys.exit(errstr)

    # Inform the user that the time data was successfully
    # written in the output file.
    infostr = \
        f"The time data were successfully written in '{output_time}'."
    logger.info(infostr)
