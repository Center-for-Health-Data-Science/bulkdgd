#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-

#    probability_density.py
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
            name = "probability_density",
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

    # Create a group of arguments for configuration files.
    config_group = \
        parser.add_argument_group(title = "Configuration files")

    #-----------------------------------------------------------------#

    # Set a help message.
    ir_help = \
        "The input CSV file containing the data frame with " \
        "the representations."

    # Add the argument to the group.
    input_group.add_argument("-ir", "--input-rep",
                             type = str,
                             required = True,
                             help = ir_help)

    #-----------------------------------------------------------------#

    # Set the default value for the argument.
    opr_default = "probability_density_representations.csv"

    # Set a help message.
    opr_help = \
        "The name of the output CSV file containing, for each " \
        "representation, its probability density for each of the " \
        "Gaussian mixture model's components, the maximum " \
        "probability density found, and the component the maximum " \
        "probability density comes from. The file will be " \
        "written in the working directory. The default file " \
        f"name is '{opr_default}'."

    # Add the argument to the group.
    output_group.add_argument("-opr", "--output-prob-rep",
                              type = str,
                              default = opr_default,
                              help = opr_help)

    #-----------------------------------------------------------------#

    # Set the default value for the argument.
    opc_default = "probability_density_components.csv"

    # Set a help message.
    opc_help = \
        "The name of the output CSV file containing, for each " \
        "component of the Gaussian mixture model, the " \
        "representation(s) having the maximum probability " \
        "density with respect to it. The file will be written " \
        "in the working directory. The default file name is " \
        f"'{opc_default}'."

    # Add the argument to the group.
    output_group.add_argument("-opc", "--output-prob-comp",
                              type = str,
                              default = opc_default,
                              help = opc_help)

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

    # Return the parser.
    return parser


#######################################################################


# Define the 'main' function.
def main(args):

    # Get the argument corresponding to the working directory.
    wd = args.work_dir

    # Get the arguments corresponding to the input files.
    input_rep = args.input_rep

    # Get the arguments corresponding to the output files.
    output_prob_rep = os.path.join(wd, args.output_prob_rep)
    output_prob_comp = os.path.join(args.output_prob_comp)

    # Get the arguments corresponding to the configuration files.
    config_file_model = args.config_file_model

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

    # Try to load the data.
    try:

        df_rep  = \
            ioutil.load_representations(\
                csv_file = input_rep,
                sep = ",",
                split = False)

    # If something went wrong
    except Exception as e:

        # Warn the user and exit.
        errstr = \
            "It was not possible to load the representations from " \
            f"'{input_rep}'. Error: {e}"
        logger.exception(errstr)
        sys.exit(errstr)

    # Inform the user that the data were successfully loaded.
    infostr = \
        "The representations were successfully loaded " \
        f"from '{input_rep}'."
    logger.info(infostr)

    #-----------------------------------------------------------------#

    # Try to get the GMM.
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
        logger.exception(errstr)
        sys.exit(errstr)

    # Inform the user that the probability densities were successfully
    # calculated.
    infostr = \
        "The probability densities for the representations were " \
        "successfully calculated."
    logger.info(infostr)

    #-----------------------------------------------------------------#

    # Try to write the probability densities for the representations
    # to the output CSV file.
    try:

        df_prob_rep.to_csv(output_prob_rep,
                           sep = ",",
                           header = True,
                           index = True)

    # If something went wrong
    except Exception as e:

        # Warn the user and exit.
        errstr = \
            "It was not possible to write the probability " \
            "densities for each representation in " \
            f"'{output_prob_rep}'. Error: {e}"
        logger.exception(errstr)
        sys.exit(errstr)

    # Inform the user that the probability densities for the
    # representations were successfully written in the output file.
    infostr = \
        "The probability densities for each representation " \
        f"were successfully written in '{output_prob_rep}'."
    logger.info(infostr)

    #-----------------------------------------------------------------#

    # Try to write the probability densities for the representations
    # having the highest probability density for each component
    # to the output CSV file.
    try:

        df_prob_comp.to_csv(output_prob_comp,
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
            f"in '{output_prob_comp}'. Error: {e}"
        logger.exception(errstr)
        sys.exit(errstr)

    # Inform the user that the probability densities for the
    # representations were successfully written in the output file.
    infostr = \
        "The probability densities for the representations having " \
        "the highest probability density for each component " \
        f"were successfully written in '{output_prob_comp}'."
    logger.info(infostr)
