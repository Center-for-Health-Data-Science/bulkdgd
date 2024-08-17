#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-

#    bulkdgd_train.py
#
#    Train the :class:`core.model.BulkDGDModel`.
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
__doc__ = "Train the :class:`core.model.BulkDGDModel`."


#######################################################################


# Import from the standard library.
import logging as log
import os
import sys
# Import from third-party packages.
import torch
# Import from 'bulkDGD'.
from bulkDGD.core import model
from bulkDGD import defaults, ioutil
from . import util


#######################################################################


# Get the module's logger.
logger = log.getLogger(__name__)


#######################################################################


# Define a function to set up the parser.
def set_parser(sub_parsers):

    # Create the argument parser.
    parser = \
        sub_parsers.add_parser(\
            name = "train",
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

    # Create a group of argument for the run options.
    run_group = \
        parser.add_argument_group(title = "Run options")

    #-----------------------------------------------------------------#

    # Set a help message.
    it_help = \
        "The input CSV file containing a data frame with " \
        "the gene expression data for the training samples."

    # Add the argument to the group.
    input_group.add_argument("-it", "--input-train",
                             type = str,
                             required = True,
                             help = it_help)

    #-----------------------------------------------------------------#

    # Set a help message.
    ie_help = \
        "The input CSV file containing a data frame with " \
        "the gene expression data for the test samples."

    # Add the argument to the group.
    input_group.add_argument("-ie", "--input-test",
                             type = str,
                             required = True,
                             help = ie_help)

    #-----------------------------------------------------------------#

    # Set the default value for the argument.
    ort_default = "representations_train.csv"

    # Set a help message.
    ort_help = \
        "The name of the output CSV file containing the data frame " \
        "with the representation of each training sample in latent " \
        "space. The file will be written in the working directory. " \
        f"The default file name is '{ort_default}'."

    # Add the argument to the group.
    output_group.add_argument("-ort", "--output-rep-train",
                              type = str,
                              default = ort_default,
                              help = ort_help)

    #-----------------------------------------------------------------#

    # Set the default value for the argument.
    ore_default = "representations_test.csv"

    # Set a help message.
    ore_help = \
        "The name of the output CSV file containing the data frame " \
        "with the representation of each test sample in latent " \
        "space. The file will be written in the working directory. " \
        f"The default file name is '{ore_default}'."

    # Add the argument to the group.
    output_group.add_argument("-ore", "--output-rep-test",
                              type = str,
                              default = ore_default,
                              help = ore_help)

    #-----------------------------------------------------------------#

    # Set the default value for the argument.
    ol_default = "loss.csv"

    # Set a help message.
    ol_help = \
        "The name of the output CSV file containing the data frame " \
        "with the per-epoch loss(es) for training and test samples. " \
        "The file will be written in the working directory. The " \
        f"default file name is '{ol_default}'."

    # Add the argument to the group.
    output_group.add_argument("-ol", "--output-loss",
                              type = str,
                              default = ol_default,
                              help = ol_help)

    #-----------------------------------------------------------------#

    # Set the default value for the argument.
    ot_default = "train_time.csv"

    # Set a help message.
    ot_help = \
        "The name of the output CSV file containing the data frame " \
        "with information about the CPU and wall clock time " \
        "spent for each training epoch the backpropagation steps " \
        "through the decoder. The file will be written in the " \
        f"working directory. The default file name is '{ot_default}'."

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
    ct_dir = defaults.CONFIG_DIRS["training"]

    # Set a help message.
    ct_help = \
        "The YAML configuration file specifying the " \
        "options for training the model. If it is a name " \
        "without an extension, it is assumed to be the name of " \
        f"a configuration file in '{ct_dir}'."

    # Add the argument to the group.
    config_group.add_argument("-ct", "--config-file-train",
                              type = str,
                              required = True,
                              help = ct_help)

    #-----------------------------------------------------------------#

    # Set a help message.
    dev_help = \
        "The device to use. If not provided, the GPU will be used " \
        "if it is available. Available devices are: 'cpu', 'cuda'."

    # Add the argument to the group.
    run_group.add_argument("-dev", "--device",
                           type = str,
                           help = dev_help)

    #-----------------------------------------------------------------#

    # Return the parser.
    return parser


#######################################################################


# Define the 'main' function.
def main(args):

    # Get the argument corresponding to the working directory.
    wd = args.work_dir

    # Get the arguments corresponding to the input files.
    input_train = args.input_train
    input_test = args.input_test

    # Get the arguments corresponding to the output files.
    output_rep_train = os.path.join(wd, args.output_rep_train)
    output_rep_test = os.path.join(wd, args.output_rep_test)
    output_loss = os.path.join(args.output_loss)
    output_time = os.path.join(args.output_time)

    # Get the arguments corresponding to the configuration files.
    config_file_model = args.config_file_model
    config_file_train = args.config_file_train

    # Get the arguments corresponding to the run options.
    device = args.device

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

        config_train = ioutil.load_config_train(config_file_train)

    # If something went wrong
    except Exception as e:

        # Warn the user and exit.
        errstr = \
            "It was not possible to load the configuration from " \
            f"'{config_file_train}'. Error: {e}"
        log.exception(errstr)
        sys.exit(errstr)

    # Inform the user that the configuration was successfully loaded.
    infostr = \
        "The configuration was successfully loaded from " \
        f"'{config_file_train}'."
    log.info(infostr)

    #-----------------------------------------------------------------#

    # Try to load the training samples.
    try:

        df_train = \
            ioutil.load_samples(csv_file = input_train,
                                sep = ",",
                                keep_samples_names = True,
                                split = False)

    # If something went wrong
    except Exception as e:

        # Warn the user and exit.
        errstr = \
            "It was not possible to load the training samples from " \
            f"'{input_train}'. Error: {e}"
        log.exception(errstr)
        sys.exit(errstr)

    # Inform the user that the data were successfully loaded.
    infostr = \
        "The training samples were successfully loaded from " \
        f"'{input_train}'."
    log.info(infostr)

    #-----------------------------------------------------------------#

    # Try to load the test samples.
    try:

        df_test = \
            ioutil.load_samples(csv_file = input_test,
                                sep = ",",
                                keep_samples_names = True,
                                split = False)

    # If something went wrong
    except Exception as e:

        # Warn the user and exit.
        errstr = \
            "It was not possible to load the test samples from " \
            f"'{input_test}'. Error: {e}"
        log.exception(errstr)
        sys.exit(errstr)

    # Inform the user that the data were successfully loaded.
    infostr = \
        "The test samples were successfully loaded from " \
        f"'{input_test}'."
    log.info(infostr)

    #-----------------------------------------------------------------#

    # Try to set the model.
    try:
        
        dgd_model = model.BulkDGDModel(**config_model)

    # If something went wrong
    except Exception as e:

        # Warn the user and exit.
        errstr = \
            f"It was not possible to set the bulkDGD model. Error: {e}"
        log.exception(errstr)
        sys.exit(errstr)

    # Inform the user that the model was successfully set.
    infostr = "The bulkDGD model was successfully set."
    log.info(infostr)

    #-----------------------------------------------------------------#

    # If no device was passed
    if device is None:
        
        # If a CPU with CUDA is available.
        if torch.cuda.is_available():

            # Set the GPU as the device.
            device = torch.device("cuda")

        # Otherwise
        else:

            # Set the CPU as the device.
            device = torch.device("cpu")

    # Try to move the model to the device.
    try:
        
        dgd_model.device = device

    # If something went wrong
    except Exception as e:

        # Warn the user and exit.
        errstr = \
            "It was not possible to move the bulkDGD model to the " \
            f"'{device}' device. Error: {e}"
        log.exception(errstr)
        sys.exit(errstr)

    # Inform the user that the model was successfully moved.
    infostr = \
        "The bulkDGD model was successfully moved to the " \
        f"'{device}' device."
    log.info(infostr)

    #-----------------------------------------------------------------#

    # Try to train the bulkDGD model.
    try:

        df_rep_train, df_rep_test, df_loss, df_time = \
            dgd_model.train(df_train = df_train,
                            df_test = df_test,
                            config_train = config_train)

    # If something went wrong
    except Exception as e:

        # Warn the user and exit.
        errstr = \
            "It was not possible to train the bulkDGD model. " \
            f"Error: {e}"
        log.exception(errstr)
        sys.exit(errstr)

    # Inform the user that the bulkDGD model was successfully trained.
    infostr = "The bulkDGD model was successfully trained."
    log.info(infostr)

    #-----------------------------------------------------------------#

    # Try to write the representations for the training samples in
    # the output CSV file.
    try:

        ioutil.save_representations(\
            df = df_rep_train,
            csv_file = output_rep_train,
            sep = ",")

    # If something went wrong
    except Exception as e:

        # Warn the user and exit.
        errstr = \
            "It was not possible to write the representations " \
            f"for the training samples in '{output_rep_train}'. " \
            f"Error: {e}"
        log.exception(errstr)
        sys.exit(errstr)

    # Inform the user that the representations for the training
    # samples were successfully written in the output file.
    infostr = \
        "The representations for the training samples were " \
        f"successfully written in '{output_rep_train}'."
    log.info(infostr)

    #-----------------------------------------------------------------#

    # Try to write the representations for the test samples in the
    # output CSV file.
    try:

        ioutil.save_representations(\
            df = df_rep_test,
            csv_file = output_rep_test,
            sep = ",")

    # If something went wrong
    except Exception as e:

        # Warn the user and exit.
        errstr = \
            "It was not possible to write the representations " \
            f"for the test samples in '{output_rep_test}'. " \
            f"Error: {e}"
        log.exception(errstr)
        sys.exit(errstr)

    # Inform the user that the representations for the test samples
    # were successfully written in the output file.
    infostr = \
        "The representations for the test samples were " \
        f"successfully written in '{output_rep_test}'."
    log.info(infostr)

    #-----------------------------------------------------------------#

    # Try to write the losses in the dedicated CSV file.
    try:

        ioutil.save_loss(\
            df = df_loss,
            csv_file = output_loss,
            sep = ",")

    # If something went wrong
    except Exception as e:

        # Warn the user and exit.
        errstr = \
            "It was not possible to write the losses " \
            f"in '{output_loss}'. Error: {e}"
        log.exception(errstr)
        sys.exit(errstr)

    # Inform the user that the losses were successfully written in
    # the output file.
    infostr = \
        f"The losses were successfully written in '{output_loss}'."
    log.info(infostr)

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
        log.exception(errstr)
        sys.exit(errstr)

    # Inform the user that the time data was successfully
    # written in the output file.
    infostr = \
        f"The time data were successfully written in '{output_time}'."
    log.info(infostr)
