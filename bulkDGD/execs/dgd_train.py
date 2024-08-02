#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-

#    dgd_train.py
#
#    Train the DGD model.
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
__doc__ = "Train the DGD model."


#######################################################################


# Import from the standard library.
import argparse
import logging as log
import os
import sys
# Import from third-party packages.
import torch
# Import from 'bulkDGD'.
from bulkDGD.core import model
from bulkDGD import defaults, ioutil, util


#######################################################################


def main():


    # Create the argument parser.
    parser = argparse.ArgumentParser(description = __doc__)

    #-----------------------------------------------------------------#

    # Add the arguments.
    it_help = \
        "The input CSV file containing a data frame with " \
        "the gene expression data for the training samples."
    parser.add_argument("-it", "--input-csv-train",
                        type = str,
                        required = True,
                        help = it_help)

    #-----------------------------------------------------------------#

    ie_help = \
        "The input CSV file containing a data frame with " \
        "the gene expression data for the test samples."
    parser.add_argument("-ie", "--input-csv-test",
                        type = str,
                        required = True,
                        help = ie_help)

    #-----------------------------------------------------------------#

    ort_default = "representations_train.csv"
    ort_help = \
        "The name of the output CSV file containing the data frame " \
        "with the representation of each training sample in latent " \
        "space. The file will be written in the working directory. " \
        f"The default file name is '{ort_default}'."
    parser.add_argument("-ort", "--output-csv-rep-train",
                        type = str,
                        default = ort_default,
                        help = ort_help)

    #-----------------------------------------------------------------#

    ore_default = "representations_test.csv"
    ore_help = \
        "The name of the output CSV file containing the data frame " \
        "with the representation of each test sample in latent " \
        "space. The file will be written in the working directory. " \
        f"The default file name is '{ore_default}'."
    parser.add_argument("-ore", "--output-csv-rep-test",
                        type = str,
                        default = ore_default,
                        help = ore_help)

    #-----------------------------------------------------------------#

    ol_default = "loss.csv"
    ol_help = \
        "The name of the output CSV file containing the data frame " \
        "with the per-epoch loss(es) for training and test samples. " \
        "The file will be written in the working directory. The " \
        f"default file name is '{ol_default}'."
    parser.add_argument("-ol", "--output-csv-loss",
                        type = str,
                        default = ol_default,
                        help = ol_help)

    #-----------------------------------------------------------------#

    ot_default = "train_time.csv"
    ot_help = \
        "The name of the output CSV file containing the data frame " \
        "with information about the CPU and wall clock time " \
        "spent for each training epoch the backpropagation steps " \
        "through the decoder. The file will be written in the " \
        f"working directory. The default file name is '{ot_default}'."
    parser.add_argument("-ot", "--output-csv-time",
                        type = str,
                        default = ot_default,
                        help = ot_help)

    #-----------------------------------------------------------------#

    cm_help = \
        "The YAML configuration file specifying the " \
        "DGD model's parameters. If it is a name without an " \
        "extension, it is assumed to be the name of a " \
        f"configuration file in '{ioutil.CONFIG_MODEL_DIR}'."
    parser.add_argument("-cm", "--config-file-model",
                        type = str,
                        required = True,
                        help = cm_help)

    #-----------------------------------------------------------------#

    ct_help = \
        "The YAML configuration file containing the " \
        "options for training the model. If it is a name " \
        "without an extension, it is assumed to be the name of " \
        f"a configuration file in '{ioutil.CONFIG_TRAIN_DIR}'."
    parser.add_argument("-ct", "--config-file-train",
                        type = str,
                        required = True,
                        help = ct_help)

    #-----------------------------------------------------------------#

    d_help = \
        "The working directory. The default is the current " \
        "working directory."
    parser.add_argument("-d", "--work-dir",
                        type = str,
                        default = os.getcwd(),
                        help = d_help)

    #-----------------------------------------------------------------#

    dev_help = \
        "The device to use. If not provided, the GPU will be used " \
        "if it is available. Available devices are: 'cpu', 'cuda'."
    parser.add_argument("-dev", "--device",
                        type = str,
                        default = None,
                        help = dev_help)

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
    input_csv_train = args.input_csv_train
    input_csv_test = args.input_csv_test
    output_csv_rep_train = args.output_csv_rep_train
    output_csv_rep_test = args.output_csv_rep_test
    output_csv_loss = args.output_csv_loss
    output_csv_time = args.output_csv_time
    config_file_model = args.config_file_model
    config_file_train = args.config_file_train
    wd = os.path.abspath(args.work_dir)
    device = args.device
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
            ioutil.load_samples(csv_file = input_csv_train,
                                sep = ",",
                                keep_samples_names = True,
                                split = False)

    # If something went wrong
    except Exception as e:

        # Warn the user and exit.
        errstr = \
            "It was not possible to load the training samples from " \
            f"'{input_csv_train}'. Error: {e}"
        log.exception(errstr)
        sys.exit(errstr)

    # Inform the user that the data were successfully loaded.
    infostr = \
        "The training samples were successfully loaded from " \
        f"'{input_csv_train}'."
    log.info(infostr)

    #-----------------------------------------------------------------#

    # Try to load the test samples.
    try:

        df_test = \
            ioutil.load_samples(csv_file = input_csv_test,
                                sep = ",",
                                keep_samples_names = True,
                                split = False)

    # If something went wrong
    except Exception as e:

        # Warn the user and exit.
        errstr = \
            "It was not possible to load the test samples from " \
            f"'{input_csv_test}'. Error: {e}"
        log.exception(errstr)
        sys.exit(errstr)

    # Inform the user that the data were successfully loaded.
    infostr = \
        "The test samples were successfully loaded from " \
        f"'{input_csv_test}'."
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
            "It was not possible to move the DGD model to the " \
            f"'{device}' device. Error: {e}"
        log.exception(errstr)
        sys.exit(errstr)

    # Inform the user that the model was successfully moved.
    infostr = \
        f"The DGD model was successfully moved to the '{device}' " \
        "device."
    log.info(infostr)

    #-----------------------------------------------------------------#

    # Try to train the DGD model.
    try:

        df_rep_train, df_rep_test, df_loss, df_time = \
            dgd_model.train(df_train = df_train,
                            df_test = df_test,
                            config_train = config_train)

    # If something went wrong
    except Exception as e:

        # Warn the user and exit.
        errstr = \
            f"It was not possible to train the DGD model. Error: {e}"
        log.exception(errstr)
        sys.exit(errstr)

    # Inform the user that the DGD model was successfully trained.
    infostr = "The DGD model was successfully trained."
    log.info(infostr)

    #-----------------------------------------------------------------#

    # Set the path to the output file.
    output_csv_rep_train_path = os.path.join(wd, output_csv_rep_train)

    # Try to write the representations for the training samples in
    # the output CSV file.
    try:

        ioutil.save_representations(\
            df = df_rep_train,
            csv_file = output_csv_rep_train_path,
            sep = ",")

    # If something went wrong
    except Exception as e:

        # Warn the user and exit.
        errstr = \
            "It was not possible to write the representations " \
            "for the training samples in " \
            f"'{output_csv_rep_train_path}'. Error: {e}"
        log.exception(errstr)
        sys.exit(errstr)

    # Inform the user that the representations for the training
    # samples were successfully written in the output file.
    infostr = \
        "The representations for the training samples were " \
        f"successfully written in '{output_csv_rep_train_path}'."
    log.info(infostr)

    #-----------------------------------------------------------------#

    # Set the path to the output file.
    output_csv_rep_test_path = os.path.join(wd, output_csv_rep_test)

    # Try to write the representations for the test samples in the
    # output CSV file.
    try:

        ioutil.save_representations(\
            df = df_rep_test,
            csv_file = output_csv_rep_test_path,
            sep = ",")

    # If something went wrong
    except Exception as e:

        # Warn the user and exit.
        errstr = \
            "It was not possible to write the representations " \
            "for the test samples in " \
            f"'{output_csv_rep_test_path}'. Error: {e}"
        log.exception(errstr)
        sys.exit(errstr)

    # Inform the user that the representations for the test samples
    # were successfully written in the output file.
    infostr = \
        "The representations for the test samples were " \
        f"successfully written in '{output_csv_rep_test_path}'."
    log.info(infostr)

    #-----------------------------------------------------------------#

    # Set the path to the output file.
    output_csv_loss_path = os.path.join(wd, output_csv_loss)

    # Try to write the loss(es) in the dedicated CSV file.
    try:

        ioutil.save_loss(\
            df = df_loss,
            csv_file = output_csv_loss,
            sep = ",")

    # If something went wrong
    except Exception as e:

        # Warn the user and exit.
        errstr = \
            "It was not possible to write the loss(es) " \
            f"in '{output_csv_loss_path}'. Error: {e}"
        log.exception(errstr)
        sys.exit(errstr)

    # Inform the user that the loss(es) was successfully written in
    # the output file.
    infostr = \
        "The loss(es) was (were) successfully written in " \
        f"'{output_csv_loss_path}'."
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
