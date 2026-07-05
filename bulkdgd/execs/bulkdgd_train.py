#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-

#    bulkdgd_train.py
#
#    Train the :class:`core.model.BulkDGD`.
#
#    Copyright (C) 2026 Valentina Sora 
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
__doc__ = "Train the :class:`core.model.BulkDGD`."


#######################################################################


# Import from the standard library.
import argparse
import logging as log
import os
import sys

# Import from third-party libraries.
import pandas as pd
import torch

# Import from 'bulkdgd'.
from bulkdgd.core import model
from bulkdgd import defaults, ioutil
from . import util


#######################################################################


# Get the module's logger.
logger = log.getLogger(__name__)


#######################################################################


# Define a function to set up the parser.
def set_parser() -> argparse.ArgumentParser:

    # Create the argument parser.
    parser = \
        argparse.ArgumentParser(\
            description = __doc__,
            formatter_class = util.CustomHelpFormatter)

    #-----------------------------------------------------------------#

    # Create a group of arguments for the input files.
    input_group = \
        parser.add_argument_group(title = "Input files")

    # Create a group of arguments for the output files.
    output_group = \
        parser.add_argument_group(title = "Output files")

    # Create a group of argument for the run options.
    run_group = \
        parser.add_argument_group(title = "Run options")

    #-----------------------------------------------------------------#

    # Set a help message.
    is_help = \
        """The input CSV file containing a data frame with
        the gene expression data for the samples."""

    # Add the argument to the group.
    input_group.add_argument("-is", "--input-samples",
                             type = str,
                             required = True,
                             help = is_help)

    #-----------------------------------------------------------------#

    # Set a help message.
    it_help = \
        """The input plain text file containing a newline-separated
        list of sample names/indexes/IDs for the training samples."""

    # Add the argument to the group.
    input_group.add_argument("-it", "--input-train",
                             type = str,
                             required = True,
                             help = it_help)

    #-----------------------------------------------------------------#

    # Set a help message.
    ie_help = \
        """The input plain text file containing a newline-separated
        list of sample names/indexes/IDs for the test samples."""

    # Add the argument to the group.
    input_group.add_argument("-ie", "--input-test",
                             type = str,
                             required = True,
                             help = ie_help)

    #-----------------------------------------------------------------#

    # Set a help message.
    ilt_help = \
        """The input CSV file containing the labels for the
        training samples. The file should have two columns, one with
        the names/indexes/IDs of the samples and one with the
        corresponding labels. The file should not have a header. This
        argument is needed only if supervised clustering metrics are
        computed during training."""

    # Add the argument to the group.
    input_group.add_argument("-ilt", "--input-labels-train",
                             type = str,
                             help = ilt_help)

    #-----------------------------------------------------------------#

    # Set a help message.
    ile_help = \
        """The input CSV file containing the labels for the
        test samples. The file should have two columns, one with
        the names/indexes/IDs of the samples and one with the
        corresponding labels. The file should not have a header. This
        argument is needed only if supervised clustering metrics are
        computed during training."""

    # Add the argument to the group.
    input_group.add_argument("-ile", "--input-labels-test",
                             type = str,
                             help = ile_help)

    #-----------------------------------------------------------------#

    # Set the default directory where the configuration files are
    # located.
    icm_dir = defaults.CONFIG_DIRS["model"]

    # Set a help message.
    icm_help = \
        f"""The YAML configuration file specifying the model's
        parameters. If it is a name without an extension, it is assumed
        to be the name of a configuration file in '{icm_dir}'."""

    # Add the argument to the group.
    input_group.add_argument("-icm", "--input-config-file-model",
                             type = str,
                             required = True,
                             help = icm_help)

    #-----------------------------------------------------------------#

    # Set the default directory where the configuration files are
    # located.
    ict_dir = defaults.CONFIG_DIRS["training"]

    # Set a help message.
    ict_help = \
        f"""The YAML configuration file specifying the options for 
        training the model. If it is a name without an extension, it is
        assumed to be the name of a configuration file in
        '{ict_dir}'."""

    # Add the argument to the group.
    input_group.add_argument("-ict", "--input-config-file-train",
                             type = str,
                             required = True,
                             help = ict_help)

    #-----------------------------------------------------------------#

    # Set the default value for the argument.
    olat_default = "latent.pth"

    # Set a help message.
    olat_help = \
        f"""The output .pth file where the latent space's parameters 
        will be saved after training. By default, the file will be
        named '{olat_default}'."""
    
    # Add the argument to the group.
    output_group.add_argument("-olat", "--output-latent",
                              type = str,
                              default = olat_default,
                              help = olat_help)

    #-----------------------------------------------------------------#

    # Set the default value for the argument.
    odec_default = "decoder.pth"

    # Set a help message.
    odec_help = \
        f"""The output .pth file where the decoder's parameters will be
        saved after training. By default, the file will be named
        '{odec_default}'."""

    # Add the argument to the group.
    output_group.add_argument("-odec", "--output-decoder",
                              type = str,
                              default = odec_default,
                              help = odec_help)

    #-----------------------------------------------------------------#

    # Set the default value for the argument.
    ort_default = "representations_train.csv"

    # Set a help message.
    ort_help = \
        f"""The output CSV file containing the data frame
        with the representation of each training sample in latent
        space. The default file name is '{ort_default}'."""

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
        f"""The output CSV file containing the data frame with the
        representation of each test sample in latent space. The
        default file name is '{ore_default}'."""

    # Add the argument to the group.
    output_group.add_argument("-ore", "--output-rep-test",
                              type = str,
                              default = ore_default,
                              help = ore_help)

    #-----------------------------------------------------------------#
    
    # Set the default value for the argument.
    opmt_default = "pred_means_train.csv"

    # Set a help message.
    opmt_help = \
        f"""The output CSV file containing the data frame with the
        predicted mean for each gene in each training sample. The
        default file name is '{opmt_default}'."""

    # Add the argument to the group.
    output_group.add_argument("-opmt", "--output-pred-means-train",
                              type = str,
                              default = opmt_default,
                              help = opmt_help)

    #-----------------------------------------------------------------#

    # Set the default value for the argument.
    opme_default = "pred_means_test.csv"

    # Set a help message.
    opme_help = \
        f"""The output CSV file containing the data frame with the
        predicted mean for each gene in each test sample. The default
        file name is '{opme_default}'."""

    # Add the argument to the group.
    output_group.add_argument("-opme", "--output-pred-means-test",
                              type = str,
                              default = opme_default,
                              help = opme_help)

    #-----------------------------------------------------------------#

    # Set the default value for the argument.
    opv_default = "pred_r_values.csv"

    # Set a help message.
    opv_help = \
        f"""The output CSV file containing the data frame
        with the r-value for each gene. The default file name is
        '{opv_default}'. If the model's output module returns
        r-values per gene and per sample, the '-opvt',
        '--output-pred-rvalues-train' and '-opve',
        '--output-pred-rvalues-test' options should be used instead."""
    
    # Add the argument to the group.
    output_group.add_argument("-opv", "--output-pred-rvalues",
                              type = str,
                              default = opv_default,
                              help = opv_help)

    #-----------------------------------------------------------------#

    # Set the default value for the argument.
    opvt_default = "pred_r_values_train.csv"

    # Set a help message.
    opvt_help = \
        f"""The output CSV file containing the data frame
        with the r-value for each gene in each training sample. The
        default file name is '{opvt_default}'. This option will be
        used only if the model's output module returns r-values per
        gene and per sample."""

    # Add the argument to the group.
    output_group.add_argument("-opvt", "--output-pred-rvalues-train",
                              type = str,
                              default = opvt_default,
                              help = opvt_help)

    #-----------------------------------------------------------------#

    # Set the default value for the argument.
    opve_default = "pred_r_values_test.csv"

    # Set a help message.
    opve_help = \
        f"""The output CSV file containing the data frame
        with the r-value for each gene in each test sample. The
        default file name is '{opve_default}'. This option will be
        used only if the model's output module returns r-values per
        gene and per sample."""

    # Add the argument to the group.
    output_group.add_argument("-opve", "--output-pred-rvalues-test",
                              type = str,
                              default = opve_default,
                              help = opve_help)

    #-----------------------------------------------------------------#

    # Set the default value for the argument.
    ol_default = "loss.csv"

    # Set a help message.
    ol_help = \
        f"""The output CSV file containing the data frame
        with the per-epoch losses for training and test samples.
        The default file name is '{ol_default}'."""

    # Add the argument to the group.
    output_group.add_argument("-ol", "--output-loss",
                              type = str,
                              default = ol_default,
                              help = ol_help)

    #-----------------------------------------------------------------#

    # Set the default value for the argument.
    omt_default = "metrics_train.csv"

    # Set a help message.
    omt_help = \
        f"""The output CSV file containing the data frame with the
        per-epoch values of the GMM clustering metrics computed during
        training for the training samples. The default file name is
        '{omt_default}'. This file will be written only if at least one
        clustering metric is computed during training (they are
        specified in the '-ict', '--input-config-file-train'
        configuration file)."""
    
    # Add the argument to the group.
    output_group.add_argument("-omt", "--output-metrics-train",
                              type = str,
                              default = omt_default,
                              help = omt_help)

    #-----------------------------------------------------------------#

    # Set the default value for the argument.
    ome_test_default = "metrics_test.csv"

    # Set a help message.
    ome_test_help = \
        f"""The output CSV file containing the data frame with the
        per-epoch values of the GMM clustering metrics computed during
        training for the test samples. The default file name is
        '{ome_test_default}'. This file will be written only if at
        least one clustering metric is computed during training (they
        are specified in the '-ict', '--input-config-file-train'
        configuration file)."""

    # Add the argument to the group.
    output_group.add_argument("-ome", "--output-metrics-test",
                              type = str,
                              default = ome_test_default,
                              help = ome_test_help)

    #-----------------------------------------------------------------#

    # Set the default value for the argument.
    ot_default = "train_time.csv"

    # Set a help message.
    ot_help = \
        f"""The output CSV file containing the data frame
        with information about the CPU and wall clock time
        spent for each training epoch. The default file name is
        '{ot_default}'."""

    # Add the argument to the group.
    output_group.add_argument("-ot", "--output-time",
                              type = str,
                              default = ot_default,
                              help = ot_help)

    #-----------------------------------------------------------------#

    # Set a help message.
    dev_help = \
        """The device to use. If not provided, the GPU will be used
        if it is available."""

    # Add the argument to the group.
    run_group.add_argument("-dev", "--device",
                           type = str,
                           help = dev_help)

    #-----------------------------------------------------------------#

    # Add the working directory and logging arguments.
    util.add_wd_and_logging_arguments(\
        parser = parser,
        command_name = "bulkdgd_train")

    #-----------------------------------------------------------------#

    # Return the parser.
    return parser


#######################################################################


# Define the 'main' function.
def main(args: argparse.Namespace) -> None:

    # Get the argument corresponding to the working directory.
    wd = args.work_dir

    # Get the arguments corresponding to the input files.
    input_samples = args.input_samples
    input_train = args.input_train
    input_test = args.input_test
    input_labels_train = args.input_labels_train
    input_labels_test = args.input_labels_test

    # Get the arguments corresponding to the configuration files.
    input_config_file_model = args.input_config_file_model
    input_config_file_train = args.input_config_file_train

    # Get the arguments corresponding to the output files.
    output_latent = os.path.join(wd, args.output_latent)
    output_decoder = os.path.join(wd, args.output_decoder)
    output_rep_train = os.path.join(wd, args.output_rep_train)
    output_rep_test = os.path.join(wd, args.output_rep_test)
    output_pred_means_train = \
        os.path.join(wd, args.output_pred_means_train)
    output_pred_means_test = \
        os.path.join(wd, args.output_pred_means_test)
    output_pred_r_values = \
        os.path.join(wd, args.output_pred_rvalues)
    output_pred_r_values_train = \
        os.path.join(wd, args.output_pred_rvalues_train)
    output_pred_r_values_test = \
        os.path.join(wd, args.output_pred_rvalues_test)
    output_loss = os.path.join(wd, args.output_loss)
    output_metrics_train = os.path.join(wd, args.output_metrics_train)
    output_metrics_test = os.path.join(wd, args.output_metrics_test)
    output_time = os.path.join(wd, args.output_time)

    # Get the arguments corresponding to the run options.
    device = args.device

    #-----------------------------------------------------------------#

    # Try to load the configuration.
    try:

        config_model = \
            ioutil.load_config_model(input_config_file_model)

    # If something went wrong
    except Exception as e:

        # Warn the user and exit.
        errstr = \
            "It was not possible to load the configuration from " \
            f"'{input_config_file_model}'. Error: {e}"
        logger.exception(errstr)
        sys.exit(errstr)

    # Inform the user that the configuration was successfully loaded.
    infostr = \
        "The configuration was successfully loaded from " \
        f"'{input_config_file_model}'."
    logger.info(infostr)

    #-----------------------------------------------------------------#

    # Try to load the configuration.
    try:

        config_train = \
            ioutil.load_config_train(input_config_file_train)

    # If something went wrong
    except Exception as e:

        # Warn the user and exit.
        errstr = \
            "It was not possible to load the configuration from " \
            f"'{input_config_file_train}'. Error: {e}"
        logger.exception(errstr)
        sys.exit(errstr)

    # Inform the user that the configuration was successfully loaded.
    infostr = \
        "The configuration was successfully loaded from " \
        f"'{input_config_file_train}'."
    logger.info(infostr)

    #-----------------------------------------------------------------#

    # Try to load the samples.
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

    # Inform the user that the samples were successfully loaded.
    infostr = \
        "The samples were successfully loaded from " \
        f"'{input_samples}'."
    logger.info(infostr)

    #-----------------------------------------------------------------#

    # Try to load the names/indexes/IDs for the training samples.
    try:

        names_train = \
            [line.strip() for line in open(input_train, "r") \
                if line.strip() != "" and not line.startswith("#")]

    # If something went wrong
    except Exception as e:

        # Warn the user and exit.
        errstr = \
            "It was not possible to load the names/indexes/IDs for " \
            f"the training samples from '{input_train}'. Error: {e}"
        logger.exception(errstr)
        sys.exit(errstr)

    # Inform the user that the data were successfully loaded.
    infostr = \
        "The names/indexes/IDs for the training samples were " \
        f"successfully loaded from '{input_train}'."
    logger.info(infostr)

    #-----------------------------------------------------------------#

    # Try to load the names/indexes/IDs for the test samples.
    try:

        names_test = \
            [line.strip() for line in open(input_test, "r") \
                if line.strip() != "" and not line.startswith("#")]
    
    # If something went wrong
    except Exception as e:

        # Warn the user and exit.
        errstr = \
            "It was not possible to load the names/indexes/IDs for " \
            f"the test samples from '{input_test}'. Error: {e}"
        logger.exception(errstr)
        sys.exit(errstr)
    
    # Inform the user that the data were successfully loaded.
    infostr = \
        "The names/indexes/IDs for the test samples were " \
        f"successfully loaded from '{input_test}'."
    logger.info(infostr)

    #-----------------------------------------------------------------#

    # Initialize the labels for the training and test samples to None.
    labels_train = None

    # If there are labels for the training samples
    if input_labels_train is not None:

        # Try to load the labels for the training samples.     
        try:

            # Load the labels.
            df_labels_train = \
                pd.read_csv(input_labels_train,
                            header = None,
                            names = ["sample", "label"])
            
        # If something went wrong
        except Exception as e:

            # Warn the user and exit.
            errstr = \
                "It was not possible to load the labels for the " \
                f"training samples from '{input_labels_train}'. " \
                f"Error: {e}"
            logger.exception(errstr)
            sys.exit(errstr)
        
        # Inform the user that the labels were successfully loaded.
        infostr = \
            "The labels for the training samples were successfully " \
            f"loaded from '{input_labels_train}'."
        
        # Convert the data frame into a dictionary mapping each sample
        # to the corresponding label.
        dict_labels_train = dict(zip(df_labels_train["sample"],
                                    df_labels_train["label"]))
        
        # The final labels will be the ones corresponding to the
        # training samples, in the same order.
        labels_train = \
            [dict_labels_train[name] for name in names_train]

    #-----------------------------------------------------------------#

    # Initialize the labels for the test samples to None.
    labels_test = None

    # If there are labels for the test samples
    if input_labels_test is not None:

        # Try to load the labels for the test samples.
        try:

            # Load the labels.
            df_labels_test = \
                pd.read_csv(input_labels_test,
                            header = None,
                            names = ["sample", "label"])
        
        # If something went wrong
        except Exception as e:

            # Warn the user and exit.
            errstr = \
                "It was not possible to load the labels for the " \
                f"test samples from '{input_labels_test}'. Error: {e}"
            logger.exception(errstr)
            sys.exit(errstr)
        
        # Inform the user that the labels were successfully loaded.
        infostr = \
            "The labels for the test samples were successfully " \
            f"loaded from '{input_labels_test}'."
        
        # Convert the data frame into a dictionary mapping each sample
        # to the corresponding label.
        dict_labels_test = dict(zip(df_labels_test["sample"],
                                    df_labels_test["label"]))
        
        # The final labels will be the ones corresponding to the test
        # samples, in the same order.
        labels_test = [dict_labels_test[name] for name in names_test]

    #-----------------------------------------------------------------#

    # Try to set the model.
    try:
        
        dgd_model = model.BulkDGD(**config_model)

    # If something went wrong
    except Exception as e:

        # Warn the user and exit.
        errstr = \
            f"It was not possible to set the BulkDGD model. Error: {e}"
        logger.exception(errstr)
        sys.exit(errstr)

    # Inform the user that the model was successfully set.
    infostr = "The BulkDGD model was successfully set."
    logger.info(infostr)

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
            "It was not possible to move the BulkDGD model to the " \
            f"'{device}' device. Error: {e}"
        logger.exception(errstr)
        sys.exit(errstr)

    # Inform the user that the model was successfully moved.
    infostr = \
        "The BulkDGD model was successfully moved to the " \
        f"'{device}' device."
    logger.info(infostr)

    #-----------------------------------------------------------------#

    # Try to train the model.
    try:

        dfs_reps, dfs_pred_means, dfs_pred_r_values, \
            df_loss, dfs_metrics, df_time = \
            dgd_model.train(df_samples = df_samples,
                            names_train = names_train,
                            names_test = names_test,
                            config_train = config_train,
                            latent_pth_file = output_latent,
                            decoder_pth_file = output_decoder,
                            labels_train = labels_train,
                            labels_test = labels_test)

        # If there are any metrics' data frames
        if dfs_metrics is not None:

            # Unpack them.
            df_metrics_train, df_metrics_test = dfs_metrics

        # Otherwise
        else:

            # Set the metrics' data frames to None.
            df_metrics_train, df_metrics_test = None, None

    # If something went wrong
    except Exception as e:

        # Warn the user and exit.
        errstr = \
            "It was not possible to train the BulkDGD model. " \
            f"Error: {e}"
        logger.exception(errstr)
        sys.exit(errstr)

    # Inform the user that the model was successfully trained.
    infostr = "The BulkDGD model was successfully trained."
    logger.info(infostr)

    #-----------------------------------------------------------------#

    # Try to write the representations for the training samples in
    # the output CSV file.
    try:

        ioutil.save_representations(\
            df = dfs_reps[0],
            csv_file = output_rep_train,
            sep = ",")

    # If something went wrong
    except Exception as e:

        # Warn the user and exit.
        errstr = \
            "It was not possible to write the representations " \
            f"for the training samples in '{output_rep_train}'. " \
            f"Error: {e}"
        logger.exception(errstr)
        sys.exit(errstr)

    # Inform the user that the representations for the training
    # samples were successfully written in the output file.
    infostr = \
        "The representations for the training samples were " \
        f"successfully written in '{output_rep_train}'."
    logger.info(infostr)

    #-----------------------------------------------------------------#

    # Try to write the representations for the test samples in the
    # output CSV file.
    try:

        ioutil.save_representations(\
            df = dfs_reps[1],
            csv_file = output_rep_test,
            sep = ",")

    # If something went wrong
    except Exception as e:

        # Warn the user and exit.
        errstr = \
            "It was not possible to write the representations " \
            f"for the test samples in '{output_rep_test}'. " \
            f"Error: {e}"
        logger.exception(errstr)
        sys.exit(errstr)

    # Inform the user that the representations for the test samples
    # were successfully written in the output file.
    infostr = \
        "The representations for the test samples were " \
        f"successfully written in '{output_rep_test}'."
    logger.info(infostr)

    #-----------------------------------------------------------------#

    # Try to write the predicted means for the training samples in
    # the output CSV file.
    try:

        ioutil.save_decoder_outputs(\
            df = dfs_pred_means[0],
            csv_file = output_pred_means_train,
            sep = ",")

    # If something went wrong
    except Exception as e:

        # Warn the user and exit.
        errstr = \
            "It was not possible to write the predicted means " \
            f"for the training samples in " \
            f"'{output_pred_means_train}'. Error: {e}"
        logger.exception(errstr)
        sys.exit(errstr)
    
    # Inform the user that the predicted means for the training
    # samples were successfully written in the output file.
    infostr = \
        "The predicted means for the training samples were " \
        f"successfully written in '{output_pred_means_train}'."
    logger.info(infostr)

    #-----------------------------------------------------------------#

    # Try to write the predicted means for the test samples in the
    # output CSV file.
    try:

        ioutil.save_decoder_outputs(\
            df = dfs_pred_means[1],
            csv_file = output_pred_means_test,
            sep = ",")
        
    # If something went wrong
    except Exception as e:
        
        # Warn the user and exit.
        errstr = \
            "It was not possible to write the predicted means " \
            f"for the test samples in '{output_pred_means_test}'. " \
            f"Error: {e}"
        logger.exception(errstr)
        sys.exit(errstr)
    
    # Inform the user that the predicted means for the test
    # samples were successfully written in the output file.
    infostr = \
        "The predicted means for the test samples were " \
        f"successfully written in '{output_pred_means_test}'."
    logger.info(infostr)

    #-----------------------------------------------------------------#

    # If thre are r-values to write
    if dfs_pred_r_values is not None:
        
        # If there is only one data frame
        if len(dfs_pred_r_values) == 1:

            # Try to write the r-values for each gene in the output
            # CSV file.
            try:

                ioutil.save_decoder_outputs(\
                    df = dfs_pred_r_values[0],
                    csv_file = output_pred_r_values,
                    sep = ",")

            # If something went wrong
            except Exception as e:

                # Warn the user and exit.
                errstr = \
                    "It was not possible to write the r-values " \
                    f"for each gene in '{output_pred_r_values}'. " \
                    f"Error: {e}"
                logger.exception(errstr)
                sys.exit(errstr)
            
            # Inform the user that the r-values for each gene
            # were successfully written in the output file.
            infostr = \
                "The r-values for each gene were successfully " \
                f"written in '{output_pred_r_values}'."
            logger.info(infostr)
        
        # If there are two data frames
        elif len(dfs_pred_r_values) == 2:

            # Try to write the r-values for each gene in each
            # training sample in the output CSV file.
            try:

                ioutil.save_decoder_outputs(\
                    df = dfs_pred_r_values[0],
                    csv_file = output_pred_r_values_train,
                    sep = ",")

            # If something went wrong
            except Exception as e:

                # Warn the user and exit.
                errstr = \
                    "It was not possible to write the r-values " \
                    f"for each gene in each training sample in " \
                    f"'{output_pred_r_values_train}'. Error: {e}"
                logger.exception(errstr)
                sys.exit(errstr)
            
            # Inform the user that the r-values for each gene
            # in each training sample were successfully
            # written in the output file.
            infostr = \
                "The r-values for each gene in each training " \
                f"sample were successfully written in " \
                f"'{output_pred_r_values_train}'."
            logger.info(infostr)
        
            # Try to write the r-values for each gene in each
            # test sample in the output CSV file.
            try:

                ioutil.save_decoder_outputs(\
                    df = dfs_pred_r_values[1],
                    csv_file = output_pred_r_values_test,
                    sep = ",")
            
            # If something went wrong
            except Exception as e:

                # Warn the user and exit.
                errstr = \
                    "It was not possible to write the r-values " \
                    f"for each gene in each test sample in " \
                    f"'{output_pred_r_values_test}'. Error: {e}"
                logger.exception(errstr)
                sys.exit(errstr)
            
            # Inform the user that the r-values for each gene
            # in each test sample were successfully written
            # in the output file.
            infostr = \
                "The r-values for each gene in each test " \
                f"sample were successfully written in " \
                f"'{output_pred_r_values_test}'."
            logger.info(infostr)

    #-----------------------------------------------------------------#

    # Try to write the losses in the dedicated CSV file.
    try:

        # Save the loss(es).
        df_loss.to_csv(output_loss,
                       sep = ",",
                       index = False,
                       header = True)

    # If something went wrong
    except Exception as e:

        # Warn the user and exit.
        errstr = \
            "It was not possible to write the losses " \
            f"in '{output_loss}'. Error: {e}"
        logger.exception(errstr)
        sys.exit(errstr)

    # Inform the user that the losses were successfully written in
    # the output file.
    infostr = \
        f"The losses were successfully written in '{output_loss}'."
    logger.info(infostr)

    #-----------------------------------------------------------------#

    # If at least one clustering metric was computed during training
    if df_metrics_train is not None and df_metrics_test is not None:

        # Try to write the metrics for the training samples in the
        # dedicated CSV file.
        try:

            df_metrics_train.to_csv(output_metrics_train,
                                    sep = ",",
                                    index = False,
                                    header = True)

        # If something went wrong
        except Exception as e:

            # Warn the user and exit.
            errstr = \
                "It was not possible to write the metrics for the " \
                f"training samples in '{output_metrics_train}'. " \
                f"Error: {e}"
            logger.exception(errstr)
            sys.exit(errstr)

        # Inform the user that the metrics for the training samples
        # were successfully written in the output file.
        infostr = \
            "The metrics for the training samples were successfully " \
            f"written in '{output_metrics_train}'."
        logger.info(infostr)

        # Try to write the metrics for the test samples in the
        # dedicated CSV file.
        try:

            df_metrics_test.to_csv(output_metrics_test,
                                   sep = ",",
                                   index = False,
                                   header = True)

        # If something went wrong
        except Exception as e:

            # Warn the user and exit.
            errstr = \
                "It was not possible to write the metrics for the " \
                f"test samples in '{output_metrics_test}'. " \
                f"Error: {e}"
            logger.exception(errstr)
            sys.exit(errstr)

        # Inform the user that the metrics for the test samples
        # were successfully written in the output file.
        infostr = \
            "The metrics for the test samples were successfully " \
            f"written in '{output_metrics_test}'."
        logger.info(infostr) 

    #-----------------------------------------------------------------#

    # Try to write the time data in the dedicated CSV file.
    try:

        df_time.to_csv(output_time,
                       sep = ",",
                       index = False,
                       header = True)

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


#######################################################################


# Define the entry point for the standalone executable.
def entry_point() -> None:

    # Build the parser.
    parser = set_parser()

    # Parse the arguments.
    args = parser.parse_args()

    # Set up the logging.
    util.set_main_logging(args = args)

    # Check if the execution should be parallelized.
    if getattr(args, "parallelize", False):

        # Run with parallelization.
        util.run_with_parallelization(\
            executable = "bulkdgd_train",
            args = args)

    # Otherwise
    else:

        # Run the main function.
        main(args = args)
