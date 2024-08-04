#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-


#######################################################################


# Import the 'logging' module.
import logging as log
# Import 'torch'.
import torch
# Import the 'model' module.
from bulkDGD.core import model
# Import the 'ioutil' module.
from bulkDGD import ioutil


#######################################################################


# Set the logging options so that every message of level INFO or above
# is emitted.
log.basicConfig(level = "INFO")


#--------------- Load the configuration for the model ----------------#


# Load the model's configuration.
config_model = ioutil.load_config_model("model_untrained.yaml")


#------------------------- Create the model --------------------------#


# Create the DGD model (Gaussian mixture model and decoder).
dgd_model = model.DGDModel(**config_model)

# If a CPU with CUDA is available.
if torch.cuda.is_available():

    # Set the GPU as the device.
    device = torch.device("cuda")

# Otherwise
else:

    # Set the CPU as the device.
    device = torch.device("cpu")

# Move the model to the device.
dgd_model.device = device

#------------------------- Load the samples --------------------------#


# Load the training samples into a data frame.
df_train_raw = \
    ioutil.load_samples(# The CSV file where the samples are stored
                        csv_file = "samples_train.csv",
                        # The field separator used in the CSV file
                        sep = ",",
                        # Whether to keep the original samples' names/
                        # indexes (if True, they are assumed to be in
                        # the first column of the data frame 
                        keep_samples_names = True,
                        # Whether to split the input data frame into
                        # two data frames, one containing only gene
                        # expression data and the other containing
                        # additional information about the samples
                        split = False)

# Load the test samples into a data frame.
df_test_raw = \
    ioutil.load_samples(# The CSV file where the samples are stored
                        csv_file = "samples_test.csv",
                        # The field separator used in the CSV file
                        sep = ",",
                        # Whether to keep the original samples' names/
                        # indexes (if True, they are assumed to be in
                        # the first column of the data frame 
                        keep_samples_names = True,
                        # Whether to split the input data frame into
                        # two data frames, one containing only gene
                        # expression data and the other containing
                        # additional information about the samples
                        split = False)


#---------------------- Preprocess the samples -----------------------#


# Preprocess the training samples.
df_train, genes_excluded_train, genes_missing_train = \
    ioutil.preprocess_samples(df_samples = df_train_raw,
                              genes_txt_file = "custom_genes.txt")

# Preprocess the test samples.
df_test, genes_excluded_test, genes_missing_test = \
    ioutil.preprocess_samples(df_samples = df_test_raw,
                              genes_txt_file = "custom_genes.txt")


#---------------- Load the configuration for training ----------------#


# Load the configuration for training the model.
config_train = ioutil.load_config_train("training")


#-------------------------- Train the model --------------------------#


# Train the DGD model.
df_rep_train, df_rep_test, df_loss, df_time = \
    dgd_model.train(df_train = df_train,
                    df_test = df_test,
                    config_train = config_train)


#------------------------- Save the outputs --------------------------#


# Save the preprocessed training samples.
ioutil.save_samples(\
   # The data frame containing the samples
   df = df_train,
   # The output CSV file
   csv_file = "samples_preprocessed_train.csv",
   # The field separator in the output CSV file
   sep = ",")

# Save the preprocessed test samples.
ioutil.save_samples(\
   # The data frame containing the samples
   df = df_test,
   # The output CSV file
   csv_file = "samples_preprocessed_test.csv",
   # The field separator in the output CSV file
   sep = ",")

# Save the representations for the training samples.
ioutil.save_representations(\
    # The data frame containing the representations
    df = df_rep_train,
    # The output CSV file
    csv_file = "representations_train.csv",
    # The field separator in the output CSV file
    sep = ",")

# Save the representations for the test samples.
ioutil.save_representations(\
    # The data frame containing the representations
    df = df_rep_test,
    # The output CSV file
    csv_file = "representations_test.csv",
    # The field separator in the output CSV file
    sep = ",")

# Save the losses.
ioutil.save_loss(\
    # The data frame containing the losses
    df = df_loss,
    # The output CSV file
    csv_file = "loss.csv",
    # The field separator in the output CSV file
    sep = ",")

# Save the time data.
ioutil.save_time(\
    # The data frame containing the time data
    df = df_time,
    # The output CSV file
    csv_file = "train_time.csv",
    # The field separator in the output CSV file
    sep = ",")
