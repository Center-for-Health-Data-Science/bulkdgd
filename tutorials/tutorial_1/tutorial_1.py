#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-


#######################################################################


# Import the 'logging' module.
import logging as log
# Import the 'util' module.
from bulkDGD import util
# Import the 'model' module.
from bulkDGD.core import model
# Import the 'ioutil' module.
from bulkDGD import ioutil


#######################################################################


# Set the logging options so that every message of level INFO or above
# is emitted.
log.basicConfig(level = "INFO")


#---------------------- Preprocess the samples -----------------------#


# Load the samples into a data frame.
df_samples = \
    ioutil.load_samples(# The CSV file where the samples are stored
                        csv_file = "samples.csv",
                        # The field separator used in the CSV file
                        sep = ",",
                        # Whether to keep the original samples' names/
                        # indexes (if True, they are assumed to be in
                        # the first column of the data frame)
                        keep_samples_names = True,
                        # Whether to split the input data frame into
                        # two data frames, one containing only gene
                        # expression data and the other containing
                        # additional information about the samples                  
                        split = False)

# Pre-process the samples.
df_preproc, genes_excluded, genes_missing = \
    ioutil.preprocess_samples(df_samples = df_samples)


#---------------------- Load the configurations ----------------------#


# Load the model's configuration.
config_model = ioutil.load_config_model("model")

# Check the configuration.
config_model = util.check_config_model(config = config_model)

# Load the configuration with the options to configure the rounds of
# optimization when searching for the best representations.
config_rep = ioutil.load_config_rep("two_opt")

# Check the configuration.
config_rep = util.check_config_rep(config = config_rep)


#----------------------- Get the trained model -----------------------#


# Get the trained bulkDGD model (Gaussian mixture model and decoder).
dgd_model = model.BulkDGDModel(**config_model)


#---------------------- Get the representations ----------------------#


# Get the representations, the predicted scaled means and
# r-values of the negative binomials modeling the genes' counts for
# the in silico samples corresponding to the representations found,
# and the time spent finding the representations.
df_rep, df_pred_means, df_pred_r_values, df_time_opt = \
    dgd_model.get_representations(\
        # The data frame with the samples
        df_samples = df_samples,
        # The configuration to find the representations                        
        config_rep = config_rep)


#------------------------- Save the outputs --------------------------#


# Save the pre-processed samples.
ioutil.save_samples(\
   # The data frame containing the samples
   df = df_preproc,
   # The output CSV file
   csv_file = "samples_preprocessed.csv",
   # The field separator in the output CSV file
   sep = ",")

# Save the representations.
ioutil.save_representations(\
    # The data frame containing the representations
    df = df_rep,
    # The output CSV file
    csv_file = "representations.csv",
    # The field separator in the output CSV file
    sep = ",")

# Save the predicted scaled means.
ioutil.save_decoder_outputs(\
    # The data frame containing the predicted scaled means
    df = df_pred_means,
    # The output CSV file
    csv_file = "pred_means.csv",
    # The field separator in the output CSV file
    sep = ",")

# Save the predicted r-values.
ioutil.save_decoder_outputs(\
    # The data frame containing the predicted r-values
    df = df_pred_r_values,
    # The output CSV file
    csv_file = "pred_r_values.csv",
    # The field separator in the output CSV file
    sep = ",")

# Save the time data.
ioutil.save_time(\
    # The data frame containing the time data
    df = df_time_opt,
    # The output CSV file
    csv_file = "time_opt.csv",
    # The field separator in the output CSV file
    sep = ",")
