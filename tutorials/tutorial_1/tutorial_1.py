#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-


# Import the 'logging' module
import logging as log
# Import Pandas
import pandas as pd
# Import the 'bulkDGD.utils.dgd' module
import bulkDGD.utils.dgd as dgd
# Import the 'bulkDGD.utils.misc' module
from bulkDGD.utils import misc


#------------------------------ Logging ------------------------------#


# Set the logging options so that every message
# above and including the INFO level is reported
log.basicConfig(level = "INFO")


#---------------------- Preprocess the samples -----------------------#


# Load the samples into a data frame
df_samples = pd.read_csv(# Name of/path to the CSV file
                         "samples.csv",
                         # Column separator used in the file
                         sep = ",",
                         # Name or numerical index of the column
                         # containing the samples' names/IDs/indexes)
                         index_col = 0,
                         # Name or numerical index of the row
                         # containing the columns' names
                         header = 0)

# Preprocess the samples
df_preproc, genes_excluded, genes_missing = \
    dgd.preprocess_samples(df_samples = df_samples)


#---------------------- Load the configurations ----------------------#


# Load the model's configuration
config_model = misc.get_config_model("config_model.yaml")

# Load the configuration with the options to configure
# the search for the best representations
config_rep = misc.get_config_rep("config_rep.yaml")


#----------------------- Get the trained model -----------------------#


# Get the trained DGD model (Gaussian mixture model
# and decoder)
gmm, dec = dgd.get_model(config_gmm = config_model["gmm"],
                         config_dec = config_model["dec"])


#---------------------- Get the representations ----------------------#


# Get the representations and the corresponding decoder outputs
df_rep, df_dec_out = \
    dgd.get_representations(\
        # The data frame containing the preprocessed samples
        df = df_preproc,
        # The trained Gaussian mixture model
        gmm = gmm,
        # The trained decoder
        dec = dec,
        # How many representations to initialize per component
        # of the Gaussian mixture model per sample
        n_rep_per_comp = config_rep["rep"]["n_rep_per_comp"],
        # The configuration to load the samples
        config_data = config_rep["data"],
        # The configuration for the first optimization
        config_opt1 = config_rep["rep"]["opt1"],
        # The configuration for the second optimization
        config_opt2 = config_rep["rep"]["opt2"])

# Save the representations
dgd.save_representations(\
    # The data frame containing the representations
    df = df_rep,
    # The output CSV file
    csv_file = "representations.csv",
    # The field separator in the output CSV file
    sep = ",")

# Save the decoder outputs
dgd.save_decoder_outputs(\
    # The data frame containing the decoder outputs
    df = df_dec_out,
    # The output CSV file
    csv_file = "decoder_outputs.csv",
    # The field separator in the output CSV file
    sep = ",")