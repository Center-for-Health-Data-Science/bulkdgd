#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-


# Import the 'logging' module
import logging as log
# Import the 'model' module
from bulkDGD.core import model
# Import the 'dea' and 'ioutil' modules
from bulkDGD.analysis import dea
from bulkDGD import ioutil


#------------------------------ Logging ------------------------------#


# Set the logging options so that every message
# above and including the INFO level is reported
log.basicConfig(level = "INFO")


#------------------------- Load the samples --------------------------#


# Load the preprocessed samples into a data frame
df_samples = \
    ioutil.load_samples(# The CSV file where the samples are stored
                        csv_file = "samples_preprocessed.csv",
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

# Get only the first 10 rows
df_samples = df_samples.iloc[:10,:]


#--------------------- Load the decoder outputs ----------------------#


# Load the decoder outputs into a data frame
df_dec_out = \
   ioutil.load_decoder_outputs(# The CSV file where the decoder outputs
                               # are stored
                               csv_file = "decoder_outputs.csv",
                               # The field separator used in the CSV
                               # file
                               sep = ",",
                               # Whether to split the input data frame
                               # into two data frame, one containing
                               # only the decoder outputs and the other
                               # containing additional information
                               # about the original samples
                               split = False)

# Get only the first ten rows
df_dec_out = df_dec_out.iloc[:10,:]


#---------------------- Load the configuration -----------------------#


# Load the model's configuration
config_model = ioutil.load_config_model("model.yaml")


#----------------------- Get the trained model -----------------------#


# Get the trained DGD model (Gaussian mixture model
# and decoder)
dgd_model = model.DGDModel(**config_model)


#----------------- Differential expression analysis ------------------#


# Get the r-values
r_values = dgd_model.r_values

# For each sample
for sample in df_samples.index:

    # Perform differential expression analysis
    dea_results, _ = \
        dea.perform_dea(# The observed gene counts for the current
                        # sample
                        obs_counts = df_samples.loc[sample,:],
                        # The predicted means - decoder outputs for
                        # the current sample
                        pred_means = df_dec_out.loc[sample,:],
                        # Which statistics should be computed and
                        # included in the results
                        statistics = \
                            ["p_values", "q_values",
                             "log2_fold_changes"],
                        # The r-values of the negative binomials
                        r_values = r_values,
                        # The resolution for the p-values calculation
                        # (the higher, the more accurate the
                        # calculation; set to 'None' for an exact
                        # calculation)
                        resolution = 1e5,
                        # The family-wise error rate for the
                        # calculation of the q-values
                        alpha = 0.05,
                        # The method used to calculate the q-values
                        method = "fdr_bh")

    # Save the results
    dea_results.to_csv(# The CSV file where to save the results
                       # for the current sample
                       f"dea_sample_{sample}.csv",
                       # The field separator to use in the output
                       # CSV file
                       sep = ",",
                       # Whether to keep the rows' names
                       index = True,
                       # Whether to keep the columns' names
                       header = True)