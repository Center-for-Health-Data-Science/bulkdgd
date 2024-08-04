#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-


#######################################################################


# Import the 'logging' module
import logging as log
# Import the 'core.model' module
from bulkDGD.core import model
# Import the 'analysis.dea' module
from bulkDGD.analysis import dea
# Import the 'ioutil' package
from bulkDGD import ioutil


#######################################################################


# Set the logging options so that every message of level INFO or above
# is emitted.
log.basicConfig(level = "INFO")


#------------------------- Load the samples --------------------------#


# Load the preprocessed samples into a data frame.
df_samples = \
    ioutil.load_samples(# The CSV file where the samples are stored
                        csv_file = "samples_preprocessed.csv",
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

# Get only the first ten rows.
df_samples = df_samples.iloc[:10,:]


#----------------- Load the predicted scaled means -------------------#


# Load the predicted scaled means into a data frame.
df_pred_means = \
   ioutil.load_decoder_outputs(# The CSV file where the predicted
                               # scaled means are stored
                               csv_file = "pred_means.csv",
                               # The field separator used in the CSV
                               # file
                               sep = ",",
                               # Whether to split the input data frame
                               # into two data frames, one containing
                               # only the predicted scaled means and
                               # the other containing additional
                               # information about the original samples
                               split = False)

# Get only the first ten rows.
df_pred_means = df_pred_means.iloc[:10,:]


#-------------------- Load the predicted r-values --------------------#


# Load the predicted r-values into a data frame.
df_pred_r_values = \
   ioutil.load_decoder_outputs(# The CSV file where the predicted
                               # r-values are stored
                               csv_file = "pred_r_values.csv",
                               # The field separator used in the CSV
                               # file
                               sep = ",",
                               # Whether to split the input data frame
                               # into two data frames, one containing
                               # only the predicted r-values and
                               # the other containing additional
                               # information about the original samples
                               split = False)

# Get only the first ten rows.
df_pred_r_values = df_pred_r_values.iloc[:10,:]


#----------------- Differential expression analysis ------------------#


# For each sample
for sample in df_samples.index:

    # Perform differential expression analysis.
    dea_results, _ = \
        dea.perform_dea(# The observed gene counts for the current
                        # sample
                        obs_counts = df_samples.loc[sample,:],
                        # The predicted scaled means for the current
                        # sample
                        pred_means = df_pred_means.loc[sample,:],
                        # The r-values for the current sample
                        r_values = df_pred_r_values.loc[sample,:],
                        # Which statistics should be computed and
                        # included in the results
                        statistics = \
                            ["p_values", "q_values",
                             "log2_fold_changes"],
                        # The resolution for the p-values calculation
                        # (the higher, the more accurate the
                        # calculation; set to 'None' for an exact
                        # calculation)
                        resolution = 1e4,
                        # The family-wise error rate for the
                        # calculation of the q-values
                        alpha = 0.05,
                        # The method used to calculate the q-values
                        method = "fdr_bh")

    # Save the results.
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
