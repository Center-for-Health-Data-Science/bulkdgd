#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-

#    bulkdgd_dea.py
#
#    Perform a differential expression analysis comparing experimental
#    samples to their "closest normal" sample found in latent space
#    by the :class:`core.model.BulkDGDModel`.
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
    "Perform a differential expression analysis comparing " \
    "experimental samples to their 'closest normal' samples " \
    "found in latent space by the :class:`core.model.BulkDGDModel`."


#######################################################################


# Import from the standard library.
import logging as log
import os
import sys
# Import from third-party packages.
from distributed import LocalCluster, Client, as_completed
# Import from 'bulkDGD'.
from bulkDGD.analysis import dea
from bulkDGD import ioutil
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
            name = "dea",
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

    # Create a group of arguments for the DEA options.
    dea_group = \
        parser.add_argument_group(title = "DEA options")

    # Create a group of arguments for the run options.
    run_group = \
        parser.add_argument_group(title = "Run options")

    #-----------------------------------------------------------------#

    # Set a help message.
    is_help = \
        "The input CSV file containing a data frame with " \
        "the gene expression data for the samples."

    # Add the argument to the group.
    input_group.add_argument("-is", "--input-samples",
                             type = str,
                             required = True,
                             help = is_help)

    #-----------------------------------------------------------------#

    # Set a help message.
    im_help = \
        "The input CSV file containing the data frame with the " \
        "predicted means of the distributions used to model the " \
        "genes' counts for each in silico control sample."

    # Add the argument to the group.
    input_group.add_argument("-im", "--input-means",
                             type = str,
                             required = True,
                             help = im_help)

    #-----------------------------------------------------------------#

    # Set a help message.
    iv_help = \
        "The input CSV file containing the data frame with the " \
        "predicted r-values of the negative binomial distributions " \
        "for each in silico control sample, if negative binomial " \
        "distributions were used to model the genes' counts."

    # Add the argument to the group.
    input_group.add_argument("-iv", "--input-rvalues",
                             type = str,
                             default = None,
                             help = iv_help)

    #-----------------------------------------------------------------#

    # Set the default value for the argument.
    op_default = "dea_"

    # Set a help message.
    op_help = \
        "The prefix of the output CSV file(s) that will contain " \
        "the results of the differential expression analysis. " \
        "Since the analysis will be performed for each sample, " \
        "one file per sample will be created. The files' names " \
        "will have the form {output_csv_prefix}{sample_name}.csv. " \
        f"The default prefix is '{op_default}'."

    # Add the argument to the group.
    output_group.add_argument("-op", "--output-prefix",
                              type = str,
                              default = op_default,
                              help = op_help)

    #-----------------------------------------------------------------#

    # Set the default value for the argument.
    pr_default = 1e4

    # Set a help message.
    pr_help = \
        "The resolution at which to sum over the probability " \
        "mass function to compute the p-values. The higher the " \
        "resolution, the more accurate the calculation. " \
        f"The default is {pr_default}."

    # Add the argument to the group.
    dea_group.add_argument("-pr", "--p-values-resolution",
                           type = lambda x: int(float(x)),
                           default = pr_default,
                           help = pr_help)

    #-----------------------------------------------------------------#

    # Set the default value for the argument.
    qa_default = 0.05

    # Set a help message.
    qa_help = \
        "The alpha value used to calculate the q-values (adjusted " \
        f"p-values). The default is {qa_default}."

    # Add the argument to the group.
    dea_group.add_argument("-qa", "--q-values-alpha",
                           type = float,
                           default = qa_default,
                           help = qa_help)

    #-----------------------------------------------------------------#

    # Set the default value for the argument.
    qm_default = "fdr_bh"

    # Set a help message.
    qm_help = \
        "The method used to calculate the q-values (i.e., to " \
        f"adjust the p-values). The default is '{qm_default}'. " \
        "The available methods can be found in the documentation " \
        "of 'statsmodels.stats.multitest.multipletests', " \
        "which is used to perform the calculation."

    # Add the argument to the group.
    dea_group.add_argument("-qm", "--q-values-method",
                           type = str,
                           default = qm_default,
                           help = qm_help)

    #-----------------------------------------------------------------#

    # Set the default value for the argument.
    n_default = 1

    # Set a help message.
    n_help = \
        "The number of processes to start. The default number " \
        f"of processes started is {n_default}."

    # Add the argument to the group.
    run_group.add_argument("-n", "--n-proc",
                           type = int,
                           default = n_default,
                           help = n_help)

    #-----------------------------------------------------------------#

    # Return the parser.
    return parser


#######################################################################


# Define the 'main' function.
def main(args):

    # Get the argument corresponding to the working directory.
    wd = args.work_dir

    # Get the arguments corresponding to the input files.
    input_samples = args.input_samples
    input_means = args.input_means
    input_rvalues = args.input_rvalues

    # Get the argument corresponding to the output file.
    output_prefix = args.output_prefix

    # Get the arguments corresponding to the DEA options.
    p_values_resolution = args.p_values_resolution
    q_values_alpha = args.q_values_alpha
    q_values_method = args.q_values_method

    # Get the arguments corresponding to the run options.
    n_proc = args.n_proc

    #-----------------------------------------------------------------#

    # Try to load the samples.
    try:

        # Get the samples (= observed gene counts).
        obs_counts = \
            ioutil.load_samples(\
                csv_file = input_samples,
                sep = ",",
                keep_samples_names = True,
                split = False)

        # Get the sample's names.
        obs_counts_names = obs_counts.index.tolist()

    # If something went wrong
    except Exception as e:

        # Warn the user and exit.
        errstr = \
            "It was not possible to load the samples from " \
            f"'{input_samples}'. Error: {e}"
        log.exception(errstr)
        sys.exit(errstr)

    # Inform the user that the samples were successfully loaded.
    infostr = \
        "The samples were successfully loaded from " \
        f"'{input_samples}'."
    log.info(infostr)

    #-----------------------------------------------------------------#

    # Try to load the predicted means.
    try:

        # Get the predicted means.
        pred_means = \
            ioutil.load_decoder_outputs(csv_file = input_means,
                                        sep = ",",
                                        split = False)

    # If something went wrong
    except Exception as e:

        # Warn the user and exit.
        errstr = \
            "It was not possible to load the predicted means from " \
            f"'{input_means}'. Error: {e}"
        log.exception(errstr)
        sys.exit(errstr)

    # Inform the user that the predicted means were successfully
    # loaded.
    infostr = \
        "The predicted means were successfully loaded from " \
        f"'{input_means}'."
    log.info(infostr)

    #-----------------------------------------------------------------#

    # If r-values were passed
    if input_rvalues is not None:

        # Try to load the predicted r-values.
        try:

            # Get the predicted r-values.
            r_values = \
                ioutil.load_decoder_outputs(\
                    csv_file = input_rvalues,
                    sep = ",",
                    split = False)

        # If something went wrong
        except Exception as e:

            # Warn the user and exit.
            errstr = \
                "It was not possible to load the predicted r-values " \
                f"from '{input_rvalues}'. Error: {e}"
            log.exception(errstr)
            sys.exit(errstr)

        # Inform the user that the predicted r-values were successfully
        # loaded.
        infostr = \
            "The predicted r-values were successfully loaded from " \
            f"'{input_rvalues}'."
        log.info(infostr)

    # Otherwise
    else:

        # The r-values will be None.
        r_values = None

    #-----------------------------------------------------------------#

    # Create the local cluster.
    cluster = LocalCluster(# Number of workers
                           n_workers = n_proc,
                           # Below which level log messages will
                           # be silenced
                           silence_logs = "ERROR",
                           # Whether to use processes, single-core
                           # or threads
                           processes = True,
                           # How many threads for each worker should
                           # be used
                           threads_per_worker = 1)

    # Open the client from the cluster.
    client = Client(cluster)

    #-----------------------------------------------------------------#

    # Set the statistics to be calculated.
    statistics = ["p_values", "q_values", "log2_fold_changes"]

    # Create a list to store the futures.
    futures = []
    
    # For each sample
    for sample_name in obs_counts_names:

        # Set the options to perform the analysis.
        dea_options = \
            {"obs_counts" : obs_counts.loc[sample_name,:],
             "pred_means" : pred_means.loc[sample_name,:],
             "sample_name" : sample_name,
             "statistics" : statistics,
             "resolution" : p_values_resolution,
             "alpha" : q_values_alpha,
             "method" : q_values_method}

        # If r-values were passed
        if r_values is not None:

            # Add the r-values for the current sample.
            dea_options["r_values"] = r_values.loc[sample_name,:]

        # Submit the calculation to the cluster.
        futures.append(\
            client.submit(dea.perform_dea,
                          **dea_options))

    #-----------------------------------------------------------------#

    # For each future
    for future, result in as_completed(futures, 
                                       with_results = True):
        
        # Get the data frame containing the DEA results for the
        # current sample and the name of the sample.
        df_stats, sample_name = result

        # Add a column containing the observed counts.
        df_stats["obs_counts"] = obs_counts.loc[sample_name,:]

        # Add a column containing the predicted means.
        df_stats["dgd_mean"] = pred_means.loc[sample_name,:]

        #-------------------------------------------------------------#

        # If the r-values were passed
        if r_values is not None:
            
            # Add a column containing the r-values
            df_stats["dgd_r"] = r_values.loc[sample_name,:]
        
        #-------------------------------------------------------------#

        # Set the path to the output file.
        output_path = \
            os.path.join(wd, f"{output_prefix}{sample_name}.csv")

        # Try to write the data frame in the output file.
        try:

            df_stats.to_csv(output_path,
                            sep = ",",
                            index = True,
                            header = True)

        # If something went wrong
        except Exception as e:

            # Warn the user and exit.
            errstr = \
                "It was not possible to write the DEA results " \
                f"for sample '{sample_name}' in '{output_path}'. " \
                f"Error: {e}"
            log.exception(errstr)
            sys.exit(errstr)

        # Inform the user that the file was successfully written.
        infostr = \
            f"The DEA results for sample '{sample_name}' were " \
            f"successfully written in '{output_path}'."
        log.info(infostr)
