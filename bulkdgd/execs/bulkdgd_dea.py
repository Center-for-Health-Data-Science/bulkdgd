#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-

#    bulkdgd_dea.py
#
#    Perform a differential expression analysis comparing experimental
#    samples to their "closest normal" sample found in latent space
#    by the :class:`core.model.BulkDGD`.
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
__doc__ = \
    "Perform a differential expression analysis comparing " \
    "experimental samples to their 'closest normal' samples " \
    "found in latent space by the :class:`core.model.BulkDGD`."


#######################################################################


# Import from the standard library.
import argparse
import logging as log
import os
import sys

# Import from third-party libraries.
from distributed import LocalCluster, Client, as_completed
import pandas as pd
import torch

# Import from 'bulkdgd'.
from bulkdgd.analysis import dea
from bulkdgd import _internals, ioutil
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
    odp_default = "dea_"

    # Set a help message.
    odp_help = \
        "The prefix of the output CSV file(s) that will contain " \
        "the results of the differential expression analysis. " \
        "Since the analysis will be performed for each sample, " \
        "one file per sample will be created. The files' names " \
        "will have the form {output_csv_prefix}{sample_name}.csv. " \
        f"The default prefix is '{odp_default}'."

    # Add the argument to the group.
    output_group.add_argument("-odp", "--output-dea-prefix",
                              type = str,
                              default = odp_default,
                              help = odp_help)
    
    #-----------------------------------------------------------------#

    # Set the default value for the argument.
    ogp_default = "gsea_"

    # Set a help message.
    ogp_help = \
        "The prefix of the output CSV file(s) that will contain " \
        "the results of the gene set enrichment analysis. By " \
        "default, the analysis will be performed for each sample " \
        "and one per sample will be created. The files' names will " \
        "have the form {output_csv_prefix}{sample_name}.csv. The " \
        f"default prefix is '{ogp_default}'. If the '-mg', " \
        "'--merge-gsea' option is passed, this option is " \
        "interpreted as the name of the output file where the " \
        "merged results will be written (stripped of any trailing " \
        "underscores or dots)."
    
    # Add the argument to the group.
    output_group.add_argument("-ogp", "--output-gsea-prefix",
                              type = str,
                              default = ogp_default,
                              help = ogp_help)

    #-----------------------------------------------------------------#

    # Set a help message.
    mg_help = \
        "Whether the results of the gene set enrichment analysis " \
        "will be merged into a single file. By default, the " \
        "results for each sample are written in separate files."
    
    # Add the argument to the group.
    output_group.add_argument("-mg", "--merge-gsea",
                              action = "store_true",
                              help = mg_help)

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
    pt_default = 0.05

    # Set a help message.
    p_val_threshold = \
        "The threshold used to select the significant genes " \
        f"based on the p-values. The default is {pt_default}."
    
    # Add the argument to the group.
    dea_group.add_argument("-pt", "--p-values-threshold",
                            type = float,
                            default = pt_default,
                            help = p_val_threshold)

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
    qt_default = 0.05

    # Set a help message.
    qt_help = \
        "The threshold used to select the significant genes " \
        f"based on the q-values. The default is {qt_default}."
    
    # Add the argument to the group.
    dea_group.add_argument("-qt", "--q-values-threshold",
                            type = float,
                            default = qt_default,
                            help = qt_help)
    
    #-----------------------------------------------------------------#

    # Set the default value for the argument.
    fct = 2

    # Set a help message.
    fct_help = \
        "The threshold used to select the significant genes " \
        f"based on the log2-fold changes. The default is {fct}."
    
    # Add the argument to the group.
    dea_group.add_argument("-fct", "--log2-fold-change-threshold",
                            type = float,
                            default = fct,
                            help = fct_help)

    #-----------------------------------------------------------------#

    # Set a help message.
    gsf_help = \
        "A list of plain text files containing the gene sets " \
        "to be used in the analysis. The files must contain " \
        "one gene symbol per line. The gene sets will be used " \
        "to perform the gene set enrichment analysis."
    
    # Add the argument to the group.
    dea_group.add_argument("-gsf", "--genes-sets-files",
                            type = str,
                            nargs = "+",
                            default = None,
                            help = gsf_help)

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

    # Set a help message.
    dev_help = \
        "The device to use. If not provided, the GPU will be used " \
        "if it is available. Otherwise, the CPU will be used. The " \
        "genes are independent of each other, so calculating the " \
        "p-values on a GPU is considerably faster."

    # Add the argument to the group.
    run_group.add_argument("-dev", "--device",
                           type = str,
                           default = None,
                           help = dev_help)

    #-----------------------------------------------------------------#

    # Add the working directory and logging arguments.
    util.add_wd_and_logging_arguments(\
        parser = parser,
        command_name = "bulkdgd_dea")

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
    input_means = args.input_means
    input_rvalues = args.input_rvalues

    # Get the arguments corresponding to the output files.
    output_dea_prefix = args.output_dea_prefix
    output_gsea_prefix = args.output_gsea_prefix
    merge_gsea = args.merge_gsea

    # Get the arguments corresponding to the DEA options.
    p_values_resolution = args.p_values_resolution
    p_values_threshold = args.p_values_threshold
    q_values_alpha = args.q_values_alpha
    q_values_method = args.q_values_method
    q_values_threshold = args.q_values_threshold
    log2_fold_change_threshold = args.log2_fold_change_threshold
    genes_sets_files = args.genes_sets_files

    # Get the arguments corresponding to the run options.
    n_proc = args.n_proc
    device = args.device

    #-----------------------------------------------------------------#

    # If no device was passed
    if device is None:

        # If a GPU is available
        if torch.cuda.is_available():

            # Set the GPU as the device.
            device = "cuda"

        # Otherwise
        else:

            # Set the CPU as the device.
            device = "cpu"

    # Inform the user about the device that will be used.
    infostr = \
        f"The p-values will be calculated on the '{device}' device."
    logger.info(infostr)

    # If the p-values will be calculated on a GPU, but more than one
    # process was requested
    if torch.device(device).type != "cpu" and n_proc > 1:

        # Warn the user - the processes would all queue up on the same
        # GPU, and each of them would keep its own context on it.
        warnstr = \
            f"The p-values will be calculated on the '{device}' " \
            f"device, but {n_proc} processes were requested. The " \
            "processes will share the same GPU, so using more than " \
            "one process is not expected to speed up the " \
            "calculation. Consider using '-n 1'."
        logger.warning(warnstr)

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
        logger.exception(errstr)
        sys.exit(errstr)

    # Inform the user that the samples were successfully loaded.
    infostr = \
        "The samples were successfully loaded from " \
        f"'{input_samples}'."
    logger.info(infostr)

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
        logger.exception(errstr)
        sys.exit(errstr)

    # Inform the user that the predicted means were successfully
    # loaded.
    infostr = \
        "The predicted means were successfully loaded from " \
        f"'{input_means}'."
    logger.info(infostr)

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
            logger.exception(errstr)
            sys.exit(errstr)

        # Inform the user that the predicted r-values were successfully
        # loaded.
        infostr = \
            "The predicted r-values were successfully loaded from " \
            f"'{input_rvalues}'."
        logger.info(infostr)

    # Otherwise
    else:

        # The r-values will be None.
        r_values = None

    #-----------------------------------------------------------------#

    # Set the list of genes sets to None.
    genes_sets = None

    # Set the list of all genes to None.
    genes_all = None

    # Initialize an empty list to store the results of the gene set
    # enrichment analysis.
    gsea_results = []

    # If a list of files containing genes sets was passed
    if genes_sets_files is not None:

        # Set the list of all genes to all columns containing genes'
        # counts in the observed gene counts data frame.
        genes_all = \
            [c for c in obs_counts.columns.tolist() \
            if c.startswith("ENSG")]

        # Initialize the dictionary of genes sets, keyed by the
        # name of the gene set (the name of the file it was loaded
        # from, without the extension).
        genes_sets = {}

        # For each file
        for genes_set_file in genes_sets_files:

            # Try to load the gene set.
            try:

                # Get the gene set.
                genes_set = \
                    _internals.load_list(list_file = genes_set_file)

                # Get the name of the gene set from the file's name.
                genes_set_name = \
                    os.path.splitext(\
                        os.path.basename(genes_set_file))[0]

                # Add the gene set to the dictionary of genes sets.
                genes_sets[genes_set_name] = genes_set

            # If something went wrong
            except Exception as e:

                # Warn the user and exit.
                errstr = \
                    "It was not possible to load the gene set from " \
                    f"'{genes_set_file}'. Error: {e}"
                logger.exception(errstr)
                sys.exit(errstr)

            # Inform the user that the gene set was successfully loaded.
            infostr = \
                "The gene set was successfully loaded from " \
                f"'{genes_set_file}'."
            logger.info(infostr)

    #-----------------------------------------------------------------#

    # Set the statistics to be calculated.
    statistics = ["p_values", "q_values", "log2_fold_changes"]

    # Create a list to store the options to analyze each sample.
    samples_options = []

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
             "method" : q_values_method,
             "device" : device}

        # If r-values were passed
        if r_values is not None:

            # Add the r-values for the current sample.
            dea_options["r_values"] = r_values.loc[sample_name,:]

        # Save the options.
        samples_options.append(dea_options)

    #-----------------------------------------------------------------#

    # If only one process was requested
    if n_proc == 1:

        # Analyze the samples in the current process.
        #
        # A cluster of one worker would gain nothing, and would cost a
        # second Python process, which would have to import the
        # third-party libraries again and - if the analysis runs on a
        # GPU - set up its own GPU context. Both are slower than the
        # analysis itself.
        results = \
            (dea.get_statistics(**dea_options) \
             for dea_options in samples_options)

        # No cluster was created.
        cluster, client = None, None

    # Otherwise
    else:

        # Create the local cluster.
        cluster = LocalCluster(# Number of workers
                               n_workers = n_proc,
                               # Below which level log messages will
                               # be silenced
                               silence_logs = "ERROR",
                               # Whether to use processes, single-core
                               # or threads
                               processes = True,
                               # How many threads for each worker
                               # should be used
                               threads_per_worker = 1)

        # Open the client from the cluster.
        client = Client(cluster)

        # Submit the analysis of each sample to the cluster.
        futures = \
            [client.submit(dea.get_statistics, **dea_options) \
             for dea_options in samples_options]

        # Get the results as they come in.
        results = \
            (result for _, result \
             in as_completed(futures, with_results = True))

    #-----------------------------------------------------------------#

    # For each result
    for result in results:

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
            os.path.join(wd, f"{output_dea_prefix}{sample_name}.csv")

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
            logger.exception(errstr)
            sys.exit(errstr)

        # Inform the user that the file was successfully written.
        infostr = \
            f"The DEA results for sample '{sample_name}' were " \
            f"successfully written in '{output_path}'."
        logger.info(infostr)

        #-----------------------------------------------------------------#

        # If gene sets were passed
        if genes_sets is not None:

            # Try to get the significant genes.
            try:

                # Get the significant genes.
                df_significant_genes = \
                    dea.get_significant_genes(\
                            df_stats = df_stats,
                            p_val = p_values_threshold,
                            q_val = q_values_threshold,
                            log2_fold_change = log2_fold_change_threshold)
        
            # If something went wrong
            except Exception as e:

                # Warn the user and exit.
                errstr = \
                    "It was not possible to get the significant " \
                    f"genes from the DEA results of sample " \
                    f"'{sample_name}'. Error: {e}"
                logger.exception(errstr)
                sys.exit(errstr)
            
            # Inform the user that the significant genes were
            # successfully obtained.
            infostr = \
                "The significant genes were successfully obtained " \
                f"from the DEA results of sample '{sample_name}'."
            logger.info(infostr)

            #---------------------------------------------------------#

            # Try to perform the gene set enrichment analysis.
            try:

                # Perform the gene set enrichment analysis.
                df_gsea_result = \
                    dea.get_enrichment_scores(\
                        df_significant_genes = df_significant_genes,
                        genes_sets = genes_sets,
                        genes_all = genes_all)
        
            # If something went wrong
            except Exception as e:

                # Warn the user and exit.
                errstr = \
                    "It was not possible to perform the gene set " \
                    f"enrichment analysis for sample '{sample_name}'. " \
                    f"Error: {e}"
                logger.exception(errstr)
                sys.exit(errstr)
        
            # Inform the user that the gene set enrichment analysis
            # was successfully performed.
            infostr = \
                "The gene set enrichment analysis was successfully " \
                f"performed for sample '{sample_name}."
            logger.info(infostr)

            #-------------------------------------------------------------#

            # If the results of the analysis should be written to separate
            # files
            if not merge_gsea:

                # Set the path to the output file.
                output_path = \
                    os.path.join(wd,
                                 f"{output_gsea_prefix}{sample_name}.csv")

                # Try to write the data frame in the output file.
                try:

                    df_gsea_result.to_csv(output_path,
                                         sep = ",",
                                         index = True,
                                         header = True)

                # If something went wrong
                except Exception as e:

                    # Warn the user and exit.
                    errstr = \
                        "It was not possible to write the gene set " \
                        f"enrichment analysis results for sample " \
                        f"'{sample_name}' in '{output_path}'. Error: {e}"
                    logger.exception(errstr)
                    sys.exit(errstr)

                # Inform the user that the file was successfully written.
                infostr = \
                    f"The gene set enrichment analysis results for " \
                    f"sample '{sample_name}' were successfully written " \
                    f"in '{output_path}'."
                logger.info(infostr)
        
            # Otherwise
            else:

                # Add a column identifying the sample the results belong
                # to.
                df_gsea_result.insert(0, "sample", sample_name)

                # Add the results to the list.
                gsea_results.append(df_gsea_result)

    #-----------------------------------------------------------------#

    # If a cluster was created to analyze the samples
    if client is not None:

        # Close it.
        client.close()
        cluster.close()

    #-----------------------------------------------------------------#

    # If the results of the gene set enrichment analysis should be
    # merged
    if genes_sets and merge_gsea:

        # Set the name of the output file.
        output_gsea_name = output_gsea_prefix.rstrip("_").rstrip(".")

        # Set the path to the output file.
        output_gsea_file = os.path.join(wd, f"{output_gsea_name}.csv")

        # Try to write the data frame in the output file.
        try:

            pd.concat(gsea_results,
                     ignore_index = True).to_csv(\
                        output_gsea_file,
                        sep = ",",
                        index = True,
                        header = True)

        # If something went wrong
        except Exception as e:

            # Warn the user and exit.
            errstr = \
                "It was not possible to write the merged gene set " \
                "enrichment analysis results in  " \
                f"'{output_gsea_file}'. Error: {e}"
            logger.exception(errstr)
            sys.exit(errstr)

        # Inform the user that the file was successfully written.
        infostr = \
            "The merged gene set enrichment analysis results were " \
            f"successfully written in '{output_gsea_file}'."
        logger.info(infostr)


#######################################################################


# Define the entry point for the standalone executable.
def entry_point() -> None:

    # Build the parser.
    parser = set_parser()

    # Parse the arguments.
    args = parser.parse_args()

    # Set up the logging.
    util.set_main_logging(\
        args = args,
        command_name = "bulkdgd_dea")

    # Run the main function.
    main(args = args)
