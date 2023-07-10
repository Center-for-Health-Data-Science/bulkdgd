#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-

#    dgd_preprocess_samples.py
#
#    Preprocess new samples to use them with the DGD model.
#
#    Copyright (C) 2023 Valentina Sora 
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


# Description of the module
__doc__ = \
    "Preprocess new samples to use them with the DGD model."

# Standard library
import argparse
import logging as log
import os
import sys
# Third-party packages
import pandas as pd
# bulkDGD
from bulkDGD.utils import dgd

def main():


    #-------------------- Command-line arguments ---------------------#


    # Create the argument parser
    parser = argparse.ArgumentParser(description = __doc__)

    # Add the arguments
    i_help = \
        "The input CSV file containing a data frame with the " \
        "samples to be preprocessed. The columns must represent " \
        "the genes (Ensembl IDs), while the rows must represent " \
        "the samples."
    parser.add_argument("-i", "--input-csv",
                        type = str,
                        required = True,
                        help = i_help)

    os_default = "samples_preprocessed.csv"
    os_help = \
        f"The name of the output CSV file containing the data frame " \
        f"with the preprocessed samples. The columns will represent " \
        f"the genes (Ensembl IDs), while the rows will represent the " \
        f"samples. The file will be written in the working " \
        f"directory. The default file name is '{os_default}'."
    parser.add_argument("-os", "--output-csv-samples",
                        type = str,
                        default = os_default,
                        help = os_help)

    oge_default = "genes_excluded.csv"
    oge_help = \
        f"The name of the output plain text file containing the " \
        f"list of genes whose expression data are excluded from the " \
        f"data frame with the preprocessed samples. This is done " \
        f"because the preprocessed samples must only contain data " \
        f"for the genes on which the DGD model was trained. The " \
        f"file will be written in the working directory. " \
        f"The default file name is '{oge_default}'."
    parser.add_argument("-oge", "--output-txt-genes-excluded",
                        type = str,
                        default = oge_default,
                        help = oge_help)

    ogm_default = "genes_missing.csv"
    ogm_help = \
        f"The name of the output plain text file containing the " \
        f"list of genes for which no available expression data " \
        f"are found in the input data frame. A default count of " \
        f"0 is assigned to these genes in the output data frame " \
        f"containing the preprocessed samples. The file will " \
        f"be written in the working directory. The " \
        f"default file name is '{ogm_default}'."
    parser.add_argument("-ogm", "--output-txt-genes-missing",
                        type = str,
                        default = ogm_default,
                        help = ogm_help)

    sc_help = \
        "The name/index of the column containing the IDs/names " \
        "of the samples, if any. By default, the program will " \
        "assume that no such column is present."
    parser.add_argument("-sc", "--samples-names-column",
                        type = str,
                        default = None,
                        help = sc_help)

    tc_help = \
        "The name of the column containing the names of the " \
        "tissues the samples belong to, if any. By default, " \
        "the program will assume that no such column is present. "
    parser.add_argument("-tc", "--tissues-column",
                        type = str,
                        default = None,
                        help = tc_help)

    d_help = \
        "The working directory. The default is the current " \
        "working directory."
    parser.add_argument("-d", "--work-dir",
                        type = str,
                        default = os.getcwd(),
                        help = d_help)

    v_help = "Verbose logging (INFO level)."
    parser.add_argument("-v", "--logging-verbose",
                        action = "store_true",
                        help = v_help)

    vv_help = \
        "Maximally verbose logging for debugging " \
        "purposes (DEBUG level)."
    parser.add_argument("-vv", "--logging-debug",
                        action = "store_true",
                        help = vv_help)

    # Parse the arguments
    args = parser.parse_args()
    input_csv = args.input_csv
    output_csv_samples = args.output_csv_samples
    output_txt_genes_excluded = args.output_txt_genes_excluded
    output_txt_genes_missing = args.output_txt_genes_missing
    samples_names_column = \
        int(args.samples_names_column) \
        if args.samples_names_column.isdigit() \
        else args.samples_names_column
    tissues_column = args.tissues_column
    wd = args.work_dir
    v = args.logging_verbose
    vv = args.logging_debug


    #---------------------------- Logging ----------------------------#


    # Get the module's logger
    logger = log.getLogger(__name__)

    # Set WARNING logging level by default
    level = log.WARNING

    # If the user requested verbose logging
    if v:

        # The minimal logging level will be INFO
        level = log.INFO

    # If the user requested logging for debug purposes
    # (-vv overrides -v if both are provided)
    if vv:

        # The minimal logging level will be DEBUG
        level = log.DEBUG

    # Set the logging level
    log.basicConfig(level = level)


    #------------------------- Data loading --------------------------#


    # If the user did not specify any column containing the samples'
    # IDs/names
    if samples_names_column is None:

        # Try to load the input samples
        try:

            samples_df = pd.read_csv(input_csv,
                                     sep = ",",
                                     index_col = False)

        # If something went wrong
        except Exception as e:

            # Warn the user and exit
            errstr = \
                f"It was not possible to load the samples from " \
                f"'{input_csv}'. Error: {e}"
            logger.exception(errstr)
            sys.exit(errstr)

    # If the user specified a column containing the samples' IDs/names
    else:

        # Try to load the input samples
        try:

            samples_df = pd.read_csv(input_csv,
                                     sep = ",",
                                     index_col = samples_names_column)

        # If something went wrong
        except Exception as e:

            # Warn the user and exit
            errstr = \
                f"It was not possible to load the samples from " \
                f"'{input_csv}'. Error: {e}"
            logger.exception(errstr)
            sys.exit(errstr)

    # If the user specified a column containing the tissues' labels
    if tissues_column is not None:

        # Take the column values
        tissues_column_values = samples_df[tissues_column].tolist()

        # Drop the column before preprocessing
        samples_df = samples_df.drop(tissues_column,
                                     axis = 1)


    #-------------------- Samples' preprocessing ---------------------#


    # Try to preprocess the samples
    try:

        preproc_df, genes_excluded, genes_missing = \
            dgd.preprocess_samples(samples_df = samples_df)

    # If something went wrong
    except Exception as e:

        # Warn the user and exit
        errstr = \
            f"It was not possible to preprocess the samples. " \
            f"Error: {e}"
        logger.exception(errstr)
        sys.exit(errstr)


    #----------------- Output - preprocessed samples -----------------#


    # Set the path to the output file
    output_csv_samples_path = os.path.join(wd, output_csv_samples)

    # If there was a column containing the tissues' labels in the
    # input data frame
    if tissues_column is not None:

        # Add it to the data frame containing the preprocessed
        # samples (as the last column)
        preproc_df.insert(len(preproc_df.columns),
                          tissues_column,
                          tissues_column_values)

        # Inform the user that the column was added
        infostr = \
            f"The column '{tissues_column}' containing the names " \
            f"of the tissues the original samples belong to was " \
            f"added to the data frame of preprocessed samples as " \
            f"the last column."
        logger.info(infostr)

    # If the user did not specify any column containing the samples'
    # IDs/names
    if samples_names_column is None:

        # Try to write out the preprocessed samples without the
        # index column
        try:

            preproc_df.to_csv(output_csv_samples_path,
                              sep = ",",
                              index = False)

        # If something went wrong
        except Exception as e:

            # Warn the user and exit
            errstr = \
                f"It was not possible to write the preprocessed " \
                f"samples to '{output_csv_samples_path}'. Error: {e}"
            logger.exception(errstr)
            sys.exit(errstr)

    # If the user specified a column containing the samples' IDs/names
    else:

        # Try to write out the preprocessed samples with the
        # index column
        try:

            preproc_df.to_csv(output_csv_samples_path,
                              sep = ",",
                              index = True)

        # If something went wrong
        except Exception as e:

            # Warn the user and exit
            errstr = \
                f"It was not possible to write the preprocessed " \
                f"samples to '{output_csv_samples_path}'. Error: {e}"
            logger.exception(errstr)
            sys.exit(errstr)

    # Inform the user that the preprocessed samples were
    # successfilly written in the output file
    infostr = \
        f"The preprocessed samples were successfully written in " \
        f"'{output_csv_samples_path}'."
    logger.info(infostr)


    #-------------------- Output - excluded genes --------------------#


    # If some genes were excluded
    if genes_excluded:

        # Set the path to the output file
        output_txt_genes_excluded_path = \
            os.path.join(wd, output_txt_genes_excluded)

        # Warn the user
        warnstr = \
            f"{len(genes_excluded)} genes found in the input " \
            f"samples are not part of the set of genes used to " \
            f"train the DGD model. Therefore, they will be removed " \
            f"from the preprocessed samples."
        logger.warning(warnstr)


        # Try to write the list of excluded genes
        try:

            with open(output_txt_genes_excluded_path, "w") as out:
                out.write("\n".join(gene for gene in genes_excluded))

        # If something went wrong
        except Exception as e:

            # Warn the user and exit
            errstr = \
                f"It was not possible to write the list of genes " \
                f"that are present in the input samples but are " \
                f"not among the genes used to train the DGD model " \
                f"to '{output_txt_genes_excluded_path}'. Error: {e}"
            logger.exception(errstr)
            sys.exit(errstr)

        # Inform the user that the genes were successfully written
        # to the output file
        infostr = \
            f"The list of genes that are present in the input " \
            f"samples but are not among the genes used to train " \
            f"the DGD model was successfully written in " \
            f"'{output_txt_genes_excluded_path}'."
        logger.info(infostr)


    #-------------------- Output - missing genes ---------------------#


    # If some genes were missing
    if genes_missing:

        # Set the path to the output file
        output_txt_genes_missing_path = \
            os.path.join(wd, output_txt_genes_missing)

        # Warn the user
        warnstr = \
            f"{len(genes_missing)} genes in the set of genes used " \
            f"to train the DGD model were not found in the input " \
            f"samples. A default count of 0 will be assigned to " \
            f"them in all preprocessed samples."
        logger.warning(warnstr)

        # Try to write the list of missing genes
        try:

            with open(output_txt_genes_missing_path, "w") as out:
                out.write("\n".join(gene for gene in genes_missing))

        # If something went wrong
        except Exception as e:

            # Warn the user and exit
            errstr = \
                f"It was not possible to write the list of genes " \
                f"that were used to train the DGD model but are " \
                f"not present in the input samples to " \
                f"'{output_txt_genes_missing_path}'."
            logger.exception(errstr)
            sys.exit(errstr)

        # Inform the user that the genes were successfully written in the
        # output file
        infostr = \
            f"The list of genes that were used to train the DGD " \
            f"model but are not present in the input samples " \
            f"was successfully written in " \
            f"'{output_txt_genes_missing_path}'."
        logger.info(infostr)