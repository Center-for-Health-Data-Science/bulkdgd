#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-

#    dgd_get_recount3_data.py
#
#    Get RNA-seq data associated with specific human samples
#    from the Recount3 platform.
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


__doc__ = \
    "Get RNA-seq data associated with specific " \
    "human samples from the Recount3 platform."


# Standard library
import argparse
import logging as log
import os
import sys
# bulDGD
from bulkDGD.utils import misc, recount3


def main():


    #-------------------- Command-line arguments ---------------------#


    # Create the argument parser
    parser = argparse.ArgumentParser(description = __doc__)

    ip_choices = ["gtex", "tcga"]
    ip_choices_str = ", ".join(f"'{choice}'" for choice in ip_choices)
    ip_help = \
        f"The name of the Recount3 project for which samples will " \
        f"be retrieved. Available projects are: {ip_choices_str}."
    parser.add_argument("-ip", "--input-project-name",
                        type = str,
                        required = True,
                        choices = ip_choices,
                        help = ip_help)

    is_help = \
        "The category of samples for which RNA-seq data will be " \
        "retrieved. For GTEx data, this is the name of the tissue " \
        "the samples belong to " \
        "For TCGA data, this is the type of cancer the samples are " \
        "associated with."
    parser.add_argument("-is", "--input-samples-category",
                        type = str,
                        required = True,
                        help = is_help)


    o_default = "{:s}_{:s}.csv"
    o_help = \
        "The name of the output CSV file containing the data frame " \
        "with the RNA-seq data for the samples. The rows " \
        "of the data frame represent samples, while the columns " \
        "represent genes (identified by their Ensembl IDs. The " \
        "file will be saved in the working directory. The default " \
        "file name is '{input_project_name}_" \
        "{input_samples_category}.csv'."
    parser.add_argument("-o", "--output-csv",
                        type = str,
                        default = None,
                        help = o_help)

    d_help = \
        "The working directory. The default is the current " \
        "working directory."
    parser.add_argument("-d", "--work-dir",
                        type = str,
                        default = os.getcwd(),
                        help = d_help)

    query_string_help = \
        f"The string that will be used to filter the samples " \
        f"according to their associated metadata using the " \
        f"'pandas.DataFrame.query()' method. The option also " \
        f"accepts a plain text file containing the string " \
        f"since it can be long for complex queries."
    parser.add_argument("--query-string",
                        type = str,
                        default = None,
                        help = query_string_help)

    save_gene_sums_help = \
        "Save the original GZ file containing the RNA-seq " \
        "data for the samples. The file will be saved in the " \
        "working directory and named " \
        "'{input_project_name}_{input_samples_category}_" \
        "gene_sums.gz'."
    parser.add_argument("--save-gene-sums",
                        action = "store_true",
                        help = save_gene_sums_help)

    save_metadata_help = \
        "Save the original GZ file containing the metadata " \
        "for the samples. The file will be saved in the working " \
        "directory and named " \
        "'{input_project_name}_{input_samples_category}_" \
        "metadata.gz'."
    parser.add_argument("--save-metadata",
                        action = "store_true",
                        help = save_metadata_help)

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
    input_project_name = args.input_project_name
    input_samples_category = args.input_samples_category
    output_csv = args.output_csv
    wd = args.work_dir
    save_gene_sums = args.save_gene_sums
    save_metadata = args.save_metadata
    query_string = args.query_string
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


    #------------------ Check the samples' category ------------------#


    # Try to get the category of samples
    try:

        cancer_type = \
            recount3.check_samples_category(\
                samples_category = input_samples_category,
                project_name = input_project_name)

    # If something went wrong
    except Exception as e:

        # Warn the user and exit
        errstr = \
            f"It was not possible to validate the provided " \
            f"category. Error: {e}"
        logger.exception(errstr)
        sys.exit(errstr)


    #--------------------- Get the RNA-seq data ----------------------#


    # Try to get the RNA-seq data for the samples from Recount3
    try:
        
        df_gene_sums = \
            recount3.get_gene_sums(\
                samples_category = input_samples_category,
                project_name = input_project_name,
                save_gene_sums = save_gene_sums,
                wd = wd)

    # If something went wrong
    except Exception as e:

        # Warn the user and exit
        errstr = \
            f"It was not possible to retrieve the RNA-seq " \
            f"data for the '{input_samples_category}' samples " \
            f"from the Recount3 platform. Error: {e}"
        logger.exception(errstr)
        sys.exit(errstr)


    #----------------------- Get the metadata ------------------------#


    # Try to get the metadata for the samples from Recount3
    try:

        df_metadata = \
            recount3.get_metadata(\
                samples_category = input_samples_category,
                project_name = input_project_name,
                save_metadata = save_metadata,
                wd = wd)

    # If something went wrong
    except Exception as e:

        # Warn the user and exit
        errstr = \
            f"It was not possible to retrieve the metadata for the " \
            f"'{input_samples_category}' samples from the Recount3 " \
            f"platform. Error: {e}"
        logger.exception(errstr)
        sys.exit(errstr)


    #---------------------- Filter by metadata -----------------------#

    
    # If the user has passed a query string or a file containing the
    # query string
    if query_string is not None:

        # Try to get the string
        try:
            
            query_string = \
                util.get_query_string(query_string = query_string)

        # If something went wrong
        except Exception as e:

            # Warn the user and exit
            errstr = \
                f"It was not possible to get the query string. " \
                f"Error: {e}"
            logger.exception(errstr)
            sys.exit(errstr)

        # Try to add the metadata to the RNA-seq data frame
        try:

            df_final = \
                recount3.merge_gene_sums_and_metadata(\
                    df_gene_sums = df_gene_sums,
                    df_metadata = df_metadata,
                    project_name = input_project_name)

        # If something went wrong
        except Exception as e:

            # Warn the user and exit
            errstr = \
                f"It was not possible to add the metadata to the " \
                f"'{input_samples_category}' samples. Error: {e}"
            logger.exception(errstr)
            sys.exit(errstr)

        # Inform the user that the metadata were successfully added
        infostr = \
            f"The metadata were successfully added to the " \
            f"'{input_samples_category}' samples. Error: {e}"
        logger.info(infostr)

        # Try to filter tha samples according to the query string
        try:
        
            df_final = \
                recount3.filter_by_metadata(\
                    df = df,
                    query_string = query_string,
                    project_name = input_project_name)

        # If something went wrong
        except Exception as e:

            # Warn the user and exit
            errstr = \
                f"It was not possible to filter the samples " \
                f"using the associated metadata. Error: {e}"
            logger.exception(errstr)
            sys.exit(errstr)

        # Inform the user that the samples were successfully filtered
        infostr = \
            f"The samples were successfully filtered using the " \
            f"associated metadata."
        logger.info(infostr)

    # Otherwise
    else:

        # The final data frame will be the one containing the gene
        # expression data
        df_final = df_gene_sums


    #------------------ Save the output data frame -------------------#


    # If the user did not pass a name for the output CSV file
    if output_csv is None:

        # Use the default output name
        output_csv_path = \
            os.path.join(wd, o_default.format(input_project_name,
                                              input_samples_category))

    # Otherwise
    else:

        # Use the user-defined one
        output_csv_path = \
            os.path.join(wd, output_csv)

    # Try to write the data frame to the output CSV file
    try:

        df_final.to_csv(output_csv_path,
                        sep = ",",
                        index = True)

    # If something went wrong
    except Exception as e:

        # Warn the user and exit
        errstr = \
            f"It was not possible to write the data frame " \
            f"in '{output_csv_path}'. Error: {e}"
        logger.exception(errstr)
        sys.exit(errstr)

    # Inform the user that the data frame was successfully
    # written to the output file
    infostr = \
        f"The data frame was successfully written in " \
        f"'{output_csv_path}'."
    logger.info(infostr)