#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-

#    dgd_get_recount3_data.py
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
    "Get RNA-seq data associated with specific human samples " \
    "for projects hosted on the Recount3 platform."


# Standard library
import argparse
import logging as log
import os
import sys
# bulDGD
from bulkDGD import ioutil, recount3


def main():


    #-------------------- Command-line arguments ---------------------#


    # Create the argument parser
    parser = argparse.ArgumentParser(description = __doc__)

    # Add the arguments
    ip_choices = ["gtex", "tcga"]
    ip_choices_str = ", ".join(f"'{choice}'" for choice in ip_choices)
    ip_help = \
        "The name of the Recount3 project for which samples will " \
        f"be retrieved. The available projects are: {ip_choices_str}."
    parser.add_argument("-ip", "--input-project-name",
                        type = str,
                        required = True,
                        choices = ip_choices,
                        help = ip_help)

    is_help = \
        "The category of samples for which RNA-seq data will be " \
        "retrieved. For GTEx data, this is the name of the tissue " \
        "the samples belong to. " \
        "For TCGA data, this is the type of cancer the samples are " \
        "associated with."
    parser.add_argument("-is", "--input-samples-category",
                        type = str,
                        required = True,
                        help = is_help)


    o_default = "{:s}_{:s}.csv"
    o_help = \
        "The name of the output CSV file containing the data frame " \
        "with the RNA-seq data for the samples. The " \
        "file will be written in the working directory. The default " \
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

    sg_help = \
        "Save the original GZ file containing the RNA-seq " \
        "data for the samples. The file will be saved in the " \
        "working directory and named " \
        "'{input_project_name}_{input_samples_category}_" \
        "gene_sums.gz'."
    parser.add_argument("-sg", "--save-gene-sums",
                        action = "store_true",
                        help = sg_help)

    sm_help = \
        "Save the original GZ file containing the metadata " \
        "for the samples. The file will be saved in the working " \
        "directory and named " \
        "'{input_project_name}_{input_samples_category}_" \
        "metadata.gz'."
    parser.add_argument("-sm", "--save-metadata",
                        action = "store_true",
                        help = sm_help)

    qs_help = \
        "The string that will be used to filter the samples " \
        "according to their associated metadata using the " \
        "'pandas.DataFrame.query()' method. The option also " \
        "accepts a plain text file containing the string " \
        "since it can be long for complex queries."
    parser.add_argument("-qs", "--query-string",
                        type = str,
                        default = None,
                        help = qs_help)

    lf_default = "dgd_get_recount3_data.log"
    lf_help = \
        "The name of the log file. The file will be written " \
        "in the working directory. The default file name is " \
        f"'{lf_default}'."
    parser.add_argument("-lf", "--log-file",
                        type = str,
                        default = lf_default,
                        help = lf_help)

    lc_help = "Show log messages also on the console."
    parser.add_argument("-lc", "--log-console",
                        action = "store_true",
                        help = lc_help)

    v_help = "Enable verbose logging (INFO level)."
    parser.add_argument("-v", "--log-verbose",
                        action = "store_true",
                        help = v_help)

    vv_help = \
        "Enable maximally verbose logging for debugging " \
        "purposes (DEBUG level)."
    parser.add_argument("-vv", "--log-debug",
                        action = "store_true",
                        help = vv_help)

    # Parse the arguments
    args = parser.parse_args()
    input_project_name = args.input_project_name
    input_samples_category = args.input_samples_category
    output_csv = args.output_csv
    wd = args.work_dir
    query_string = args.query_string
    save_gene_sums = args.save_gene_sums
    save_metadata = args.save_metadata
    log_file = args.log_file
    log_console = args.log_console
    v = args.log_verbose
    vv = args.log_debug


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

    # Initialize the logging handlers to a list containing only
    # the FileHandler (to log to the log file)
    handlers = [log.FileHandler(# The log file
                                filename = log_file,
                                # How to open the log file ('w' means
                                # re-create it every time the
                                # executable is called)
                                mode = "w")]

    # If the user requested logging to the console, too
    if log_console:

        # Append a StreamHandler to the list
        handlers.append(log.StreamHandler())

    # Set the logging level
    log.basicConfig(# The level below which log messages are silenced
                    level = level,
                    # The format of the log strings
                    format = "{asctime}:{levelname}:{name}:{message}",
                    # The format for dates/time
                    datefmt="%Y-%m-%d,%H:%M",
                    # The format style
                    style = "{",
                    # The handlers
                    handlers = handlers)


    #------------------ Check the samples' category ------------------#


    # Try to validate the category of samples to be retrieved
    try:

        cancer_type = \
            recount3.check_samples_category(\
                samples_category = input_samples_category,
                project_name = input_project_name)

    # If something went wrong
    except Exception as e:

        # Warn the user and exit
        errstr = \
            "It was not possible to validate the provided " \
            f"category '{input_samples_category}' for the " \
            f"project '{input_project_name}'. Error: {e}"
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
            "It was not possible to retrieve the RNA-seq " \
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
            "It was not possible to retrieve the metadata for the " \
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
                recount3.get_query_string(query_string = query_string)

        # If something went wrong
        except Exception as e:

            # Warn the user and exit
            errstr = \
                "It was not possible to get the query string. " \
                f"Error: {e}"
            logger.exception(errstr)
            sys.exit(errstr)

        # Try to filter the samples metadata to the RNA-seq data frame
        try:

            # Add the metadata to the RNA-seq data frame
            df_merged = \
                recount3.merge_gene_sums_and_metadata(\
                    df_gene_sums = df_gene_sums,
                    df_metadata = df_metadata,
                    project_name = input_project_name)

            # Filter the samples
            df_final = \
                recount3.filter_by_metadata(\
                    df = df_merged,
                    query_string = query_string,
                    project_name = input_project_name)

        # If something went wrong
        except Exception as e:

            # Warn the user and exit
            errstr = \
                "It was not possible to filter the samples " \
                f"using the associated metadata. Error: {e}"
            logger.exception(errstr)
            sys.exit(errstr)

        # Inform the user that the samples were successfully filtered
        infostr = \
            "The samples were successfully filtered using the " \
            "associated metadata."
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

        ioutil.save_samples(df = df_final,
                            csv_file = output_csv_path,
                            sep = ",")

    # If something went wrong
    except Exception as e:

        # Warn the user and exit
        errstr = \
            "It was not possible to write the output data " \
            f"frame in '{output_csv_path}'. Error: {e}"
        logger.exception(errstr)
        sys.exit(errstr)

    # Inform the user that the data frame was successfully
    # written to the output file
    infostr = \
        "The output data frame was successfully written in " \
        f"'{output_csv_path}'."
    logger.info(infostr)