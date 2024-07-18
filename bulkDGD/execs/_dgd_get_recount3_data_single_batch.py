#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-

#    _dgd_get_recount3_data_single_batch.py
#
#    Get RNA-seq data associated with a single set of human samples
#    for projects hosted on the Recount3 platform.
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
    "Get RNA-seq data associated with a single set of human samples " \
    "for projects hosted on the Recount3 platform."


#######################################################################


# Import from the standard library.
import argparse
import logging as log
import os
import sys
# Import from third-party packages.
import pandas as pd
# Import from 'bulDGD'.
from bulkDGD import defaults, ioutil, recount3, util


#######################################################################


def main():


    # Create the argument parser.
    parser = argparse.ArgumentParser()

    #-----------------------------------------------------------------#

    # Add the arguments.
    ip_choices = ["gtex", "tcga", "sra"]
    ip_choices_str = ", ".join(f"'{choice}'" for choice in ip_choices)
    ip_help = \
        "The name of the Recount3 project for which samples will " \
        f"be retrieved. The available projects are: {ip_choices_str}."
    parser.add_argument("-ip", "--input-project-name",
                        required = True,
                        choices = ip_choices,
                        help = ip_help)

    #-----------------------------------------------------------------#

    is_help = \
        "The category of samples for which RNA-seq data will be " \
        "retrieved. For GTEx data, this is the name of the tissue " \
        "the samples belong to. " \
        "For TCGA data, this is the type of cancer the samples are " \
        "associated with." \
        "For SRA data, this is the code associated with the project."
    parser.add_argument("-is", "--input-samples-category",
                        required = True,
                        help = is_help)

    #-----------------------------------------------------------------#

    o_help = \
        "The name of the output CSV file containing the data frame " \
        "with the RNA-seq data for the samples. The file will be " \
        "written in the working directory. The default file name is " \
        "'{input_project_name}_{input_samples_category}.csv'."
    parser.add_argument("-o", "--output-csv",
                        default = None,
                        help = o_help)

    #-----------------------------------------------------------------#

    d_help = \
        "The working directory. The default is the current " \
        "working directory."
    parser.add_argument("-d", "--work-dir",
                        default = os.getcwd(),
                        help = d_help)

    #-----------------------------------------------------------------#

    sg_help = \
        "Save the original GZ file containing the RNA-seq " \
        "data for the samples. The file will be saved in the " \
        "working directory and named " \
        "'{input_project_name}_{input_samples_category}_" \
        "gene_sums.gz'."
    parser.add_argument("-sg", "--save-gene-sums",
                        action = "store_true",
                        help = sg_help)

    #-----------------------------------------------------------------#

    sm_help = \
        "Save the original GZ file containing the metadata " \
        "for the samples. The file will be saved in the working " \
        "directory and named " \
        "'{input_project_name}_{input_samples_category}_" \
        "metadata.gz'."
    parser.add_argument("-sm", "--save-metadata",
                        action = "store_true",
                        help = sm_help)

    #-----------------------------------------------------------------#

    qs_help = \
        "The string that will be used to filter the samples " \
        "according to their associated metadata using the " \
        "'pandas.DataFrame.query()' method. The option also " \
        "accepts a plain text file containing the string " \
        "since it can be long for complex queries."
    parser.add_argument("-qs", "--query-string",
                        default = None,
                        help = qs_help)

    #-----------------------------------------------------------------#

    mk_help = \
        "A vertical line (|)-separated list of names of metadata " \
        "columns to keep in the final data frame. All the other " \
        "metadata columns will be dropped from the data frame. If " \
        "neither this option nor the '-md', '--metadata-to-drop' " \
        "is passed, all metadata columns are kept in the final data " \
        "frame."
    parser.add_argument("-mk", "--metadata-to-keep",
                        default = None,
                        help = mk_help)

    #-----------------------------------------------------------------#

    md_help = \
        "A vertical line (|)-separated list of names of metadata " \
        "columns to drop in the final data frame. All the other " \
        "metadata columns will be kept in the final data frame. Use " \
        "the '_all_' reserved keyword to drop all metadata columns " \
        "from the data frame. If neither this option nor the '-mk', " \
        "'--metadata-to-keep' option is passed, all metadata " \
        "columns are kept in the final data frame."
    parser.add_argument("-md", "--metadata-to-drop",
                        default = None,
                        help = md_help)

    #-----------------------------------------------------------------#

    lf_default = "_dgd_get_recount3_data_single_batch.log"
    lf_help = \
        "The name of the log file. The file will be written " \
        "in the working directory. The default file name is " \
        f"'{lf_default}'."
    parser.add_argument("-lf", "--log-file",
                        default = lf_default,
                        help = lf_help)

    #-----------------------------------------------------------------#

    lc_help = "Show log messages also on the console."
    parser.add_argument("-lc", "--log-console",
                        action = "store_true",
                        help = lc_help)

    #-----------------------------------------------------------------#

    v_help = "Enable verbose logging (INFO level)."
    parser.add_argument("-v", "--log-verbose",
                        action = "store_true",
                        help = v_help)
    
    #-----------------------------------------------------------------#

    vv_help = \
        "Enable maximally verbose logging for debugging " \
        "purposes (DEBUG level)."
    parser.add_argument("-vv", "--log-debug",
                        action = "store_true",
                        help = vv_help)

    #-----------------------------------------------------------------#

    # Parse the arguments.
    args = parser.parse_args()
    input_project_name = args.input_project_name
    input_samples_category = args.input_samples_category
    output_csv = args.output_csv
    wd = os.path.abspath(args.work_dir)
    query_string = args.query_string
    metadata_to_keep = args.metadata_to_keep
    metadata_to_drop = args.metadata_to_drop
    save_gene_sums = args.save_gene_sums
    save_metadata = args.save_metadata
    log_file = args.log_file
    log_console = args.log_console
    v = args.log_verbose
    vv = args.log_debug

    #-----------------------------------------------------------------#

    # Get the module's logger.
    logger = log.getLogger("dgd_get_recount3_data")

    # Set WARNING logging level by default.
    log_level = log.WARNING

    # If the user requested verbose logging
    if v:

        # The minimal logging level will be INFO.
        log_level = log.INFO

    # If the user requested logging for debug purposes
    # (-vv overrides -v if both are provided)
    if vv:

        # The minimal logging level will be DEBUG.
        log_level = log.DEBUG

    # Configure the logging.
    handlers = \
        util.get_handlers(\
            log_console = log_console,
            log_console_level = log.ERROR,
            log_file_class = log.FileHandler,
            log_file_options = {"filename" : log_file,
                                "mode" : "w"},
            log_file_level = log_level)

    # Set the logging configuration.
    log.basicConfig(level = log_level,
                    format = defaults.LOG_FMT,
                    datefmt = defaults.LOG_DATEFMT,
                    style = defaults.LOG_STYLE,
                    handlers = handlers)

    #-----------------------------------------------------------------#

    # Try to get the RNA-seq data for the samples from Recount3.
    try:
        
        df_gene_sums = \
            recount3.get_gene_sums(\
                project_name = input_project_name,
                samples_category = input_samples_category,
                save_gene_sums = save_gene_sums,
                wd = wd)

    # If something went wrong
    except Exception as e:

        # Log it an exit.
        errstr = \
            "It was not possible to get the RNA-seq data from " \
            f"Recount3. Error: {e}"
        logger.exception(errstr)
        sys.exit(errstr)

    #-----------------------------------------------------------------#

    # Try to get the metadata for the samples from Recount3.
    try:
        
        df_metadata = \
            recount3.get_metadata(\
                project_name = input_project_name,
                samples_category = input_samples_category,
                save_metadata = save_metadata,
                wd = wd)

    # If something went wrong
    except Exception as e:

        # Log it an exit.
        errstr = \
            "It was not possible to get the metadata from Recount3. " \
            f"Error: {e}"
        logger.exception(errstr)
        sys.exit(errstr)

    #-----------------------------------------------------------------#

    # Try to merge the RNA-seq data frame and the metadata data frame.
    try:
        
        df_final = \
            recount3.merge_gene_sums_and_metadata(\
                df_gene_sums = df_gene_sums,
                df_metadata = df_metadata)

    # If something went wrong
    except Exception as e:

        # Log it and exit.
        errstr = \
            "It was not possible to combine the RNA-seq data " \
            f"with the metadata. Error: {e}"
        logger.exception(errstr)
        sys.exit(errstr)

    #-----------------------------------------------------------------#
    
    # If the user has passed a query string or a file containing the
    # query string
    if query_string is not None:

        # Try to get the query string.
        try:
            
            query_string = \
                recount3.get_query_string(\
                    query_string = query_string)

        # If something went wrong
        except Exception as e:

            # Log it and exit.
            errstr = \
                "It was not possible to get the query string. " \
                f"Error: {e}"
            logger.exception(errstr)
            sys.exit(errstr)

        # Try to add the metadata to the RNA-seq data frame.
        try:

            # Filter the samples.
            df_final = \
                recount3.filter_by_metadata(\
                    df = df_final,
                    query_string = query_string)

        # If something went wrong
        except Exception as e:

            # Log it an exit.
            errstr = \
                "It was not possible to filter the RNA-seq " \
                f"by their associated metadata. Error: {e}"
            logger.exception(errstr)
            sys.exit(errstr)

    #-----------------------------------------------------------------#

    # If the user passed a list of metadata columns to keep in the
    # final data frame
    if metadata_to_keep is not None:

        # Get the list of metadata columns to keep.
        metadata_to_keep = \
            [m.lstrip("'").rstrip("'") for m \
             in metadata_to_keep.rstrip().split("|")]

        # Get the columns to keep in the final data frame.
        columns_to_keep = \
            [col for col in df_final.columns \
             if col.startswith("ENSG")] + \
            [col for col in df_final.columns \
             if not col.startswith("ENSG") \
             and col in metadata_to_keep]

        # Try to keep only the selected columns
        try:
            
            df_final = df_final.loc[:,columns_to_keep]

        # If something went wrong
        except Exception as e:

            # Get a string representing the metadata columns to keep.
            metadata_to_keep_str = \
                ", ".join([f"'{m}'" for m in metadata_to_keep])

            # Log it and exit.
            errstr = \
                "It was not possible to keep only the following " \
                "metadata columns in the final data frame: " \
                f"{metadata_to_keep_str}. Error: {e}"
            logger.exception(errstr)
            sys.exit(errstr)

    #-----------------------------------------------------------------#

    # If the user passed a list of metadata columns to drop in the
    # final data frame
    if metadata_to_drop is not None:

        # If the user wants to drop all metadata columns
        if metadata_to_drop == "_all_":

            # Get the columns to keep in the final data frame.
            columns_to_keep = \
                [col for col in df_final.columns \
                 if col.startswith("ENSG")] 

        # Otherwise
        else:

            # Get the list of metadata columns.
            metadata_to_drop = \
                [m.lstrip("'").rstrip("'") for m \
                 in metadata_to_drop.rstrip().split("|")]

            # Get the columns to keep in the final data frame.
            columns_to_keep = \
                [col for col in df_final.columns \
                 if col.startswith("ENSG")] + \
                [col for col in df_final.columns \
                 if not col.startswith("ENSG") \
                 and col not in metadata_to_drop]

        # Try to keep only the selected columns.
        try:
            
            df_final = df_final.loc[:, columns_to_keep]

        # If something went wrong
        except Exception as e:

            # Get a string representing the metadata columns to
            # drop.
            metadata_to_drop_str = \
                ", ".join([f"'{m}'" for m in metadata_to_drop])

            # Log it and exit.
            errstr = \
                "It was not possible to drop the following " \
                "metadata columns from the final data frame: " \
                f"{metadata_to_drop_str}. Error: {e}"
            logger.exception(errstr)
            sys.exit(errstr)

    #-----------------------------------------------------------------#

    # If the user did not pass a name for the output CSV file
    if output_csv is None:

        # Use the default output name.
        output_csv_path = \
            os.path.join(wd, o_default.format(input_project_name,
                                              input_samples_category))

    # Otherwise
    else:

        # Use the user-defined one.
        output_csv_path = \
            os.path.join(wd, output_csv)

    # Try to write the data frame to the output CSV file.
    try:
        
        ioutil.save_samples(df = df_final,
                            csv_file = output_csv_path,
                            sep = ",")

    # If something went wrong
    except Exception as e:

        # Log it and exit.
        errstr = \
            "It was not possible to save the final data frame in " \
            f"'{output_csv_path}'. Error: {e}"
        logger.exception(errstr)
        sys.exit(errstr)
