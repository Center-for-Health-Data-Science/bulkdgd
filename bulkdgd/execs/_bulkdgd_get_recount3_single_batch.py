#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-

#    _bulkdgd_recount3_single_batch.py
#
#    Get RNA-seq data associated with a single set of human samples
#    for projects hosted on the Recount3 platform.
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
    "Get RNA-seq data associated with a single set of human samples " \
    "for projects hosted on the Recount3 platform."


#######################################################################


# Import from the standard library.
import argparse
import logging as log
import os
import sys

# Import from third-party libraries.
import pandas as pd

# Import from 'bulDGD'.
from bulkdgd import ioutil, recount3
from . import defaults, util


#######################################################################


# Define the 'main' function.
def main() -> None:

    # Create the argument parser.
    parser = \
        argparse.ArgumentParser(\
            description = __doc__,
            formatter_class = util.CustomHelpFormatter)

    #-----------------------------------------------------------------#

    # Create a group of arguments for the input options.
    input_group = \
        parser.add_argument_group(title = "Input options")

    # Create a group of arguments for the output options.
    output_group = \
        parser.add_argument_group(title = "Output options")

    #-----------------------------------------------------------------#

    # Set the choices for the argument.
    ip_choices = ["GTEx", "TCGA", "SRA"]
    ip_choices_str = ", ".join(f"'{choice}'" for choice in ip_choices)

    # Set a help message.
    ip_help = \
        "The name of the Recount3 project for which samples will " \
        f"be retrieved. The available projects are: {ip_choices_str}."

    # Add the argument to the group.
    input_group.add_argument("-ip", "--input-project-name",
                             required = True,
                             choices = ip_choices,
                             help = ip_help)

    #-----------------------------------------------------------------#

    # Set a help message.
    is_help = \
        "The category of samples for which RNA-seq data will be " \
        "retrieved. For GTEx data, this is the name of the tissue " \
        "the samples belong to. " \
        "For TCGA data, this is the type of cancer the samples are " \
        "associated with." \
        "For SRA data, this is the code associated with the project."

    # Add the argument to the group.
    input_group.add_argument("-is", "--input-samples-category",
                             required = True,
                             help = is_help)

    #-----------------------------------------------------------------#

    # Set the default value for the argument.
    os_default = "{input_project_name}_{input_samples_category}.csv"

    # Set a help message.
    os_help = \
        "The name of the output CSV file containing the data frame " \
        "with the RNA-seq data for the samples. The file will be " \
        "written in the working directory. The default file name is " \
        f"'{os_default}'."

    # Add the argument to the group.
    output_group.add_argument("-os", "--output-samples",
                              help = os_help)

    #-----------------------------------------------------------------#

    # Set a help message.
    sg_help = \
        """Save the original GZ file containing the RNA-seq data for
        the samples. For each batch of samples, the corresponding file
        will be saved in the working directory and named
        '{recount3_project_name}_{recount3_samples_category}_gene_sums.gz'."""

    # Add the argument to the group.
    output_group.add_argument("-sg", "--save-gene-sums",
                              action = "store_true",
                              help = sg_help)

    #-----------------------------------------------------------------#

    # Set the help message.
    sq_help = \
        """Save the original GZ file containing the quality control
        metrics for the samples. For each batch of samples, the
        corresponding file will be saved in the working directory and
        named '{recount3_project_name}_{recount3_samples_category}_qc.gz'."""
    
    # Add the argument to the group.
    output_group.add_argument("-sq", "--save-qc",
                              action = "store_true",
                              help = sq_help)


    #-----------------------------------------------------------------#

    # Set a help message.
    sm_help = \
        """Save the original GZ file containing the metadata for the
        samples. For each batch of samples, the corresponding file will
        be saved in the working directory and named
        '{recount3_project_name}_{recount3_samples_category}_metadata.gz'."""

    # Add the argument to the group.
    output_group.add_argument("-sm", "--save-metadata",
                              action = "store_true",
                              help = sm_help)

    #-----------------------------------------------------------------#

    # Add arguments for the working directory and logging.
    util.add_wd_and_logging_arguments(\
        parser = parser,
        command_name = "_get_recount3_single_batch")

    #-----------------------------------------------------------------#

    # Parse the arguments.
    args = parser.parse_args()

    # Get the argument corresponding to the working directory.
    wd = args.work_dir

    # Get the arguments corresponding to the input options.
    input_project_name = args.input_project_name
    input_samples_category = args.input_samples_category

    # Get the arguments corresponding to the output options.
    output_samples = os.path.join(wd, args.output_samples)
    save_gene_sums = args.save_gene_sums
    save_qc = args.save_qc
    save_metadata = args.save_metadata

    #-----------------------------------------------------------------#

    # Get the module's logger.
    logger = log.getLogger("recount3")

    # Set WARNING logging level by default.
    log_level = log.WARNING

    # If the user requested verbose logging
    if args.log_verbose:

        # The minimal logging level will be INFO.
        log_level = log.INFO

    # If the user requested logging for debug purposes
    # (-vv overrides -v if both are provided)
    if args.log_debug:

        # The minimal logging level will be DEBUG.
        log_level = log.DEBUG

    # Configure the logging.
    handlers = \
        util.get_handlers(\
            log_console = args.log_console,
            log_console_level = log.ERROR,
            log_file_class = log.FileHandler,
            log_file_options = {"filename" : args.log_file,
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

    # Try to get the quality control metrics for the samples from
    # Recount3.
    try:

        df_qc = \
            recount3.get_qc(
                project_name = input_project_name,
                samples_category = input_samples_category,
                save_qc = save_qc,
                wd = wd)

    # If something went wrong
    except Exception as e:

        # Log it an exit.
        errstr = \
            "It was not possible to get the quality control metrics " \
            f"from Recount3. Error: {e}"
        logger.exception(errstr)
        sys.exit(errstr)

    #-----------------------------------------------------------------#

    # Try to get the metadata for the samples from Recount3.
    try:
        
        df_metadata = \
            recount3.get_metadata(
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

    # Try to compute the read counts.
    try:

        df_read_counts = \
            recount3.get_read_counts(
                df_raw_counts = df_gene_sums,
                avg_mapped_read_length = \
                    df_qc["star.average_mapped_length"],
                do_round = True)
    
    # If something went wrong
    except Exception as e:

        # Log it and exit.
        errstr = \
            "It was not possible to compute the read counts. " \
            f"Error: {e}"
        logger.exception(errstr)
        sys.exit(errstr)

    #-----------------------------------------------------------------#

    # Try to merge the RNA-seq data frame and the metadata data frame.
    try:

        # Combine the read counts data frame with the metadata data
        # frame.
        df_final = pd.concat([df_read_counts, df_metadata],
                             axis = 1)

    # If something went wrong
    except Exception as e:

        # Log it and exit.
        errstr = \
            "It was not possible to combine the RNA-seq data " \
            f"with the metadata. Error: {e}"
        logger.exception(errstr)
        sys.exit(errstr)

    #-----------------------------------------------------------------#

    # If the user did not pass a name for the output CSV file
    if output_samples is None:

        # Use the default output name.
        output_samples = \
            os.path.join(\
                wd,
                os_default.format(\
                    input_project_name = input_project_name,
                    input_samples_category = input_samples_category))

    # Otherwise
    else:

        # Use the user-defined one.
        output_samples = \
            os.path.join(wd, output_samples)

    # Try to write the data frame to the output CSV file.
    try:
        
        ioutil.save_samples(df = df_final,
                            csv_file = output_samples,
                            sep = ",")

    # If something went wrong
    except Exception as e:

        # Log it and exit.
        errstr = \
            "It was not possible to save the final data frame in " \
            f"'{output_samples}'. Error: {e}"
        logger.exception(errstr)
        sys.exit(errstr)