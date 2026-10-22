#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-

#    bulkdgd_get_recount3.py
#
#    Build a dataset of read counts and harmonized metadata from
#    studies hosted on the Recount3 platform.
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
    "Build a dataset of read counts and harmonized metadata from " \
    "studies hosted on the Recount3 platform."


#######################################################################


# Import from the standard library.
import argparse
import logging as log
import os
import sys
import warnings

# Import from third-party libraries.
import pandas as pd

# Import from 'bulkdgd'.
from bulkdgd import defaults, ioutil, recount3
from . import util


#######################################################################


# Ignore the warnings about fragmented data frames.
warnings.filterwarnings("ignore",
                        category = pd.errors.PerformanceWarning)

# Ignore the warnings about pandas' future behavior.
warnings.filterwarnings("ignore",
                        category = FutureWarning)

# If pandas has the warning about setting values on copies
if hasattr(pd.errors,
           "SettingWithCopyWarning"):

    # Ignore the warnings.
    warnings.filterwarnings(\
        "ignore",
        category = pd.errors.SettingWithCopyWarning)


#######################################################################


# Get the module's logger.
logger = log.getLogger(__name__)


#######################################################################


# Define a function to set up the parser.
def set_parser() -> argparse.ArgumentParser:
    """Set up the argument parser.

    Returns
    -------
    parser : :class:`argparse.ArgumentParser`
        The argument parser.
    """

    # Create the argument parser.
    parser = \
        argparse.ArgumentParser(\
            description = __doc__,
            formatter_class = util.CustomHelpFormatter)

    #-----------------------------------------------------------------#

    # Create a group of arguments for input files.
    input_group = \
        parser.add_argument_group(title = "Input files")

    # Create a group of arguments for output files.
    output_group = \
        parser.add_argument_group(title = "Output files")

    # Create a group of arguments for the run options.
    run_group = \
        parser.add_argument_group(title = "Run options")

    #-----------------------------------------------------------------#

    # Set the default directory where the configuration files are
    # located.
    ic_dir = defaults.CONFIG_DIRS["recount3"]

    # Set a help message.
    ic_help = \
        "The YAML configuration file selecting the studies and " \
        "samples the dataset is built from and harmonizing their " \
        "metadata. If it is a name without an extension, it is " \
        "assumed to be the name of a configuration file in " \
        f"'{ic_dir}'."

    # Add the argument to the group.
    input_group.add_argument("-ic", "--input-config-file",
                             type = str,
                             required = True,
                             help = ic_help)

    #-----------------------------------------------------------------#

    # Set a help message.
    ip_help = \
        "The Recount3 project the studies belong to. You can " \
        "choose between: 'GTEx', whose studies are tissues, " \
        "'TCGA', whose studies are cancer types, and 'SRA', whose " \
        "studies are SRA projects."

    # Add the argument to the group.
    input_group.add_argument("-ip", "--input-project-name",
                             type = str,
                             required = True,
                             choices = recount3.RECOUNT3_PROJECTS,
                             help = ip_help)

    #-----------------------------------------------------------------#

    # Set the default value for the argument.
    ig_default = 29

    # Set a help message.
    ig_help = \
        "The GENCODE release the counts were annotated with. You " \
        "can choose between: 26 and 29. The default is " \
        f"{ig_default}."

    # Add the argument to the group.
    input_group.add_argument(\
        "-ig", "--input-gencode-release",
        type = int,
        default = ig_default,
        choices = sorted(recount3.RECOUNT3_GENCODE_RELEASES),
        help = ig_help)

    #-----------------------------------------------------------------#

    # Set a help message.
    iec_help = \
        "An existing CSV file of read counts, with samples as " \
        "rows, to take the metadata's samples from. Only the " \
        "studies of the samples it lacks are retrieved. A " \
        "relative path is taken from the current directory, not " \
        "from the working directory. By default, the counts of " \
        "all the studies are retrieved."

    # Add the argument to the group.
    input_group.add_argument("-iec", "--input-existing-counts",
                             type = str,
                             default = None,
                             help = iec_help)

    #-----------------------------------------------------------------#

    # Set the default value for the argument.
    oc_default = "counts.csv"

    # Set a help message.
    oc_help = \
        "The output CSV file of the read counts, with samples as " \
        "rows and genes as columns. The default file name is " \
        f"'{oc_default}'."

    # Add the argument to the group.
    output_group.add_argument("-oc", "--output-counts",
                              type = str,
                              default = oc_default,
                              help = oc_help)

    #-----------------------------------------------------------------#

    # Set the default value for the argument.
    om_default = "metadata.csv"

    # Set a help message.
    om_help = \
        "The output CSV file of the harmonized metadata, with " \
        "samples as rows. The default file name is " \
        f"'{om_default}'."

    # Add the argument to the group.
    output_group.add_argument("-om", "--output-metadata",
                              type = str,
                              default = om_default,
                              help = om_help)

    #-----------------------------------------------------------------#

    # Set the default value for the argument.
    occ_default = "counts_columns.txt"

    # Set a help message.
    occ_help = \
        "The output plain text file of the counts' columns (the " \
        "genes), one per line. The default file name is " \
        f"'{occ_default}'."

    # Add the argument to the group.
    output_group.add_argument("-occ", "--output-counts-columns",
                              type = str,
                              default = occ_default,
                              help = occ_help)

    #-----------------------------------------------------------------#

    # Set the default value for the argument.
    omc_default = "metadata_columns.txt"

    # Set a help message.
    omc_help = \
        "The output plain text file of the metadata's columns, one " \
        f"per line. The default file name is '{omc_default}'."

    # Add the argument to the group.
    output_group.add_argument("-omc", "--output-metadata-columns",
                              type = str,
                              default = omc_default,
                              help = omc_help)

    #-----------------------------------------------------------------#

    # Set the default value for the argument.
    osi_default = "samples_ids.txt"

    # Set a help message.
    osi_help = \
        "The output plain text file of the samples' IDs, one per " \
        "line, in the counts' order. The default file name is " \
        f"'{osi_default}'."

    # Add the argument to the group.
    output_group.add_argument("-osi", "--output-samples-ids",
                              type = str,
                              default = osi_default,
                              help = osi_help)

    #-----------------------------------------------------------------#

    # Set a help message.
    odc_help = \
        "The directory of the RNA-seq data files downloaded from " \
        "Recount3, which are not downloaded again if present. The " \
        "default is the working directory."

    # Add the argument to the group.
    output_group.add_argument("-odc", "--output-dir-counts",
                              type = str,
                              default = None,
                              help = odc_help)

    #-----------------------------------------------------------------#

    # Set a help message.
    odq_help = \
        "The directory of the QC metadata files downloaded from " \
        "Recount3, which are not downloaded again if present. The " \
        "default is the working directory."

    # Add the argument to the group.
    output_group.add_argument("-odq", "--output-dir-qc",
                              type = str,
                              default = None,
                              help = odq_help)

    #-----------------------------------------------------------------#

    # Set a help message.
    odm_help = \
        "The directory of the metadata files downloaded from " \
        "Recount3 (and, for SRA studies, from ENA, GEO and the SRA " \
        "Run Selector), which are not downloaded again if present. " \
        "The default is the working directory."

    # Add the argument to the group.
    output_group.add_argument("-odm", "--output-dir-metadata",
                              type = str,
                              default = None,
                              help = odm_help)

    #-----------------------------------------------------------------#

    # Set a help message.
    st_help = \
        "The studies to take from the configuration file: tissues " \
        "for GTEx, cancer types for TCGA, or SRA projects' codes " \
        "for SRA. By default, all of them are taken."

    # Add the argument to the group.
    run_group.add_argument("-st", "--studies",
                           type = str,
                           nargs = "+",
                           default = None,
                           help = st_help)

    #-----------------------------------------------------------------#

    # Set a help message.
    mo_help = "Write only the metadata, without getting the counts."

    # Add the argument to the group.
    run_group.add_argument("-mo", "--metadata-only",
                           action = "store_true",
                           help = mo_help)

    #-----------------------------------------------------------------#

    # Set a help message.
    se_help = \
        "Skip the run if all its output files already exist. By " \
        "default, the output files are overwritten."

    # Add the argument to the group.
    run_group.add_argument("-se", "--skip-existing",
                           action = "store_true",
                           help = se_help)

    #-----------------------------------------------------------------#

    # Add the working directory and logging arguments.
    util.add_wd_and_logging_arguments(\
        parser = parser,
        command_name = "bulkdgd_get_recount3")

    #-----------------------------------------------------------------#

    # Return the parser.
    return parser


#######################################################################


# Define the 'main' function.
def main(args: argparse.Namespace) -> None:
    """Build the dataset and write it out.

    Parameters
    ----------
    args : :class:`argparse.Namespace`
        The parsed arguments.
    """

    # Get the argument corresponding to the working directory.
    wd = args.work_dir

    # Get the arguments corresponding to the input files.
    input_config_file = args.input_config_file
    input_project_name = args.input_project_name
    input_gencode_release = args.input_gencode_release
    input_existing_counts = args.input_existing_counts

    # Get the arguments corresponding to the output files.
    output_counts = \
        os.path.join(wd,
                     args.output_counts)
    output_metadata = \
        os.path.join(wd,
                     args.output_metadata)
    output_counts_columns = \
        os.path.join(wd,
                     args.output_counts_columns)
    output_metadata_columns = \
        os.path.join(wd,
                     args.output_metadata_columns)
    output_samples_ids = \
        os.path.join(wd,
                     args.output_samples_ids)

    # Get the arguments corresponding to the download directories (the
    # working directory if not passed).
    output_dir_counts, output_dir_qc, output_dir_metadata = \
        [wd if d is None else os.path.join(wd,
                                           d) \
         for d in (args.output_dir_counts,
                   args.output_dir_qc,
                   args.output_dir_metadata)]

    # Get the arguments corresponding to the run options.
    studies = args.studies
    metadata_only = args.metadata_only
    skip_existing = args.skip_existing

    #-----------------------------------------------------------------#

    # Get the files the run writes.
    output_files = \
        [output_metadata, output_metadata_columns] + \
        ([] if metadata_only \
         else [output_counts,
               output_counts_columns,
               output_samples_ids])

    # If they all exist and are to be kept
    if skip_existing \
    and all(os.path.exists(f) for f in output_files):

        # Inform the user that the run is skipped.
        infostr = "All output files already exist. Skipping."
        logger.info(infostr)

        # Return.
        return

    # For each file the run writes
    for output_file in output_files:

        # Create its directory if it does not exist.
        os.makedirs(os.path.dirname(output_file),
                    exist_ok = True)

    #-----------------------------------------------------------------#

    # Try to load the configuration.
    try:

        config = \
            ioutil.load_config_recount3(\
                config_file = input_config_file,
                project_name = input_project_name)

    # If something went wrong
    except Exception as e:

        # Warn the user and exit.
        errstr = \
            "It was not possible to load the configuration from " \
            f"'{input_config_file}'. Error: {e}"
        logger.exception(errstr)
        sys.exit(errstr)

    # Inform the user that the configuration was successfully loaded.
    infostr = \
        "The configuration was successfully loaded from " \
        f"'{input_config_file}'."
    logger.info(infostr)

    #-----------------------------------------------------------------#

    # Try to get the metadata.
    try:

        df_metadata = \
            recount3.get_dataset_metadata(\
                config = config,
                project_name = input_project_name,
                wd_metadata = output_dir_metadata,
                samples_categories = studies)

    # If something went wrong
    except Exception as e:

        # Warn the user and exit.
        errstr = \
            "It was not possible to get the dataset's metadata. " \
            f"Error: {e}"
        logger.exception(errstr)
        sys.exit(errstr)

    # Inform the user that the metadata were successfully generated.
    infostr = \
        f"The metadata of {len(df_metadata):,} samples were " \
        "successfully generated."
    logger.info(infostr)

    #-----------------------------------------------------------------#

    # Try to write the metadata and their columns.
    try:

        # Write the metadata.
        recount3.save_dataset_metadata(df = df_metadata,
                                       csv_file = output_metadata)

        # Write the metadata's columns.
        with open(output_metadata_columns,
                  "w") as out:
            out.write("\n".join(map(str,
                                    df_metadata.columns)))

    # If something went wrong
    except Exception as e:

        # Warn the user and exit.
        errstr = \
            "It was not possible to write the metadata in " \
            f"'{output_metadata}' and their columns in " \
            f"'{output_metadata_columns}'. Error: {e}"
        logger.exception(errstr)
        sys.exit(errstr)

    # Inform the user that the metadata were successfully written.
    infostr = \
        "The metadata were successfully written in " \
        f"'{output_metadata}' and their columns in " \
        f"'{output_metadata_columns}'."
    logger.info(infostr)

    #-----------------------------------------------------------------#

    # If only the metadata are written
    if metadata_only:

        # Return.
        return

    #-----------------------------------------------------------------#

    # Initialize the existing counts.
    df_counts_existing = None

    # If an existing file of counts was passed
    if input_existing_counts is not None:

        # Try to load the existing counts.
        try:

            df_counts_existing = \
                pd.read_csv(input_existing_counts,
                            sep = ",",
                            header = 0,
                            index_col = 0,
                            low_memory = False)

        # If something went wrong
        except Exception as e:

            # Warn the user and exit.
            errstr = \
                "It was not possible to load the existing counts " \
                f"from '{input_existing_counts}'. Error: {e}"
            logger.exception(errstr)
            sys.exit(errstr)

        # Inform the user that the existing counts were loaded.
        infostr = \
            "The existing counts were successfully loaded from " \
            f"'{input_existing_counts}'."
        logger.info(infostr)

    #-----------------------------------------------------------------#

    # Try to get the counts.
    try:

        df_counts = \
            recount3.get_dataset_counts(\
                df_metadata = df_metadata,
                project_name = input_project_name,
                gencode_release = input_gencode_release,
                wd_gene_sums = output_dir_counts,
                wd_qc = output_dir_qc,
                df_counts_existing = df_counts_existing)

    # If something went wrong
    except Exception as e:

        # Warn the user and exit.
        errstr = \
            "It was not possible to get the dataset's counts. " \
            f"Error: {e}"
        logger.exception(errstr)
        sys.exit(errstr)

    #-----------------------------------------------------------------#

    # Try to write the counts, their columns and the samples' IDs.
    try:

        # Write the counts.
        recount3.save_dataset_counts(df = df_counts,
                                     csv_file = output_counts)

        # Write the counts' columns.
        with open(output_counts_columns,
                  "w") as out:
            out.write("\n".join(map(str,
                                    df_counts.columns)))

        # Write the samples' IDs, in the counts' order.
        with open(output_samples_ids,
                  "w") as out:
            out.write("\n".join(df_counts.index.astype(str)))

    # If something went wrong
    except Exception as e:

        # Warn the user and exit.
        errstr = \
            "It was not possible to write the counts in " \
            f"'{output_counts}', their columns in " \
            f"'{output_counts_columns}' and the samples' IDs in " \
            f"'{output_samples_ids}'. Error: {e}"
        logger.exception(errstr)
        sys.exit(errstr)

    # Inform the user that the counts were successfully written.
    infostr = \
        f"The counts of {len(df_counts):,} samples were " \
        f"successfully written in '{output_counts}', their " \
        f"columns in '{output_counts_columns}' and the samples' " \
        f"IDs in '{output_samples_ids}'."
    logger.info(infostr)


#######################################################################


# Define the entry point for the standalone executable.
def entry_point() -> None:
    """Run the executable."""

    # Build the parser.
    parser = set_parser()

    # Parse the arguments.
    args = parser.parse_args()

    # Set up the logging.
    util.set_main_logging(\
        args = args,
        command_name = "bulkdgd_get_recount3")

    # Run the main function.
    main(args = args)
