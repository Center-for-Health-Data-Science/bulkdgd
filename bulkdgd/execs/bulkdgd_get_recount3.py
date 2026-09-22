#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-

#    bulkdgd_get_recount3.py
#
#    Get RNA-seq data associated with specific human samples for
#    projects hosted on the Recount3 platform.
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
    "Get RNA-seq data associated with specific human samples for " \
    "projects hosted on the Recount3 platform."


#######################################################################


# Import from the standard library.
import argparse
import logging as log
import os
import sys

# Import from third-party libraries.
from distributed import LocalCluster, Client, as_completed

# Import from 'bulkdgd'.
from bulkdgd import recount3
from . import util


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

    # Set a help message.
    ib_help = "A CSV file used to download samples' data in bulk."

    # Add the argument to the group.
    input_group.add_argument("-ib", "--input-samples-batches",
                             type = str,
                             default = None,
                             help = ib_help)

    #-----------------------------------------------------------------#

    # Set a help message.
    sg_help = \
        """Save the original GZ file containing the RNA-seq data for
        the samples. For each batch of samples, the corresponding file
        will be saved in the working directory and named
        '{recount3_project_name}_""" \
        "{recount3_samples_category}_gene_sums.gz'."

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
        named '{recount3_project_name}_""" \
        "{recount3_samples_category}_qc.gz'."

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
        '{recount3_project_name}_""" \
        "{recount3_samples_category}_metadata.gz'."

    # Add the argument to the group.
    output_group.add_argument("-sm", "--save-metadata",
                              action = "store_true",
                              help = sm_help)

    #-----------------------------------------------------------------#

    # Set the default value for the argument.
    n_default = 1

    # Set a help message.
    n_help = \
        f"""The number of processes to start. The default number of
        processes started is {n_default}."""

    # Add the argument to the group.
    run_group.add_argument("-n", "--n-proc",
                           type = int,
                           default = n_default,
                           help = n_help)

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
    """Get the RNA-seq data for each batch of samples.

    Parameters
    ----------
    args : :class:`argparse.Namespace`
        The parsed arguments.
    """

    # Get the argument corresponding to the input file.
    input_samples_batches = args.input_samples_batches

    # Get the arguments corresponding to the output files.
    save_gene_sums = args.save_gene_sums
    save_qc = args.save_qc
    save_metadata = args.save_metadata

    # Get the argument corresponding to the working directory.
    wd = args.work_dir

    # Get the arguments corresponding to the run options.
    n_proc = args.n_proc

    # Get the logging arguments to pass to each batch's run.
    log_console = args.log_console
    v = args.log_verbose
    vv = args.log_debug

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

    # Try to load the samples' batches.
    try:

        df = recount3.load_samples_batches(\
            samples_file = input_samples_batches)

    # If something went wrong
    except Exception as e:

        # Warn the user and exit.
        errstr = \
            "It was not possible to load the samples' batches " \
            f"from '{input_samples_batches}'. Error: {e}"
        logger.exception(errstr)
        sys.exit(errstr)

    # Inform the user that the batches were successfully loaded.
    infostr = \
        "The samples' batches were successfully loaded from " \
        f"'{input_samples_batches}'."
    logger.info(infostr)

    #-----------------------------------------------------------------#

    # Create a list to store the futures.
    futures = []

    # Create a set to store the names of the output/log files.
    output_names = set()

    # For each row of the data frame containing the samples' batches
    for num_batch, row in enumerate(df.itertuples(index = False), 1):

        # Get the name of the project.
        project_name = row.recount3_project_name

        # Get the samples' category.
        samples_category = row.recount3_samples_category

        #-------------------------------------------------------------#

        # Get the overall name for the output/log files.
        output_name = f"{project_name}_{samples_category}"

        # Set a counter to make the name unique.
        counter = 1

        # While the name already exists
        while output_name in output_names:

            # Add the counter to the name.
            output_name = \
                f"{project_name}_{samples_category}_{counter}"

            # Update the counter.
            counter += 1

        # Add the name to the set of names.
        output_names.add(output_name)

        #-------------------------------------------------------------#

        # Get the name of the output file.
        output_csv_name = f"{output_name}.csv"

        # Get the path to the output file.
        output_csv_path = os.path.join(wd, output_csv_name)

        # Inform the user that the results will be written to the
        # specified output file.
        infostr = \
            f"The results for batch # {num_batch} ('{project_name}' " \
            f"project, '{samples_category}' samples) will be " \
            f"written in '{output_csv_path}'."
        logger.info(infostr)

        #-------------------------------------------------------------#

        # Get the name of the log file.
        log_file_name = f"{output_name}.log"

        # Get the path to the log file.
        log_file_path = os.path.join(wd, log_file_name)

        # Inform the user about the log file for the current batch
        # of samples.
        infostr = \
            f"The log messages for batch # {num_batch} " \
            f"('{project_name}' project, '{samples_category}' " \
            f"samples) will be written in '{log_file_path}'."
        logger.info(infostr)

        #-------------------------------------------------------------#

        # Set the arguments for the batch's run.
        args = \
            ["-ip", str(project_name),
             "-is", str(samples_category),
             "-os", output_csv_path,
             "-d", wd,
             "-lf", log_file_path]

        # If the user wants to save the original 'gene_sums' files
        if save_gene_sums:

            # Add the option for it to the list of arguments.
            args.append("-sg")

        # If the user wants to save the original 'qc' files
        if save_qc:

            # Add the option for it to the list of arguments.
            args.append("-sq")

        # If the user wants to save the original 'metadata' files
        if save_metadata:

            # Add the option for it to the list of arguments.
            args.append("-sm")

        # If the user requested logging to the console
        if log_console:

            # Add the option for it to the list of arguments.
            args.append("-lc")

        # If the user requested verbose logging
        if v:

            # Add the option for it to the list of arguments.
            args.append("-v")

        # If the user requested debug-level logging
        if vv:

            # Add the option for it to the list of arguments.
            args.append("-vv")

        #-------------------------------------------------------------#

        # Submit the calculation.
        futures.append(\
            client.submit(\
                util.run_executable,
                executable = "_bulkdgd_get_recount3_single_batch",
                arguments = args,
                extra_return_values = [num_batch, log_file_path]))

    #-----------------------------------------------------------------#

    # Get the futures as they are completed.
    for _, result in as_completed(futures,
                                  with_results = True):

        # Get the process, the batch number and the log file from the
        # current future.
        process, num_batch, log_file_path = result

        # Check the process' return code.
        try:

            process.check_returncode()

        # If something went wrong
        except Exception as e:

            # Log the error.
            errstr = \
                f"The run for batch # {num_batch} failed. Please " \
                f"check the log file '{log_file_path}' for " \
                f"more details. Error: {e}"
            logger.error(errstr)

            # Go to the next future.
            continue

        # Inform the user that the run completed successfully.
        infostr = \
            f"The run for batch # {num_batch} completed successfully."
        logger.info(infostr)

    #-----------------------------------------------------------------#

    # Close the client.
    client.close()

    # Close the cluster.
    cluster.close()


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
