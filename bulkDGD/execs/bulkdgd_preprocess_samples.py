#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-

#    bulkdgd_preprocess_samples.py
#
#    Pre-process new samples to use them with the
#    :class:`core.model.BulkDGDModel`.
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
        "Pre-process new samples to use them with the " \
        ":class:`core.model.BulkDGDModel`."


#######################################################################


# Import from the standard library.
import logging as log
import os
import sys
# Import from 'bulkDGD'.
from bulkDGD import ioutil
from . import util


#######################################################################


# Get the module's logger.
logger = log.getLogger(__name__)


#######################################################################


# Define a function to set up the parser.
def set_sub_parser(sub_parsers):

    # Create the argument parser.
    parser = \
        sub_parsers.add_parser(\
            name = "samples",
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

    #-----------------------------------------------------------------#

    # Set a help message.
    is_help = \
        "The input CSV file containing a data frame with the " \
        "samples to be preprocessed."

    # Add the argument to the group.
    input_group.add_argument("-is", "--input-samples",
                             type = str,
                             required = True,
                             help = is_help)

    #-----------------------------------------------------------------#

    # Set the default value for the argument.
    os_default = "samples_preprocessed.csv"

    # Set a help message.
    os_help = \
        "The name of the output CSV file containing the data frame " \
        "with the preprocessed samples. The file will be written " \
        "in the working directory. The default file name is " \
        f"'{os_default}'."

    # Add the argument to the group.
    output_group.add_argument("-os", "--output-samples",
                              type = str,
                              default = os_default,
                              help = os_help)

    #-----------------------------------------------------------------#

    # Set the default value for the argument.
    oge_default = "genes_excluded.txt"

    # Set a help message.
    oge_help = \
        "The name of the output plain text file containing the " \
        "list of genes whose expression data are excluded from the " \
        "data frame with the preprocessed samples. The " \
        "file will be written in the working directory. " \
        f"The default file name is '{oge_default}'."

    # Add the argument to the group.
    output_group.add_argument("-oe", "--output-genes-excluded",
                              type = str,
                              default = oge_default,
                              help = oge_help)

    #-----------------------------------------------------------------#

    # Set the default value for the argument.
    ogm_default = "genes_missing.txt"

    # Set a help message.
    ogm_help = \
        "The name of the output plain text file containing the list " \
        "of genes for which no expression data are found in the " \
        "input data frame. A default count of 0 is assigned to " \
        "these genes in the output data frame containing the " \
        "preprocessed samples. The file will be written in the " \
        f"working directory. The default file name is '{ogm_default}'."

    # Add the argument to the group.
    output_group.add_argument("-om", "--output-genes-missing",
                              type = str,
                              default = ogm_default,
                              help = ogm_help)

    #-----------------------------------------------------------------#

    # Return the parser.
    return parser


#######################################################################


# Define the 'main' function.
def main(args):

    # Get the argument corresponding to the working directory.
    wd = args.work_dir

    # Get the argument corresponding to the input file.
    input_samples = args.input_samples

    # Get the arguments corresponding to the output files.
    output_samples = \
        os.path.join(wd, args.output_samples)
    output_genes_excluded = \
        os.path.join(wd, args.output_genes_excluded)
    output_genes_missing = \
        os.path.join(wd, args.output_genes_missing)

    #-----------------------------------------------------------------#

    # Try to load the input samples.
    try:

        df_samples = \
            ioutil.load_samples(csv_file = input_samples,
                                sep = ",",
                                keep_samples_names = True,
                                split = False)

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
        f"The samples were successfully loaded from '{input_samples}'."
    logger.info(infostr)

    #-----------------------------------------------------------------#

    # Try to preprocess the samples.
    try:

        df_preproc, genes_excluded, genes_missing = \
            ioutil.preprocess_samples(df_samples = df_samples)

    # If something went wrong
    except Exception as e:

        # Warn the user and exit.
        errstr = \
            "It was not possible to pre-process the samples. " \
            f"Error: {e}"
        logger.exception(errstr)
        sys.exit(errstr)

    # Inform the user that the samples were successfully pre-processed.
    infostr = "The samples were successfully pre-processed."
    logger.info(infostr)

    #-----------------------------------------------------------------#

    # Try to write out the preprocessed samples.
    try:

        ioutil.save_samples(df = df_preproc,
                            csv_file = output_samples,
                            sep = ",")

    # If something went wrong
    except Exception as e:

        # Warn the user and exit.
        errstr = \
            "It was not possible to write the preprocessed " \
            f"samples in '{output_samples}'. Error: {e}"
        logger.exception(errstr)
        sys.exit(errstr)

    # Inform the user that the preprocessed samples were
    # successfully written in the output file.
    infostr = \
        "The preprocessed samples were successfully written in " \
        f"'{output_samples}'."
    logger.info(infostr)

    #-----------------------------------------------------------------#

    # If some genes were excluded
    if genes_excluded:

        # Try to write the list of excluded genes.
        try:

            with open(output_genes_excluded, "w") as out:
                out.write("\n".join(gene for gene in genes_excluded))

        # If something went wrong
        except Exception as e:

            # Warn the user and exit.
            errstr = \
                "It was not possible to write the list of genes " \
                "that are present in the input samples but are " \
                "not among the genes included in the bulkDGD model " \
                f"in '{output_genes_excluded}'. Error: {e}"
            logger.exception(errstr)
            sys.exit(errstr)

        # Inform the user that the genes were successfully written
        # to the output file.
        infostr = \
            "The list of genes that are present in the input " \
            "samples but are not among the genes included in " \
            "the bulkDGD model was successfully written in " \
            f"'{output_genes_excluded}'."
        logger.info(infostr)

    #-----------------------------------------------------------------#

    # If some genes were missing
    if genes_missing:

        # Try to write the list of missing genes.
        try:

            with open(output_genes_missing, "w") as out:
                out.write("\n".join(gene for gene in genes_missing))

        # If something went wrong
        except Exception as e:

            # Warn the user and exit.
            errstr = \
                "It was not possible to write the list of genes " \
                "that are included in the bulkDGD model but are " \
                "not present in the input samples in " \
                f"'{output_genes_missing}'. Error: {e}"
            logger.exception(errstr)
            sys.exit(errstr)

        # Inform the user that the genes were successfully written in
        # the output file.
        infostr = \
            "The list of genes that are included in the bulkDGD " \
            "model but are not present in the input samples " \
            "was successfully written in " \
            f"'{output_genes_missing}'."
        logger.info(infostr)
