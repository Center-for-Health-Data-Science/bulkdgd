#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-

#    dgd_get_genes_list.py
#
#    Create a customized list of genes that can be used to build a
#    new bulkDGD model.
#
#    The code was originally developed by Nafisa Barmakhshad, and
#    adapted and converted into an executable for this package by
#    Valentina Sora.
#
#    Copyright (C) 2024 Valentina Sora 
#                       <sora.valentina1@gmail.com>
#                       Nafisa Barmakhshad
#                       <nafisa.barmakhshad@gmail.com>
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
    "Create a customized list of genes that can be used to build a " \
    "new bulkDGD model."


#######################################################################


# Import from the standard library.
import argparse
import logging as log
import logging.handlers as loghandlers
import os
import sys
# Import from third-party packages.
import dask
import pandas as pd
# Import from 'bulkDGD'.
from bulkDGD import defaults, ioutil, genes, util


#######################################################################


def main():


    # Create the argument parser.
    parser = argparse.ArgumentParser(description = __doc__)

    #-----------------------------------------------------------------#

    # Add the arguments.
    ol_default = "genes_list.txt"
    ol_help = \
        "The name of the output plain text file containing the " \
        "list of genes of interest, identified using their Ensembl " \
        "IDs. The file will be written in the working directory. " \
        f"The default file name is '{ol_default}'."
    parser.add_argument("-ol", "--output-txt-list",
                        type = str,
                        default = ol_default,
                        help = ol_help)

    #-----------------------------------------------------------------#

    oa_default = "genes_attributes.csv"
    oa_help = \
        "The name of the output CSV file containing the attributes " \
        "retrieved from the Ensembl database for the genes of " \
        "interest. The file will be written in the working " \
        f"directory. The default file name is '{oa_default}'."
    parser.add_argument("-oa", "--output-csv-attributes",
                        type = str,
                        default = oa_default,
                        help = oa_help)

    #-----------------------------------------------------------------#

    cg_help = \
        "The YAML configuration file containing the options used " \
        "to query the Ensembl database for the genes of interest. " \
        "If it is a name without an extension, it is assumed to be " \
        "the name of a configuration file in " \
        f"'{ioutil.CONFIG_GENES_DIR}'."
    parser.add_argument("-cg", "--config-file-genes",
                        type = str,
                        required = True,
                        help = cg_help)

    #-----------------------------------------------------------------#

    d_help = \
        "The working directory. The default is the current " \
        "working directory."
    parser.add_argument("-d", "--work-dir",
                        type = str,
                        default = os.getcwd(),
                        help = d_help)

    #-----------------------------------------------------------------#

    lf_default = "dgd_get_genes_list.log"
    lf_help = \
        "The name of the log file. The file will be written " \
        "in the working directory. The default file name is " \
        f"'{lf_default}'."
    parser.add_argument("-lf", "--log-file",
                        type = str,
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
    output_txt_list = args.output_txt_list
    output_csv_attributes = args.output_csv_attributes
    config_file_genes = args.config_file_genes
    wd = os.path.abspath(args.work_dir)
    log_file = os.path.join(wd, args.log_file)
    log_console = args.log_console
    v = args.log_verbose
    vv = args.log_debug

    #-----------------------------------------------------------------#

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
            log_console_level = log_level,
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

    # Try to load the configuration.
    try:

        config_genes = ioutil.load_config_genes(config_file_genes)

    # If something went wrong
    except Exception as e:

        # Warn the user and exit.
        errstr = \
            "It was not possible to load the configuration from " \
            f"'{config_file_genes}'. Error: {e}"
        log.exception(errstr)
        sys.exit(errstr)

    # Inform the user that the configuration was successfully loaded.
    infostr = \
        "The configuration was successfully loaded from " \
        f"'{config_file_genes}'."
    log.info(infostr)

    #-----------------------------------------------------------------#

    # Get the attributes to be retrieved for the genes.
    attributes = config_genes.get("attributes", [])

    # If the 'ensembl_gene_id' attribute is not present
    if "ensembl_gene_id" not in attributes:

        # Add it.
        attributes.append("ensembl_gene_id")

    # Get the filters to be used when retrieving the genes' attributes.
    filters = config_genes.get("filters", {})

    #-----------------------------------------------------------------#

    # Try to get the genes' attributes from Ensembl.
    try:

        genes_attributes = \
            genes.get_genes_attributes(\
                attributes = attributes,
                filters = filters,
                dataset = "hsapiens_gene_ensembl")

    # If something went wrong
    except Exception as e:

        # Warn the user and exit.
        errstr = \
            "It was not possible to get the genes' attributes " \
            f"from the Ensembl database. Error: {e}"
        log.exception(errstr)
        sys.exit(errstr)

    # Inform the user that the genes' attributes were successfully
    # retrieved from Ensembl.
    infostr = \
        "The genes' attributes were successfully retrieved from " \
        "the Ensembl database."
    log.info(infostr)

    #-----------------------------------------------------------------#

    # Get the list of Ensembl IDs of the genes retrieved.
    genes_list = \
        genes_attributes["Gene stable ID"].unique().tolist()

    #-----------------------------------------------------------------#

    # Try to write out the data frame containing the genes' attribues.
    try:

        genes_attributes.to_csv(output_csv_attributes,
                                sep = ",",
                                index = False,
                                header = True)

    # If something went wrong
    except Exception as e:

        # Warn the user and exit.
        errstr = \
            "It was not possible to write the data frame containing " \
            f"the genes' attributes in '{output_csv_attributes}'. " \
            f"Error: {e}"
        log.exception(errstr)
        sys.exit(errstr)

    # Inform the user that the data frame was successfully saved.
    infostr = \
        "The data frame containing the genes' attributes was " \
        f"successfully written in '{output_csv_attributes}'."
    log.info(infostr)

    #-----------------------------------------------------------------#

    # Try to write out the plain txt file contaning the genes' list.
    try:

        with open(output_txt_list, "w") as out:
            out.write("\n".join(genes_list))

    # If something went wrong
    except Exception as e:

        # Warn the user and exit.
        errstr = \
            "It was not possible to write the genes' list in " \
            f"'{output_txt_list}'. Error: {e}"
        log.exception(errstr)
        sys.exit(errstr)

    # Inform the user that the list was successfully saved.
    infostr = \
        "The genes' list was successfully written in " \
        f"'{output_txt_list}'."
    log.info(infostr)


#######################################################################


if __name__ == "__main__":
    main()
