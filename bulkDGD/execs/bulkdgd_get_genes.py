#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-

#    bulkdgd_get_genes.py
#
#    Create a customized list of genes that can be used to build a
#    new :class:`core.model.BulkDGDModel`.
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
    "Get a customized list of genes that can be used to build a " \
    "new :class:`core.model.BulkDGDModel`."


#######################################################################


# Import from the standard library.
import logging as log
import os
import sys
# Import from 'bulkDGD'.
from bulkDGD import defaults, ioutil, genes
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
            name = "genes",
            description = __doc__,
            help = __doc__,
            formatter_class = util.CustomHelpFormatter)

    #-----------------------------------------------------------------#

    # Create a group of arguments for the output files.
    output_group = \
        parser.add_argument_group(title = "Output files")

    # Create a group of arguments for the configuration files.
    config_group = \
        parser.add_argument_group(title = "Configuration files")

    #-----------------------------------------------------------------#

    # Set the default value for the argument.
    ol_default = "genes_list.txt"

    # Set a help message.
    ol_help = \
        "The name of the output plain text file containing the " \
        "list of genes of interest, identified using their Ensembl " \
        f"IDs. The default file name is '{ol_default}'."

    # Add the argument to the group.
    output_group.add_argument("-ol", "--output-list",
                              type = str,
                              default = ol_default,
                              help = ol_help)

    #-----------------------------------------------------------------#

    # Set the default value for the argument.
    oa_default = "genes_attributes.csv"

    # Set a help message.
    oa_help = \
        "The name of the output CSV file containing the attributes " \
        "retrieved from the Ensembl database for the genes of " \
        f"interest. The default file name is '{oa_default}'."

    # Add the argument to the group.
    output_group.add_argument("-oa", "--output-attributes",
                              type = str,
                              default = oa_default,
                              help = oa_help)

    #-----------------------------------------------------------------#

    # Set the default directory where the configuration files are
    # located.
    cg_dir = defaults.CONFIG_DIRS["genes"]

    # Set a help message.
    cg_help = \
        "The YAML configuration file containing the options used " \
        "to query the Ensembl database for the genes of interest. " \
        "If it is a name without an extension, it is assumed to be " \
        f"the name of a configuration file in '{cg_dir}'."

    # Add the argument to the group.
    config_group.add_argument("-cg", "--config-file-genes",
                              type = str,
                              required = True,
                              help = cg_help)

    #-----------------------------------------------------------------#

    # Return the parser.
    return parser


#######################################################################


# Define the 'main' function.
def main(args):

    # Get the argument corresponding to the working directory.
    wd = args.work_dir

    # Get the arguments corresponding to the configuration files.
    config_file_genes = args.config_file_genes

    # Get the arguments corresponding to the output files.
    output_list = os.path.join(wd, args.output_list)
    output_attributes = os.path.join(wd, args.output_attributes)

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
        logger.exception(errstr)
        sys.exit(errstr)

    # Inform the user that the configuration was successfully loaded.
    infostr = \
        "The configuration was successfully loaded from " \
        f"'{config_file_genes}'."
    logger.info(infostr)

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
        logger.exception(errstr)
        sys.exit(errstr)

    # Inform the user that the genes' attributes were successfully
    # retrieved from Ensembl.
    infostr = \
        "The genes' attributes were successfully retrieved from " \
        "the Ensembl database."
    logger.info(infostr)

    #-----------------------------------------------------------------#

    # Get the list of Ensembl IDs of the genes retrieved.
    genes_list = \
        genes_attributes["Gene stable ID"].unique().tolist()

    #-----------------------------------------------------------------#

    # Try to write out the data frame containing the genes' attributes.
    try:

        genes_attributes.to_csv(output_attributes,
                                sep = ",",
                                index = False,
                                header = True)

    # If something went wrong
    except Exception as e:

        # Warn the user and exit.
        errstr = \
            "It was not possible to write the data frame containing " \
            f"the genes' attributes in '{output_attributes}'. " \
            f"Error: {e}"
        logger.exception(errstr)
        sys.exit(errstr)

    # Inform the user that the data frame was successfully saved.
    infostr = \
        "The data frame containing the genes' attributes was " \
        f"successfully written in '{output_attributes}'."
    logger.info(infostr)

    #-----------------------------------------------------------------#

    # Try to write out the plain txt file containing the genes' list.
    try:

        with open(output_list, "w") as out:
            out.write("\n".join(genes_list))

    # If something went wrong
    except Exception as e:

        # Warn the user and exit.
        errstr = \
            "It was not possible to write the genes' list in " \
            f"'{output_list}'. Error: {e}"
        logger.exception(errstr)
        sys.exit(errstr)

    # Inform the user that the list was successfully saved.
    infostr = \
        f"The genes' list was successfully written in '{output_list}'."
    logger.info(infostr)
