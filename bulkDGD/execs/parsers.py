#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-

#    parsers.py
#
#    Parser-related utilities.
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
__doc__ = "Parser-related utilities."


#######################################################################

# Import from the standard library.
import argparse
# Import from 'bulkDGD'.
from . import (
    bulkdgd_get,
    bulkdgd_get_genes,
    bulkdgd_get_recount3,
    bulkdgd_find,
    bulkdgd_find_probability_density,
    bulkdgd_find_representations,
    bulkdgd_dea,
    bulkdgd_preprocess,
    bulkdgd_preprocess_samples,
    bulkdgd_reduction,
    bulkdgd_train,
    util)


#######################################################################


# Set a mapping between the commands and their modules, 'main'
# functions, and sub-commands.
COMMANDS = \
    {# Set the options for the 'get' command.
     "get" : \
         # Set the module corresponding to the command.
        {"module" : bulkdgd_get,
         # Set the sub-commands.
         "commands" :
            {# Set the 'get recount3' command.
             "recount3" : \
                 # Set the module corresponding to the command.
                {"module" : bulkdgd_get_recount3,
                 # Set the 'main' function corresponding to the
                 # command.
                 "main" : bulkdgd_get_recount3.main},
              # Set the 'get genes' command.
             "genes" : \
                {# Set the module corresponding to the command.
                 "module" : bulkdgd_get_genes,
                 # Set the 'main' function corresponding to the
                 # command.
                 "main" : bulkdgd_get_genes.main}}},
     
     # Set the options for the 'get' command.
     "find" : \
        {# Set the module corresponding to the command.
         "module" : bulkdgd_find,
         # Set the sub-commands.
         "commands" :
            {# Set the 'find representations' command.
             "representations" : 
                {# Set the module corresponding to the command.
                 "module" : bulkdgd_find_representations,
                 # Set the 'main' function corresponding to the
                 # command.
                 "main" : bulkdgd_find_representations.main},
             # Set the 'find probability_density' command.
             "probability_density" : \
                {# Set the module corresponding to the command.
                 "module" : bulkdgd_find_probability_density,
                 # Set the 'main' function corresponding to the
                 # command.
                 "main" : bulkdgd_find_probability_density.main}}},

     # Set the options for the 'preprocess' command.
     "preprocess": \
        {# Set the module corresponding to the command.
         "module" : bulkdgd_preprocess,
         # Set the sub-commands.
         "commands" : \
            {# Set the 'preprocess samples' command.
             "samples" : \
                {# Set the module corresponding to the command.
                 "module" : bulkdgd_preprocess_samples,
                 # Set the 'main' function corresponding to the
                 # command.
                 "main" : bulkdgd_preprocess_samples.main}}},

     # Set the options for the 'reduction' command.
     "reduction" : \
        {# Set the module corresponding to the command.
         "module" : bulkdgd_reduction,
         # Set the sub-commands.
         "commands" : \
             {# Set the 'reduction pca' command.
              "pca" : \
                 {# Set the module corresponding to the command.
                  "module" : bulkdgd_reduction,
                  # Set the 'main' function corresponding to the
                  # command.
                  "main" : bulkdgd_reduction.main,
                  # Set the options to use in the 'main' function.
                  "main_options" : \
                    {"dim_red_name" : "pca"},
                  # Set the options to use for the argument parser.
                  "parser_options" : \
                    {"dim_red_name" : "pca"}},
              # Set the 'reduction kpca' command.
              "kpca" : \
                 {# Set the module corresponding to the command.
                  "module" : bulkdgd_reduction,
                  # Set the 'main' function corresponding to the
                  # command.
                  "main" : bulkdgd_reduction.main,
                  # Set the options to use in the 'main' function.
                  "main_options" : \
                    {"dim_red_name" : "kpca"},
                  # Set the options to use for the argument parser.
                  "parser_options" : \
                    {"dim_red_name" : "kpca"}},
              # Set the 'reduction mds' command.
              "mds" : \
                 {# Set the module corresponding to the command.
                  "module" : bulkdgd_reduction,
                  # Set the 'main' function corresponding to the
                  # command.
                  "main" : bulkdgd_reduction.main,
                  # Set the options to use in the 'main' function.
                  "main_options" : \
                    {"dim_red_name" : "mds"},
                  # Set the options to use for the argument parser.
                  "parser_options" : \
                    {"dim_red_name" : "mds"}},
              # Set the 'reduction tsne' command.
              "tsne" : \
                 {# Set the module corresponding to the command.
                  "module" : bulkdgd_reduction,
                  # Set the 'main' function corresponding to the
                  # command.
                  "main" : bulkdgd_reduction.main,
                  # Set the options to use in the 'main' function.
                  "main_options" : \
                    {"dim_red_name" : "tsne"},
                  # Set the options to use for the argument parser.
                  "parser_options" : \
                    {"dim_red_name" : "tsne"}}}},

     # Set the options for the 'dea' command.
     "dea" : \
        {# Set the module corresponding to the command.
         "module" : bulkdgd_dea,
         # Set the 'main' function corresponding to the command.
         "main" : bulkdgd_dea.main},

     # Set the options for the 'train' command.
     "train" : \
        {# Set the module corresponding to the command.
         "module" : bulkdgd_train,
         # Set the 'main' function corresponding to the command.
         "main" : bulkdgd_train.main}
    }


#######################################################################


def set_main_parser(description):
    """Create and configure the main parser for the command-line
    interface.

    Parameters:
    -----------
    description: :class:`str`
        The description of the main parser.
    
    Returns:
    --------
    parser: :class:`argparse.ArgumentParser`
        The main parser.
    """

    # Create the main parser.
    parser = \
        argparse.ArgumentParser(\
            description = description,
            formatter_class = util.CustomHelpFormatter)

    #-----------------------------------------------------------------#

    # Add the sub-parsers to the main parser.
    sub_parsers = parser.add_subparsers(dest = "command")

    #-----------------------------------------------------------------#

    # For each command and the associated options
    for command_name, command_options in COMMANDS.items():

        # Get the module corresponding to the command.
        module = command_options.get("module")

        # Get the options to be used to set up the command's parser.
        parser_options = command_options.get("parser_options", {})

        # Get sub-commands of the current command.
        sub_commands = command_options.get("commands")

        # Add the parser for the command.
        command_parser = \
            module.set_parser(sub_parsers = sub_parsers,
                              **parser_options)

        # If there are sub-commands
        if sub_commands is not None:

            # Create sub-sub-parsers.
            sub_sub_parsers = \
                command_parser.add_subparsers(dest = "sub_command")

            # For each sub-command
            for sub_command_name, sub_command_options \
                in sub_commands.items():

                # Get the module corresponding to the sub-command.
                sub_module = sub_command_options.get("module")

                # Get the options to be used to set up the
                # sub-command's parser.
                sub_parser_options = \
                    sub_command_options.get("parser_options", {})

                # Add the parser for the sub-command.
                sub_command_parser = \
                    sub_module.set_sub_parser(\
                        sub_parsers = sub_sub_parsers,
                        **sub_parser_options)

                # Add the arguments for the working directory and
                # logging.
                util.add_wd_and_logging_arguments(\
                    parser = sub_command_parser,
                    command_name = \
                        f"bulkdgd_{command_name}_{sub_command_name}")

        else:

            # Add the arguments for the working directory and logging.
            util.add_wd_and_logging_arguments(\
                parser = command_parser,
                command_name = f"bulkdgd_{command_name}")
    
    #-----------------------------------------------------------------#

    # Return the main parser.
    return parser
