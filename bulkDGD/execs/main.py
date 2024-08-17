#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-

#    bulkdgd.py
#
#    Main command.
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
__doc__ = "Main command."


#######################################################################


# Import from the standard library.
import argparse
import logging as log
import logging.handlers as loghandlers
import os
# Import from third-party packages.
import dask
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
    util,
    defaults)


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


# Define the 'main' function.
def main():

    # Create the main parser.
    parser = \
        argparse.ArgumentParser(\
            description = __doc__,
            formatter_class = util.CustomHelpFormatter)

    #-----------------------------------------------------------------#

    # Add the sub-parsers to the main parser.
    sub_parsers = parser.add_subparsers(dest = "command")

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

    # Parse the arguments.
    args = parser.parse_args()

    #-----------------------------------------------------------------#

    # Get the 'command' argument.
    command = args.command

    #-----------------------------------------------------------------#

    # If the user passed a command
    if command is not None:

        # If we have a sub-command
        if hasattr(args, "sub_command"):

            # Get the 'main' function of the sub-command.
            main_function = \
                COMMANDS[command]["commands"][args.sub_command]["main"]

            # Get the options to use with the 'main' function.
            main_options = \
                COMMANDS[command]["commands"][args.sub_command].get(\
                    "main_options", {})

        # Otherwise
        else:

            # Get the 'main' function of the command.
            main_function = COMMANDS[command]["main"]

            # Get the options to use with the 'main' function.
            main_options = \
                COMMANDS[command].get(\
                    "main_options", {})

        #-------------------------------------------------------------#

        # Get the log file.
        log_file = os.path.join(args.work_dir, args.log_file)

        #-------------------------------------------------------------#

        # Set WARNING logging level by default.
        log_level = log.WARNING

        # If the user requested verbose logging
        if args.log_verbose:

            # The minimal logging level will be INFO.
            log_level = log.INFO

        # If the user requested logging for debug purposes
        if args.log_debug:

            # The minimal logging level will be DEBUG.
            log_level = log.DEBUG

        #-------------------------------------------------------------#

        # If the command is run with Dask
        if (args.command in ("dea",)) or \
           (args.command in ("get",) \
             and getattr(args, "sub_command") in ("recount3",)):

            # Get the logging configuration for Dask.
            dask_logging_config = \
                util.get_dask_logging_config(\
                    log_console = args.log_console,
                    log_file = log_file)

            # Set the configuration for Dask-specific logging.
            dask.config.set(\
                {"distributed.logging" : dask_logging_config})

            # Set the appropriate class for the file handler.
            log_file_class = loghandlers.RotatingFileHandler

        # Otherwise
        else:

            # Set the appropriate class for the file handler.
            log_file_class = log.FileHandler

        #-------------------------------------------------------------#
        
        # Configure the logging (for non-Dask operations).
        handlers = \
            util.get_handlers(\
                log_console = args.log_console,
                log_console_level = log_level,
                log_file_class = log_file_class,
                log_file_options = {"filename" : log_file,
                                    "mode" : "w"},
                log_file_level = log_level)

        #-------------------------------------------------------------#

        # Set the logging configuration.
        log.basicConfig(level = log_level,
                        format = defaults.LOG_FMT,
                        datefmt = defaults.LOG_DATEFMT,
                        style = defaults.LOG_STYLE,
                        handlers = handlers)

        #-------------------------------------------------------------#

        # Add the arguments parsed to the options to be passed to the
        # 'main' function.
        main_options["args"] = args
        
        # Call the 'main' function of the command.
        main_function(**main_options)

    #-----------------------------------------------------------------#

    # Otherwise
    else:

        # Print the help message.
        parser.print_help()
