#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-

#    _bulkdgd_exec.py
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


# Import from 'bulkDGD'.
from . import parsers, util


#######################################################################


# Define the 'main' function.
def main():

    # Set the main parser.
    parser = \
        parsers.set_main_parser(description = "Main command.")

    #-----------------------------------------------------------------#
    
    # Parse the arguments.
    args = parser.parse_args()

    #-----------------------------------------------------------------#

    # Get the 'command' argument.
    command = getattr(args, "command", None)

    # Get the 'sub_command' argument.
    sub_command = getattr(args, "sub_command", None)

    #-----------------------------------------------------------------#

    # If we have a sub-command
    if sub_command:

        # Get the 'main' function of the sub-command.
        main_function = \
            parsers.COMMANDS[command]["commands"][sub_command]["main"]

        # Get the options to use with the 'main' function.
        main_options = \
            parsers.COMMANDS[command]["commands"][sub_command].get(\
                "main_options", {})

    # Otherwise
    else:

        # Get the 'main' function of the command.
        main_function = parsers.COMMANDS[command]["main"]

        # Get the options to use with the 'main' function.
        main_options = \
            parsers.COMMANDS[command].get(\
                "main_options", {})

    #-----------------------------------------------------------------#

    # Set the logging.
    util.set_main_logging(args = args)
 
    #-------------------------------------------------------------#
 
    # Run the function.
    main_function(args = args,
                  **main_options)


#######################################################################
    

# If the script is run as the main script
if __name__ == "__main__":

    # Run the 'main' function.
    main()
