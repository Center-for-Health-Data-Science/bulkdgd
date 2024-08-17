#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-

#    util.py
#
#    Utilities for the executables.
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
__doc__ = "Utilities for the executables."


#######################################################################


# Import from the standard library.
import argparse
import logging as log
import os
import subprocess
# Import from 'bulkDGD'.
from . import defaults


########################### PUBLIC CLASSES ############################


class LevelContentFilter(log.Filter):

    """
    A custom :class:`logging.Filter` class to filter log records based
    on their level and content.
    """

    def __init__(self,
                 level = "WARNING",
                 start = None,
                 end = None,
                 content = None):
        """Initialize the filter.

        Parameters
        ----------
        level : :class:`str` or :obj:`logging` level, \
            :const:`logging.WARNING`
            The level at which and below which messages should be
            filtered out.

        start : :class:`list`, optional
            A list of strings. Log records starting with any of these
            strings will be filtered out.

            Note that the log record is stripped of any leading and
            trailing blank spaces before checking whether it starts
            with any of the provided strings.

        end : :class:`list`, optional
            A list of strings. Log records ending with any of these
            strings will be filtered out.

            Note that the log record is stripped of any leading and
            trailing blank spaces before checking whether it ends with
            any of the provided strings.

        content : :class:`list`, optional
            A list of strings. Log records containing any of these
            strings will be filtered out.
        """

        # Call the parent class' initialization method.
        super().__init__()

        # Save the level below which messages should be ignored.
        self.level = log._nameToLevel[level]

        # Save the strings the log record needs to start with to be
        # ignored - if the log record starts with any of the specified
        # strings, it will be ignored.
        self.start = start

        # Save the strings the log record needs to end with to be
        # ignored - if the log record ends with any of the specified
        # strings, it will be ignored.
        self.end = end

        # Save the strings the log record needs to contain to be
        # ignored - if the log record contains any of the specified
        # strings, it will be ignored.
        self.content = content


    def filter(self,
               record):
        """Decide whether a log record will be emitted or not.

        Parameters
        ----------
        record : :class:`logging.LogRecord`
            The log record.

        Returns
        -------
        emit_record : :class:`bool`
            Whether the log record will be emitted or not.
        """

        # If the record's level is below or equal to the selected
        # level
        if record.levelno <= self.level:

            # If we specified strings the record needs to start with
            # to be ignored
            if self.start is not None:

                # For each string
                for string in self.start:

                    # If the record starts with the selected string
                    if record.msg.strip().startswith(string):

                        # Do not log the record.
                        return False

            #---------------------------------------------------------#

            # If we specified strings the record needs to end with to
            # be ignored
            if self.end is not None:

                # For each string
                for string in self.end:

                    # If the record ends with the selected string
                    if record.msg.strip().endswith(string):

                        # Do not log the record.
                        return False

            #---------------------------------------------------------#

            # If we specified strings the record needs to contain to be
            # ignored
            if self.content is not None:

                # For each string
                for string in self.content:

                    # If the record contains the selected string
                    if string in record:

                        # Do not log the record.
                        return False

        #-------------------------------------------------------------#

        # If we reached this point, log the record.
        return True


class CustomHelpFormatter(argparse.HelpFormatter):
    
    """
    A custom :class:`argparse.HelpFormatter` class to format the
    help messages displayed for command-line utilities.
    """

    def __init__(self,
                 *args,
                 **kwargs):
        """Initialize a new instance of the class.
        """

        # Call the parent class' initialization method.
        super().__init__(*args, **kwargs)


    def start_section(self,
                      heading):
        """Format the start of each section of the help message.

        Parameters
        ----------
        heading : :class:`str`
            The section's heading.
        """

        # If we are at the section for positional arguments
        if heading == "positional arguments":

            # Change the heading.
            heading = "Positional arguments"

        # If we are at the section for optional arguments
        elif heading == "optional arguments":

            # Change the heading.
            heading = "Optional arguments"

        # If we are at the section for generic options
        elif heading == "options":

            # Change the heading.
            heading = "Help options"

        #-------------------------------------------------------------#

        # Call the parent class' 'start_section' method with the new
        # heading.
        super().start_section(heading)


########################## PUBLIC FUNCTIONS ###########################


def get_handlers(log_console = True,
                 log_console_level = defaults.LOG_LEVEL,
                 log_file_class = None,
                 log_file_options = None,
                 log_file_level = defaults.LOG_LEVEL):
    """Get the handlers to use when logging.

    Parameters
    ----------
    log_console : :class:`bool`, :const:`True`
        Whether to write log messages to the console.

    log_console_level : :class:`str`, \
        :const:`bulkDGD.execs.defaults.LOG_LEVEL`
        The level below which log messages will not be logged on the
        console.

        By default, it takes the value of
        :const:`bulkDGD.execs.defaults.LOG_LEVEL`.

    log_file_class : :class:`logging.FileHandler`, optional
        A :class:`logging.FileHandler` class to construct the handler
        that will log to a file.

        If not provided, the log messages will not be written to any
        file.

    log_file_options : :class:`dict`, optional
        A dictionary of options to set up the handler that will log to
        a file.

        It must be provided if ``log_file_class`` is provided.

    log_file_level : :class:`str`, \
        :const:`bulkDGD.execs.defaults.LOG_LEVEL`
        The level below which log messages will not be logged to the
        file.

        By default, it takes the value of
        :const:`bulkDGD.execs.defaults.LOG_LEVEL`.

    Results
    -------
    handlers : :class:`list`
        A list of handlers.
    """

    # Create a list to store the handlers.
    handlers = []

    #-----------------------------------------------------------------#

    # If the user requested logging to the console
    if log_console:

        # Create a 'StreamHandler'.
        handler = log.StreamHandler()

        # Set the handler's level.
        handler.setLevel(log_console_level)

        # Add the handler to the list of handlers.
        handlers.append(handler)

    #-----------------------------------------------------------------#

    # If the user specified a class to create a log file
    if log_file_class is not None:

        # Try to get the path to the log file.
        try:

            log_file = log_file_options["filename"]

        # If the user did not pass any options to initialize the
        # instance of the class
        except TypeError as e:

            # Raise an error.
            errstr = \
                "If 'log_file_class' is provided, a dictionary of " \
                "'log_file_options' must be passed as well. " \
                f"Error: {e}"
            raise TypeError(errstr)

        # If the user did not pass the 'filename' option
        except KeyError as e:

            # Raise an error.
            errstr = \
                "'filename' must be present in teh dictionary of " \
                f"'log_file_options. Error: {e}"
            raise KeyError(errstr)

        # If the file already exists
        if os.path.exists(log_file):

            # Remove it.
            os.remove(log_file)

        #-------------------------------------------------------------#

        # Create the corresponding handler.
        handler = log_file_class(**log_file_options)

        # Set the handler's level.
        handler.setLevel(log_file_level)

        # Add the handler to the list of handlers.
        handlers.append(handler)

    #-----------------------------------------------------------------#
    
    # Return the list of handlers.
    return handlers


def get_dask_logging_config(log_console = True,
                            log_file = None,
                            log_level = "ERROR"):
    """Get the logging configuration for Dask/distributed loggers.

    Parameters
    ----------
    log_console : :class:`bool`, :const:`True`
        Whether to log messages to the console.

    log_file : :class:`str`, optional
        The name of the log file where to write the log messages.

        If not provided, the log messages will not be written to
        any file.

    log_level : :class:`str`, ``"ERROR"``
        The level below which log messages should be silenced.

    Returns
    -------
    dask_logging_config : :class:`dict`
        The logging configuration for Dask/distributed loggers.
    """

    # Initialize the logging configuration - it follows the
    # "configuration dictionary schema" provided in
    # https://docs.python.org/3/library/logging.config.html
    # #configuration-dictionary-schema.
    dask_logging_config = {\
        
        # Set the version of the logging configuration - so far,
        # only version 1 is supported.
        "version": 1,
        
        # Set the formatters for the log records - each key
        # represents the name of a 'logging.Formatter' object and
        # the dictionary associated with it contains the options to
        # initialize it
        "formatters" : {\
            
            # Set a generic formatter.
            "generic_formatter" : \
                defaults.CONFIG_FORMATTERS["generic_formatter"],
        }, 

        # Set the filters to be used for log records - each key
        # represents the name of a 'logging.Filter' object and the
        # dictionary associated with it contains the options to
        # initialize it.
        "filters" : {\
            
            # Use the custom filter for the 'distributed.worker'
            # logger.
            "distributed_filter" : \
                defaults.CONFIG_FILTERS["distributed_filter"],
        },
       
        # The handlers to be used when logging - each key represents
        # the name of a 'logging.Handler' object and the dictionary
        # associated with it contains the options to initialize it.
        "handlers": {
            
            # Set a handler to log to a rotating file.
            "rotating_file_handler": \
                {# Set the class of the handler to initialize.
                 "class": "logging.handlers.RotatingFileHandler",
                 # Set the name of the rotating file.
                 "filename": log_file,
                 # Set the formatter for the log records.
                 "formatter" : "generic_formatter"},
            
            # Set a handler to log to the console.
            "stream_handler": \
                {# Set the class of the handler to initialize.
                 "class": "logging.StreamHandler",
                 # Set the formatter for the log records.
                 "formatter" : "generic_formatter"},
        },

        # Set the loggers to configure - each key represents the name
        # of a 'logging.Logger' object and the dictionary associated
        # with it contains the options to configure it. The dictionary
        # of loggers is empty because we are going to fill it later.
        "loggers" : {},
    }

    #-----------------------------------------------------------------#

    # Initialize an empty list to store the handlers to be used.
    handlers = []

    #-----------------------------------------------------------------#

    # If the user requested logging to the console
    if log_console:

        # Add the corresponding handler to the list.
        handlers.append("stream_handler")

    # Otherwise
    else:

        # Remove the corresponding handler from the configuration.
        dask_logging_config["handlers"].pop("stream_handler")

    #-----------------------------------------------------------------#

    # If the user requested logging to a file
    if log_file is not None:

        # Add the corresponding handler to the list.
        handlers.append("rotating_file_handler")

    # Otherwise
    else:

        # Remove the corresponding handler from the configuration.
        dask_logging_config["handlers"].pop("rotating_file_handler")

    #-----------------------------------------------------------------#

    # For each Dask logger
    for logger in defaults.DASK_LOGGERS:
  
        # Configure it.
        dask_logging_config["loggers"][logger] = \
            {"level" : log_level,
             "filters" : ["distributed_filter"],
             "handlers" : handlers}

    #-----------------------------------------------------------------#

    # Return the configuration.
    return dask_logging_config


def add_wd_and_logging_arguments(parser,
                                 command_name):
    """Add the options for logging and for the working directory to
    a parser.

    Parameters
    ----------
    parser : :class:`argparse.ArgumentParser`
        The parser.

    command_name : :class:`str`
        The name of the command.

    Returns
    -------
    parser : :class:`argparse.ArgumentParser`
        The updated parser.
    """

    # Set a group of arguments for the working directory.
    wd_group = parser.add_argument_group(\
        title = "Working directory options")

    # Set a group of arguments for logging.
    log_group = parser.add_argument_group(\
        title = "Logging options")

    #-----------------------------------------------------------------#

    # Set a help message.
    d_help = \
        "The working directory. The default is the current " \
        "working directory."

    # Add the argument to the group.
    wd_group.add_argument("-d", "--work-dir",
                          type = lambda x: os.path.abspath(x),
                          default = os.getcwd(),
                          help = d_help)

    #-----------------------------------------------------------------#

    # Set the default value for the argument.
    lf_default = f"{command_name}.log"

    # Set a help message.
    lf_help = \
        "The name of the log file. The file will be written " \
        "in the working directory. The default file name is " \
        f"'{lf_default}'."

    # Add the argument to the group.
    log_group.add_argument("-lf", "--log-file",
                           type = str,
                           default = lf_default,
                           help = lf_help)

    #-----------------------------------------------------------------#

    # Set a help message.
    lc_help = "Show log messages on the console."

    # Add the argument to the group.
    log_group.add_argument("-lc", "--log-console",
                           action = "store_true",
                           help = lc_help)

    #-----------------------------------------------------------------#

    # Set a help message.
    v_help = "Enable verbose logging (INFO level)."

    # Add the argument to the group.
    log_group.add_argument("-v", "--log-verbose",
                           action = "store_true",
                           help = v_help)

    #-----------------------------------------------------------------#

    # Set a help message.
    vv_help = \
        "Enable maximally verbose logging for debugging " \
        "purposes (DEBUG level)."

    # Add the argument to the group.
    log_group.add_argument("-vv", "--log-debug",
                           action = "store_true",
                           help = vv_help)


def process_arg_input_columns(val):
    """Process the value passed to the '-ic', '--input-columns' 
    argument in the 'bulkdgd reduction' sub-commands.

    Parameters
    ----------
    val : :class:`str` or :const:`None`
        The value as a string or :const:`None`.

    Returns
    -------
    val : :class:`str` or :class:`int` or :const:`None`
        The processed value.
    """

    # Process and return the value.
    return val if (val is None or isinstance(val, str)) else \
           [item.strip() for item in val.split(",")]



def process_arg_groups_column(val):
    """Process the value passed to the '-gc', '--groups-column'
    argument in the 'bulkdgd reduction' sub-commands.

    Parameters
    ----------
    val : :class:`str` or :const:`None`
        The value as a string or :const:`None`.

    Returns
    -------
    val : :class:`str` or :class:`int` or :const:`None`
        The processed value.
    """

    # Process and return the value.
    return val if (val is None or not val.isdigit()) else int(val)


def process_arg_groups(val):
    """Process the value passed to the '-gr', '--groups' argument in
    the 'bulkdgd reduction' sub-commands.

    Parameters
    ----------
    val : :class:`str` or :const:`None`
        The value as a string or :const:`None`.

    Returns
    -------
    val : :class:`str` or :class:`int` or :const:`None`
        The processed value.
    """

    # Process and return the value.
    return val if val is None else \
           [item.strip() for item in val.split(",")]


def run_executable(executable,
                   arguments,
                   extra_return_values = None):
    """Run an executable.

    Parameters
    ----------
    executable : :class:`str`
        The executable.

    arguments : :class:`list`
        A list of arguments to run the executable with.

    extra_return_values : :class:`list`, optional
        A list of extra values to be returned by the function,
        together with the ``subprocess.CompletedProcess`` instance
        representing the completed process.

    Returns
    -------
    completed_process : :class:`subprocess.CompletedProcess`
        The completed process.

    Plus as many values as ``extra_return_values`` contains, if
    ``extra_return_values`` was passed.
    """

    # Launch the executable.
    completed_process = \
        subprocess.run([executable] + arguments)

    # Return the completed process and any other value that was
    # passed.
    return (completed_process, *extra_return_values)
