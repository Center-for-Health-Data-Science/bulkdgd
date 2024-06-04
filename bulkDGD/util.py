#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-

#    util.py
#
#    General utilities.
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
__doc__ = "General utilities."


#######################################################################


# Import from the standard library.
import functools
import logging as log
import os
import subprocess
# Import from bulkDGD.
from . import defaults


########################### PUBLIC CLASSES ############################


class LevelContentFilter(log.Filter):

    """
    A custom ``logging.Filter`` class to filter log records based on
    their level and content.
    """


    def __init__(self,
                 level = "WARNING",
                 start = None,
                 end = None,
                 content = None):
        """Initialize the filter.

        Parameters
        ----------
        level : ``str`` or ``logging`` level, ``logging.WARNING``
            The level at which and below which messages should be
            filtered out.

        start : ``list``, optional
            A list of strings. Log records starting with any of these
            strings will be filtered out.

            Note that the log record is stripped of any leading and
            traling blank spaces before checking whether it starts
            with any of the provided strings.

        end : ``list``, optional
            A list of strings. Log records ending with any of these
            strings will be filtered out.

            Note that the log record is stripped of any leading and
            trailing blank spaces before checking whether it ends with
            any of the provided strings.

        content : ``list``, optional
            A list of strings. Log records containing any of these
            strings will be filtered out.
        """

        # Call the parent class' initialization method.
        super().__init__()

        # Save the level below which messages from the specified
        # loggers should be ignored.
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
        record : ``logging.LogRecord``
            The log record.

        Returns
        -------
        emit_record : ``bool``
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


########################## PUBLIC FUNCTIONS ###########################


def get_handlers(log_console = True,
                 log_file_class = None,
                 log_file_options = None,
                 log_level = defaults.LOG_LEVEL):
    """Get the handlers to use when logging.

    Parameters
    ----------
    log_console : ``bool``, ``True``
        Whether to write log messages to the console.

    log_file_class : ``logging.FileHandler``, optional
        A ``FileHandler`` class to construct the handler that will
        log to a file.

        If not provided, the log messages will not be written to any
        file.

    log_file_options : ``dict``, optional
        A dictionary of options to set up the handler that will log to
        a file.

        It must be provided if ``log_file_class`` is provided.

    log_level : ``str``, ``bulkDGD.defaults.LOG_LEVEL``
        The level below which log messages will be ignored.

        By default, it takes the value of
        ``bulkDGD.defaults.LOG_LEVEL``.

    Results
    -------
    handlers : ``list``
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
        handler.setLevel(log_level)

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
                "'log_file_options' must be passed as well."
            raise TypeError(errstr)

        # If the user did not pass the 'filename' option
        except KeyError as e:

            # Raise an error.
            errstr = \
                "'filename' must be present in teh dictionary of " \
                "'log_file_options."
            raise KeyError(errstr)

        # If the file already exists
        if os.path.exists(log_file):

            # Remove it.
            os.remove(log_file)

        #-------------------------------------------------------------#

        # Create the corresponding handler.
        handler = log_file_class(**log_file_options)

        # Set the handler's level.
        handler.setLevel(log_level)

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
    log_console : ``bool``, ``True``
        Whether to log messages to the console.

    log_file : ``str``, optional
        The name of the log file where to write the log messages.

        If not provided, the log messages will not be written to
        any file.

    log_level : ``str``, ``"ERROR"``
        The level below which log messages should be silenced.

    Returns
    -------
    dask_logging_config : ``dict``
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

    # Initnialize an empty list to store the handlers to be used.
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


def uniquify_file_path(file_path):
    """If ``file_path`` exists, number it uniquely.

    Parameters
    ----------
    file_path : ``str``
        The file path.

    Returns
    -------
    unique_file_path : ``str``
        A unique file path generated from the original file path.
    """
    
    # Get the file's name and extension.
    file_name, file_ext = os.path.splitext(file_path)

    # Set the counter to 1.
    counter = 1

    # If the file already exists
    while os.path.exists(file_path):

        # Set the path to the new unique file.
        file_path = file_name + "_" + str(counter) + file_ext

        # Update the counter.
        counter += 1

    # Return the new path.
    return file_path


def run_executable(executable,
                   arguments,
                   extra_return_values = None):
    """Run an executable.

    Parameters
    ----------
    executable : ``str``
        The executable.

    arguments : ``list``
        A list of arguments to run the executable with.

    extra_return_values : ``list``, optional
        A list of extra values to be returned by the function,
        together with the ``subprocess.CompletedProcess`` instance
        representing the completed process.

    Returns
    -------
    completed_process : ``subprocess.CompletedProcess``
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
