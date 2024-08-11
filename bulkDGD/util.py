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
import copy
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
                 log_console_level = defaults.LOG_LEVEL,
                 log_file_class = None,
                 log_file_options = None,
                 log_file_level = defaults.LOG_LEVEL):
    """Get the handlers to use when logging.

    Parameters
    ----------
    log_console : ``bool``, ``True``
        Whether to write log messages to the console.

    log_console_level : ``str``, ``bulkDGD.defaults.LOG_LEVEL``
        The level below which log messages will not be logged on the
        console.

        By default, it takes the value of
        ``bulkDGD.defaults.LOG_LEVEL``.

    log_file_class : ``logging.FileHandler``, optional
        A ``FileHandler`` class to construct the handler that will
        log to a file.

        If not provided, the log messages will not be written to any
        file.

    log_file_options : ``dict``, optional
        A dictionary of options to set up the handler that will log to
        a file.

        It must be provided if ``log_file_class`` is provided.

    log_file_level : ``str``, ``bulkDGD.defaults.LOG_LEVEL``
        The level below which log messages will not be logged to the
        file.

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


def check_config_against_template(config,
                                  template,
                                  ignore_if_missing,
                                  ignore_if_varying):
    """Check a configuration against the configuration's template.

    Parameters
    ----------
    config : ``dict``
        The configuration.

    template : ``dict``
        A template describing how the configuration should be
        structured.

    ignore_if_missing : ``set``
        A set containing the fields that can be missing from the
        configuration.

    ignore_if_varying : ``set``
        A set containing the fields that can vary in the
        configuration.

    Returns
    -------
    config : ``dict``
        The configuration loaded from the file provided by
        the user, if all checks were successful.
    """

    # Define the recursion.
    def recurse(config,
                template,
                key):

        # If both the current configuration dictionary and the template
        # dictionary are dictionaries (they are either the full
        # dictionaries or sub-dictionaries of the original ones in the
        # recursive calls)
        if (isinstance(config, dict) \
        and isinstance(template, dict)):

            # Get the fields (= keys) in the configuration.
            config_fields = set(config.keys())
            
            # Get the fields (= keys) in the template.
            template_fields = set(template.keys())
            
            # Get the fields (= keys) unique to the configuration
            # (= not found in the template).
            unique_config_fields = \
                config_fields - template_fields

            # Get the fields (= keys) unique to the template
            # (= not found in the configuration).
            unique_template_fields = \
                template_fields - config_fields

            #---------------------------------------------------------#

            # If any unique field was found in the configuration
            if len(unique_config_fields) != 0:

                # Raise an exception.
                fields = \
                    ", ".join([f"'{f}'" for f \
                               in unique_config_fields])
                errstr = \
                    f"Unrecognized field(s) in configuration: " \
                    f"{fields}."
                raise KeyError(errstr)

            #---------------------------------------------------------#

            # If any field was found in the template but not in the
            # configuration.
            if len(unique_template_fields) != 0:

                # Get the fields that need to be present.
                needed_fields = \
                    ignore_if_missing.union(unique_template_fields) \
                    - ignore_if_missing

                # If some fields that are needed are missing.
                if needed_fields:

                    # Raise an exception.
                    fields = \
                        ", ".join([f"'{f}'" for f in needed_fields])
                    errstr = \
                        f"Missing field(s) in configuration: " \
                        f"{fields}."
                    raise KeyError(errstr)

            #---------------------------------------------------------#

            # For each key, value pair in the configuration
            for k, val_config in config.items():

                # If the element corresponds to a dictionary of
                # options that can vary
                if k in ignore_if_varying:

                    # Just ignore it.
                    continue

                # Get the corresponding value in the template
                # (we are sure that there will be a value for
                # the given key, because we checked the equality
                # of all keys between configuration and template
                # earlier).
                val_template = template[k]

                # Recursively check the values.
                recurse(config = val_config,
                        template = val_template,
                        key = k)

        #-------------------------------------------------------------#

        # If the user-supplied configuration is a dictionary, but the
        # template is not
        elif (isinstance(config, dict) \
        and not isinstance(template, dict)) and template is not None:

            # Raise an exception.
            errstr = \
                f"The configuration contains sub-fields " \
                f"for the field '{str(template)}', which is " \
                f"not supposed to have sub-fields."
            raise TypeError(errstr)

        #-------------------------------------------------------------#

        # If the template is a dictionary, but the configuration is
        # not
        elif (not isinstance(config, dict) \
        and isinstance(template, dict)):

            # Raise an exception.
            errstr = \
                f"The configuration does not contain " \
                f"sub-fields for the field '{str(config)}', " \
                f"while it is supposed to have the following " \
                f"sub-fields: {str(template)}."
            raise TypeError(errstr)

        #-------------------------------------------------------------#

        # If both the configuration and the template are not
        # dictionaries (we are in a "leaf" value, not a key of a
        # a nested dictionary)
        elif (not isinstance(config, dict) \
        and not isinstance(template, dict)):

            # If the type of value found in the configuration does not
            # match the type set in the template
            if not isinstance(config, template):

                # Raise an exception.
                errstr = \
                    f"'{key}' must be of type '{str(template)}', "\
                    f"not {str(type(config))} ('{config}')."
                raise TypeError(errstr)

        #-------------------------------------------------------------#

        # Return the dictionary.
        return config

    #-----------------------------------------------------------------#

    # Recurse and return the result.
    return recurse(config = config,
                   template = template,
                   key = None)


def recursive_map_dict(d,
                       func,
                       keys = None):
    """Recursively traverse a dictionary mapping a function to the
    dictionary's leaf values (the function substitutes the values with
    the return value of the function applied to those values).

    Parameters
    ----------
    d : ``dict``
        The input dictionary.

    func : any callable
        A callable taking as keyword arguments the values of a
        dictionary and returning a single value.

    keys : ``list``, ``set``, optional
        A list of specific keys on whose items the mapping should be
        performed.

        This means that all values associated with keys different
        from those in the list will not be affected.

        If ``None``, all keys and associated values will be considered.
    
    Returns
    -------
    new_d : ``dict``
        The new dictionary.
    """

    # Define the recursion.
    def recurse(d,
                func,
                keys):

        # If the current object is a dictionary
        if isinstance(d, dict):
            
            # Get the keys of the items on which the mapping will be
            # performed. If no keys are passed, all keys in the
            # dictionary will be considered.
            sel_keys = keys if keys else d.keys()

            # For each key, value pair in the dictionary
            for k, v in list(d.items()):

                # If the value is a dictionary
                if isinstance(v, dict):

                    # If the key is in the selected keys
                    if k in sel_keys:

                        # Substitute the value with the return value
                        # of 'func' applied to it.
                        d[k] = func(**v)
                    
                    # Otherwise
                    else:

                        # Recursively check the sub-dictionaries
                        # in the current dictionary.
                        recurse(d = v,
                                func = func,
                                keys = sel_keys)

    #-----------------------------------------------------------------#

    # Create a copy of the input dictionary.
    new_d = copy.deepcopy(d)

    #-----------------------------------------------------------------#

    # Recurse through the new dictionary.
    recurse(d = new_d,
            func = func,
            keys = keys)

    #-----------------------------------------------------------------#

    # Return the new dictionary.
    return new_d


#######################################################################


def load_list(list_file):
    """Load a list of newline-separated entities from a plain text
    file.

    Parameters
    ----------
    list_file : ``str``
        The plain text file containing the entities of interest.

    Returns
    -------
    list_entities : ``list``
        The list of entities.
    """

    # Return the list of entities from the file (exclude blank
    # and comment lines).
    return \
        [l.rstrip("\n") for l in open(list_file, "r") \
         if (not l.startswith("#") and not re.match(r"^\s*$", l))]


#######################################################################


def check_config_model(config):
    """Check the model's configuration.

    Parameters
    ----------
    config : ``dict``
        The model's configuration.

    Returns
    -------
    config : ``dict``
        The model's configuration.
    """

    # Set the fields that can be missing from the configuration,
    # so that they do not raise an exception if they are not found.
    ignore_if_missing = {"gmm_pth_file", "dec_pth_file"}

    # Set the fields that can vary in the configuration, so that they
    # do not raise an exception if they are different than expected.
    ignore_if_varying = {"gmm_options", "dec_options"}

    #-----------------------------------------------------------------#

    # Check the configuration against the template.
    config = check_config_against_template(\
                config = config,
                template = defaults.CONFIG_MODEL_TEMPLATE,
                ignore_if_missing = ignore_if_missing,
                ignore_if_varying = ignore_if_varying)

    #-----------------------------------------------------------------#

    # Return the configuration.
    return config


def check_config_rep(config):
    """Check the configuration containing the options for the
    optimization round(s) to find the best representations for a
    set of samples.

    Parameters
    ----------
    config : ``dict``
        The model's configuration.

    Returns
    -------
    config : ``dict``
        The model's configuration.
    """

    # Set the names of the supported optimization schemes.
    optimization_schemes = ["one_opt", "two_opt"]

    #-----------------------------------------------------------------#

    # Get the optimization scheme from the configuration.
    opt_scheme = config.get("scheme")

    # If no scheme was defined
    if not opt_scheme:

        # Raise an error.
        errstr = \
            "The configuration must include a 'scheme' key " \
            "associated with the name of the optimization " \
            "scheme the configuration refers to."
        raise KeyError(errstr)

    # If the optimization scheme is not supported
    if opt_scheme not in optimization_schemes:

        # Raise an error.
        supported_schemes = \
            ", ".join([f"'{s}'" for s in optimization_schemes])
        errstr = \
            f"Unsupported optimization scheme '{opt_scheme}'. " \
            "Supported optimization schemes are: " \
            f"'{supported_schemes}."
        raise ValueError(errstr)

    #-----------------------------------------------------------------#

    # Check the configuration against the template.
    config = util.check_config_against_template(\
                config = config,
                template = defaults.CONFIG_REP_TEMPLATE[opt_scheme],
                ignore_if_missing = set(),
                ignore_if_varying = set())

    #-----------------------------------------------------------------#

    # Return the configuration.
    return config


def check_config_train(config):
    """Check the configuration containing the options for training the
    model.

    Parameters
    ----------
    config : ``dict``
        The model's configuration.

    Returns
    -------
    config : ``dict``
        The model's configuration.
    """

    # Check the configuration against the template.
    config = util.check_config_against_template(\
                config = config,
                template = defaults.CONFIG_TRAIN_TEMPLATE,
                ignore_if_missing = set(),
                ignore_if_varying = set())

    #-----------------------------------------------------------------#

    # Return the configuration.
    return config


def check_config_plot(config):
    """Check the configuration for a plot.

    Parameters
    ----------
    config : ``dict``
        The model's configuration.

    Returns
    -------
    config : ``dict``
        The model's configuration.
    """

    # Return the configuration.
    return config


def check_config_genes(config):
    """Check the configuration for creating a new list of genes.

    Parameters
    ----------
    config : ``dict``
        The model's configuration.

    Returns
    -------
    config : ``dict``
        The model's configuration.
    """

    # Return the configuration.
    return config
