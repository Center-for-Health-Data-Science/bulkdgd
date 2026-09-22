#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-

#    util.py
#
#    Utilities for the executables.
#
#    Copyright (C) 2025 Valentina Sora
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
import copy
import glob
import logging.handlers as loghandlers
import logging as log
import os
import subprocess
from typing import Optional

# Import from third-party libraries.
import dask

# Import from 'bulkdgd'.
from . import defaults


#######################################################################


# Get the module's logger.
logger = log.getLogger(__name__)


########################### PUBLIC CLASSES ############################


class LevelContentFilter(log.Filter):

    """
    A custom :class:`logging.Filter` class to filter log records based
    on their level and content.
    """

    def __init__(self,
                 level: str = "WARNING",
                 start: Optional[list[str]] = None,
                 end: Optional[list[str]] = None,
                 content: Optional[list[str]] = None) -> None:
        """Initialize the filter.

        Parameters
        ----------
        level : :class:`str`, ``"WARNING"``
            The level at and below which records are filtered out.

        start : :class:`list`, optional
            The strings that filtered-out records start with, once
            stripped of leading and trailing blank spaces.

        end : :class:`list`, optional
            The strings that filtered-out records end with, once
            stripped of leading and trailing blank spaces.

        content : :class:`list`, optional
            The strings that filtered-out records contain.
        """

        # Call the parent class' initialization method.
        super().__init__()

        # Save the level at and below which records are ignored.
        self.level = log._nameToLevel[level]

        # Save the strings marking ignored records by their start.
        self.start = start

        # Save the strings marking ignored records by their end.
        self.end = end

        # Save the strings marking ignored records by their content.
        self.content = content


    def filter(self,
               record: log.LogRecord) -> bool:
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

            # If there are strings marking ignored records by their
            # start
            if self.start is not None:

                # For each string
                for string in self.start:

                    # If the record starts with the selected string
                    if record.msg.strip().startswith(string):

                        # Do not log the record.
                        return False

            #---------------------------------------------------------#

            # If there are strings marking ignored records by their end
            if self.end is not None:

                # For each string
                for string in self.end:

                    # If the record ends with the selected string
                    if record.msg.strip().endswith(string):

                        # Do not log the record.
                        return False

            #---------------------------------------------------------#

            # If there are strings marking ignored records by their
            # content
            if self.content is not None:

                # For each string
                for string in self.content:

                    # If the record contains the selected string
                    if string in record.getMessage():

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

    def start_section(self,
                      heading: str) -> None:
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


def get_handlers(
    log_console: bool = True,
    log_console_level: str = defaults.LOG_LEVEL,
    log_file_class: Optional[type[log.Handler]] = None,
    log_file_options: Optional[dict[str, object]] = None,
    log_file_level: str = defaults.LOG_LEVEL) -> list[log.Handler]:
    """Get the handlers to use when logging.

    Parameters
    ----------
    log_console : :class:`bool`, :const:`True`
        Whether to write log messages to the console.

    log_console_level : :class:`str`, \
        :const:`bulkdgd.execs.defaults.LOG_LEVEL`
        The level below which log messages will not be logged on the
        console.

    log_file_class : :class:`logging.FileHandler`, optional
        The class of the handler logging to a file. If not provided,
        no log file is written.

    log_file_options : :class:`dict`, optional
        The options to set up the handler logging to a file. It must
        be provided if ``log_file_class`` is provided.

    log_file_level : :class:`str`, \
        :const:`bulkdgd.execs.defaults.LOG_LEVEL`
        The level below which log messages will not be logged to the
        file.

    Returns
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
                "'filename' must be present in the dictionary of " \
                f"'log_file_options'. Error: {e}"
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


def get_dask_logging_config(
        log_console: bool = True,
        log_file: Optional[str] = None,
        log_level: str = "ERROR") -> dict[str, object]:
    """Get the logging configuration for Dask/distributed loggers.

    Parameters
    ----------
    log_console : :class:`bool`, :const:`True`
        Whether to log messages to the console.

    log_file : :class:`str`, optional
        The log file. If not provided, no log file is written.

    log_level : :class:`str`, ``"ERROR"``
        The level below which log messages should be silenced.

    Returns
    -------
    dask_logging_config : :class:`dict[str, object]`
        The logging configuration for Dask/distributed loggers.
    """

    # Initialize the logging configuration ('logging.config'
    # dictionary schema).
    dask_logging_config = {\

        # Set the version of the schema.
        "version": 1,

        # Set the formatters, keyed by name.
        "formatters" : {\

            # Set a generic formatter.
            "generic_formatter" : \
                defaults.CONFIG_FORMATTERS["generic_formatter"],
        },

        # Set the filters, keyed by name.
        "filters" : {\

            # Set the custom filter for the Dask loggers.
            "distributed_filter" : \
                defaults.CONFIG_FILTERS["distributed_filter"],
        },

        # Set the handlers, keyed by name.
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

        # Set the loggers, keyed by name (filled below).
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
    for logger_name in defaults.DASK_LOGGERS:

        # Configure it.
        dask_logging_config["loggers"][logger_name] = \
            {"level" : log_level,
             "filters" : ["distributed_filter"],
             "handlers" : handlers}

    #-----------------------------------------------------------------#

    # Return the configuration.
    return dask_logging_config


def add_wd_and_logging_arguments(parser: argparse.ArgumentParser,
                                 command_name: str) -> \
                                    argparse.ArgumentParser:
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
        "The name of the log file. The default file name is " \
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

    #-----------------------------------------------------------------#

    # If the command is not 'bulkdgd_dea', 'bulkdgd_get_recount3' or
    # '_get_recount3_single_batch'
    if command_name not in \
        ("bulkdgd_dea",
         "bulkdgd_get_recount3",
         "_get_recount3_single_batch"):

        # Add a group of arguments for the parallelization.
        parallel_group = \
            parser.add_argument_group(\
                title = "Parallelization options")

        # Set a help message.
        p_help = "Whether to run the command in parallel."

        # Add the argument to the group.
        parallel_group.add_argument("-p", "--parallelize",
                                    action = "store_true",
                                    help = p_help)

        # Set the default value for the argument.
        n_default = 1

        # Set a help message.
        n_help = \
            "The number of processes to start. The default number " \
            f"of processes started is {n_default}."

        # Add the argument to the group.
        parallel_group.add_argument("-n", "--n-proc",
                                    type = int,
                                    default = n_default,
                                    help = n_help)

        # Set a help message.
        ds_help = \
            "The directories containing the input/configuration " \
            "files. It can be either a list of names or paths, a " \
            "pattern that the names or paths match, or a plain " \
            "text file containing the names of or the paths to the " \
            "directories. If names are given, the directories are " \
            "assumed to be inside the working directory. If paths " \
            "are given, they are assumed to be relative to the " \
            "working directory."

        # Add the argument to the group.
        parallel_group.add_argument("-ds", "--dirs",
                                    type = str,
                                    nargs = "+",
                                    help = ds_help)

    #-----------------------------------------------------------------#

    # Return the parser.
    return parser


def set_main_logging(args: argparse.Namespace,
                     command_name: Optional[str] = None) -> None:
    """Set up the main logging configuration.

    Parameters
    ----------
    args : :class:`argparse.Namespace`
        The parsed arguments.

    command_name : :class:`str`, optional
        The name of the command.
    """

    # Get the log file.
    log_file = os.path.join(args.work_dir, args.log_file)

    #-----------------------------------------------------------------#

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

    #-----------------------------------------------------------------#

    # If the command uses Dask for parallelization
    if command_name in \
        ("bulkdgd_dea", "bulkdgd_get_recount3"):

        # Get the logging configuration for Dask.
        dask_logging_config = \
            get_dask_logging_config(\
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

    #-----------------------------------------------------------------#

    # Get the handlers for non-Dask logging.
    handlers = \
        get_handlers(\
            log_console = args.log_console,
            log_console_level = log_level,
            log_file_class = log_file_class,
            log_file_options = {"filename" : log_file,
                                "mode" : "w"},
            log_file_level = log_level)

    #-----------------------------------------------------------------#

    # Set the logging configuration.
    log.basicConfig(level = log_level,
                    format = defaults.LOG_FMT,
                    datefmt = defaults.LOG_DATEFMT,
                    style = defaults.LOG_STYLE,
                    handlers = handlers)


def process_arg_input_columns(val: Optional[str]) -> \
        Optional[str | list[str]]:
    """Process the value passed to the '-ic', '--input-columns'
    argument in the 'bulkdgd reduction' sub-commands.

    Parameters
    ----------
    val : :class:`str` or :const:`None`
        The value as a string or :const:`None`.

    Returns
    -------
    val : :class:`str` or :class:`list` or :obj:`None`
        The processed value.
    """

    # Process and return the value.
    return val if (val is None or "," not in val) else \
           [item.strip() for item in val.split(",")]


def process_arg_groups_column(val: Optional[str]) -> \
        Optional[str | int]:
    """Process the value passed to the '-gc', '--groups-column'
    argument in the 'bulkdgd reduction' sub-commands.

    Parameters
    ----------
    val : :class:`str` or :const:`None`
        The value as a string or :const:`None`.

    Returns
    -------
    val : :class:`str` or :class:`int` or :obj:`None`
        The processed value.
    """

    # Process and return the value.
    return val if (val is None or not val.isdigit()) else int(val)


def process_arg_groups(val: Optional[str]) -> Optional[list[str]]:
    """Process the value passed to the '-gr', '--groups' argument in
    the 'bulkdgd reduction' sub-commands.

    Parameters
    ----------
    val : :class:`str` or :const:`None`
        The value as a string or :const:`None`.

    Returns
    -------
    val : :class:`list` or :obj:`None`
        The processed value.
    """

    # Process and return the value.
    return val if val is None else \
           [item.strip() for item in val.split(",")]


def run_executable(
    executable: str,
    arguments: list[str],
    extra_return_values: Optional[list[object]] = None,
    shell: bool = False) -> tuple[subprocess.CompletedProcess, ...]:
    """Run an executable.

    Parameters
    ----------
    executable : :class:`str`
        The executable.

    arguments : :class:`list`
        A list of arguments to run the executable with.

    extra_return_values : :class:`list`, optional
        The extra values to return after the completed process.

    shell : :class:`bool`, :obj:`False`
        Whether to run the executable in a shell.

    Returns
    -------
    completed_process : :class:`subprocess.CompletedProcess`
        The completed process.

    *extra_return_values
        The values in ``extra_return_values``, if passed.
    """

    # If the user requested to run the executable in a shell
    if shell:

        # Create the command line.
        line = f"{executable} {' '.join(arguments)}"

    # Otherwise
    else:

        # Create the command line.
        line = [executable] + arguments

    #-----------------------------------------------------------------#

    # Launch the executable.
    completed_process = \
        subprocess.run(line,
                       shell = shell)

    #-----------------------------------------------------------------#

    # If the user did not pass any extra return values
    if extra_return_values is None:

        # Use an empty list.
        extra_return_values = []

    #-----------------------------------------------------------------#

    # Return the completed process and the extra values.
    return (completed_process, *extra_return_values)


def get_dirs(dir_names: list[str] | str,
             wd: str) -> list[str]:
    """Get the full paths to a list of directories.

    Parameters
    ----------
    dir_names : :class:`list` or :class:`str`
        The names of the directories, a pattern matching them, or a
        plain text file listing them.

    wd : :class:`str`
        The working directory.

    Returns
    -------
    dirs : :class:`list`
        A list of the full paths to the directories.

    Raises
    ------
    :class:`TypeError`
        If ``dir_names`` is neither a list nor a string.
    """

    # If the user passed a list of directory names
    if isinstance(dir_names, list):

        # Get the full paths to the directories.
        dirs = \
            [os.path.abspath(os.path.join(wd, dir_name)) \
             for dir_name in dir_names]

    #-----------------------------------------------------------------#

    # If the user passed a pattern or the name of a file
    elif isinstance(dir_names, str):

        # If the string has a file extension
        if os.path.splitext(dir_names)[1]:

            # Read the file.
            with open(dir_names, "r") as f:

                # Get the full paths to the listed directories,
                # skipping empty lines.
                dirs = \
                    [os.path.abspath(\
                        os.path.join(wd, dir_name.strip()))
                    for dir_name in f.readlines() \
                    if dir_name.strip()]

        #-------------------------------------------------------------#

        # Otherwise
        else:

            # Get the full paths to the directories.
            dirs = \
                glob.glob(os.path.abspath(os.path.join(wd, dir_names)))

    #-----------------------------------------------------------------#

    # Otherwise
    else:

        # Raise an error.
        errstr = \
            "The argument 'dir_names' must be either a list of " \
            "directory names, a string representing a pattern to " \
            "match the directories, or a string representing a " \
            "file containing the names of the directories."
        raise TypeError(errstr)

    #-----------------------------------------------------------------#

    # Return the full paths to the directories.
    return dirs


def get_file_path(file_name: str,
                  wd: str,
                  main_wd: str) -> Optional[str]:
    """Get the full path to a file.

    Parameters
    ----------
    file_name : :class:`str`
        The name of the file.

    wd : :class:`str`
        The working directory.

    main_wd : :class:`str`
        The main working directory.

    Returns
    -------
    file_path : :class:`str` or :obj:`None`
        The full path to the file or :obj:`None` if ``file_name`` is
        not a file.
    """

    # Get the full path to the file in the working directory.
    file_path = os.path.abspath(os.path.join(wd, file_name))

    # Get the full path to the file in the main working directory.
    file_path_main = os.path.abspath(os.path.join(main_wd, file_name))

    # If the file exists in the working directory
    if os.path.exists(file_path):

        # Return the full path to the file.
        return file_path

    #-----------------------------------------------------------------#

    # If the file exists in the main working directory
    elif os.path.exists(file_path_main):

        # Return the full path to the file.
        return file_path_main

    #-----------------------------------------------------------------#

    # Otherwise
    else:

        # Return None.
        return None


def set_executable_args(args: argparse.Namespace,
                        parser: argparse.ArgumentParser,
                        wd: str,
                        main_wd: str) -> list[str]:
    """Set the arguments for an executable.

    Parameters
    ----------
    args : :class:`argparse.Namespace`
        The parsed arguments.

    parser : :class:`argparse.ArgumentParser`
        The parser the arguments were parsed with.

    wd : :class:`str`
        The working directory.

    main_wd : :class:`str`
        The main working directory.

    Returns
    -------
    arguments : :class:`list`
        The arguments, as a list to pass to :func:`subprocess.run`.
    """

    # Get the arguments as a dictionary.
    kwargs = copy.deepcopy(vars(args))

    # Get the arguments whose options take several words.
    dests_nargs = \
        {action.dest for action in parser._actions
         if action.nargs not in (None, 0)}

    #-----------------------------------------------------------------#

    # Initialize the arguments as an empty list.
    arguments = []

    #-----------------------------------------------------------------#

    # For each argument
    for arg, val in dict(kwargs).items():

        # If the value is None
        if val is None:

            # Remove it.
            kwargs.pop(arg)

        # If the argument starts with 'input_' or 'config_' and its
        # value is a string
        elif (arg.startswith("input_") \
              or arg.startswith("config_")) \
        and isinstance(val, str):

            # Get the full path to the file.
            file_path = \
                get_file_path(file_name = val,
                              wd = wd,
                              main_wd = main_wd)

            # If the file exists
            if file_path is not None:

                # Set the full path to the file.
                kwargs[arg] = file_path

        #-------------------------------------------------------------#

        # If the argument starts with 'output_' or is 'log_file'
        elif arg.startswith("output_") \
        or arg == "log_file":

            # Set the full path to the file.
            kwargs[arg] = os.path.abspath(os.path.join(wd, val))

    #-----------------------------------------------------------------#

    # Set the argument for the working directory.
    kwargs["work_dir"] = wd

    # Remove the argument for the directories.
    kwargs.pop("dirs", None)

    # Remove the argument for the parallelization.
    kwargs.pop("parallelize", None)

    # Remove the argument for the number of processes.
    kwargs.pop("n_proc", None)

    #-----------------------------------------------------------------#

    # For each argument
    for arg, val in kwargs.items():

        # Add the argument to the list of arguments.
        arguments.append(f"--{arg.replace('_', '-')}")

        # If the value is a list
        if isinstance(val, list):

            # If the option takes several words
            if arg in dests_nargs:

                # Add each element of the list to the list of
                # arguments.
                arguments.extend([str(item) for item in val])

            # Otherwise
            else:

                # Add the elements as one comma-separated value.
                arguments.append(",".join([str(item) for item in val]))

        # If the value is a boolean
        elif isinstance(val, bool):

            # If the value is False
            if not val:

                # Remove the flag.
                arguments.pop()

        # Otherwise
        else:

            # Add the value to the list of arguments.
            arguments.append(str(val))

    #-----------------------------------------------------------------#

    # Return the list of arguments.
    return arguments


def run_with_parallelization(executable: str,
                             args: argparse.Namespace,
                             parser: argparse.ArgumentParser) -> None:
    """Run an executable with parallelization using Dask.

    Parameters
    ----------
    executable : :class:`str`
        The name of the executable.

    args : :class:`argparse.Namespace`
        The parsed arguments.

    parser : :class:`argparse.ArgumentParser`
        The parser the arguments were parsed with.
    """

    # Create a list to store the futures.
    futures = []

    #-----------------------------------------------------------------#

    # Get the 'work_dir' argument.
    wd = getattr(args,
                 "work_dir",
                 None)

    # Get the 'n_proc' argument.
    n_proc = getattr(args,
                     "n_proc",
                     None)

    #-----------------------------------------------------------------#

    # Keep the launched commands' log messages off the console.
    args.log_console = False

    #-----------------------------------------------------------------#

    # Get the full paths to the directories.
    work_dirs = \
        get_dirs(dir_names = args.dirs,
                 wd = wd)

    #-----------------------------------------------------------------#

    # Import from 'distributed'.
    from distributed import \
        LocalCluster, Client, as_completed

    # Create the local cluster.
    cluster = \
        LocalCluster(\
            # Number of workers.
            n_workers = n_proc,
            # Below which level log messages will
            # be silenced.
            silence_logs = "ERROR",
            # Whether to use processes, single-core
            # or threads.
            processes = True,
            # How many threads for each worker
            # should be used.
            threads_per_worker = 1)

    # Open the client from the cluster.
    client = Client(cluster)

    #-----------------------------------------------------------------#

    # For each working directory
    for work_dir in work_dirs:

        # Set the arguments.
        arguments = \
            set_executable_args(args = args,
                                parser = parser,
                                wd = work_dir,
                                main_wd = wd)

        # Inform the user that the calculation is being submitted.
        infostr = \
            f"Submitting the calculation in " \
            f"'{work_dir}' directory. Command " \
            f"line: {' '.join(arguments)}."
        logger.info(infostr)

        # Submit the calculation.
        futures.append(\
            client.submit(\
                run_executable,
                executable = executable,
                arguments = arguments,
                extra_return_values = [work_dir]))

    #-----------------------------------------------------------------#

    # Get the futures as they are completed.
    for _, result in \
        as_completed(futures,
                     with_results = True):

        # Get the process and the working directory.
        process, work_dir = result

        # Check the process' return code.
        try:

            process.check_returncode()

        # If something went wrong
        except Exception:

            # Log the error.
            errstr = \
                f"The run in the '{work_dir}' " \
                "directory failed. Please check " \
                "the log file in the directory " \
                "for more information."
            logger.error(errstr)

            # Go to the next future.
            continue

        # Inform the user that the run completed
        # successfully.
        infostr = \
            f"The run in the directory " \
            f"'{work_dir}' completed " \
            "successfully."
        logger.info(infostr)

    #-----------------------------------------------------------------#

    # Close the client.
    client.close()

    # Close the cluster.
    cluster.close()