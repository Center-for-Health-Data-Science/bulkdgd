#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-

#    defaults.py
#
#    Default values for the executables.
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
__doc__ = "Default values for the executables."


#######################################################################


# Set the list of Dask loggers.
DASK_LOGGERS = [\
    "distributed.batched",
    "distributed.client",
    "distributed.comm",
    "distributed.core",
    "distributed.diskutils",
    "distributed.http.proxy",
    "distributed.nanny",
    "distributed.scheduler",
    "distributed.worker",
    "bokeh.server"]

#---------------------------------------------------------------------#

# Set the default level for logging.
LOG_LEVEL = "WARNING"

# Set the default format for log records.
LOG_FMT = "{asctime}:{levelname}:{name}:{message}"

# Set the default format for date and time in log records.
LOG_DATEFMT = "%Y-%m-%d,%H:%M"

# Set the style of the log records.
LOG_STYLE = "{"

#---------------------------------------------------------------------#

# Set the configuration for named formatters used when logging.
CONFIG_FORMATTERS = \
    {# Set a generic formatter for log messages.
     "generic_formatter" : \
    
        {# Set a format string in the given 'style' for the logged
         # output as a whole.
         "fmt" : LOG_FMT,

         # Set a format string in the given 'style' for the date/time
         # portion of the logged output.
         "datefmt" : LOG_DATEFMT,

         # The 'style' be one of '%', '{' or '$' and determines how
         # the format string will be merged with its data.
         "style" : LOG_STYLE},
    }

#---------------------------------------------------------------------#

# Set the configuration for named filters used when logging.
CONFIG_FILTERS = \
    {# Ignore messages starting with specific strings from the Dask
     # loggers.
     "distributed_filter" : \

        {# Set the custom 'factory' for the filter.
         "()" : "bulkDGD.util.LevelContentFilter",
         
         # Set the level of the records that should be ignored.
         "level" : "INFO",

         # Set how the record should start for them to be ignored.
         "start" : ["Starting Worker plugin",
                    "---",
                    "Local Directory:",
                    "Memory:",
                    "Threads:",
                    "Registered to:",
                    "Stopping worker",
                    "Waiting to connect to:",
                    "Listening to:",
                    "Start worker at:",
                    "Worker name:",
                    "dashboard at:",
                    "Worker closed",
                    "Found state lock",
                    "Starting established connection",
                    "Keep-alive"]},
    }
