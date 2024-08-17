#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-

#    bulkdgd_find.py
#
#    'Find' command.
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
__doc__ = "'Find' command."


#######################################################################


# Import from the standard library.
import logging as log
# Import from 'bulkDGD'.
from . import util


#######################################################################


# Get the module's logger.
logger = log.getLogger(__name__)


#######################################################################


# Define a function to set up the parser.
def set_parser(sub_parsers):

    # Create the argument parser.
    parser = \
        sub_parsers.add_parser(\
            name = "find",
            description = __doc__,
            help = __doc__,
            formatter_class = util.CustomHelpFormatter)

    # Return the parser.
    return parser
