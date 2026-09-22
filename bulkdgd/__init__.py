#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-

#    __init__.py
#
#    Simple __init__.py file.
#
#    Copyright (C) 2026 Valentina Sora
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


# Import from the standard library.
import importlib.metadata as _importlib_metadata

# Import everything from the 'defaults' module.
from .defaults import *

# Import the functions to seed the generators (called before a model
# is built).
from .reproducibility import set_seeds, get_seeds_state


# Try to get the package's version.
try:

    # Get it from the installed package's metadata.
    __version__ = _importlib_metadata.version("bulkdgd")

# If the package is not installed
except _importlib_metadata.PackageNotFoundError:

    # Set the version to 'unknown'.
    __version__ = "unknown"
