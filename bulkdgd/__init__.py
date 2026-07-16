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

# Import what seeds a run's generators. It is here, and not in 'core',
# because it has to be called before a model is built: by the time
# 'BulkDGD' exists its decoder's weights have already been drawn, and
# seeding afterwards seeds nothing that has already happened.
from .reproducibility import set_seeds, get_seeds_state


# Set the package's version, read from the installed distribution's
# metadata (falls back to 'unknown' for an unpacked source checkout
# that was never installed).
try:

    __version__ = _importlib_metadata.version("bulkdgd")

except _importlib_metadata.PackageNotFoundError:

    __version__ = "unknown"
