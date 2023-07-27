#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-

#    setup.py
#
#    bulkDGD setup.
#
#    Copyright (C) 2023 Valentina Sora 
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


# Standard library
from setuptools import setup, find_packages


# Name of the package 
name = "bulkDGD"

# URL where to find the package
url = \
    f"https://github.com/Center-for-Health-Data-Science/" \
    f"{name}-private"

# Package author(s)
author = \
    "Valentina Sora, Viktoria Schuster, " \
    "Inigo Prada-Luengo, Anders Krogh"

# Package version
version = "0.0.1"

# A brief description of the package
description = \
    "A generative model for human gene expression from bulk " \
    "RNA-Seq data."

# Directory of the package
package_dir = {name : name}

# Which packages are included
packages = [name]

# Which package data to include
package_data = \
    {name : ["core/*"
             "execs/*",
             "utils/*"]}

# Command-line executables
entry_points = \
    {"console_scripts" : \
        [f"dgd_get_recount3_data = " \
         f"{name}.execs.dgd_get_recount3_data:main",
         f"dgd_preprocess_samples = " \
         f"{name}.execs.dgd_preprocess_samples:main",
         f"dgd_get_representations = " \
         f"{name}.execs.dgd_get_representations:main",
         f"dgd_perform_dea = " \
         f"{name}.execs.dgd_perform_dea:main",
         f"dgd_get_probability_density = " \
         f"{name}.execs.dgd_get_probability_density:main",
        ],
    }

# Required dependencies
install_requires = ["dask",
                    "distributed",
                    "numpy",
                    "pandas",
                    "requests",
                    "scipy",
                    "statsmodels",
                    "torch",
                    "PyYAML"]

# Run the setup
setup(name = name,
      url = url,
      author = author,
      version = version,
      description = description,
      include_package_data = True,
      package_data = package_data,
      package_dir = package_dir,
      packages = packages,
      entry_points = entry_points,
      install_requires = install_requires)