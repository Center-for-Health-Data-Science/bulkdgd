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


# The name of the project
name = "bulkDGD"

# The URL where to find the project
url = \
    f"https://github.com/Center-for-Health-Data-Science/" \
    f"{name}"

# The project's author(s)
author = \
    "Valentina Sora, Viktoria Schuster, " \
    "Inigo Prada-Luengo, Anders Krogh"

# The project's version
version = "0.0.2"

# A brief description of the project
description = \
    "A generative model for human gene expression from bulk " \
    "RNA-Seq data."

# The directory of the project
package_dir = {name : name}

# Which package data to include
package_data = \
    {name : ["analysis/*"
             "core/*"
             "execs/*",
             "ioutil/*",
             "plotting/*",
             "recount3/*"]}

# Command-line executables
entry_points = \
    {"console_scripts" : \
        ["dgd_get_recount3_data = " \
         f"{name}.execs.dgd_get_recount3_data:main",
         "dgd_preprocess_samples = " \
         f"{name}.execs.dgd_preprocess_samples:main",
         "dgd_get_representations = " \
         f"{name}.execs.dgd_get_representations:main",
         "dgd_perform_dea = " \
         f"{name}.execs.dgd_perform_dea:main",
         "dgd_perform_pca = " \
         f"{name}.execs.dgd_perform_pca:main",
         "dgd_get_probability_density = " \
         f"{name}.execs.dgd_get_probability_density:main",
        ],
    }

# Any required dependencies
install_requires = ["dask",
                    "distributed",
                    "matplotlib",
                    "numpy",
                    "pandas",
                    "requests",
                    "seaborn",
                    "scikit-learn",
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
      entry_points = entry_points,
      install_requires = install_requires)