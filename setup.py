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
    "Inigo Prada-Luengo, Anders Lykkebo-Valløe, " \
    "Anders Krogh"

# The project's version
version = "0.0.3"

# A brief description of the project
description = \
    "A generative model for human gene expression from bulk " \
    "RNA-Seq data."

# Which packages are included
packages = \
    ["bulkDGD",
     "bulkDGD.analysis",
     "bulkDGD.core",
     "bulkDGD.execs",
     "bulkDGD.ioutil",
     "bulkDGD.plotting",
     "bulkDGD.recount3"]

# Which package data to include
package_data = \
    {"bulkDGD.ioutil" : ["configs/model/*.yaml",
                         "configs/model/*.md",
                         "configs/plot/*.yaml",
                         "configs/plot/*.md",
                         "configs/representations/*.yaml",
                         "configs/representations/*.md",
                         "data/*.pth",
                         "data/*.txt",
                         "data/*.md",],
     "bulkDGD.recount3" : ["data/*.txt",
                           "data/*.md"]}

# Command-line executables
entry_points = \
    {"console_scripts" : \
        [# Public executables
         "dgd_get_recount3_data = " \
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
        # "Private executable" - not intended to be called
        # directly by end users
        "_dgd_get_recount3_data_single_batch = " \
         f"{name}.execs._dgd_get_recount3_data_single_batch:main",],
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
      packages = packages,
      package_data = package_data,
      entry_points = entry_points,
      install_requires = install_requires)