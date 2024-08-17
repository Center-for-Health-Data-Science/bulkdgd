#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-

#    setup.py
#
#    bulkDGD setup.
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


# Import from the standard library.
from setuptools import setup


#######################################################################


# Set the name of the project.
name = "bulkDGD"

# Set the URL where to find the project.
url = \
    f"https://github.com/Center-for-Health-Data-Science/{name}"

# Set the project's author(s).
author = \
    "Valentina Sora, Viktoria Schuster, Iñigo Prada-Luengo, " \
    "Anders Lykkebo-Valløe, Andreas Bjerregaard, Anders Krogh"

# Set the project's version.
version = "2.0.0"

# Set a brief description of the project.
description = \
    "A generative model for human gene expression from bulk " \
    "RNA-seq data."

# Set which packages are included.
packages = \
    ["bulkDGD",
     "bulkDGD.analysis",
     "bulkDGD.core",
     "bulkDGD.execs",
     "bulkDGD.genes",
     "bulkDGD.ioutil",
     "bulkDGD.plotting",
     "bulkDGD.recount3"]

# Set which package data to include.
package_data = \
    {# Main package
     "bulkDGD" : \
        [# Configuration files
         "configs/dimensionality_reduction/*yaml",
         "configs/genes/*.yaml",
         "configs/model/*.yaml",
         "configs/plotting/*.yaml",
         "configs/representations/*.yaml",
         "configs/training/*.yaml",
         # Data files
         "data/model/genes/*.txt",
         "data/model/gmm/*.pth",
         "data/model/dec/*pth",
         "data/*.md",],
     
     # 'recount3' package
     "bulkDGD.recount3" : \
        [# Data files
         "data/*.txt",
         "data/*.md"]}

# Set the command-line executables.
entry_points = \
    {"console_scripts" : \
        [# Public executables
         "bulkdgd = " \
         f"{name}.execs.main:main",

         # "Private" executables - not intended to be called
         # directly by end users
         "_bulkdgd_recount3_single_batch = " \
         f"{name}.execs._bulkdgd_recount3_single_batch:main"],
    }

# Set any required dependencies.
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


#######################################################################


# Run the setup.
setup(name = name,
      url = url,
      author = author,
      version = version,
      description = description,
      packages = packages,
      package_data = package_data,
      entry_points = entry_points,
      install_requires = install_requires)
