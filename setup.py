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
from setuptools import setup, find_packages


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
version = "1.0.4"

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
     "bulkDGD.ioutil",
     "bulkDGD.plotting",
     "bulkDGD.recount3"]

# Set which package data to include.
package_data = \
    {"bulkDGD.ioutil" : ["configs/model/*.yaml",
                         "configs/plot/*.yaml",
                         "configs/representations/*.yaml",
                         "configs/training/*.yaml",
                         "data/*.pth",
                         "data/*.txt",
                         "data/*.md",],
     "bulkDGD.recount3" : ["data/*.txt",
                           "data/*.md"]}

# Set the command-line executables.
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

         "dgd_train = " \
         f"{name}.execs.dgd_train:main",

         # "Private" executables - not intended to be called
         # directly by end users
         "_dgd_get_recount3_data_single_batch = " \
         f"{name}.execs._dgd_get_recount3_data_single_batch:main"],
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
