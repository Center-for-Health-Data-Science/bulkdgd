#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-

#    setup.py
#
#    bulkdgd setup.
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


#######################################################################


# Import from the standard library.
import pathlib

# Import from third-party packages.
from setuptools import setup


#######################################################################


# Set the name of the project.
name = "bulkdgd"

# Set the URL where to find the project.
url = \
    f"https://github.com/Center-for-Health-Data-Science/{name}"

# Set the project's author(s).
author = \
    "Valentina Sora, Adrian Sousa-Poza, Viktoria Schuster, " \
    "Iñigo Prada-Luengo, Anders Lykkebo-Valløe, " \
    "Andreas Bjerregaard, Anders Krogh"

# Set the maintainer's contact e-mail.
author_email = "sora.valentina1@gmail.com"

# Set the project's version.
version = "2.0.1"

# Set a brief description of the project.
description = \
    "A generative model for human gene expression from bulk " \
    "RNA-seq data."

# Set the long description from the README file, to be rendered
# on the project's PyPI page.
long_description = \
    pathlib.Path(__file__).parent.joinpath("README.md").read_text(\
        encoding = "utf-8")
long_description_content_type = "text/markdown"

# Set the minimum Python version required (the code uses the
# PEP 604 'X | Y' union type syntax, which requires Python 3.10+).
python_requires = ">=3.10"

# Set the project's classifiers.
classifiers = \
    ["Development Status :: 5 - Production/Stable",
     "Intended Audience :: Science/Research",
     "License :: OSI Approved :: GNU General Public License v3 " \
     "or later (GPLv3+)",
     "Operating System :: OS Independent",
     "Programming Language :: Python :: 3",
     "Programming Language :: Python :: 3.10",
     "Programming Language :: Python :: 3.11",
     "Programming Language :: Python :: 3.12",
     "Topic :: Scientific/Engineering :: Bio-Informatics"]

# Set links to relevant project pages, shown on the PyPI page.
project_urls = \
    {"Documentation" : "https://bulkdgd.readthedocs.io/en/latest/",
     "Source" : url,
     "Bug Tracker" : f"{url}/issues"}

# Set which packages are included.
packages = \
    ["bulkdgd",
     "bulkdgd._internals",
     "bulkdgd.analysis",
     "bulkdgd.core",
     "bulkdgd.execs",
     "bulkdgd.genes",
     "bulkdgd.ioutil",
     "bulkdgd.plotting",
     "bulkdgd.recount3"]

# Set which package data to include.
package_data = \
    {# Main package
     "bulkdgd" : \
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
         "data/*.md"],
     
     # 'recount3' package
     "bulkdgd.recount3" : \
        [# Data files
         "data/*.txt",
         "data/*.md"]}

# Set the command-line executables.
entry_points = \
    {"console_scripts" : \
        [# Public executables

         # Get genes.
         "bulkdgd_get_genes = " \
         f"{name}.execs.bulkdgd_get_genes:entry_point",

         # Get Recount3 data.
         "bulkdgd_get_recount3 = " \
         f"{name}.execs.bulkdgd_get_recount3:entry_point",

         # Find representations.
         "bulkdgd_find_representations = " \
         f"{name}.execs.bulkdgd_find_representations:entry_point",

         # Find probability densities.
         "bulkdgd_find_probdens = " \
         f"{name}.execs.bulkdgd_find_probdens:entry_point",

         # Preprocess samples.
         "bulkdgd_preprocess_samples = " \
         f"{name}.execs.bulkdgd_preprocess_samples:entry_point",

         # Dimensionality reduction - PCA.
         "bulkdgd_reduction_pca = " \
         f"{name}.execs.bulkdgd_reduction:entry_point_pca",

         # Dimensionality reduction - KPCA.
         "bulkdgd_reduction_kpca = " \
         f"{name}.execs.bulkdgd_reduction:entry_point_kpca",

         # Dimensionality reduction - MDS.
         "bulkdgd_reduction_mds = " \
         f"{name}.execs.bulkdgd_reduction:entry_point_mds",

         # Dimensionality reduction - t-SNE.
         "bulkdgd_reduction_tsne = " \
         f"{name}.execs.bulkdgd_reduction:entry_point_tsne",

         # Dimensionality reduction - UMAP.
         "bulkdgd_reduction_umap = " \
         f"{name}.execs.bulkdgd_reduction:entry_point_umap",

         # Differential expression analysis.
         "bulkdgd_dea = " \
         f"{name}.execs.bulkdgd_dea:entry_point",

         # Train the model.
         "bulkdgd_train = " \
         f"{name}.execs.bulkdgd_train:entry_point",

         # "Private" executables - not intended to be called
         # directly by end users.
         "_bulkdgd_get_recount3_single_batch = " \
         f"{name}.execs._bulkdgd_get_recount3_single_batch:main",],
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
                    "tgmm",
                    "torch",
                    "umap-learn",
                    "PyYAML"]


#######################################################################


# Run the setup.
setup(name = name,
      url = url,
      author = author,
      author_email = author_email,
      version = version,
      description = description,
      long_description = long_description,
      long_description_content_type = long_description_content_type,
      python_requires = python_requires,
      classifiers = classifiers,
      project_urls = project_urls,
      packages = packages,
      package_data = package_data,
      entry_points = entry_points,
      install_requires = install_requires)
