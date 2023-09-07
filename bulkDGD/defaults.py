#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-

#    defaults.py
#
#    Default values.
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


# Package name
pkg_name = "bulkDGD"

# Standard library
from pkg_resources import resource_filename, Requirement


#------------------------- Public constants --------------------------#


# The directory containing the configuration files specifying the
# DGD model's parameters and the files containing the trained
# model
CONFIG_MODEL_DIR = \
    resource_filename(Requirement(pkg_name),
                      "configs/model")

# The directory containing the configuration files specifying the
# options for data loading and optimization when finding the
# best representations
CONFIG_REP_DIR = \
    resource_filename(Requirement(pkg_name),
                      "configs/representations")

# The default configuration file for plotting the results of the PCA
CONFIG_PLOT_PCA = \
    resource_filename(Requirement(pkg_name),
                      "configs/plot/pca_scatter.yaml")

# Default PyTorch file containing the trained Gaussian mixture model
GMM_FILE = \
    resource_filename(Requirement(pkg_name),
                      "data/model/gmm.pth")

# Default PyTorch file containing the trained decoder
DEC_FILE = \
    resource_filename(Requirement(pkg_name),
                      "data/model/dec.pth")

# File containing the Ensembl IDs of the genes included in the DGD
# model
GENES_FILE = \
    resource_filename(Requirement(pkg_name),
                      "data/model/genes.txt")


# File containing the supported sample types (cancer types/tissues)
RECOUNT3_SUPPORTED_CATEGORIES_FILE = \
    {"gtex" : resource_filename(Requirement(pkg_name),
                                "data/recount3/gtex_tissues.txt"),
     "tcga" : resource_filename(Requirement(pkg_name),
                                "data/recount3/tcga_cancer_types.txt")}
    

# File containing the list of fields found in the metadata
RECOUNT3_METADATA_FIELDS_FILE = \
    {"gtex" : \
        resource_filename(Requirement(pkg_name),
                          "data/recount3/gtex_metadata_fields.txt"),
     "tcga" : \
        resource_filename(Requirement(pkg_name),
                          "data/recount3/tcga_metadata_fields.txt")}
    
# URL pointing to where the RNA-seq data for the samples
# are stored on the Recount3 platform
RECOUNT3_GENE_SUMS_URL = \
    "http://duffel.rail.bio/recount3/human/data_sources/{:s}/" \
    "gene_sums/{:s}/{:s}/{:s}.gene_sums.{:s}.G026.gz"

# URL pointing to where the metadata for the samples are stored
# on the Recount3 platform
RECOUNT3_METADATA_URL = \
    "http://duffel.rail.bio/recount3/human/data_sources/{:s}/" \
    "metadata/{:s}/{:s}/{:s}.{:s}.MD.gz"

# Path to the output GZ file containing the samples' gene
# expression data, if the user decided to save it
RECOUNT3_GENE_SUMS_FILE = "{:s}_{:s}_gene_sums.gz"

# Path to the output CSV file containing the samples'
# metadata, if the user decided to save it
RECOUNT3_METADATA_FILE = "{:s}_{:s}_metadata.gz"