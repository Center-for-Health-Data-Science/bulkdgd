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


# Standard library
import os


# File containing the supported sample types (cancer types/tissues)
RECOUNT3_SUPPORTED_CATEGORIES_FILE = \
    {"gtex" : os.path.join(os.path.dirname(__file__), 
                           "data/gtex_tissues.txt"),
     "tcga" : os.path.join(os.path.dirname(__file__),
                           "data/tcga_cancer_types.txt")}
    

# File containing the list of fields found in the metadata
RECOUNT3_METADATA_FIELDS_FILE = \
    {"gtex" : os.path.join(os.path.dirname(__file__),
                           "data/gtex_metadata_fields.txt"),
     "tcga" : os.path.join(os.path.dirname(__file__),
                           "data/tcga_metadata_fields.txt")}
    
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