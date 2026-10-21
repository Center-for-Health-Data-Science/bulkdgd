#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-

#    defaults.py
#
#    Default URLs/files for interacting with the Recount3 platform,
#    downloading data from it and building datasets.
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


# Set the module's description.
__doc__ = \
    "Default URLs/files for interacting with the Recount3 platform, " \
    "downloading data from it and building datasets."


#######################################################################


# Import from the standard library.
import os


#######################################################################


# Set the available Gencode releases used in Recount3.
RECOUNT3_GENCODE_RELEASES = {26, 29}

#---------------------------------------------------------------------#

# Set the files listing each project's metadata fields.
RECOUNT3_METADATA_FIELDS_FILE = {

    "GTEx" : os.path.join(os.path.dirname(__file__),
                          "data/gtex_metadata_fields.txt"),

    "TCGA" : os.path.join(os.path.dirname(__file__),
                          "data/tcga_metadata_fields.txt"),

    "SRA" : os.path.join(os.path.dirname(__file__),
                         "data/sra_metadata_fields.txt"),

    }

#---------------------------------------------------------------------#

# Set the URL pointing to where the RNA-seq data for the samples are
# stored on the Recount3 platform.
RECOUNT3_GENE_SUMS_URL = \
    "http://duffel.rail.bio/recount3/human/data_sources/{:s}/" \
    "gene_sums/{:s}/{:s}/{:s}.gene_sums.{:s}.G0{:d}.gz"

#---------------------------------------------------------------------#

# Set the URL pointing to where the QC metadata for the samples are
# stored on the Recount3 platform.
RECOUNT3_QC_URL = \
    "http://duffel.rail.bio/recount3/human/data_sources/{:s}/" \
    "metadata/{:s}/{:s}/{:s}.recount_qc.{:s}.MD.gz"

#---------------------------------------------------------------------#

# Set the URL pointing to where the metadata for the samples are stored
# on the Recount3 platform.
RECOUNT3_METADATA_URL = \
    "http://duffel.rail.bio/recount3/human/data_sources/{:s}/" \
    "metadata/{:s}/{:s}/{:s}.{:s}.MD.gz"

#---------------------------------------------------------------------#

# Set the name of the output GZ file containing the samples' RNA-seq
# data, if the user decided to save it.
RECOUNT3_GENE_SUMS_FILE = "{:s}_{:s}_gene_sums.gz"

#---------------------------------------------------------------------#

# Set the name of the output GZ file containing the samples' QC
# metadata, if the user decided to save it.
RECOUNT3_QC_FILE = "{:s}_{:s}_qc.gz"

#---------------------------------------------------------------------#

# Set the name of the output GZ file containing the samples' metadata,
# if the user decided to save it.
RECOUNT3_METADATA_FILE = "{:s}_{:s}_metadata.gz"


#######################################################################


# Set the supported Recount3 projects.
RECOUNT3_PROJECTS = ["GTEx", "TCGA", "SRA"]

#---------------------------------------------------------------------#

# Set the URL of a GEO series' SOFT file.
GEO_SOFT_URL = \
    "https://ftp.ncbi.nlm.nih.gov/geo/series/{:s}nnn/{:s}/" \
    "soft/{:s}_family.soft.gz"

# Set the name of a saved GEO SOFT file.
GEO_SOFT_FILE = "{:s}_family.soft.gz"

#---------------------------------------------------------------------#

# Set the URL of ENA's report on an SRA study's runs.
ENA_FILEREPORT_URL = \
    "https://www.ebi.ac.uk/ena/portal/api/filereport" \
    "?accession={:s}&result=read_run&fields={:s}&format=tsv"

# Set the fields retrieved from ENA for each run.
ENA_FILEREPORT_FIELDS = \
    ["run_accession",
     "experiment_accession",
     "sample_accession",
     "study_accession",
     "library_strategy",
     "library_selection",
     "library_source"]

# Set the name of a saved ENA report.
ENA_FILEREPORT_FILE = "{:s}_ena_filereport.tsv"

# Set the time to wait after each request to ENA, in seconds.
ENA_REQUEST_PAUSE = 0.34

#---------------------------------------------------------------------#

# Set the URL of an SRA study's Run Selector table.
SRA_RUN_SELECTOR_URL = \
    "https://trace.ncbi.nlm.nih.gov/Traces/sra-db-be/runinfo?acc={:s}"

# Set the name of a saved Run Selector table.
SRA_RUN_SELECTOR_FILE = "{:s}_run_selector.csv"

#---------------------------------------------------------------------#

# Set the SRA metadata's library columns and their new names.
SRA_LIBRARY_COLUMNS = \
    {"library_name" : "sample_sequencing_library_name",
     "library_strategy" : "sample_sequencing_library_strategy",
     "library_source" : "sample_sequencing_library_source",
     "library_selection" : "sample_sequencing_library_selection",
     "library_layout" : "sample_sequencing_library_layout"}

# Set the ENA fields filling the SRA metadata's library columns.
SRA_ENA_LIBRARY_COLUMNS = \
    {"library_strategy" : "sample_sequencing_library_strategy",
     "library_selection" : "sample_sequencing_library_selection",
     "library_source" : "sample_sequencing_library_source"}

# Set the new values of the SRA metadata's library columns.
SRA_LIBRARY_VALUES = {

    "library_selection" : \
        {"cDNA" : "cDNA",
         "PolyA" : "polyA",
         "RT-PCR" : "RT-PCR",
         "other" : "unspecified",
         "RANDOM" : "random",
         "Oligo-dT" : "oligo-dT",
         "Inverse rRNA" : "inverse rRNA",
         "Hybrid Selection" : "hybrid selection",
         "unspecified" : "unspecified",
         "RANDOM PCR" : "random PCR",
         "PCR" : "PCR"},

    "library_source" : \
        {"TRANSCRIPTOMIC" : "transcriptomic"},

    "library_strategy" : \
        {"RNA-Seq" : "RNA-seq"},

    }

# Set the columns placed first in the SRA metadata, if present.
SRA_FIRST_COLUMNS = \
    ["recount3_project_name",
     "recount3_samples_category",
     "sample_identifier_ena_experiment",
     "sample_identifier_ena_biosample",
     "sample_identifier_run_numeric_id",
     "sample_sequencing_library_strategy",
     "sample_sequencing_library_selection",
     "sample_sequencing_library_source"]

#---------------------------------------------------------------------#

# Set the text appended to the SRA studies' comments.
SRA_COMMENT_HEAD = \
    ' Data for the "sample_sequencing_library_name", ' \
    '"sample_sequencing_library_strategy", ' \
    '"sample_sequencing_library_source", ' \
    '"sample_sequencing_library_selection", and ' \
    '"sample_sequencing_library_layout" fields were taken ' \
    'from the Recount3 metadata file. '

# Set the text appended to the SRA studies' comments that already
# give the library selection's source.
SRA_COMMENT_HEAD_OWN_SELECTION = \
    ' Data for the "sample_sequencing_library_name", ' \
    '"sample_sequencing_library_strategy", ' \
    '"sample_sequencing_library_source" and ' \
    '"sample_sequencing_library_layout" fields were taken ' \
    'from the Recount3 metadata file. '

# Set the text closing the SRA studies' comments.
SRA_COMMENT_TAIL = \
    'The "subject_demographics_sex_or_gender" and ' \
    '"subject_demographics_age_bin_10_years" fields were ' \
    'automatically created. '

# Set the text closing the comments of the SRA studies found in ENA.
SRA_COMMENT_TAIL_ENA = \
    SRA_COMMENT_TAIL + \
    'Data for the "sample_identifier_ena_biosample", ' \
    '"sample_identifier_ena_experiment" and ' \
    '"sample_identifier_run_numeric_id" fields were taken from ' \
    'ENA.'

#---------------------------------------------------------------------#

# Set the width of the age bins, in years.
AGE_BIN_WIDTH = 10

# Set the limit the age bins' edges stay below, in years.
AGE_BIN_MAX = 120

#---------------------------------------------------------------------#

# Set the values taken as missing when merging or joining columns.
MERGE_NA_VALUES = {"na", "NA", ""}

# Set the default separator of the joined values.
JOIN_SEPARATOR = "|"
