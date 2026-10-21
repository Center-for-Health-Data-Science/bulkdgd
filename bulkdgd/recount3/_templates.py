#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-

#    _templates.py
#
#    Templates for the configurations of datasets built from Recount3
#    studies.
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
    "Templates for the configurations of datasets built from " \
    "Recount3 studies."


#######################################################################


# A template is a dictionary mapping each option to its specification:
#
# - "type": the types the value can have.
# - "required": whether the option must be present.
# - "condition"/"message": a check on the value and its error message.
# - "section": the template a dictionary value must follow.
# - "keys"/"values": the types of a free dictionary's keys and the
#   specification of its values.
# - "items": the specification of a list's items.


#######################################################################


# Set the types of a value in the metadata.
_SCALAR = (str, int, float, bool, type(None))

# Set the types of a metadata value used as a key.
_SCALAR_KEY = (str, int, float, bool)

#---------------------------------------------------------------------#

# Set the specification of a list of strings.
_STRINGS = {
    "type" : (list,),
    "items" : {"type" : (str,)},
    }

# Set the specification of the old values of a column mapped to new
# values.
_VALUES_MAPPING = {
    "type" : (dict,),
    "keys" : _SCALAR_KEY,
    "values" : {"type" : _SCALAR},
    }

#---------------------------------------------------------------------#

# Set the specification of the studies to take.
_STUDIES = {
    "type" : (list,),
    "items" : {"type" : (str,)},
    "required" : True,
    "condition" : lambda v: len(v) > 0,
    "message" : "must be a non-empty list",
    }

# Set the specification of the query selecting the samples.
_QUERY_STRING = {
    "type" : (str,),
    }

# Set the specification of the columns to keep, mapped to their new
# names.
_METADATA_TO_KEEP = {
    "type" : (dict,),
    "keys" : (str,),
    "values" : {"type" : (str,)},
    }

# Set the specification of the columns to add, each mapping queries
# (or '_all_'/'_other_') to the values to set.
_METADATA_TO_ADD = {
    "type" : (dict,),
    "keys" : (str,),
    "values" : {
        "type" : (dict,),
        "keys" : (str,),
        "values" : {"type" : _SCALAR},
        },
    }

# Set the specification of the columns whose values are mapped to new
# values.
_MAPPING = {
    "type" : (dict,),
    "keys" : (str,),
    "values" : _VALUES_MAPPING,
    }

# Set the specification of the columns whose values get a prefix or a
# suffix.
_MAPPING_AFFIX = {
    "type" : (dict,),
    "keys" : (str,),
    "values" : {"type" : (str,)},
    }


#######################################################################


# Set the template for a column split at separators (TCGA).
_PROCESS_COLUMN = {

    # The separators' names mapped to the new columns.
    "columns" : {
        "type" : (dict,),
        "keys" : (str,),
        "values" : {"type" : (str,)},
        "required" : True,
        },

    # The separators.
    "separators" : {
        "type" : (list,),
        "items" : {"type" : (str,)},
        "required" : True,
        },

    # Whether to keep the new columns in the dataset.
    "keep_columns" : {
        "type" : (bool,),
        },
    }

#---------------------------------------------------------------------#

# Set the template for a column joined into a new column (TCGA).
_JOIN_COLUMN = {

    # The column.
    "column" : {
        "type" : (str,),
        "required" : True,
        },

    # The column's values mapped to new values.
    "mapping" : _VALUES_MAPPING,

    # The text put before each value, followed by a colon.
    "prefix" : {
        "type" : (str,),
        },

    # The text put before each value.
    "phrase_prefix" : {
        "type" : (str,),
        },

    # The text put after each value.
    "phrase_suffix" : {
        "type" : (str,),
        },
    }

# Set the template for a column merged into a new column with a
# pattern (TCGA).
_MERGE_COLUMN = {

    # The column.
    "column" : {
        "type" : (str,),
        "required" : True,
        },

    # The pattern whose first group is taken.
    "pattern" : {
        "type" : (str,),
        "required" : True,
        },
    }

# Set the template for a new column built from other columns (TCGA).
_MERGED_COLUMN = {

    # The columns whose values are joined.
    "columns_to_join" : {
        "type" : (list,),
        "items" : {
            "type" : (dict,),
            "section" : _JOIN_COLUMN,
            },
        },

    # The separator of the joined values.
    "separator" : {
        "type" : (str,),
        },

    # The columns whose values fill each other's missing values.
    "columns_to_merge" : {
        "type" : (list,),
        "items" : {
            "type" : (str, dict),
            "section" : _MERGE_COLUMN,
            },
        },

    # The merged values mapped to new values.
    "mapping" : _VALUES_MAPPING,

    # Whether to number the merged values from 1.
    "serial" : {
        "type" : (bool,),
        },
    }


#######################################################################


# Set the template for a block of values recovered from another study
# (SRA).
_FROM_STUDY = {

    # The columns mapped to their values, each mapped to its runs.
    "values" : {
        "type" : (dict,),
        "keys" : (str,),
        "values" : {
            "type" : (dict,),
            "keys" : _SCALAR_KEY,
            "values" : _STRINGS,
            },
        },

    # How the runs were matched (a note).
    "matched_on" : {
        "type" : (str,),
        },

    # Where the values come from (a note).
    "source" : {
        "type" : (str,),
        },

    # The columns the values are for (a note).
    "fields" : _STRINGS,
    }

#---------------------------------------------------------------------#

# Set the template for an SRA study's options.
CONFIG_SRA_STUDY = {

    # The study-level metadata.
    "metadata_study" : {
        "type" : (dict,),
        "keys" : (str,),
        "values" : {"type" : _SCALAR},
        },

    # The query selecting the samples.
    "query_string" : _QUERY_STRING,

    # The query selecting the samples in the GEO metadata.
    "query_string_geo" : _QUERY_STRING,

    # The column used as the samples' IDs while processing.
    "_use_as_external_id" : {
        "type" : (str,),
        },

    # The runs to drop.
    "samples_to_drop" : _STRINGS,

    # The columns to keep.
    "metadata_to_keep" : _METADATA_TO_KEEP,

    # The GEO metadata's columns to keep.
    "metadata_to_keep_geo" : _METADATA_TO_KEEP,

    # The columns to add.
    "metadata_to_add" : _METADATA_TO_ADD,

    # The columns to add from the GEO metadata.
    "metadata_to_add_geo" : _METADATA_TO_ADD,

    # The columns' values mapped to new values.
    "mapping" : _MAPPING,

    # The GEO metadata's columns' values mapped to new values.
    "mapping_geo" : _MAPPING,

    # The columns whose values get a prefix.
    "mapping_with_prefix" : _MAPPING_AFFIX,

    # The columns whose values get a suffix.
    "mapping_with_suffix" : _MAPPING_AFFIX,

    # The columns whose substrings are replaced.
    "mapping_values" : {
        "type" : (dict,),
        "keys" : (str,),
        "values" : {
            "type" : (dict,),
            "keys" : (str,),
            "values" : {"type" : (str,)},
            },
        },

    # The columns whose values are numbered from 1, in order.
    "serial_mapping" : {
        "type" : (dict,),
        "keys" : (str,),
        "values" : {
            "type" : (list,),
            "items" : {"type" : _SCALAR},
            },
        },

    # The values recovered from other studies.
    "metadata_from_study" : {
        "type" : (dict,),
        "keys" : (str,),
        "values" : {
            "type" : (dict,),
            "section" : _FROM_STUDY,
            },
        },
    }


#######################################################################


# Set the template for a GTEx dataset's configuration.
CONFIG_GTEX = {

    # The tissues.
    "studies" : _STUDIES,

    # The query selecting the samples.
    "query_string" : _QUERY_STRING,

    # The columns to keep.
    "metadata_to_keep" : _METADATA_TO_KEEP,

    # The columns' values mapped to new values.
    "mapping" : _MAPPING,

    # The columns to add.
    "metadata_to_add" : _METADATA_TO_ADD,
    }

#---------------------------------------------------------------------#

# Set the template for a TCGA dataset's configuration.
CONFIG_TCGA = {

    # The cancer types.
    "studies" : _STUDIES,

    # The query selecting the samples.
    "query_string" : _QUERY_STRING,

    # The columns split at separators.
    "metadata_to_process" : {
        "type" : (dict,),
        "keys" : (str,),
        "values" : {
            "type" : (dict,),
            "section" : _PROCESS_COLUMN,
            },
        },

    # The new columns built from other columns.
    "metadata_to_merge" : {
        "type" : (dict,),
        "keys" : (str,),
        "values" : {
            "type" : (dict,),
            "section" : _MERGED_COLUMN,
            },
        },

    # The columns to keep.
    "metadata_to_keep" : _METADATA_TO_KEEP,

    # The columns whose values are lowercased.
    "mapping_to_lower" : _STRINGS,

    # The columns' values mapped to new values.
    "mapping" : _MAPPING,

    # The columns to add.
    "metadata_to_add" : _METADATA_TO_ADD,
    }
