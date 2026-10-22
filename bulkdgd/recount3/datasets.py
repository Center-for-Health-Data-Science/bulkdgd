#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-

#    datasets.py
#
#    Utilities to build datasets from studies hosted on the Recount3
#    platform.
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
    "Utilities to build datasets from studies hosted on the " \
    "Recount3 platform."

# Set the module's public names.
__all__ = \
    ["get_dataset_metadata",
     "get_dataset_counts",
     "save_dataset_metadata",
     "save_dataset_counts"]


#######################################################################


# Import from the standard library.
import copy
import logging as log
import os
import re
from typing import Optional

# Import from third-party libraries.
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv

# Import from 'bulkdgd'.
from . import _util, defaults, util


#######################################################################


# Get the module's logger.
logger = log.getLogger(__name__)


########################## PRIVATE FUNCTIONS ##########################


def _check_columns(df: pd.DataFrame,
                   columns: list[str],
                   option: str,
                   where: str) -> None:
    """Raise an error if columns named in an option are not in the
    metadata.

    Parameters
    ----------
    df : :class:`pandas.DataFrame`
        The metadata.

    columns : :class:`list`
        The columns.

    option : :class:`str`
        The option naming the columns.

    where : :class:`str`
        The project or study the metadata belong to.
    """

    # Get the columns not in the metadata.
    missing = [col for col in columns if col not in df.columns]

    # If there are any
    if missing:

        # Raise an error.
        errstr = \
            f"{where}: the column(s) " \
            f"{', '.join(repr(col) for col in missing)} in " \
            f"'{option}' are not in the metadata."
        raise ValueError(errstr)


def _check_kept_columns(df: pd.DataFrame,
                        columns: list[str],
                        renamed: dict[str, str],
                        where: str) -> None:
    """Raise an error if columns to keep are not in the metadata,
    naming the renamed ones by their original names.

    Parameters
    ----------
    df : :class:`pandas.DataFrame`
        The metadata, with the columns renamed.

    columns : :class:`list`
        The columns to keep.

    renamed : :class:`dict`
        The original names of the columns mapped to their new names.

    where : :class:`str`
        The project or study the metadata belong to.
    """

    # Get the columns not in the metadata.
    missing = [col for col in columns if col not in df.columns]

    # If there are any
    if missing:

        # Name each by its original name(s), if renamed.
        missing_str = \
            ", ".join(\
                " or ".join(\
                    repr(old) for old, new in renamed.items() \
                    if new == col) + f" (as {col!r})" \
                if col in renamed.values() else repr(col) \
                for col in missing)

        # Raise an error.
        errstr = \
            f"{where}: the column(s) to keep {missing_str} are not " \
            "in the metadata."
        raise ValueError(errstr)


def _add_metadata(df: pd.DataFrame,
                  metadata_to_add: dict[str, dict[str, object]],
                  as_strings: bool) -> pd.DataFrame:
    """Add columns whose values are set for the samples matching
    queries.

    Parameters
    ----------
    df : :class:`pandas.DataFrame`
        The metadata.

    metadata_to_add : :class:`dict`
        The columns to add, each mapping queries (or ``"_all_"`` and
        ``"_other_"``) to the values to set.

    as_strings : :class:`bool`
        Whether to set the values as strings (:obj:`None` as a
        missing value).

    Returns
    -------
    df : :class:`pandas.DataFrame`
        The metadata with the new columns.
    """

    # Drop the columns to add that are already there.
    df = df.drop(columns = [col for col in metadata_to_add \
                            if col in df.columns])

    #-----------------------------------------------------------------#

    # For each column to add and its values
    for col, values in metadata_to_add.items():

        # If the value is set for all samples
        if "_all_" in values:

            # Get the value.
            value = values["_all_"]

            # Set it for all samples.
            df.loc[:,col] = \
                value if not as_strings \
                else (pd.NA if value is None else str(value))

            # Go to the next column.
            continue

        #-------------------------------------------------------------#

        # Initialize the samples matched so far.
        indices = []

        # For each query and its value
        for query, value in values.items():

            # Get the value to set.
            value = \
                value if not as_strings \
                else (pd.NA if value is None else str(value))

            # If the value is for the samples not matched so far
            if query == "_other_":

                # Set it for those samples.
                df.loc[~df.index.isin(indices), col] = value

            # Otherwise
            else:

                # Get the samples matching the query.
                index = df.query(query).index

                # If there are none
                if index.empty:

                    # Warn the user.
                    warnstr = f"No samples match the query '{query}'."
                    logger.warning(warnstr)

                # Set the value for the samples.
                df.loc[index, col] = value

                # Add the samples to those matched so far.
                indices.extend(index.tolist())

    #-----------------------------------------------------------------#

    # Return the metadata.
    return df


def _map_values(df: pd.DataFrame,
                mapping: dict[str, dict[object, object]],
                option: str,
                where: str) -> pd.DataFrame:
    """Map the values of columns to new values.

    Parameters
    ----------
    df : :class:`pandas.DataFrame`
        The metadata.

    mapping : :class:`dict`
        The columns, each mapping old values to new values.

    option : :class:`str`
        The option holding the mapping.

    where : :class:`str`
        The project or study the metadata belong to.

    Returns
    -------
    df : :class:`pandas.DataFrame`
        The metadata with the values mapped.
    """

    # Check that the columns are in the metadata.
    _check_columns(df = df,
                   columns = list(mapping),
                   option = option,
                   where = where)

    # For each column and its mapping
    for col, values in mapping.items():

        # Replace the values.
        df[col] = df[col].replace(values)

    # Return the metadata.
    return df


def _keep_metadata(df: pd.DataFrame,
                   metadata_to_keep: dict[str, str]) -> pd.DataFrame:
    """Rename the columns to keep, dropping the columns already having
    their new names.

    Parameters
    ----------
    df : :class:`pandas.DataFrame`
        The metadata.

    metadata_to_keep : :class:`dict`
        The columns to keep mapped to their new names.

    Returns
    -------
    df : :class:`pandas.DataFrame`
        The metadata with the columns renamed.
    """

    # For each column to keep and its new name
    for old_col, new_col in metadata_to_keep.items():

        # If another column already has the new name
        if new_col in df.columns and old_col != new_col:

            # Drop it.
            df = df.drop(columns = new_col)

    # Return the metadata with the columns renamed.
    return df.rename(columns = metadata_to_keep)


def _split_by_separators(text: str,
                         seps: list[str]) -> dict[str, str]:
    """Split a text at separators.

    Parameters
    ----------
    text : :class:`str`
        The text.

    seps : :class:`list`
        The separators, in the order they appear in the text.

    Returns
    -------
    parts : :class:`dict`
        The names of the separators found mapped to the lowercased
        text following them.
    """

    # Initialize the start/end indexes of the parts.
    indexes = []

    # Initialize the separators found.
    seps_used = []

    #-----------------------------------------------------------------#

    # For each separator
    for sep in seps:

        # Find it in the text, not followed by a letter or digit if
        # it ends in one.
        match = re.search(re.escape(sep) \
                          + (r"(?![0-9A-Za-z])" if sep[-1:].isalnum()
                             else ""),
                          text)

        # If it is not there
        if match is None:

            # Go to the next separator.
            continue

        # Get where it starts.
        index = match.start()

        # If it is the first separator found
        if not seps_used:

            # Start a part after it.
            indexes.append([index + len(sep) + 1])

        # If it is the last separator
        elif sep == seps[-1]:

            # End the previous part before it.
            indexes[-1].append(index)

            # Add the part after it, to the end of the text.
            indexes.append([index + len(sep) + 1, len(text)])

        # Otherwise
        else:

            # If the previous part has no end yet
            if len(indexes[-1]) == 1:

                # End it before the separator.
                indexes[-1].append(index)

            # Start a part after it.
            indexes.append([index + len(sep) + 1])

        # Add the separator to those found.
        seps_used.append(sep)

    #-----------------------------------------------------------------#

    # Initialize the parts.
    parts = {}

    # For each separator found and its part's indexes
    for sep, index in zip(seps_used, indexes):

        # Get the part's start and end (the text's end if not set).
        start, end = index if len(index) == 2 else (index[0], len(text))

        # Add the lowercased part under the separator's name.
        parts[_util.get_separator_name(sep)] = text[start:end].lower()

    #-----------------------------------------------------------------#

    # Return the parts.
    return parts


def _concat_metadata(project_name: str,
                     studies: list[str],
                     wd: str) -> pd.DataFrame:
    """Get the Recount3 metadata of the studies of a project.

    Parameters
    ----------
    project_name : :class:`str`, {``"gtex"``, ``"tcga"``}
        The project.

    studies : :class:`list`
        The studies.

    wd : :class:`str`
        The directory where the metadata files are saved and, if
        already there, read from.

    Returns
    -------
    df : :class:`pandas.DataFrame`
        The studies' metadata, one study after the other.
    """

    # Initialize the list of the studies' metadata.
    dfs = []

    # For each study
    for study in studies:

        # Get its metadata.
        dfs.append(\
            util.get_metadata(project_name = project_name,
                              samples_category = study,
                              save_metadata = True,
                              wd = wd))

        # Inform the user that the metadata were retrieved.
        infostr = \
            f"The metadata of the {project_name.upper()} study " \
            f"'{study}' were successfully retrieved."
        logger.info(infostr)

    # Return the metadata of all studies.
    return pd.concat(dfs,
                     axis = 0)


#---------------------------------------------------------------------#


def _get_metadata_gtex(config: dict[str, object],
                       wd: str) -> pd.DataFrame:
    """Get the harmonized metadata of GTEx studies.

    Parameters
    ----------
    config : :class:`dict`
        The dataset's configuration.

    wd : :class:`str`
        The directory where the metadata files are saved and, if
        already there, read from.

    Returns
    -------
    df : :class:`pandas.DataFrame`
        The harmonized metadata.
    """

    # Get the options.
    query_string = config.get("query_string")
    metadata_to_keep = config.get("metadata_to_keep", {})
    metadata_to_add = config.get("metadata_to_add", {})
    mapping = config.get("mapping", {})

    #-----------------------------------------------------------------#

    # Get the studies' metadata.
    df = _concat_metadata(project_name = "gtex",
                          studies = config["studies"],
                          wd = wd)

    # Get each sample's study.
    studies = df["study"]

    #-----------------------------------------------------------------#

    # If a query was passed
    if query_string is not None:

        # Keep the samples matching it.
        df = df.query(query_string)

    # If there are columns to add
    if metadata_to_add:

        # Add them.
        df = _add_metadata(df = df,
                           metadata_to_add = metadata_to_add,
                           as_strings = False)

    # If there are values to map
    if mapping:

        # Map them.
        df = _map_values(df = df,
                         mapping = mapping,
                         option = "mapping",
                         where = "GTEx")

    #-----------------------------------------------------------------#

    # If there are columns to keep
    if metadata_to_keep:

        # Rename them.
        df = _keep_metadata(df = df,
                            metadata_to_keep = metadata_to_keep)

        # Get the columns to keep, the added ones included.
        cols_to_keep = \
            list(metadata_to_keep.values()) + list(metadata_to_add)

        # Check that they are in the metadata.
        _check_kept_columns(df = df,
                            columns = cols_to_keep,
                            renamed = metadata_to_keep,
                            where = "GTEx")

        # Drop the other columns.
        df = df[cols_to_keep]

    #-----------------------------------------------------------------#

    # If there are the subjects' GTEx IDs
    if "subject_identifier_GTEx_id" in df.columns:

        # Number the subjects from 1.
        df["subject_identifier_numeric_id"] = \
            df["subject_identifier_GTEx_id"].astype(\
                "category").cat.codes + 1

    # If there are the subjects' numbers
    if "subject_identifier_numeric_id" in df.columns:

        # Make them integers.
        df["subject_identifier_numeric_id"] = \
            df["subject_identifier_numeric_id"].astype("Int64")

    # If there are the subjects' sexes
    if "subject_demographics_sex" in df.columns:

        # Add the subjects' sex or gender.
        df["subject_demographics_sex_or_gender"] = \
            df["subject_demographics_sex"]

    #-----------------------------------------------------------------#

    # Add the project.
    df["study_recount3_project_name"] = "gtex"

    # Add the samples' studies.
    df["study_recount3_samples_category"] = studies

    #-----------------------------------------------------------------#

    # Return the metadata.
    return df


def _process_columns_tcga(df: pd.DataFrame,
                          metadata_to_process: dict[str, object]) -> \
        pd.DataFrame:
    """Split the columns of TCGA metadata at separators into new
    columns.

    Parameters
    ----------
    df : :class:`pandas.DataFrame`
        The metadata.

    metadata_to_process : :class:`dict`
        The columns to split, each with its separators and the new
        columns named after them.

    Returns
    -------
    df : :class:`pandas.DataFrame`
        The metadata with the new columns.
    """

    # Check that the columns are in the metadata.
    _check_columns(df = df,
                   columns = list(metadata_to_process),
                   option = "metadata_to_process",
                   where = "TCGA")

    # For each column to split and its options
    for col, options in metadata_to_process.items():

        # Initialize the parts of the samples' values, by separator.
        new_values = {}

        # For each sample and its value
        for index, val in zip(df.index, df[col]):

            # If the value is a non-blank string
            if isinstance(val, str) and val.strip():

                # For each separator found and its part of the value
                for name, part in \
                    _split_by_separators(\
                        text = val,
                        seps = options["separators"]).items():

                    # Add the part.
                    new_values.setdefault(name, {})[index] = part

        # For each separator's name and its new column
        for name, new_col in options["columns"].items():

            # Add the column.
            df[new_col] = pd.Series(new_values.get(name, {}))

    # Return the metadata.
    return df


def _merge_columns_tcga(df: pd.DataFrame,
                        metadata_to_merge: dict[str, object]) -> \
        pd.DataFrame:
    """Build new columns of TCGA metadata by joining or merging other
    columns.

    Parameters
    ----------
    df : :class:`pandas.DataFrame`
        The metadata.

    metadata_to_merge : :class:`dict`
        The new columns, each with the columns joined or merged into
        it.

    Returns
    -------
    df : :class:`pandas.DataFrame`
        The metadata with the new columns.
    """

    # For each new column and its options
    for new_col_name, options in metadata_to_merge.items():

        # If the columns are joined
        if "columns_to_join" in options:

            # Get the columns.
            items = options["columns_to_join"]

            # Check that they are in the metadata.
            _check_columns(\
                df = df,
                columns = [item["column"] for item in items],
                option = f"metadata_to_merge.{new_col_name}",
                where = "TCGA")

            # Initialize the columns' values.
            translated_cols = {}

            # For each column and its options
            for item in items:

                # Get the column.
                col = item["column"]

                # Get its values, placeholders as missing.
                col_series = \
                    df[col].where(\
                        ~df[col].astype(str).isin(\
                            defaults.MERGE_NA_VALUES))

                # If there is a mapping
                if item.get("mapping"):

                    # Match its old values case-insensitively.
                    lower_mapping = \
                        {str(k).lower() : v \
                         for k, v in item["mapping"].items()}

                    # Map the values.
                    col_series = \
                        col_series.map(\
                            lambda val, \
                                   mapping = lower_mapping: \
                                mapping.get(str(val).lower(), val) \
                                if pd.notna(val) else val)

                # If there is a prefix
                if item.get("prefix"):

                    # Prepend it to the values, with a colon.
                    col_series = \
                        col_series.map(\
                            lambda val, \
                                   prefix = item["prefix"]: \
                                f"{prefix}: {val}" \
                                if pd.notna(val) else val)

                # If there is a phrase prefix
                if item.get("phrase_prefix"):

                    # Prepend it to the values.
                    col_series = \
                        col_series.map(\
                            lambda val, \
                                   prefix = item["phrase_prefix"]: \
                                f"{prefix} {val}" \
                                if pd.notna(val) else val)

                # If there is a phrase suffix
                if item.get("phrase_suffix"):

                    # Append it to the values.
                    col_series = \
                        col_series.map(\
                            lambda val, \
                                   suffix = item["phrase_suffix"]: \
                                f"{val} {suffix}" \
                                if pd.notna(val) else val)

                # Store the values.
                translated_cols[col] = col_series

            # Get the separator of the joined values.
            separator = \
                options.get("separator", defaults.JOIN_SEPARATOR)

            # Join each sample's values (missing if there are none).
            df[new_col_name] = \
                pd.DataFrame(translated_cols).apply(\
                    lambda row, \
                           separator = separator: \
                        separator.join(\
                            [str(v) for v in row if pd.notna(v)]) \
                        if [str(v) for v in row if pd.notna(v)] \
                        else np.nan,
                    axis = 1)

            # Go to the next column.
            continue

        #-------------------------------------------------------------#

        # Get the columns to merge.
        items = options["columns_to_merge"]

        # Check that they are in the metadata.
        _check_columns(\
            df = df,
            columns = [item["column"] if isinstance(item, dict) \
                       else item for item in items],
            option = f"metadata_to_merge.{new_col_name}",
            where = "TCGA")

        # Initialize the columns' values.
        cols_values = []

        # For each column
        for item in items:

            # If a pattern is matched in the column
            if isinstance(item, dict):

                # Get the first group the pattern matches.
                col_values = \
                    df[item["column"]].astype(str).str.extract(\
                        item["pattern"],
                        expand = False)

            # Otherwise
            else:

                # Get the column's values.
                col_values = df[item]

            # Add the values, placeholders as missing.
            cols_values.append(\
                col_values.where(\
                    ~col_values.astype(str).isin(\
                        defaults.MERGE_NA_VALUES)))

        # Start from the first column's values.
        new_col = cols_values[0]

        # For each other column's values
        for col_values in cols_values[1:]:

            # Fill the missing values with them.
            new_col = new_col.fillna(col_values)

        # If there is a mapping
        if options.get("mapping"):

            # Map the values.
            new_col = new_col.replace(options["mapping"])

        # If the values are numbered
        if options.get("serial", False):

            # Number the sorted unique values from 1.
            new_col = \
                new_col.map(\
                    {v : k for k, v \
                     in enumerate(sorted(new_col.dropna().unique()),
                                  1)})

        # Add the new column.
        df[new_col_name] = new_col

    #-----------------------------------------------------------------#

    # Return the metadata.
    return df


def _get_metadata_tcga(config: dict[str, object],
                       wd: str) -> pd.DataFrame:
    """Get the harmonized metadata of TCGA studies.

    Parameters
    ----------
    config : :class:`dict`
        The dataset's configuration.

    wd : :class:`str`
        The directory where the metadata files are saved and, if
        already there, read from.

    Returns
    -------
    df : :class:`pandas.DataFrame`
        The harmonized metadata.
    """

    # Get the query and the columns to split and build.
    query_string = config.get("query_string")
    metadata_to_process = config.get("metadata_to_process", {})
    metadata_to_merge = config.get("metadata_to_merge", {})

    # Get the columns to keep and add.
    metadata_to_keep = config.get("metadata_to_keep", {})
    metadata_to_add = config.get("metadata_to_add", {})

    # Get the columns to lowercase and the values to map.
    mapping_to_lower = config.get("mapping_to_lower", [])
    mapping = config.get("mapping", {})

    #-----------------------------------------------------------------#

    # Get the studies' metadata.
    df = _concat_metadata(project_name = "tcga",
                          studies = config["studies"],
                          wd = wd)

    # Get each sample's study.
    studies = df["study"]

    #-----------------------------------------------------------------#

    # If a query was passed
    if query_string is not None:

        # Keep the samples matching it.
        df = df.query(query_string)

    # If there are columns to split
    if metadata_to_process:

        # Split them.
        df = _process_columns_tcga(\
                df = df,
                metadata_to_process = metadata_to_process)

    # If there are columns to build from other columns
    if metadata_to_merge:

        # Build them.
        df = _merge_columns_tcga(\
                df = df,
                metadata_to_merge = metadata_to_merge)

    # If there are columns to lowercase
    if mapping_to_lower:

        # Check that they are in the metadata.
        _check_columns(df = df,
                       columns = mapping_to_lower,
                       option = "mapping_to_lower",
                       where = "TCGA")

        # For each column
        for col in mapping_to_lower:

            # Lowercase its values.
            df[col] = df[col].str.lower()

    # If there are columns to add
    if metadata_to_add:

        # Add them.
        df = _add_metadata(df = df,
                           metadata_to_add = metadata_to_add,
                           as_strings = False)

    # If there are values to map
    if mapping:

        # Map them.
        df = _map_values(df = df,
                         mapping = mapping,
                         option = "mapping",
                         where = "TCGA")

    #-----------------------------------------------------------------#

    # If there are columns to keep
    if metadata_to_keep:

        # Rename them.
        df = _keep_metadata(df = df,
                            metadata_to_keep = metadata_to_keep)

        # Get the columns to keep, the added and built ones included.
        cols_to_keep = \
            list(metadata_to_keep.values()) + \
            list(metadata_to_add) + \
            list(metadata_to_merge)

        # For each split column and its options
        for options in metadata_to_process.values():

            # If its new columns are kept
            if options.get("keep_columns", True):

                # Add them to the columns to keep.
                cols_to_keep += list(options["columns"].values())

        # Check that they are in the metadata.
        _check_kept_columns(df = df,
                            columns = cols_to_keep,
                            renamed = metadata_to_keep,
                            where = "TCGA")

        # Drop the other columns.
        df = df[cols_to_keep]

    #-----------------------------------------------------------------#

    # If there are the subjects' ages
    if "subject_demographics_age_years" in df.columns:

        # Copy them, as strings, to a new column.
        df["subject_age_years"] = \
            df["subject_demographics_age_years"].astype(str).replace(\
                "nan", np.nan)

        # Make the ages numbers, placeholders as missing.
        df["subject_demographics_age_years"] = \
            df["subject_demographics_age_years"].replace(\
                "None", np.nan).replace(\
                "NA", np.nan).replace(\
                "", np.nan).astype("Float64")

    # Set the name of the vital status column.
    vital_status_col = \
        "subject_status_main_condition_clinical_progression_" \
        "vital_status"

    # If it is there
    if vital_status_col in df.columns:

        # Set the placeholders as missing.
        df[vital_status_col] = \
            df[vital_status_col].replace(\
                {"NA" : np.nan, "None" : np.nan, "" : np.nan})

    # For each survival/follow-up time column, in days
    for time_col in \
        ("subject_status_main_condition_clinical_progression_"
         "time_to_death_days",
         "subject_status_main_condition_clinical_progression_"
         "time_to_last_follow_up_days"):

        # If it is there
        if time_col in df.columns:

            # Make the times numbers, placeholders as missing.
            df[time_col] = \
                df[time_col].replace(\
                    {"NA" : np.nan,
                     "None" : np.nan,
                     "" : np.nan}).astype("Float64")

    # If there are the subjects' numbers
    if "subject_identifier_numeric_id" in df.columns:

        # Make them integers.
        df["subject_identifier_numeric_id"] = \
            df["subject_identifier_numeric_id"].astype("Int64")

    # If there are the adjacent normal flags
    if "sample_collection_is_adjacent_normal" in df.columns:

        # Set the missing flags to 'no'.
        df["sample_collection_is_adjacent_normal"] = \
            df["sample_collection_is_adjacent_normal"].fillna("no")

    #-----------------------------------------------------------------#

    # Add the project.
    df["study_recount3_project_name"] = "tcga"

    # Add the samples' studies.
    df["study_recount3_samples_category"] = studies

    # If there are the subjects' genders
    if "subject_demographics_gender" in df.columns:

        # Add the subjects' sex or gender.
        df["subject_demographics_sex_or_gender"] = \
            df["subject_demographics_gender"]

    #-----------------------------------------------------------------#

    # Return the metadata.
    return df


#---------------------------------------------------------------------#


def _get_geo_table(geo_code: str,
                   wd: str) -> pd.DataFrame:
    """Get a GEO series' samples, indexed by their SRA experiments.

    Parameters
    ----------
    geo_code : :class:`str`
        The GEO series' code.

    wd : :class:`str`
        The directory where the SOFT file is saved and, if already
        there, read from.

    Returns
    -------
    df_geo : :class:`pandas.DataFrame`
        The samples, indexed by SRA experiment.
    """

    # Get the samples.
    df_geo = util.get_geo_metadata(geo_code = geo_code,
                                   wd = wd)

    # Get each sample's SRA experiment.
    df_geo["external_id"] = \
        df_geo["Sample_relation"].str.extract(\
            r"SRA:.*?(SRX\d+)",
            expand = False)

    # Index the samples by their SRA experiments.
    df_geo = df_geo.set_index("external_id",
                              drop = False)

    # Keep the SRA experiments as a separate column.
    df_geo["_external_id"] = df_geo.index

    # Return the samples.
    return df_geo


def _merge_geo(df: pd.DataFrame,
               df_geo: pd.DataFrame,
               columns: list[str],
               from_run_selector: bool) -> pd.DataFrame:
    """Add columns of a GEO series' samples to an SRA study's
    metadata.

    Parameters
    ----------
    df : :class:`pandas.DataFrame`
        The study's metadata.

    df_geo : :class:`pandas.DataFrame`
        The GEO series' samples.

    columns : :class:`list`
        The columns to add.

    from_run_selector : :class:`bool`
        Whether the study's metadata come from its Run Selector
        table.

    Returns
    -------
    df : :class:`pandas.DataFrame`
        The metadata with the columns added.
    """

    # Set the columns to take, with the samples' titles.
    cols_to_take = ["Sample_title"] + columns

    # If the metadata have the samples' titles
    if "sample_title" in df.columns:

        # Add the columns by matching the samples' titles.
        df = df.merge(df_geo[cols_to_take],
                      how = "left",
                      left_on = "sample_title",
                      right_on = "Sample_title")

        # Index the samples by their IDs.
        df = df.set_index("external_id",
                          drop = False)

    # If the metadata come from the Run Selector table
    elif from_run_selector:

        # Add the columns by matching the samples' experiments,
        # keeping the matched samples only.
        df = df.merge(df_geo[cols_to_take + ["_external_id"]],
                      how = "inner",
                      left_on = "Experiment",
                      right_on = "_external_id")

        # Index the samples by their runs.
        df = df.set_index("Run",
                          drop = False)

    # Return the metadata.
    return df


def _process_metadata_sra(options: dict[str, object],
                          sra_code: str,
                          wd: str) -> pd.DataFrame:
    """Get the harmonized metadata of an SRA study.

    Parameters
    ----------
    options : :class:`dict`
        The study's options.

    sra_code : :class:`str`
        The study's code.

    wd : :class:`str`
        The directory where the metadata files are saved and, if
        already there, read from.

    Returns
    -------
    df : :class:`pandas.DataFrame`
        The study's harmonized metadata.
    """

    # Get the study-level metadata and the queries.
    metadata_study = options.get("metadata_study", {})
    query_string = options.get("query_string")
    query_string_geo = options.get("query_string_geo")

    # Get the columns to keep and add.
    metadata_to_keep = options.get("metadata_to_keep", {})
    metadata_to_keep_geo = options.get("metadata_to_keep_geo", {})
    metadata_to_add = options.get("metadata_to_add", {})
    metadata_to_add_geo = options.get("metadata_to_add_geo", {})

    # Get the runs to drop and the values from other studies.
    samples_to_drop = options.get("samples_to_drop", [])
    metadata_from_study = options.get("metadata_from_study", {})

    # Get the values to map.
    mapping = options.get("mapping", {})
    mapping_with_prefix = options.get("mapping_with_prefix", {})
    mapping_with_suffix = options.get("mapping_with_suffix", {})
    mapping_geo = options.get("mapping_geo", {})
    serial_mapping = options.get("serial_mapping", {})
    mapping_values = options.get("mapping_values", {})

    # Get the column indexing the samples meanwhile, if any.
    use_as_external_id = options.get("_use_as_external_id")

    # Get the study's GEO code.
    geo_code = metadata_study.get("study_geo_code")

    #-----------------------------------------------------------------#

    # Initialize whether the metadata come from the Run Selector table.
    from_run_selector = False

    #-----------------------------------------------------------------#

    # Try to get the study's Recount3 metadata.
    try:

        # Get the metadata, with all values as strings.
        df = util.get_metadata(project_name = "sra",
                               samples_category = sra_code,
                               save_metadata = True,
                               wd = wd).astype(str)

        # Set the missing values back.
        df = df.replace({"nan" : np.nan})

        # Keep the library columns, under their new names.
        metadata_to_keep = \
            {**metadata_to_keep, **defaults.SRA_LIBRARY_COLUMNS}

        # Keep the samples' run IDs as a column.
        df["_run_id"] = df.index

        # If the samples are indexed by another column meanwhile
        if use_as_external_id is not None:

            # Index them by the column, keeping it.
            df = df.set_index(use_as_external_id,
                              drop = False)

    # If the study has no Recount3 metadata
    except pd.errors.EmptyDataError:

        # Warn the user.
        warnstr = \
            "No Recount3 metadata found for the study with SRA " \
            f"code '{sra_code}'."
        logger.warning(warnstr)

        # Mark the metadata as coming from the Run Selector table.
        from_run_selector = True

        # Get the Run Selector table.
        df = util.get_run_selector(sra_code = sra_code,
                                   wd = wd)

        # Add the project.
        df.insert(loc = 0,
                  column = "recount3_project_name",
                  value = "sra")

        # Add the study.
        df.insert(loc = 1,
                  column = "recount3_samples_category",
                  value = sra_code)

        # Index the samples by their runs.
        df = df.set_index("Run",
                          drop = False)

        # Keep the samples' run IDs as a column.
        df["_run_id"] = df.index

        # Inform the user that the table will be used.
        infostr = \
            "The Run Selector table for the study with SRA code " \
            f"'{sra_code}' was successfully retrieved and will be " \
            "used as metadata."
        logger.info(infostr)

    #-----------------------------------------------------------------#

    # Get each run's experiment and BioSample from ENA.
    df_ena = util.get_ena_metadata(sra_code = sra_code,
                                   wd = wd)

    # If ENA has the study
    if not df_ena.empty:

        # Add each run's experiment.
        df["sample_identifier_ena_experiment"] = \
            df["_run_id"].map(df_ena["experiment_accession"])

        # Add each run's BioSample.
        df["sample_identifier_ena_biosample"] = \
            df["_run_id"].map(df_ena["sample_accession"])

        # Get each run's experiment, indexed by run.
        numbering = \
            df.set_index("_run_id")[\
                ["sample_identifier_ena_experiment"]]

        # Sort by experiment, then by run.
        numbering = \
            numbering.sort_values(\
                ["sample_identifier_ena_experiment", "_run_id"])

        # Number the runs within each experiment from 1.
        df["sample_identifier_run_numeric_id"] = \
            df["_run_id"].map(\
                numbering.groupby(\
                    "sample_identifier_ena_experiment").cumcount() + 1)

        # For each ENA field filling a library column
        for ena_col, target_col in \
            defaults.SRA_ENA_LIBRARY_COLUMNS.items():

            # Get each run's value.
            values = df["_run_id"].map(df_ena[ena_col])

            # If the column is there
            if target_col in df.columns:

                # Fill its missing values.
                df[target_col] = df[target_col].fillna(values)

            # Otherwise
            else:

                # Add the column.
                df[target_col] = values

    #-----------------------------------------------------------------#

    # If there is a GEO code and options using the GEO metadata
    if geo_code is not None \
    and (query_string_geo \
         or metadata_to_keep_geo \
         or metadata_to_add_geo):

        # Get the GEO series' samples.
        df_geo = _get_geo_table(geo_code = geo_code,
                                wd = wd)

        # If a GEO query was passed
        if query_string_geo:

            # Keep the GEO samples matching it.
            df_geo = df_geo.query(query_string_geo)

        # If the metadata come from the Run Selector table
        if from_run_selector:

            # Keep the samples whose experiments are in the GEO series.
            df = df[df["Experiment"].isin(df_geo["external_id"])]

        # Keep the samples' IDs as a column.
        df["external_id"] = df.index

    #-----------------------------------------------------------------#

    # If a query was passed
    if query_string is not None:

        # Keep the samples matching it.
        df = df.query(query_string)

    # If there are runs to drop
    if samples_to_drop:

        # Drop them.
        df = df.drop(index = samples_to_drop)

        # Inform the user that the runs were dropped.
        infostr = \
            f"Dropped {len(samples_to_drop)} run(s) of the study " \
            f"'{sra_code}'."
        logger.info(infostr)

    #-----------------------------------------------------------------#

    # If there are GEO columns to add
    if geo_code is not None and metadata_to_add_geo:

        # Add them to the GEO samples.
        df_geo_add = \
            _add_metadata(df = df_geo.copy(),
                          metadata_to_add = metadata_to_add_geo,
                          as_strings = True)

        # Add them to the metadata.
        df = _merge_geo(df = df,
                        df_geo = df_geo_add,
                        columns = list(metadata_to_add_geo),
                        from_run_selector = from_run_selector)

    # If there are GEO columns to keep
    if geo_code is not None and metadata_to_keep_geo:

        # Copy the GEO samples.
        df_geo_keep = df_geo.copy()

        # If there are GEO values to map
        if mapping_geo:

            # Map them.
            df_geo_keep = \
                _map_values(df = df_geo_keep,
                            mapping = mapping_geo,
                            option = "mapping_geo",
                            where = sra_code)

        # Rename the columns to keep.
        df_geo_keep = \
            _keep_metadata(df = df_geo_keep,
                           metadata_to_keep = metadata_to_keep_geo)

        # Add them to the metadata.
        df = _merge_geo(df = df,
                        df_geo = df_geo_keep,
                        columns = list(metadata_to_keep_geo.values()),
                        from_run_selector = from_run_selector)

    #-----------------------------------------------------------------#

    # If there are columns to add
    if metadata_to_add:

        # Add them.
        df = _add_metadata(df = df,
                           metadata_to_add = metadata_to_add,
                           as_strings = True)

    # If there are values to map
    if mapping:

        # Map them.
        df = _map_values(df = df,
                         mapping = mapping,
                         option = "mapping",
                         where = sra_code)

    # For each column whose values get a prefix
    for col, prefix in mapping_with_prefix.items():

        # Add the prefix to the values.
        df[col] = \
            df[col].replace(\
                {val : f"{prefix}{str(val)}" \
                 for val in df[col].unique() if not pd.isna(val)})

    # For each column whose values get a suffix
    for col, suffix in mapping_with_suffix.items():

        # Add the suffix to the values.
        df[col] = \
            df[col].replace(\
                {val : f"{str(val)}{suffix}" \
                 for val in df[col].unique() if not pd.isna(val)})

    # For each column whose values are numbered
    for col, values in serial_mapping.items():

        # Number the values from 1, in order.
        df[col] = \
            df[col].replace(\
                {v : k for k, v in enumerate(values, 1)})

    #-----------------------------------------------------------------#

    # If there are columns to keep
    if metadata_to_keep:

        # For each column to keep and its new name
        for old_col, new_col in metadata_to_keep.items():

            # If another column already has the new name and is not
            # one of the added columns
            if new_col in df.columns and old_col != new_col \
            and new_col not in metadata_to_add:

                # Drop it.
                df = df.drop(columns = new_col)

        # Rename the columns to keep, except those added.
        df = df.rename(\
            columns = {old : new \
                       for old, new in metadata_to_keep.items() \
                       if new not in metadata_to_add})

    # For each study-level metadata field and its value
    for col, val in metadata_study.items():

        # Add the field.
        df[col] = val

    #-----------------------------------------------------------------#

    # For each library column, by its old and new name
    for raw_col, new_col in defaults.SRA_LIBRARY_COLUMNS.items():

        # Get the column's current name.
        col = new_col if new_col in df.columns else raw_col

        # If it is there and its values are renamed
        if col in df.columns and raw_col in defaults.SRA_LIBRARY_VALUES:

            # Rename the values.
            df[col] = \
                df[col].replace(defaults.SRA_LIBRARY_VALUES[raw_col])

    # For each column whose substrings are replaced
    for col, values in mapping_values.items():

        # For each old and new substring
        for old_value, new_value in values.items():

            # Replace the substring.
            df[col] = df[col].str.replace(old_value,
                                          new_value)

    #-----------------------------------------------------------------#

    # Index the samples by their run IDs.
    df = df.set_index("_run_id").rename_axis("external_id")

    # Get the columns to keep: the first ones, if present, then the
    # study-level, kept and added ones, without repetitions.
    cols_to_keep = \
        list(dict.fromkeys(\
            [col for col in defaults.SRA_FIRST_COLUMNS \
             if col in df.columns] + \
            list(metadata_study) + \
            list(metadata_to_keep.values()) + \
            list(metadata_to_add) + \
            list(metadata_to_keep_geo.values()) + \
            list(metadata_to_add_geo)))

    # Check that they are in the metadata.
    _check_kept_columns(df = df,
                        columns = cols_to_keep,
                        renamed = {**metadata_to_keep,
                                   **metadata_to_keep_geo},
                        where = sra_code)

    # Drop the other columns.
    df = df[cols_to_keep]

    # Name the project and study columns as study-level fields.
    df = df.rename(\
        columns = \
            {"recount3_project_name" : "study_recount3_project_name",
             "recount3_samples_category" : \
                "study_recount3_samples_category"})

    # Convert the columns to the best types.
    df = df.convert_dtypes()

    #-----------------------------------------------------------------#

    # If there are values recovered from other studies
    if metadata_from_study:

        # Initialize the counts of filled and replaced values.
        n_filled = n_replaced = 0

        # For each study the values come from and its values
        for donor, block in metadata_from_study.items():

            # For each column and its values
            for field, by_value in (block.get("values") or {}).items():

                # If the column is not there
                if field not in df.columns:

                    # Add it, empty.
                    df[field] = pd.NA

                # For each value and its runs
                for value, runs in by_value.items():

                    # Get the runs not in the metadata.
                    missing = [r for r in runs if r not in df.index]

                    # If there are any
                    if missing:

                        # Raise an error.
                        errstr = \
                            f"{sra_code}: the values recovered from " \
                            f"'{donor}' for '{field}' name " \
                            f"{len(missing)} run(s) not in the " \
                            "study's metadata: " \
                            f"{', '.join(sorted(missing)[:10])}" \
                            f"{'...' if len(missing) > 10 else ''}."
                        raise ValueError(errstr)

                    # Count the runs already having a value.
                    already = int(df.loc[runs, field].notna().sum())

                    # Update the counts.
                    n_replaced += already
                    n_filled += len(runs) - already

                    # Set the value.
                    df.loc[runs, field] = str(value)

        # Inform the user about the values set.
        infostr = \
            f"{sra_code}: filled {n_filled} value(s) and replaced " \
            f"{n_replaced} with the values recovered from " \
            f"{', '.join(sorted(metadata_from_study))}."
        logger.info(infostr)

    #-----------------------------------------------------------------#

    # Return the metadata.
    return df


def _get_metadata_sra(config: dict[str, object],
                      wd: str) -> pd.DataFrame:
    """Get the harmonized metadata of SRA studies.

    Parameters
    ----------
    config : :class:`dict`
        The dataset's configuration.

    wd : :class:`str`
        The directory where the metadata files are saved and, if
        already there, read from.

    Returns
    -------
    df : :class:`pandas.DataFrame`
        The harmonized metadata.
    """

    # Initialize the metadata.
    df = pd.DataFrame()

    # For each study and its options
    for sra_code, options in config.items():

        # Inform the user that the study is being processed.
        infostr = \
            "Processing the metadata of the study with SRA code " \
            f"'{sra_code}'..."
        logger.info(infostr)

        # Get the study's metadata.
        df_study = \
            _process_metadata_sra(options = copy.deepcopy(options),
                                  sra_code = sra_code,
                                  wd = wd)

        # If any column is named NaN
        if any(c != c for c in df_study.columns):

            # Warn the user.
            warnstr = \
                "The metadata of the study with SRA code " \
                f"'{sra_code}' have column(s) named NaN."
            logger.warning(warnstr)

        # Add the study's metadata, one study after the other.
        df = pd.concat([df, df_study],
                       axis = 0)

        # Inform the user that the study was processed.
        infostr = \
            f"The metadata of the study with SRA code '{sra_code}' " \
            "were successfully processed."
        logger.info(infostr)

    #-----------------------------------------------------------------#

    # Set the column of the studies.
    study_col = "study_recount3_samples_category"

    # For each column whose missing values are reported
    for col in ["subject_status_main_condition",
                "sample_organ_or_tissue"]:

        # If it is not there
        if col not in df.columns:

            # Go to the next column.
            continue

        # Get the studies with missing values.
        studies_nan = \
            df.loc[df[col].isna(), study_col].unique().tolist()

        # If there are any
        if studies_nan:

            # Warn the user.
            warnstr = \
                "The following studies have NaN values in the " \
                f"'{col}' column: " \
                f"{', '.join(repr(s) for s in studies_nan)}."
            logger.warning(warnstr)

    #-----------------------------------------------------------------#

    # Get the columns of the subjects' sex and gender, if present.
    sex_cols = \
        [col for col in ["subject_demographics_sex",
                         "subject_demographics_gender"] \
         if col in df.columns]

    # If any is there
    if sex_cols:

        # Add the subjects' sex, or gender where the sex is missing.
        df["subject_demographics_sex_or_gender"] = \
            df[sex_cols[0]] if len(sex_cols) == 1 \
            else df[sex_cols[0]].fillna(df[sex_cols[1]])

        # Get the samples with neither.
        samples_nan = \
            df.index[\
                df["subject_demographics_sex_or_gender"].isna()\
                ].tolist()

        # If there are any
        if samples_nan:

            # Warn the user.
            warnstr = \
                "The following samples have NaN values in the " \
                "'subject_demographics_sex_or_gender' column: " \
                f"{', '.join(map(str, samples_nan))}."
            logger.warning(warnstr)

    #-----------------------------------------------------------------#

    # If there are the subjects' ages
    if "subject_demographics_age_years" in df.columns:

        # Set the age bins' edges.
        bins = list(range(0,
                          defaults.AGE_BIN_MAX + 1,
                          defaults.AGE_BIN_WIDTH))

        # Add the subjects' age bins, each closed on the left and
        # labeled by its first and last year.
        df["subject_demographics_age_bin_10_years"] = \
            pd.cut(pd.to_numeric(df["subject_demographics_age_years"],
                                 errors = "coerce"),
                   bins = bins,
                   right = False,
                   labels = \
                    [f"{low}-{low + defaults.AGE_BIN_WIDTH - 1}" \
                     for low in bins[:-1]]).astype(object)

    #-----------------------------------------------------------------#

    # For each flag column
    for flag_col in ("sample_collection_is_adjacent_normal",
                     "sample_collection_is_metastatic"):

        # If it is there
        if flag_col in df.columns:

            # Set the missing flags to 'no'.
            df[flag_col] = df[flag_col].fillna("no")

    #-----------------------------------------------------------------#

    # If there are the subjects' numbers
    if "subject_identifier_numeric_id" in df.columns:

        # Make them numbers, the non-numeric ones missing.
        df["subject_identifier_numeric_id"] = \
            pd.to_numeric(df["subject_identifier_numeric_id"],
                          errors = "coerce")

        # Get the numbered samples.
        df_numbered = \
            df[df["subject_identifier_numeric_id"].notna()]

        # Get the studies whose subjects are not numbered from 1 to
        # their number.
        not_continuous = \
            [study for study, ids in \
             df_numbered.groupby(study_col)[\
                "subject_identifier_numeric_id"] \
             if ids.nunique() != ids.max()]

        # If there are any
        if not_continuous:

            # Warn the user.
            warnstr = \
                "The following studies do not number their " \
                "subjects in 'subject_identifier_numeric_id' from " \
                "1 to the number of subjects: " \
                f"{', '.join(repr(s) for s in not_continuous)}."
            logger.warning(warnstr)

    #-----------------------------------------------------------------#

    # If there are the studies' comments
    if "study_comment" in df.columns:

        # Get whether each comment names its library selection's source.
        says_its_own = \
            df["study_comment"].str.contains(\
                '"sample_sequencing_library_selection"',
                regex = False,
                na = False)

        # Get whether ENA has each study.
        has_ena = \
            df.groupby(study_col)[\
                "sample_identifier_ena_experiment"].transform(\
                    lambda s: s.notna().any()) \
            if "sample_identifier_ena_experiment" in df.columns \
            else pd.Series(False,
                           index = df.index)

        # Append to the comments where the library and derived fields
        # come from.
        df["study_comment"] = \
            df["study_comment"] + \
            says_its_own.map(\
                {True : defaults.SRA_COMMENT_HEAD_OWN_SELECTION,
                 False : defaults.SRA_COMMENT_HEAD}) + \
            np.where(has_ena,
                     defaults.SRA_COMMENT_TAIL_ENA,
                     defaults.SRA_COMMENT_TAIL)

    #-----------------------------------------------------------------#

    # For each column
    for col in df.columns:

        # If it does not hold objects or strings
        if not pd.api.types.is_string_dtype(df[col].dtype):

            # Go to the next column.
            continue

        # Replace the newlines with spaces.
        df[col] = df[col].replace(to_replace = r"\s*\n\s*",
                                  value = " ",
                                  regex = True)

    #-----------------------------------------------------------------#

    # Return the metadata.
    return df


def _get_counts(df_metadata: pd.DataFrame,
                project_name: str,
                studies: list[str],
                gencode_release: int,
                wd_gene_sums: Optional[str],
                wd_qc: Optional[str]) -> pd.DataFrame:
    """Get the read counts of the samples of some studies in a
    dataset's metadata.

    Parameters
    ----------
    df_metadata : :class:`pandas.DataFrame`
        The dataset's metadata.

    project_name : :class:`str`, {``"GTEx"``, ``"TCGA"``, ``"SRA"``}
        The Recount3 project the studies belong to.

    studies : :class:`list`
        The studies.

    gencode_release : :class:`int`, {``29``, ``26``}
        The GENCODE release the counts were annotated with.

    wd_gene_sums : :class:`str`, optional
        The directory where the RNA-seq data files are saved and, if
        already there, read from.

    wd_qc : :class:`str`, optional
        The directory where the QC metadata files are saved and, if
        already there, read from.

    Returns
    -------
    df_counts : :class:`pandas.DataFrame`
        The read counts, with the samples in the metadata's order,
        study by study.
    """

    # Inform the user that the counts are being retrieved.
    infostr = \
        "Getting the counts (GENCODE release " \
        f"{gencode_release})..."
    logger.info(infostr)

    # Initialize the studies' counts.
    dfs = []

    #-----------------------------------------------------------------#

    # For each study
    for study in studies:

        # Get the study's RNA-seq data.
        df_gene_sums = \
            util.get_gene_sums(project_name = project_name,
                               samples_category = study,
                               save_gene_sums = True,
                               wd = wd_gene_sums,
                               gencode_release = gencode_release)

        # If there are missing values
        if df_gene_sums.isna().any().any():

            # Get the samples having them.
            nan_samples = \
                df_gene_sums.index[\
                    df_gene_sums.isna().any(axis = 1)].tolist()

            # Warn the user.
            warnstr = \
                f"In study '{study}', the counts of the following " \
                "samples have NaN values: " \
                f"{', '.join(map(str, nan_samples))}."
            logger.warning(warnstr)

        # Get the study's QC metadata.
        df_qc = util.get_qc(project_name = project_name,
                            samples_category = study,
                            save_qc = True,
                            wd = wd_qc)

        # Convert the RNA-seq data to read counts.
        df_read_counts = \
            util.get_read_counts(\
                df_raw_counts = df_gene_sums,
                avg_mapped_read_length = \
                    df_qc["star.average_mapped_length"],
                do_round = True)

        # Make the samples' IDs stripped strings.
        df_read_counts.index = \
            pd.Index(df_read_counts.index).astype(str).str.strip()

        #-------------------------------------------------------------#

        # Get the study's samples in the metadata, as stripped
        # strings.
        metadata_ids = \
            pd.Index(\
                df_metadata.loc[\
                    df_metadata["study_recount3_samples_category"] \
                    == study].index).astype(str).str.strip()

        # Keep the samples having counts, in the metadata's order.
        df_read_counts = \
            df_read_counts.reindex(\
                metadata_ids[metadata_ids.isin(df_read_counts.index)])

        # Get the samples without counts.
        missing_ids = \
            metadata_ids[~metadata_ids.isin(df_read_counts.index)]

        # If there are any
        if len(missing_ids) > 0:

            # Warn the user.
            warnstr = \
                f"In study '{study}', {len(missing_ids)} sample(s) " \
                "of the metadata are not in the counts: " \
                f"{', '.join(missing_ids)}."
            logger.warning(warnstr)

        # Inform the user about the samples matched.
        infostr = \
            f"Study '{study}': {len(metadata_ids)} sample(s) in the " \
            f"metadata, {len(df_read_counts)} with counts."
        logger.info(infostr)

        # Add the study's counts.
        dfs.append(df_read_counts)

    #-----------------------------------------------------------------#

    # Return the counts of all the studies.
    return pd.concat(dfs,
                     axis = 0)


########################## PUBLIC FUNCTIONS ###########################


def get_dataset_metadata(
        config: dict[str, object],
        project_name: str,
        wd_metadata: Optional[str] = None,
        samples_categories: Optional[list[str]] = None) -> \
            pd.DataFrame:
    """Get the harmonized metadata of a dataset built from Recount3
    studies.

    Parameters
    ----------
    config : :class:`dict`
        The dataset's configuration (see
        :func:`bulkdgd.ioutil.load_config_recount3`).

    project_name : :class:`str`, {``"GTEx"``, ``"TCGA"``, ``"SRA"``}
        The Recount3 project the studies belong to.

    wd_metadata : :class:`str`, optional
        The directory where the metadata files are saved and, if
        already there, read from. If not passed, it is the current
        working directory.

    samples_categories : :class:`list`, optional
        The studies to take from the configuration (tissues for GTEx,
        cancer types for TCGA, study codes for SRA). If not passed,
        all the studies are taken.

    Returns
    -------
    df_metadata : :class:`pandas.DataFrame`
        The harmonized metadata, one row per sample.
    """

    # If the project is not supported
    if project_name not in defaults.RECOUNT3_PROJECTS:

        # Get the supported projects as a string.
        projects_str = \
            ", ".join(repr(p) for p in defaults.RECOUNT3_PROJECTS)

        # Raise an error.
        errstr = \
            f"Unknown project '{project_name}'. The supported " \
            f"projects are: {projects_str}."
        raise ValueError(errstr)

    # Create a deep copy of the configuration.
    config = copy.deepcopy(config)

    #-----------------------------------------------------------------#

    # Get the studies in the configuration.
    studies = \
        list(config) if project_name == "SRA" else config["studies"]

    # If only some studies are taken
    if samples_categories is not None:

        # Get those not in the configuration.
        missing = \
            [s for s in samples_categories if s not in studies]

        # If there are any
        if missing:

            # Raise an error.
            errstr = \
                "These studies are not in the configuration: " \
                f"{', '.join(repr(s) for s in missing)}."
            raise ValueError(errstr)

        # If the configuration maps the studies to their options
        if project_name == "SRA":

            # Keep the studies taken.
            config = \
                {s : config[s] for s in studies \
                 if s in samples_categories}

        # Otherwise
        else:

            # Keep the studies taken.
            config["studies"] = \
                [s for s in studies if s in samples_categories]

    #-----------------------------------------------------------------#

    # If no directory was passed
    if wd_metadata is None:

        # Use the current working directory.
        wd_metadata = os.getcwd()

    # Create the directory if it does not exist.
    os.makedirs(wd_metadata,
                exist_ok = True)

    #-----------------------------------------------------------------#

    # Set the function getting the project's metadata.
    get_metadata_func = \
        {"GTEx" : _get_metadata_gtex,
         "TCGA" : _get_metadata_tcga,
         "SRA" : _get_metadata_sra}[project_name]

    # Return the metadata.
    return get_metadata_func(config = config,
                             wd = wd_metadata)


def get_dataset_counts(
        df_metadata: pd.DataFrame,
        project_name: str,
        gencode_release: int = 29,
        wd_gene_sums: Optional[str] = None,
        wd_qc: Optional[str] = None,
        df_counts_existing: Optional[pd.DataFrame] = None) -> \
            pd.DataFrame:
    """Get the read counts of the samples in a dataset's metadata.

    Parameters
    ----------
    df_metadata : :class:`pandas.DataFrame`
        The dataset's metadata (see :func:`get_dataset_metadata`).

    project_name : :class:`str`, {``"GTEx"``, ``"TCGA"``, ``"SRA"``}
        The Recount3 project the studies belong to.

    gencode_release : :class:`int`, {``29``, ``26``}, ``29``
        The GENCODE release the counts were annotated with.

    wd_gene_sums : :class:`str`, optional
        The directory where the RNA-seq data files are saved and, if
        already there, read from. If not passed, it is the current
        working directory.

    wd_qc : :class:`str`, optional
        The directory where the QC metadata files are saved and, if
        already there, read from. If not passed, it is the current
        working directory.

    df_counts_existing : :class:`pandas.DataFrame`, optional
        Existing read counts, with samples as rows. If passed, only
        the studies of the samples they lack are retrieved, and the
        samples not in the metadata are dropped.

    Returns
    -------
    df_counts : :class:`pandas.DataFrame`
        The read counts, with samples as rows and genes as columns.
    """

    # Set the column of the studies.
    study_col = "study_recount3_samples_category"

    #-----------------------------------------------------------------#

    # If there are no existing counts
    if df_counts_existing is None:

        # Return the counts of all the studies.
        return _get_counts(\
            df_metadata = df_metadata,
            project_name = project_name,
            studies = df_metadata[study_col].unique().tolist(),
            gencode_release = gencode_release,
            wd_gene_sums = wd_gene_sums,
            wd_qc = wd_qc)

    #-----------------------------------------------------------------#

    # Start from the existing counts.
    df_counts = df_counts_existing

    # Get the samples the counts lack.
    missing = df_metadata.index.difference(df_counts.index)

    # If there are any
    if len(missing) > 0:

        # Get their studies.
        studies_missing = \
            df_metadata.loc[missing, study_col].unique().tolist()

        # Inform the user that the studies' counts will be retrieved.
        infostr = \
            f"{len(missing)} sample(s) of the metadata are not in " \
            "the existing counts. Getting the counts of their " \
            f"{len(studies_missing)} study/studies " \
            f"({', '.join(map(str, studies_missing))}). The " \
            f"counts of the other {len(df_counts):,} samples are " \
            "kept as they are."
        logger.info(infostr)

        # Get the studies' counts, with the genes in the existing
        # counts' order.
        df_new = \
            _get_counts(\
                df_metadata = \
                    df_metadata[\
                        df_metadata[study_col].isin(studies_missing)],
                project_name = project_name,
                studies = studies_missing,
                gencode_release = gencode_release,
                wd_gene_sums = wd_gene_sums,
                wd_qc = wd_qc)[df_counts.columns]

        # Add the missing samples' counts.
        df_counts = \
            pd.concat(\
                [df_counts,
                 df_new.loc[df_new.index.intersection(missing)]],
                axis = 0)

        # Get the samples still without counts.
        still_missing = \
            df_metadata.index.difference(df_counts.index)

        # If there are any
        if len(still_missing) > 0:

            # Raise an error.
            errstr = \
                f"{len(still_missing)} sample(s) of the metadata " \
                "are still without counts after fetching " \
                f"{', '.join(map(str, studies_missing))}: " \
                f"{', '.join(map(str, still_missing[:5]))}" \
                f"{', ...' if len(still_missing) > 5 else ''}."
            raise ValueError(errstr)

    #-----------------------------------------------------------------#

    # Get the samples not in the metadata.
    dropped = df_counts.index.difference(df_metadata.index)

    # Keep the metadata's samples, in its order.
    df_counts = df_counts.loc[df_metadata.index]

    # Inform the user about the samples kept.
    infostr = \
        f"Kept {len(df_counts):,} samples and dropped " \
        f"{len(dropped):,} not in the metadata."
    logger.info(infostr)

    #-----------------------------------------------------------------#

    # Return the counts.
    return df_counts


def save_dataset_metadata(df: pd.DataFrame,
                          csv_file: str) -> None:
    """Save a dataset's metadata to a CSV file.

    Parameters
    ----------
    df : :class:`pandas.DataFrame`
        The metadata.

    csv_file : :class:`str`
        The output CSV file.
    """

    # Create the file's directory if it does not exist.
    os.makedirs(os.path.dirname(csv_file) or ".",
                exist_ok = True)

    # Copy the metadata.
    df = df.copy()

    #-----------------------------------------------------------------#

    # For each column holding objects or strings
    for col in df.select_dtypes(include = ["object",
                                           "string"]).columns:

        # If it is not a comment column
        if not col.endswith("_comment"):

            # Go to the next column.
            continue

        # Collapse the repeated whitespace in the comments.
        df[col] = \
            df[col].apply(\
                lambda v: re.sub(r"\s+", " ", v).strip() \
                if isinstance(v, str) else v)

    #-----------------------------------------------------------------#

    # Write the metadata.
    df.to_csv(csv_file,
              sep = ",",
              index = True,
              header = True)


def save_dataset_counts(df: pd.DataFrame,
                        csv_file: str) -> None:
    """Save a dataset's read counts to a CSV file, with the samples'
    IDs in the first column.

    Parameters
    ----------
    df : :class:`pandas.DataFrame`
        The read counts, with samples as rows.

    csv_file : :class:`str`
        The output CSV file.
    """

    # Create the file's directory if it does not exist.
    os.makedirs(os.path.dirname(csv_file) or ".",
                exist_ok = True)

    # Build the table, with the samples' IDs first.
    table = \
        pa.Table.from_pandas(df,
                             preserve_index = False).add_column(\
            0,
            "external_id",
            pa.array(df.index.astype(str).tolist()))

    # Write the table, in small batches to avoid overflowing PyArrow's
    # buffers with wide tables.
    pacsv.write_csv(\
        table,
        csv_file,
        write_options = \
            pacsv.WriteOptions(\
                include_header = True,
                quoting_style = "none",
                batch_size = min(1024, max(1, table.num_rows))))
