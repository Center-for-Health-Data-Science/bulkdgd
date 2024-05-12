#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-

#    util.py
#
#    Utilities to interact with the Recount3 platform and manipulate
#    the data retrieved from it.
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


# Set the module's description.
__doc__ = \
    "Utilities to interact with the Recount3 platform and " \
    "manipulate the data retrieved from it."


#######################################################################


# Import from the standard library.
import logging as log
import os
import sys
import re
import tempfile
# Import from third-party packages.
import pandas as pd
import requests as rq
# Import from 'bulkDGD'.
from . import defaults
from bulkDGD import util


#######################################################################


# Get the module's logger.
logger = log.getLogger(__name__)


########################## PRIVATE FUNCTIONS ##########################


def _get_metadata_fields(project_name,
                         df = None):
    """Get the metadata fields for the samples and check whether they
    are valid.

    Parameters
    ----------
    project_name : ``str``, {``"gtex"``, ``"tcga"``, ``"sra"``}
        The name of the project of interest.

    df : ``pandas.DataFrame``, optional
        The data frame containing the samples.

        If provided, and ``project_name`` is ``"sra"``, the function
        will check whether there are any ``sample_attributes`` in
        the data frame, and consider them as separate columns.

    Returns
    -------
    metadata_fields : ``list``
        The list of metadata fields.
    """

    # Get the file containing the valid metadata fields for the
    # project of interest.
    metadata_fields_file = \
        defaults.RECOUNT3_METADATA_FIELDS_FILE[project_name]

    #-----------------------------------------------------------------#

    # Open the file.
    with open(metadata_fields_file, "r") as f:

        # Get the fields (ignore empty lines and comment lines).
        metadata_fields = \
            [l.rstrip("\n") for l in f \
             if not (re.match(r"^\s*$", l) or l.startswith("#"))]

        #-------------------------------------------------------------#

        # Initialize an empty set to store each sample's attributes.
        samples_attributes = set()

        # If:
        # - The project is 'sra'.
        # - A data frame was passed.
        # - There is a 'sample_attributes' column in the data frame.
        if project_name == "sra" \
        and df is not None \
        and "sample_attributes" in df.columns:

            # Add the names of the samples' attributes to the set.
            samples_attributes.add(\
                [item.split(";;")[0].replace(" ", "_") for item \
                 in df["sample_attributes"].split("|")])

        #-------------------------------------------------------------#

        # Return all the metadata fields found.
        return metadata_fields + sorted(samples_attributes)


########################## PUBLIC FUNCTIONS ###########################


def load_samples_batches(csv_file):
    """Load a comma-separated CSV file containing a data frame with
    information about the batches of samples to be downloaded from
    Recount3.

    The data frame is expected to have at least two columns:

    * ``"input_project_name"``, containing the name of the project
      the samples belong to.
    * ``"input_samples_category"``, containing the name of the
      category the samples belong to.

    A third column, ``"query_string"``, may be present. This should
    contain the query string that should be used to filter each batch
    of samples by their metadata.

    If no ``"query_string"`` column is present, the samples will not
    be filtered.

    Parameters
    ----------
    csv_file : ``str``
        The input CSV file.

    Returns
    -------
    df : ``pandas.DataFrame``
        The data frame parsed from the CSV file.
    """

    # Set the columns taken into consideration in the data frame.
    supported_columns = \
        ["input_project_name",
         "input_samples_category",
         "query_string"]

    #-----------------------------------------------------------------#

    # Load the data frame.
    df = pd.read_csv(csv_file,
                     sep = ",",
                     header = 0,
                     comment = "#",
                     index_col = False)

    #-----------------------------------------------------------------#

    # For each required column
    for col in ["input_project_name", "input_samples_category"]:

        # If it does not exist
        if col not in df.columns:

            # Raise an error.
            errstr = \
                f"The column '{col}' must be present in the input " \
                f"CSV file '{csv_file}'."
            raise ValueError(errstr)

    #-----------------------------------------------------------------#

    # For each project found in the data frame
    for project_name in df["input_project_name"].unique():

        # Get the unique samples' categories found for that project
        # in the data frame.
        unique_samples_categories = \
            df.loc[df["input_project_name"] == project_name][\
                "input_samples_category"].unique()

        # For each samples' category
        for samples_category in unique_samples_categories:

            # Check whether it is valid.
            check_samples_category(\
                samples_category = samples_category,
                project_name = project_name)

    #-----------------------------------------------------------------#

    # If there are extra columns
    if set(df.columns) != set(supported_columns):

        # Get the extra columns.
        extra_columns = set(df.columns) - set(supported_columns)

        # Drop the extra columns.
        df = df.drop(extra_columns)

        # Get the string representing the extra columns (for logging
        # purposes).
        extra_columns_str = \
            ", ".join([f"'{col}'" for col in extra_columns])

        # Warn the user that the columns were dropped.
        warnstr = \
            "These extra columns were found in the input CSV file " \
            f"'{csv_file}': {extra_columns_str}. They will be " \
            "ignored."
        logger.warning(warnstr)

    #-----------------------------------------------------------------#

    # Return the data frame.
    return df


def check_samples_category(samples_category,
                           project_name):
    """Check that the category of samples requested by the user is
    present for the project of choice.

    Parameters
    ----------
    samples_category : ``str``
        The category of samples requested.

    project_name : ``str``, {``"gtex"``, ``"tcga"``, ``"sra"``}
        The name of the project of interest.
    """

    # Get the list of supported categories for the project of interest.
    supported_categories = \
        [l.rstrip("\n") for l in \
         open(\
            defaults.RECOUNT3_SUPPORTED_CATEGORIES_FILE[\
                project_name], "r") \
         if (not l.startswith("#") and not re.match(r"^\s*$", l))]

    #-----------------------------------------------------------------#

    # If the category provided by the user is not among the supported
    # categories
    if not samples_category in supported_categories:

        # Format the string representing the supported categories.
        supported_categories_str = \
            ", ".join([f"'{cat}'" for cat in supported_categories])

        # Raise an error.
        errstr = \
            f"The category '{samples_category}' is not a " \
            f"supported category for '{project_name}'. The " \
            f"supported categories for '{project_name}' are " \
            f"{supported_categories_str}."
        raise ValueError(errstr)


def get_query_string(query_string):
    """Get the string that will be used to filter the samples
    according to their metadata.

    Parameters
    ----------
    query_str : ``str``
        The query string or the path to a plain text file containing
        the query string.

    Returns
    -------
    ``str``
        The query string.
    """

    # If the user passed a plain text file
    if query_string.endswith(".txt"):

        # Open the file.
        with open(query_string, "r") as f:

            # Initialize an empty string to store the file's content.
            content = ""

            # For each line in the file
            for line in f:

                # If a line is empty
                if re.match(r"^\s*$", line):

                    # Ignore it.
                    continue

                # If the line is a comment line
                if line.startswith("#"):

                    # Ignore it.
                    continue

                # Add the line to the content string, converting each
                # newline character into a whitespace separator, in
                # case the content is split on multiple lines.
                content += line.replace("\n", " ")

            # Let the user see the query string.
            infostr = \
                "The query string was loaded from " \
                f"'{query_string}': '{content}'."
            logger.info(infostr)
            
            # Return the content of the file (= the query string).
            return content

    #-----------------------------------------------------------------#

    # If the user has passed a sting
    else:
        
        # Let the user see the query string.
        infostr = f"Query string: '{query_string}'."
        logger.info(infostr)

        # Return the string.
        return query_string


def get_gene_sums(project_name,
                  samples_category,
                  save_gene_sums = True,
                  wd = None):
    """Get RNA-seq counts for samples deposited in the Recount3
    platform.

    Parameters
    ----------
    project_name : ``str``, {``"gtex"``, ``"tcga"``, ``"sra"``}
        The name of the project of interest.

    samples_category : ``str``
        The category of samples requested.

    save_gene_sums : ``bool``, ``True``
        If ``True``, save the original RNA-seq data file in the
        working directory.

        The file name will be 
        ``"{project_name}_{samples_category}_gene_sums.gz"``.

    wd : ``str``, optional
        The working directory where the original RNA-seq data
        file will be saved, if ``save_gene_sums`` is ``True``.

        If not specified, it will be the current working directory.

    Returns
    -------
    ``pandas.DataFrame``
        A data frame containing the RNA-seq counts for the samples
        associated with the given category.
    """

    # Set the name of the file that will contain the RNA-seq data.
    f_gene_sums_name = \
        defaults.RECOUNT3_GENE_SUMS_FILE.format(project_name,
                                                samples_category)

    #-----------------------------------------------------------------#

    # If no working directory was specified
    if wd is None:

        # The working directory will be the current working directory.
        wd = os.getcwd()

    #-----------------------------------------------------------------#

    # Set the path to the file.
    f_gene_sums_path = os.path.join(wd, f_gene_sums_name)

    #-----------------------------------------------------------------#

    # If the file already exists in the working directory
    if os.path.exists(f_gene_sums_path):

        # Warn the user that the file exists.
        warnstr = \
            f"'{f_gene_sums_name}' already exists in '{wd}'. " \
            "The RNA-seq data will be read from this file."
        logger.warning(warnstr)

        # Read the file content into a data frame and transpose it so
        # that the samples represent the rows.
        df_gene_sums = pd.read_csv(f_gene_sums_path,
                                   sep = "\t",
                                   skiprows = 2,
                                   index_col = 0,
                                   compression = "gzip",
                                   low_memory = False).T

        # Return the data frame.
        return df_gene_sums

    #-----------------------------------------------------------------#

    # Get the URL where to find the RNA-seq data.
    gene_sums_url = \
        defaults.RECOUNT3_GENE_SUMS_URL.format(\
            project_name,
            samples_category[-2:],
            samples_category,
            project_name,
            samples_category)

    #-----------------------------------------------------------------#

    # Get the RNA-seq data.
    gene_sums = rq.get(gene_sums_url)

    #-----------------------------------------------------------------#

    # If there was a problem retrieving the RNA-seq data
    if not gene_sums.ok:

        # Raise an error.
        errstr = \
            "It was not possible to retieve the RNA-seq " \
            "data from the Recount3 platform. Error code: " \
            f"{gene_sums.status_code}. URL: {gene_sums_url}. "
        raise Exception(errstr)

    # Inform the user that the data were successfully retrieved.
    infostr = \
        "The RNA-seq data were successfully retieved from the " \
        "the Recount3 platform."
    logger.info(infostr)

    #-----------------------------------------------------------------#

    # If the user wants to save the original file
    if save_gene_sums:

        # Open a new file.
        with open(f_gene_sums_path, "wb") as f:

            # Write the content to the file.
            f.write(gene_sums.content)

            # Inform the user that the data were saved.
            infostr = \
                "The RNA-seq data were successfully saved " \
                f"in '{f_gene_sums_path}'."
            logger.info(infostr)

        # Read the file's content into a data frame and transpose it so
        # that the samples represent the rows.
        df_gene_sums = pd.read_csv(f_gene_sums_path,
                                   sep = "\t",
                                   skiprows = 2,
                                   index_col = 0,
                                   compression = "gzip",
                                   low_memory = False).T

    #-----------------------------------------------------------------#

    # Otherwise
    else:

        # Open a new temporary file.
        with tempfile.NamedTemporaryFile("wb") as f:

            # Write the content to the file.
            f.write(gene_sums.content)

            # Get the file's path.
            f_gene_sums_path = f.name

            # Read the file's content into a data frame and transpose
            # it so that the samples represent the rows.
            df_gene_sums = pd.read_csv(f_gene_sums_path,
                                       sep = "\t",
                                       skiprows = 2,
                                       index_col = 0,
                                       compression = "gzip",
                                       low_memory = False).T

    #-----------------------------------------------------------------#

    # Return the data frame.
    return df_gene_sums


def get_metadata(project_name,
                 samples_category,
                 save_metadata = True,
                 wd = None):
    """Get samples' metadata from the Recount3 platform.

    Parameters
    ----------
    project_name : ``str``, {``"gtex"``, ``"tcga"``, ``"sra"``}
        The name of the project of interest.

    samples_category : ``str``
        The category of samples requested.

    save_metadata : ``bool``, ``True``
        If ``True``, save the original metadata file in the working
        directory.

    wd : ``str``, optional
        The working directory where the original metadata file will be
        saved, if ``save_metadata`` is ``True``.

        If not specified, it will be the current working directory.

    Returns
    -------
    ``pandas.DataFrame``
        A data frame containing the metadata for the samples associated
        with the given category.
    """

    # Set the name of the file that will contain the metadata.
    f_metadata_name = \
        defaults.RECOUNT3_METADATA_FILE.format(project_name,
                                               samples_category)

    #-----------------------------------------------------------------#

    # If no working directory was specified
    if wd is None:

        # The working directory will be the current working directory.
        wd = os.getcwd()

    #-----------------------------------------------------------------#

    # Set the path to the file.
    f_metadata_path = os.path.join(wd, f_metadata_name)

    #-----------------------------------------------------------------#

    # If the file already exists in the working directory
    if os.path.exists(f_metadata_path):

        # Warn the user that the file exists.
        warnstr = \
            f"'{f_metadata_name}' already exists in '{wd}'. " \
            "The metadata will be read from this file."
        logger.warning(warnstr)

        # Read the file content into a data frame.
        df_metadata = pd.read_csv(f_metadata_path,
                                  sep = "\t",
                                  index_col = "external_id",
                                  compression = "gzip",
                                  low_memory = False)

        # Return the data frame.
        return df_metadata

    #-----------------------------------------------------------------#

    # Get the URL where to find the metadata.
    metadata_url = \
        defaults.RECOUNT3_METADATA_URL.format(\
            project_name,
            samples_category[-2:],
            samples_category,
            f"{project_name}.{project_name}",
            samples_category)

    #-----------------------------------------------------------------#

    # Get the metadata.
    metadata = rq.get(metadata_url)

    #-----------------------------------------------------------------#

    # If there was a problem retrieving the metadata
    if not metadata.ok:

        # Raise an error.
        errstr = \
            "It was not possible to retieve the metadata from the " \
            "Recount3 platform. Error code: " \
            f"{metadata.status_code}. URL: {metadata_url}."
        raise Exception(errstr)

    # Inform the user that the data were successfully retrieved.
    infostr = \
        "The metadata were successfully retieved from the Recount3 " \
        "platform."
    logger.info(infostr)

    #-----------------------------------------------------------------#

    # If the user wants to save the metadata
    if save_metadata:
            
        # Open a new file.
        with open(f_metadata_path, "wb") as f:

            # Write the content to the file.
            f.write(metadata.content)

            # Inform the user that the file was saved.
            infostr = \
                "The metadata were successfully saved in " \
                f"'{f_metadata_path}'."
            logger.info(infostr)

        # Read the file's content into a data frame.
        df_metadata = pd.read_csv(f_metadata_path,
                                  sep = "\t",
                                  index_col = "external_id",
                                  compression = "gzip",
                                  low_memory = False)

    #-----------------------------------------------------------------#
    
    # Otherwise
    else:

        # Open a new temporary file.
        with tempfile.NamedTemporaryFile("wb") as f:

            # Write the content to the file.
            f.write(metadata.content)

            # Read the file's content into a data frame.
            df_metadata = pd.read_csv(f.name,
                                      sep = "\t",
                                      index_col = "external_id",
                                      compression = "gzip",
                                      low_memory = False)

    #-----------------------------------------------------------------#

    # If there is no 'sample_attributes' column in the data frame
    if "sample_attributes" not in df_metadata.columns:

        # Simply return the data frame as it is.
        return df_metadata

    #-----------------------------------------------------------------#
    
    # Inform the string that samples' attributes were found.
    infostr = \
        "Samples' attributes were found in the metadata (see below)."
    logger.info(infostr)

    # Define a function to parse the 'sample_attribute' column in the
    # metadata.
    parse_sample_attributes = \
        lambda attr_str: dict(\
            (item.split(";;")[0].replace(" ", "_"), 
             item.split(";;")[1]) \
            for item in attr_str.split("|"))

    # Parse the samples' attributes from the data frame and covert
    # them into a DataFrame.
    df_sample_attrs = \
        df_metadata["sample_attributes"].apply(\
            parse_sample_attributes).apply(pd.Series)

    # For each attribute
    for col in df_sample_attrs.columns:

        # Get a string representing the unique values found in the
        # column.
        unique_values_str = \
            ", ".join([f"'{val}'" \
                       for val in df_sample_attrs[col].unique()])

        # Log the attribute and its unique values.
        infostr = \
            f"Sample attribute '{col}' found. Unique values: " \
            f"{unique_values_str}."
        logger.info(infostr)

    #-----------------------------------------------------------------#

    # Get the standard metadata fields.
    metadata_fields = \
        _get_metadata_fields(project_name = project_name)

    # Get any attributes that are already found in metadata.
    attrs_to_drop = \
        df_sample_attrs.columns[\
            [col_name in df_metadata.columns \
             for col_name in df_sample_attrs.columns]]

    # Drop them from the data frame of attributes.
    df_sample_attrs = df_sample_attrs.drop(attrs_to_drop,
                                           axis = 1)

    # Add the metadata columns to the data frame of metadata.
    df_metadata = df_metadata.join(df_sample_attrs)

    #-----------------------------------------------------------------#

    # If the user wants to save the metadata
    if save_metadata:

        # Inform the user that the updated metadata will be saved
        # in a separate file.
        infostr = \
            "The metadata with the 'sample_attributes' split into " \
            "different columns will be saved in a separate file."
        logger.info(infostr)
            
        # Set the name of the file that will contain the metadata.
        f_metadata_name = \
            defaults.RECOUNT3_METADATA_UPDATED_FILE.format(\
                project_name,
                samples_category)

        # Set the path to the file.
        f_metadata_path = os.path.join(wd, f_metadata_name)

        # If the file already exists in the working directory.
        if os.path.exists(f_metadata_path):

            # Warn the user that the file will be overwritten.
            infostr = \
                f"The metadata file '{f_metadata_name}' will " \
                f"be overwritten in '{wd}'."
            logger.info(infostr)

        # Write the data frame to the output file.
        df_metadata.to_csv(f_metadata_path,
                           sep = "\t",
                           compression = "gzip")

        # Inform the user that the file was written.
        infostr = \
            "The metadata with the 'sample_attributes' column " \
            "split into different columns were successfully " \
            f"written in '{f_metadata_path}'."
        logger.info(infostr)

    #-----------------------------------------------------------------#
        
    # Return the data frame containing the updated metadata.
    return df_metadata


def merge_gene_sums_and_metadata(df_gene_sums,
                                 df_metadata,
                                 project_name):
    """Add the metadata for samples deposited in the Recount3 platform.

    Parameters
    ----------
    df_gene_sums : ``pandas.DataFrame``
        The data frame containing the RNA-seq counts for the samples.

    df_metadata : ``pandas.DataFrame``
        The data frame containing the metadata for the samples.

    project_name : ``str``, {``"gtex"``, ``"tcga"``, ``"sra"``}
        The name of the project of interest.

    Returns
    -------
    df : ``pandas.DataFrame``
        The data frame containing both RNA-seq counts and metadata
        for the samples.
    """

    # Add the metadata to the original data frame.
    df_final =  pd.concat(objs = [df_gene_sums, df_metadata],
                          axis = 1)

    #-----------------------------------------------------------------#

    # Return the data frame.
    return df_final


def filter_by_metadata(df,
                       query_string,
                       project_name):
    """Filter samples using the associated metadata.

    Parameters
    ----------
    df : ``pandas.DataFrame``
        A data frame containing both RNA-seq counts and metadata for
        a set of samples.

    query_string : ``str``
        A string to query the data frame with.

    project_name : ``str``, {``"gtex"``, ``"tcga"``, ``"sra"``}
        The name of the project of interest.

    Returns
    -------
    ``pandas.DataFrame``
        The filtered data frame. This data frame will only contain the
        RNA-seq counts (no metadata).
    """

    # Filter the data frame based on the query string.
    df = df.query(query_string)

    #-----------------------------------------------------------------#

    # Get the fields containing metadata.
    metadata_fields = \
        [col for col in df.columns \
         if not col.startswith("ENSG")]

    # Remove the index column from the fields containing metadata.
    metadata_fields.remove("external_id")

    # Drop these columns from the data frame.
    df = df.drop(metadata_fields,
                 axis = 1)

    #-----------------------------------------------------------------#

    # Return the data frame.
    return df
