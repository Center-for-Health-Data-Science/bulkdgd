#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-

#    utils.py
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


# Description of the module
__doc__ = \
    "This module contains utilities to interact with the " \
    "`Recount3 platform <https://rna.recount.bio/>`_ and " \
    "manipulate the data downloaded from it."


# Standard library
import logging as log
import os
import sys
import re
import tempfile
# Third-party packages
import pandas as pd
import requests as rq
# bulkDGD
from . import defaults


# Get the module's logger
logger = log.getLogger(__name__)


#----------------------------- Functions -----------------------------#


def get_recount3_data_input_file(data, project_id_subset=None):
    """The function takes a path for a csv file or a DataFrame as input.
    Columns should contain 
    [str:<input_project_name>; str:<input_samples_category>; str:<tissue>; str:<query_string>].
    Returns a DataFrame with columns forced to type and sorted according to <tissue>.

    Parameters
    ----------
    data : ``str``|``pandas.DataFrame``
        Inut is either a path to a csv file or a pandas.DataFrame.

    project_id_subset : ``list``: [``str``]
        A list of strings denoting the subset of project id's to be handled.
    """
    # Check that input type belongs to either string or DataFrame
    if isinstance(data, pd.DataFrame):
        df=data
        print("Input detected as DataFrame.")

    elif isinstance(data, str):
        df = pd.read_csv(
            data, 
            sep=",", 
            header=0, 
            index_col="input_samples_category", 
            comment='#'
            )
    else:
        # Warn the user and exit
        errstr = \
            "Input type was neither detected as string or DataFrame " 
        # logger.exception(errstr)
        sys.exit(errstr)

    # Check that each relevant columns exists
    def check_column_exists(df, column_name):
        """Check if a column exists in a DataFrame.
        If it does not exist, check if the index holds the column,
        and if this is true, the reset_index is performed to turn the 
        index into a column.

        Parameters
        ----------
        df : ``pandas.DataFrame``
            The DataFrame to be checked.

        column_name : ``str``
            The name of the column of interest.
        """
        try:
            _ = df[column_name]
            return df
        except KeyError:
            try:
                df.reset_index(inplace=True)
                _ = df[column_name]
                return df
            except Exception as e:
                # Warn the user and exit
                errstr = \
                    "It was not possible to validate the provided " \
                    f"column '{column_name}' for the " \
                    f"input DataFrame. Error: {e}"
                # logger.exception(errstr)
                sys.exit(errstr)

    df = check_column_exists(df, 'input_samples_category')
    df = check_column_exists(df, 'input_project_name')
    df = check_column_exists(df, 'query_string')
    df = check_column_exists(df, 'tissue')

    # Force types for each relevant column
    try:
        df = df.astype({
            'input_samples_category':'string',
            'input_project_name':'string',
            'query_string': 'string',
            'tissue':'string'
            })
    except Exception as e:
        print(f"Error: {e}")
    
    # Trim the DataFrame to contain only relevant columns
    try:
        df = df.loc[:,
            [
                'input_samples_category',
                'input_project_name',
                'tissue',
                'query_string'
            ]
            ]
    except Exception as e:
        print(f"Error: {e}")

    # Sort the values according to tissue
    try:
        df.sort_values(
            ['tissue'], 
            inplace=True
            )
    except Exception as e:
        print(f"Error: {e}")
    
    # Use only a subset of the projects if the user has defined such a subset
    if project_id_subset is not None:
        try:
            df = df[df['input_samples_category'].isin(project_id_subset)]
        except Exception as e:
            print(f"Error: {e}")            

    return(df)


def check_samples_category(samples_category,
                           project_name):
    """Check that the category of samples requested by the user (cancer
    type or tissue is present for the project of choice (GTEX, TCGA,
    etc.).

    Parameters
    ----------
    samples_category : ``str``
        The category of samples requested.

    project_name : ``str``, {``"gtex"``, ``"tcga"``}
        The name of the project of interest.
    """

    # Get the list of accepted categories for the project of
    # interest
    supported_categories = \
        [l.rstrip("\n") for l in \
         open(\
            defaults.RECOUNT3_SUPPORTED_CATEGORIES_FILE[\
                project_name], "r") \
         if (not l.startswith("#") and not re.match(r"^\s*$", l))]

    # If the category provided by the user is not among the
    # accepted categories
    if not samples_category in supported_categories:

        # Raise an error
        errstr = \
            f"The category '{samples_category}' is not a " \
            f"supported category for '{project_name}'. The " \
            f"supported categories for '{project_name}' are " \
            f"{', '.join(supported_categories)}."
        raise ValueError(errstr)


def get_query_string(query_string):
    """Get the string that will be used to filter the samples
    according to their metadata.

    Parameters
    ----------
    query_str : ``str``
        The query string that will be used to filter the samples
        according to their associated metadata, or the path to a
        plain text file containing the query string.

    Returns
    -------
    ``str``
        The query string.
    """

    # If the user has passed a plain text file
    if query_string.endswith(".txt"):

        # Open the file
        with open(query_string, "r") as f:

            # Initialize an empty string to store
            # the file content
            content = ""

            # For each line in the file
            for line in f:

                # If a line is empty
                if re.match(r"^\s*$", line):

                    # Ignore it
                    continue

                # If the line is a comment line
                if line.startswith("#"):

                    # Ignore it
                    continue

                # Add the line to the content string,
                # converting each newline character
                # into a whitespace separator, in
                # case the content is split on
                # multiple lines
                content += line.replace("\n", " ")

            # Let the user see the query string.
            infostr = \
                f"Query file found: \n" \
                f"{content}.\n\n"
            logger.info(infostr)
            # Return the content
            return content

    # If the user has passed a sting
    else:
        # Let the user see the query string.
        infostr = \
            f"\n\nQuery string found: \n" \
            f"{query_string}.\n\n"
        logger.info(infostr)

        # Just return the string
        return query_string


def get_metadata_fields(project_name, project_specific = False):
    """Get the fields of the metadata for the samples from a
    plain text file.

    Parameters
    ----------
    project_name : ``str``, {``"gtex"``, ``"tcga"``}
        The name of the project of interest.

    Returns
    -------
    ``list``
        List of supported metadata fields for the samples
        of the project of interest.
    """

    # If project_specific is True, use project specific file for metadata fields
    if project_specific:
        path_metadata_fields = defaults.RECOUNT3_METADATA_FIELDS_FILE__PROJECT_SPECIFIC[\
                project_name]
    else:
        path_metadata_fields = defaults.RECOUNT3_METADATA_FIELDS_FILE[\
                project_name]

    # Open the file
    with open(path_metadata_fields, "r") as f:

        # Return the fields (ignore empty lines
        # and comment lines)
        return [l.rstrip("\n") for l in f \
                if not (re.match(r"^\s*$", l) \
                        or l.startswith("#"))]


def get_gene_sums(samples_category,
                  project_name,
                  save_gene_sums,
                  wd):
    """Get RNA-seq data for the samples deposited
    in the Recount3 platform.

    Parameters
    ----------
    samples_category : ``str``
        The category of samples requested.

    project_name : ``str``, {``"gtex"``, ``"tcga"``}
        The name of the project of interest.

    save_gene_sums : ``bool``
        If ``True``, save the original RNA-seq data
        file in the working directory.

    wd : ``str``
        Working directory.

    Returns
    -------
    ``pandas.DataFrame``
        Data frame containing the RNA-seq counts for
        the samples associated with the given category.
    """


    #------------- Check if the RNA-seq data file exists -------------#


    # Set the name of the file that will contain the RNA-seq data
    f_gene_sums_name = \
        defaults.RECOUNT3_GENE_SUMS_FILE.format(project_name,
                                                samples_category)

    # Set the path to the file
    f_gene_sums_path = os.path.join(wd, f_gene_sums_name)

    # If the file already exists in the working directory
    if os.path.exists(f_gene_sums_path):

        # Warn the user that the file will be overwritten
        warnstr = \
            f"'{f_gene_sums_name}' already exists in '{wd}'. " \
            f"The RNA-seq data will be read from this file."
        logger.warning(warnstr)

        # Read the file content into a data frame and transpose
        # it so that the samples represent the rows
        df_gene_sums = pd.read_csv(f_gene_sums_path,
                                   sep = "\t",
                                   skiprows = 2,
                                   index_col = 0,
                                   compression = "gzip").T

        # Return the data frame
        return df_gene_sums


    #------------------- Download the RNA-seq data -------------------#


    # Get the URL where to find the RNA-seq data
    gene_sums_url = \
        defaults.RECOUNT3_GENE_SUMS_URL.format(\
            project_name,
            samples_category[-2:],
            samples_category,
            project_name,
            samples_category)

    # Get the RNA-seq data
    gene_sums = rq.get(gene_sums_url)

    # If there was a problem retrieving the RNA-seq data
    if not gene_sums.ok:

        # Raise an error
        errstr = \
            f"It was not possible to retieve the RNA-seq " \
            f"data from the Recount3 platform. Error code: " \
            f"{gene_sums.status_code}. URL: {gene_sums_url}. "
        raise Exception(errstr)

    # Inform the user that the data were successfully retrieved
    infostr = \
        f"The RNA-seq data were successfully retieved from the " \
        f"the Recount3 platform."
    logger.info(infostr)


    #------------------ Save the RNA-seq data file -------------------#


    # If the user wants to save the gene sums file
    if save_gene_sums:

        # Open a new file
        with open(f_gene_sums_path, "wb") as f:

            # Write the content to the file
            f.write(gene_sums.content)

            # Inform the user that the file was written
            infostr = \
                f"The RNA-seq data were successfully written " \
                f"in {f_gene_sums_path}."
            logger.info(infostr)

        # Read the file content into a data frame and transpose
        # it so that the samples represent the rows
        df_gene_sums = pd.read_csv(f_gene_sums_path,
                                   sep = "\t",
                                   skiprows = 2,
                                   index_col = 0,
                                   compression = "gzip").T


    #--------------- Do not save the RNA-seq data file ---------------#


    # Otherwise
    else:

        # Open a new temporary file
        with tempfile.NamedTemporaryFile("wb") as f:

            # Write the content to the file
            f.write(gene_sums.content)

            # Get the file path
            f_gene_sums_path = f.name

            # Read the file content into a data frame and transpose
            # it so that the samples represent the rows
            df_gene_sums = pd.read_csv(f_gene_sums_path,
                                       sep = "\t",
                                       skiprows = 2,
                                       index_col = 0,
                                       compression = "gzip").T

    # Return the data frame
    return df_gene_sums



def get_metadata(samples_category,
                 project_name,
                 save_metadata,
                 wd):
    """Get the samples' metadata from the Recount3 platform.

    Parameters
    ----------
    samples_category : ``str``
        The category of samples requested.

    project_name : ``str``, {``"gtex"``, ``"tcga"``}
        The name of the project of interest.

    save_metadata : ``bool``
        If ``True``, save the original metadata file in
        the working directory.

    wd : ``str``
        Working directory.

    Returns
    -------
    ``pandas.DataFrame``
        Data frame containing the metadata for the samples
        associated with the given category.
    """


    #--------------- Check if the metadata file exists ---------------#


    # Set the name of the file that will contain the metadata
    f_metadata_name = \
        defaults.RECOUNT3_METADATA_FILE.format(project_name,
                                               samples_category)

    # Set the path to the file
    f_metadata_path = os.path.join(wd, f_metadata_name)

    # If the file already exists in the working directory
    if os.path.exists(f_metadata_path):

        # Warn the user that the file will be overwritten
        warnstr = \
            f"'{f_metadata_name}' already exists in '{wd}'. " \
            f"The metadata will be read from this file."
        logger.warning(warnstr)

        # Read the file content into a data frame
        df_metadata = pd.read_csv(f_metadata_path,
                                  sep = "\t",
                                  index_col = "external_id",
                                  compression = "gzip")

        # Return the data frame
        return df_metadata


    #--------------------- Download the metadata ---------------------#


    # Get the URL where to find the metadata
    metadata_url = \
        defaults.RECOUNT3_METADATA_URL.format(\
            project_name,
            samples_category[-2:],
            samples_category,
            f"{project_name}.{project_name}",
            samples_category)

    # Get the metadata
    metadata = rq.get(metadata_url)

    # If there was a problem retrieving the metadata
    if not metadata.ok:

        # Raise an error
        errstr = \
            f"It was not possible to retieve the metadata " \
            f"from the Recount3 platform. Error code: " \
            f"{metadata.status_code}. URL: {metadata_url}."
        raise Exception(errstr)

    # Inform the user that the data were retrieved
    # successfully
    infostr = \
        f"The metadata were successfully retieved " \
        f"from the Recount3 platform."
    logger.info(infostr)


    #-------------------- Save the metadata file ---------------------#


    # If the user provided a path to a CSV file where
    # to save the metadata
    if save_metadata:
            
        # Open a new file
        with open(f_metadata_path, "wb") as f:

            # Write the content to the file
            f.write(metadata.content)

            # Inform the user that the file was written
            infostr = \
                f"The metadata were successfully " \
                f"written in {f_metadata_path}."
            logger.info(infostr)

        # Read the file content into a data frame
        df_metadata = pd.read_csv(f_metadata_path,
                                  sep = "\t",
                                  index_col = "external_id",
                                  compression = "gzip")


    #----------------- Do not save the metadata file -----------------#


    # Otherwise
    else:

        # Open a new temporary file
        with tempfile.NamedTemporaryFile("wb") as f:

            # Write the content to the file
            f.write(metadata.content)

            # Read the file content into a data frame
            df_metadata = pd.read_csv(f.name,
                                      sep = "\t",
                                      index_col = "external_id",
                                      compression = "gzip")

    # Return the data frame
    return df_metadata



#--------------- Update metadata fields ---------------#

def update_metadata_fields(test_attributes, project_name):
    """Get the fields of the metadata for the samples from a
    plain text file. If new fields are available from the 
    input attributes, then append these fields to the file.

    Parameters
    ----------
    test_attributes : `'pandas.DataFrame.columns'`
        From sra sample_attributes, holding potential new 
        metadata fields as columns.

    project_name : ``str``, {``"gtex"``, ``"tcga"``}
        The name of the project of interest.

    Returns
    -------
    ``file``
        Updated file of supported metadata fields for the 
        samples of the project of interest.
    """

    # Copy metadata_fields file to project specific version


    # Open the file
    with open(defaults.RECOUNT3_METADATA_FIELDS_FILE[\
                project_name], "r") as f:

        # Retrieve the fields as a list of strings 
        # (ignore empty lines and comment lines)
        metadata_fields = \
             [l.rstrip("\n") for l in f \
                if not (re.match(r"^\s*$", l) \
                        or l.startswith("#"))]

    
    # Check for new fields in test_attributes, which are not found in metadata_fields
    new_fields = [field for field in test_attributes if field not in metadata_fields]
    
    # If new fields, add these together with the standard SRA metadata fields to project specific metadata fields
    if new_fields: 
        # Let the user know which fields will be added.
        new_fields_report_string = ", ".join(new_fields)
        infostr = \
            f"New fields will be added to metadata fields file: \n" \
            f"{new_fields_report_string}.\n\n"
        logger.info(infostr)

        # Save fields to project specific metadata fields file
        with open(defaults.RECOUNT3_METADATA_FIELDS_FILE__PROJECT_SPECIFIC[project_name], "w") as f:  # open in write mode
            f.write("# Fields found in the SRA metadata files downloaded from the Recount3 platform." + "\n")
            f.write(" " + "\n")
            for field in metadata_fields:
                f.write(field + "\n")
            f.write("\n")
            f.write("# Fields found in the sample_attributes column in the project speciic metadata." + "\n")
            f.write(" " + "\n")
            for field in new_fields:
                f.write(field + "\n")



#--------------- Function for parsing attributes ---------------#

def parse_sample_attributes(attr_str):
    return dict(item.split(';;') for item in attr_str.split('|'))


#--------------- Parse attributes, add to columns, update metadata fields ---------------#

def add_sample_attributes_to_metadata(df, save_metadata, project_name, samples_category, wd):
    # Parse sample_attributes
    df_attr = df['sample_attributes'] \
        .apply(parse_sample_attributes) \
        .apply(pd.Series)
    
    # Report the columns and levels of the attributes column
    attribute_report_string = "\n".join([f"{col} \n Levels: {df_attr[col].unique()}\n\n" for col in df_attr.columns])
    infostr = \
        f"\n\nAttributes found: \n\n" \
        f"{attribute_report_string}\n"
    logger.info(infostr)


    # Update metadata fields
    update_metadata_fields(test_attributes = df_attr.columns, project_name = project_name)


    # Drop any attributes that are allready found in metadata
    attr_drop_bool = [col_name in df.columns for col_name in df_attr.columns]
    attr_to_drop = df_attr.columns[attr_drop_bool]
    if not attr_to_drop.empty:
            # Inform the user
            warnstr = \
                f"Attributes \n{attr_to_drop} \n" \
                f"were previously added to metadata. " \
                "These will not be added again."
            logger.warning(warnstr)

            df_attr = df_attr.drop(attr_to_drop, axis = 1)
    
    # If no attributes will be added, return original metadata
    if df_attr.empty:
            warnstr = \
                f"No attributes will be added to metadata.\n"
            logger.warning(warnstr)
            return(df)

    # Join to metadata columns
    df = \
        df.join(
            df_attr
        )

    
    #-------------------- Save the metadata file ---------------------#


    # If the user provided a path to a CSV file where
    # to save the metadata
    if save_metadata:
            
        # Try writing the content to the file
        try:
            
            # Set the name of the file that will contain the metadata
            f_metadata_name = \
                defaults.RECOUNT3_METADATA_FILE.format(project_name, samples_category)


            # Set the path to the file
            f_metadata_path = os.path.join(wd, f_metadata_name)

            # If the file already exists in the working directory
            if os.path.exists(f_metadata_path):

                # Warn the user that the file will be overwritten
                infostr = \
                    f"The metadata file '{f_metadata_name}' will be overwritten in '{wd}'. " \
                    f"Metadata with attributes added to columns will replace the file."
                logger.info(infostr)

            # Write data frame to file
            df.to_csv(
                f_metadata_path,
                sep = "\t",
                compression = "gzip")

            # Inform the user that the file was written
            infostr = \
                f"The metadata were successfully " \
                f"written in {f_metadata_path}."
            logger.info(infostr)
        except Exception as e:
            # Warn the user and exit
            errstr = \
                "It was not possible to write the metadata to disc for " \
                f"'{samples_category}' samples from the Recount3 " \
                f"platform. Error: {e}"
            logger.exception(errstr)
            sys.exit(errstr)
    
    # Return the updated metadata
    return(df)



def merge_gene_sums_and_metadata(df_gene_sums,
                                 df_metadata,
                                 project_name):
    """Add the metadata for TCGA samples deposited
    in the Recount3 platform.

    Parameters
    ----------
    df_gene_sums : ``pandas.DataFrame``
        Data frame containing the RNA-seq data
        for the samples.

    df_metadata : ``pandas.DataFrame``
        Data frame containing the metadata for the
        samples.

    project_name : ``str``, {``"gtex"``, ``"tcga"``}
        The name of the project of interest.

    Returns
    -------
    ``pandas.DataFrame``
        Data frame containing both RNA-seq counts
        and metadata for the samples.
    """

    # Get the supported metadata fields
    metadata_fields = \
        get_metadata_fields(project_name = project_name, project_specific=True)

    # If the fields in the metadata file do not correspond
    # to the ones saved
    if len(set(df_metadata.columns.tolist() + \
                    ["external_id"]).difference( \
                    set(metadata_fields))) > 0:

        # Get the project specific metadata fields' file
        fields_file = \
            defaults.RECOUNT3_METADATA_FIELDS_FILE__PROJECT_SPECIFIC[project_name]

        # Warn the user about the inconsistency and raise
        # an exception
        errstr = \
            "The fields found in the metadata downloaded " \
            "from Recount3 do not correspond to the ones " \
            f"in the '{fields_file}' file. Please update " \
            "the file and rerun."
        raise ValueError(errstr)

    # Add the metadata to the original data frame
    df_final =  pd.concat(objs = [df_gene_sums, df_metadata],
                          axis = 1)

    # Return the data frame
    return df_final



def filter_by_metadata(df,
                       query_string,
                       project_name):
    """Filter samples using the associated metadata.

    Parameters
    ----------
    df : ``pandas.DataFrame``
        Data frame containing both RNA-seq data
        and metadata for a set of samples.

    query_string : ``str``
        String to query the data frame with.

    project_name : ``str``, {``"gtex"``, ``"tcga"``}
        The name of the project of interest.

    Returns
    -------
    ``pandas.DataFrame``
        Filtered data frame. This data frame will only
        contain the RNA-seq counts.
    """

    # Filter the data frame based on the query string
    df = df.query(query_string)

    # Get the supported metadata fields
    metadata_fields = \
        get_metadata_fields(project_name = project_name, project_specific = True)

    # Remove the metadata columns, apart from the index
    # column
    metadata_fields.remove("external_id")
    df = df.drop(metadata_fields,
                 axis = 1)

    # Return the data frame
    return df