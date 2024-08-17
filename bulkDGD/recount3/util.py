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
import re
import tempfile
# Import from third-party packages.
import pandas as pd
import requests as rq
import yaml
# Import from 'bulkDGD'.
from . import defaults


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
    project_name : :class:`str`, {``"gtex"``, ``"tcga"``, ``"sra"``}
        The name of the project of interest.

    df : :class:`pandas.DataFrame`, optional
        The data frame containing the samples.

        If provided, and ``project_name`` is ``"sra"``, the function
        will check whether there are any ``sample_attributes`` in
        the data frame, and consider them as separate columns.

    Returns
    -------
    metadata_fields : :class:`list`
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
            [line.rstrip("\n") for line in f \
             if not (re.match(r"^\s*$", line) or \
                     line.startswith("#"))]

        #-------------------------------------------------------------#

        # For each entity that may have attributes
        for entity in ["sample", "experiment"]:

            # Initialize an empty set to store the entity's attributes.
            attributes = set()

            # Set the name of the column that may contain the
            # attributes.
            column_attrs = f"{entity}_attributes"

            # If:
            # - The project is 'sra'.
            # - A data frame was passed.
            # - There is an '{entity}_attributes' column in the data
            # frame.
            if project_name == "sra" \
            and df is not None \
            and column_attrs in df.columns:

                # Add the names of the samples' attributes to the set.
                attributes.add(\
                    [item.split(";;")[0].replace(" ", "_") for item \
                     in df[column_attrs].split("|")])

                # Add the attributes to the list of metadata fields
                # found.
                metadata_fields += sorted(attributes)

        #-------------------------------------------------------------#

        # Return all the metadata fields found.
        return metadata_fields


def _load_samples_batches_csv(csv_file):
    """Load the information for batches of samples to be downloaded
    from the Recount3 platform from a CSV file.

    Parameters
    ----------
    yaml_file : :class:`str`
        The CSV file.

    Returns
    -------
    df : :class:`pandas.DataFrame`
        A data frame containing the information for the batches of
        samples.
    """

    # Set the columns taken into consideration in the data frame.
    supported_columns = \
        ["recount3_project_name",
         "recount3_samples_category",
         "query_string",
         "metadata_to_keep",
         "metadata_to_drop"]

    #-----------------------------------------------------------------#

    # Load the data frame.
    df = pd.read_csv(csv_file,
                     sep = ",",
                     header = 0,
                     comment = "#",
                     index_col = False).fillna("")

    #-----------------------------------------------------------------#

    # For each required column
    for col in ["recount3_project_name", "recount3_samples_category"]:

        # If it does not exist
        if col not in df.columns:

            # Raise an error.
            errstr = \
                f"The column '{col}' must be present in the input " \
                f"CSV file '{csv_file}'."
            raise ValueError(errstr)

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


def _load_samples_batches_yaml(yaml_file):
    """Load the information for batches of samples to be downloaded
    from the Recount3 platform from a YAML file.

    Parameters
    ----------
    yaml_file : :class:`str`
        The YAML file.

    Returns
    -------
    df : :class:`pandas.DataFrame`
        A data frame containing the information for the batches of
        samples.
    """

    # Set the columns that the final data frame will have.
    columns = \
        ["recount3_project_name",
         "recount3_samples_category",
         "query_string",
         "metadata_to_keep",
         "metadata_to_drop"]

    #-----------------------------------------------------------------#

    # Set an empty list to store the data for the final data frame.
    data = []

    #-----------------------------------------------------------------#
    
    # Load the batches of samples.
    samples_batches = yaml.safe_load(open(yaml_file, "r"))

    #-----------------------------------------------------------------#

    # For each Recount3 project's name.
    for project_name in samples_batches:

        # Get the conditions that apply to all samples' categories
        # for the project.
        conditions_all = samples_batches[project_name].pop("all", {})

        # Get the query string to filter all samples belonging to
        # the project.
        qs = conditions_all.get("query_string", "")

        # Get the metadata columns to be kept in all samples belonging
        # to the project.
        mtk = \
            "|".join(conditions_all.get("metadata_to_keep", []))

        # Get the metadata columns to be dropped from all samples
        # belonging to the project.
        mtd = \
            "|".join(conditions_all.get("metadata_to_drop", []))
        
        # For each category of samples in the project
        for samples_category in samples_batches[project_name]:

            # Get the data for the samples belonging to the category.
            samples_data = \
                samples_batches[project_name][samples_category]

            # Add each piece of data to the final list.
            data.append(\
                {"recount3_project_name" : \
                    project_name,
                 "recount3_samples_category" : \
                    samples_category,
                 "query_string" : \
                    samples_data.get("query_string", qs),
                 "metadata_to_keep" : \
                    mtk + "|".join(samples_data.get(\
                        "metadata_to_keep", [])),
                 "metadata_to_drop" : \
                    mtd + "|".join(samples_data.get(\
                        "metadata_to_drop", []))})

    #-----------------------------------------------------------------#

    # Create the final data frame from the list.
    df = pd.DataFrame(data,
                      columns = columns)

    #-----------------------------------------------------------------#

    # Return the data frame.
    return df


########################## PUBLIC FUNCTIONS ###########################


def load_samples_batches(samples_file):
    """Load a file with information about the batches of samples to be
    downloaded from Recount3.

    The file can be either a CSV file or a YAML file.

    See the Notes section below for more details about their format.

    Parameters
    ----------
    samples_file : :class:`str`
        The input file.

    Returns
    -------
    df : :class:`pandas.DataFrame`
        A data frame containing the information parsed from the
        file.

    Notes
    -----
    **CSV file**

    If the input file is a CSV file, it should contain a
    comma-separated data frame.

    The data frame is expected to have at least two columns:

    * ``"recount3_project_name"``, containing the name of the project
      the samples belong to.
    * ``"recount3_samples_category"``, containing the name of the
      category the samples belong to (it is a tissue type for GTEx
      data, a cancer type for TCGA data, and a project code for SRA
      data)
    
    These additional three columns may also be present:
    
    * ``"query_string"``, containing the query string that should be
      used to filter each batch of samples by their metadata. The
      query string is passed to the ``pandas.DataFrame.query()``
      method.

      If no ``"query_string"`` column is present, the samples will not
      be filtered.

    * ``metadata_to_keep``, containing a vertical line (|)-separated
      list of names of metadata columns that will be kept in the 
      final data frames, together with the columns containing gene
      expression data.

      ``"recount3_project_name"`` and ``"recount3_samples_category"``
      are valid column names, and, if passed, the final data frames
      will also include them (each data frame will, of course, contain
      only one repeated value for each of these columns, since it
      contains samples from a single category of a single project).

      By default, all metadata columns (plus the
      ``"recount3_project_name"`` and ``"recount3_samples_category"`` 
      columns) are kept in the final data frames.

    * ``metadata_to_drop``, containing a vertical line (|)-separated
      list of names of metadata columns that will be dropped from the
      final data frames.

      The reserved keyword ``'_all_'`` can be used to drop all metadata
      columns from the final data frame of a specific batch of samples.

      ``"recount3_project_name"`` and ``"recount3_samples_category"``
      are valid column names and, if passed, will result in these
      columns being dropped.

    **YAML file**

    If the file is a YAML file, it should have the format exemplified
    below. We recommend using a YAML file over a CSV file when you have
    several studies for which different filtering conditions should be
    applied.

    .. code-block:: yaml

        # SRA studies - it can be omitted if no SRA studies are
        # included.
        sra:

          # Conditions applied to all SRA studies.
          all:

            # Which metadata to keep in all studies (if found). It is
            # a vertical line (|)-separated list of names of metadata
            # columns that will be kept in the  final data frames,
            # together with the columns containing gene expression
            # data.
            #
            # "recount3_project_name"`` and "recount3_samples_category"
            # are valid column names, and, if passed, the final data
            # frames will also include them (each data frame will, of
            # course, contain only one repeated value for each of these
            # columns, since it contains samples from a single category
            # of a single project).
            #
            # By default, all metadata columns (plus the
            # "recount3_project_name" and `"recount3_samples_category"
            # columns) are kept in the final data frames.
            metadata_to_keep:

              # Keep in all studies.
              - source_name
              ...

            # Which metadata to drop from all studies (if found). It is
            # a vertical line (|)-separated list of names of metadata
            # columns that will be dropped from the final data frames.
            #
            # The reserved keyword '_all_' can be used to drop all
            # columns from the data frames.
            #
            # "recount3_project_name" and "recount3_samples_category"
            # are valid column names and, if passed, will result in
            # these columns being dropped.
            metadata_to_drop:

              # Found in all studies.
              - age
              ...

          # Conditions applied to SRA study SRP179061.
          SRP179061:
            
            # The query string that should be used to filter each batch
            # of samples by their metadata. The query string is passed
            # to the 'pandas.DataFrame.query()' method for filtering.

            # If no query string  is present, the samples will not
            # be filtered.
            query_string: diagnosis == 'Control'

            # Which metadata to keep in this study (if found), It
            # follows the same rules as the 'metadata_to_keep' field
            # in the 'all' section.
            metadata_to_keep:
            - tissue

            # Which metadata to drop from this study (if found), It
            # follows the same rules as the 'metadata_to_drop' field
            # in the 'all' section.
            metadata_to_drop:
            - Sex

        # GTEx studies - it can be omitted if no GTEx studies are
        # included.
        gtex:

          # Same format as for SRA studies - single studies are
          # identified by the tissue type each study refers to.
          ...

        # TCGA studies - it can be omitted if no TCGA studies are
        # included.
        tcga:

          # Same format as for SRA studies - single studies are
          # identified by the cancer type each study refers to.
          ...

    """

    # Get the extension of the input file.
    _, samples_file_ext = os.path.splitext(samples_file)

    # If the file is a CSV file
    if samples_file_ext == ".csv":

        # Create the data frame from the file and return it.
        return _load_samples_batches_csv(csv_file = samples_file)

    # If the file is a YAML file
    elif samples_file_ext == ".yaml":

        # Create the data frame from the file and return it.
        return _load_samples_batches_yaml(yaml_file = samples_file)

    # Otherwise
    else:

        # Raise an error.
        errstr = \
            f"The file '{samples_file}' must be either a CSV file " \
            "('.csv' extension) or a YAML file ('.yaml' extension)."
        raise ValueError(errstr)


def get_query_string(query_string):
    """Get the string that will be used to filter the samples
    according to their metadata.

    Parameters
    ----------
    query_str : :class:`str`
        The query string or the path to a plain text file containing
        the query string.

    Returns
    -------
    query_str : :class:`str`
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
    project_name : :class:`str`, {``"gtex"``, ``"tcga"``, ``"sra"``}
        The name of the project of interest.

    samples_category : :class:`str`
        The category of samples requested.

    save_gene_sums : :class:`bool`, :obj:`True`
        If :obj:`True`, save the original RNA-seq data file in the
        working directory.

        The file name will be 
        ``"{project_name}_{samples_category}_gene_sums.gz"``.

    wd : :class:`str`, optional
        The working directory where the original RNA-seq data
        file will be saved, if ``save_gene_sums`` is :obj:`True`.

        If not specified, it will be the current working directory.

    Returns
    -------
    df_gene_sums : :class:`pandas.DataFrame`
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
            "It was not possible to retrieve the RNA-seq " \
            "data from the Recount3 platform. Error code: " \
            f"{gene_sums.status_code}. URL: {gene_sums_url}. "
        raise Exception(errstr)

    # Inform the user that the data were successfully retrieved.
    infostr = \
        "The RNA-seq data were successfully retrieved from the " \
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
    project_name : :class:`str`, {``"gtex"``, ``"tcga"``, ``"sra"``}
        The name of the project of interest.

    samples_category : :class:`str`
        The category of samples requested.

    save_metadata : :class:`bool`, :obj:`True`
        If :obj:`True`, save the original metadata file in the working
        directory.

    wd : :class:`str`, optional
        The working directory where the original metadata file will be
        saved, if ``save_metadata`` is :obj:`True`.

        If not specified, it will be the current working directory.

    Returns
    -------
    df_metadata : :class:`pandas.DataFrame`
        A data frame containing the metadata for the samples associated
        with the given category.

    Notes
    -----
    The ``"recount3_project_name"`` and the
    ``"recount3_samples_category"`` columns are automatically added to
    the metadata returned by the function and contain the
    ``project_name`` and ``samples_category`` of the samples,
    respectively.
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

        # Add the column containing the project's name.
        df_metadata.insert(loc = 0,
                           column = "recount3_project_name",
                           value = project_name)

        # Add the column containing the samples' category.
        df_metadata.insert(loc = 1,
                           column = "recount3_samples_category",
                           value = samples_category)

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
            "It was not possible to retrieve the metadata from the " \
            "Recount3 platform. Error code: " \
            f"{metadata.status_code}. URL: {metadata_url}."
        raise Exception(errstr)

    # Inform the user that the data were successfully retrieved.
    infostr = \
        "The metadata were successfully retrieved from the Recount3 " \
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

    # Add the column containing the project's name.
    df_metadata.insert(loc = 0,
                       column = "recount3_project_name",
                       value = project_name)

    # Add the column containing the samples' category.
    df_metadata.insert(loc = 1,
                       column = "recount3_samples_category",
                       value = samples_category)

    #-----------------------------------------------------------------#

    # For each entity that may have attributes
    for entity in ["sample", "experiment"]:

        # Set the name of the column that may contain the attributes.
        column_attrs = f"{entity}_attributes"

        # If the column exists in the data frame containing the
        # metadata
        if column_attrs in df_metadata.columns:

            #---------------------------------------------------------#
            
            # Inform the user that attributes were found.
            infostr = \
                f"{entity.capitalize()}s' attributes were found in " \
                "the metadata (see below)."
            logger.info(infostr)

            #---------------------------------------------------------#

            # Define a function to parse the 'sample_attribute' column
            # in the metadata.
            def parse_attributes(attr_str):
                return dict((item.split(";;")[0].replace(" ", "_"), \
                             item.split(";;")[1]) for item \
                             in str(attr_str).split("|") \
                             if item != "nan")

            # Parse the samples' attributes from the data frame and
            # convert them into a data frame.
            df_attrs = \
                df_metadata[column_attrs].apply(\
                    parse_attributes).apply(pd.Series)

            # For each attribute
            for col in df_attrs.columns:

                # Get a string representing the unique values found in
                # the column.
                unique_values_str = \
                    ", ".join(\
                        [f"'{v}'" for v in df_attrs[col].unique()])

                # Log the attribute and its unique values.
                infostr = \
                    f"{entity} attribute '{col}' found. Unique " \
                    f"values: {unique_values_str}."
                logger.info(infostr)

            #---------------------------------------------------------#

            # Get any attributes that are already found in the
            # metadata.
            attrs_to_drop = \
                df_attrs.columns[\
                    [col_name in df_metadata.columns \
                     for col_name in df_attrs.columns]]

            # Drop them from the data frame of attributes.
            df_attrs = df_attrs.drop(labels = attrs_to_drop,
                                     axis = 1)

            # Add the metadata columns to the data frame of metadata.
            df_metadata = df_metadata.join(df_attrs)

            # Drop the original column from the data frame containing
            # the metadata.
            df_metadata = df_metadata.drop(labels = [column_attrs],
                                           axis = 1)

    #-----------------------------------------------------------------#

    # If the user wants to save the metadata
    if save_metadata:

        # Inform the user that the updated metadata will be saved
        # in a separate file.
        infostr = \
            "The metadata with the sample/experiment attributes " \
            "split into different columns will be saved in a " \
            "separate file."
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
            "The metadata with the sample/experiment attributes " \
            "split into different columns were successfully " \
            f"written in '{f_metadata_path}'."
        logger.info(infostr)

    #-----------------------------------------------------------------#
        
    # Return the data frame containing the updated metadata.
    return df_metadata


def merge_gene_sums_and_metadata(df_gene_sums,
                                 df_metadata):
    """Add the metadata for samples deposited in the Recount3 platform.

    Parameters
    ----------
    df_gene_sums : :class:`pandas.DataFrame`
        The data frame containing the RNA-seq counts for the samples.

    df_metadata : :class:`pandas.DataFrame`
        The data frame containing the metadata for the samples.

    Returns
    -------
    df_merged : :class:`pandas.DataFrame`
        The data frame containing both RNA-seq counts and metadata
        for the samples.
    """

    # Add the metadata to the original data frame. Drop duplicated
    # columns.
    df_final =  pd.concat(objs = [df_gene_sums, df_metadata],
                          axis = 1)

    #-----------------------------------------------------------------#

    # Return the data frame.
    return df_final


def filter_by_metadata(df,
                       query_string):
    """Filter samples using the associated metadata.

    Parameters
    ----------
    df : :class:`pandas.DataFrame`
        A data frame containing both RNA-seq counts and metadata for
        a set of samples.

    query_string : :class:`str`
        A string to query the data frame with.

    Returns
    -------
    df_filtered : :class:`pandas.DataFrame`
        The filtered data frame.
    """

    # Filter the data frame based on the query string.
    df = df.query(query_string)

    #-----------------------------------------------------------------#

    # Return the data frame.
    return df
