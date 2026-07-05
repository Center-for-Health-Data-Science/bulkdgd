#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-

#    util.py
#
#    Utilities to interact with the Recount3 platform and manipulate
#    the data retrieved from it.
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
    "Utilities to interact with the Recount3 platform and " \
    "manipulate the data retrieved from it."


#######################################################################


# Import from the standard library.
import logging as log
import os
from typing import Optional

# Import from third-party libraries.
import pandas as pd
import requests as rq

# Import from 'bulkdgd'.
from . import defaults


#######################################################################


# Get the module's logger.
logger = log.getLogger(__name__)


########################## PUBLIC FUNCTIONS ###########################


def load_samples_batches(samples_file: str) -> pd.DataFrame:
    """Load a data frame with information about the batches of samples
    to be downloaded from Recount3 from a CSV file.

    Parameters
    ----------
    samples_file : :class:`str`
        The input CSV file. It is expected to have the following
        columns:

        * ``"recount3_project_name"``, containing the name of the
          project the samples belong to.
        
        * ``"recount3_samples_category"``, containing the name of the
          category the samples belong to (it is a tissue type for
          GTEx data, a cancer type for TCGA data, and a project code
          for SRA data)

    Returns
    -------
    df : :class:`pandas.DataFrame`
        A data frame containing the information parsed from the
        file.

    Notes
    -----

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
    df = pd.read_csv(samples_file,
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
                f"CSV file '{samples_file}'."
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
            f"'{samples_file}': {extra_columns_str}. They will be " \
            "ignored."
        logger.warning(warnstr)

    #-----------------------------------------------------------------#

    # Return the data frame.
    return df


def get_gene_sums(project_name: str,
                  samples_category: str,
                  save_gene_sums: bool = True,
                  wd: Optional[str] = None,
                  gencode_release: int = 29) -> pd.DataFrame:
    """Get RNA-seq counts for samples deposited in the Recount3
    platform.

    Parameters
    ----------
    project_name : :class:`str`, {``"GTEx"``, ``"TCGA"``, ``"SRA"``}
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
    
    gencode_release : :class:`int`, :obj:`29`
        The Gencode release according to which the RNA-seq data
        were annotated.

    Returns
    -------
    df_gene_sums : :class:`pandas.DataFrame`
        A data frame containing the RNA-seq counts for the samples
        associated with the given category.
    """
        
    # If the given Gencode release is not supported in Recount3
    if gencode_release not in defaults.RECOUNT3_GENCODE_RELEASES:

        # Raise an error.
        errstr = \
            "The given Gencode release is not supported in " \
            "Recount3. Supported releases are: " \
            f"{', '.join(
                map(str, defaults.RECOUNT3_GENCODE_RELEASES))}."
        raise Exception(errstr)

    #-----------------------------------------------------------------#

    # Set the name of the file that will contain the RNA-seq data.
    f_gene_sums_name = \
        defaults.RECOUNT3_GENE_SUMS_FILE.format(
            project_name.lower(),
            samples_category)

    #-----------------------------------------------------------------#

    # If no working directory was specified
    if wd is None:

        # The working directory will be the current working directory.
        wd = os.getcwd()
    
    # Otherwise
    else:
        
        # Create it if it does not exist.
        os.makedirs(wd,
                    exist_ok = True)

    #-----------------------------------------------------------------#

    # Set the path to the file.
    f_gene_sums_path = os.path.join(wd, f_gene_sums_name)

    #-----------------------------------------------------------------#

    # If the file already exists in the working directory
    if os.path.exists(f_gene_sums_path):

        # Inform the user that the file exists.
        infostr = \
            f"'{f_gene_sums_name}' already exists in '{wd}'. " \
            "The RNA-seq data will be read from this file."
        logger.info(infostr)

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
    
    # Otherwise
    else:

        # Get the URL where to find the RNA-seq data.
        gene_sums_url = \
            defaults.RECOUNT3_GENE_SUMS_URL.format(\
                project_name.lower(),
                samples_category[-2:],
                samples_category,
                project_name.lower(),
                samples_category,
                gencode_release)

        # Read the data frame from the URL.
        df_gene_sums = pd.read_csv(gene_sums_url,
                                   sep = "\t",
                                   skiprows = 2,
                                   index_col = 0,
                                   compression = "gzip",
                                   low_memory = False).T

        # If the user wants to save the original file
        if save_gene_sums:
            
            # Get the response.
            gene_sums = rq.get(gene_sums_url)

            # Open a new file.
            with open(f_gene_sums_path, "wb") as f:

                # Write the content to the file.
                f.write(gene_sums.content)

                # Inform the user that the data were saved.
                infostr = \
                    "The RNA-seq data were successfully saved " \
                    f"in '{f_gene_sums_path}'."
                logger.info(infostr)

    #-----------------------------------------------------------------#

    # Rename the index.
    df_gene_sums = df_gene_sums.rename_axis("external_id")

    # Return the data frame.
    return df_gene_sums


def get_qc(project_name: str,
           samples_category: str,
           save_qc: bool = True,
           wd: Optional[str] = None) -> pd.DataFrame:
    """Get QC metadata for samples deposited in the Recount3
    platform.

    Parameters
    ----------
    project_name : :class:`str`, {``"GTEx"``, ``"TCGA"``, ``"SRA"``}
        The name of the project of interest.

    samples_category : :class:`str`
        The category of samples requested.

    save_qcs : :class:`bool`, :obj:`True`
        If :obj:`True`, save the original QC metadata file in the
        working directory.

        The file name will be 
        ``"{project_name}_{samples_category}_qc.gz"``.

    wd : :class:`str`, optional
        The working directory where the original QC metadata file
        will be saved, if ``save_qc`` is :obj:`True`.

        If not specified, it will be the current working directory.

    Returns
    -------
    df_qc : :class:`pandas.DataFrame`
        A data frame containing the QC metadata for the samples
        associated with the given category.
    """

    # Set the name of the file that will contain the QC metadata.
    f_qc_name = \
        defaults.RECOUNT3_QC_FILE.format(project_name.lower(),
                                         samples_category)

    #-----------------------------------------------------------------#

    # If no working directory was specified
    if wd is None:

        # The working directory will be the current working directory.
        wd = os.getcwd()

    # Otherwise
    else:
        
        # Create it if it does not exist.
        os.makedirs(wd,
                    exist_ok = True)

    #-----------------------------------------------------------------#

    # Set the path to the file.
    f_qc_path = os.path.join(wd, f_qc_name)

    #-----------------------------------------------------------------#

    # If the file already exists in the working directory
    if os.path.exists(f_qc_path):

        # Inform the user that the file exists.
        infostr = \
            f"'{f_qc_name}' already exists in '{wd}'. " \
            "The QC metadata will be read from this file."
        logger.info(infostr)
        
        # Read the file content into a data frame and transpose it so
        # that the samples represent the rows.
        df_qc = pd.read_csv(f_qc_path,
                            sep = "\t",
                            compression = "gzip",
                            low_memory = False,
                            index_col = "external_id")

        # Return the data frame.
        return df_qc
    
    #-----------------------------------------------------------------#

    # Otherwise
    else:

        # Get the URL where to find the QC metadata.
        qc_url = \
            defaults.RECOUNT3_QC_URL.format(\
                project_name.lower(),
                samples_category[-2:],
                samples_category,
                project_name.lower(),
                samples_category)

        # Read the data frame from the URL.
        df_qc = pd.read_csv(qc_url,
                            sep = "\t",
                            compression = "gzip",
                            low_memory = False,
                            index_col = "external_id")

        # If the user wants to save the original file
        if save_qc:

            # Get the QC metadata.
            qc = rq.get(qc_url)

            # Open a new file.
            with open(f_qc_path, "wb") as f:

                # Write the content to the file.
                f.write(qc.content)

                # Inform the user that the data were saved.
                infostr = \
                    "The QC metadata were successfully saved " \
                    f"in '{f_qc_path}'."
                logger.info(infostr)

    #-----------------------------------------------------------------#

    # Return the data frame.
    return df_qc


def get_metadata(project_name: str,
                 samples_category: str,
                 save_metadata: bool = True,
                 wd: Optional[str] = None) -> pd.DataFrame:
    """Get samples' metadata from the Recount3 platform.

    Parameters
    ----------
    project_name : :class:`str`, {``"GTEx"``, ``"TCGA"``, ``"SRA"``}
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
        defaults.RECOUNT3_METADATA_FILE.format(project_name.lower(),
                                               samples_category)

    #-----------------------------------------------------------------#

    # If no working directory was specified
    if wd is None:

        # The working directory will be the current working directory.
        wd = os.getcwd()

    # Otherwise
    else:
        
        # Create it if it does not exist.
        os.makedirs(wd,
                    exist_ok = True)

    #-----------------------------------------------------------------#

    # Set the path to the file.
    f_metadata_path = os.path.join(wd, f_metadata_name)

    #-----------------------------------------------------------------#

    # If the file already exists in the working directory
    if os.path.exists(f_metadata_path):

        # Inform the user that the file exists.
        infostr = \
            f"'{f_metadata_name}' already exists in '{wd}'. " \
            "The metadata will be read from this file."
        logger.info(infostr)
        
        # Read the file content into a data frame.
        df_metadata = pd.read_csv(f_metadata_path,
                                  sep = "\t",
                                  index_col = "external_id",
                                  compression = "gzip",
                                  low_memory = False,
                                  dtype = str,
                                  keep_default_na = False)
        
        # Return the data frame.
        return df_metadata

    #-----------------------------------------------------------------#

    # Otherwise
    else:

        # Get the URL where to find the metadata.
        metadata_url = \
            defaults.RECOUNT3_METADATA_URL.format(\
                project_name.lower(),
                samples_category[-2:],
                samples_category,
                f"{project_name.lower()}.{project_name.lower()}",
                samples_category)
            
        # Log the URL.
        logger.info(f"Retrieving metadata from: {metadata_url}.")

        #-------------------------------------------------------------#

        # Read the file's content into a data frame.
        df_metadata = pd.read_csv(metadata_url,
                                  sep = "\t",
                                  index_col = "external_id",
                                  compression = "gzip",
                                  low_memory = False,
                                  dtype = str,
                                  keep_default_na = False)
        
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
            
            # Inform the user that attributes were found.
            infostr = \
                f"{entity.capitalize()}s' attributes were found in " \
                "the metadata (see below)."
            logger.info(infostr)

            #---------------------------------------------------------#

            # Define a function to parse the attributes' column
            # in the metadata.
            def parse_attributes(attr_str: str) -> dict[str, str]:
                return dict((item.split(";;")[0].replace(" ", "_"), \
                             item.split(";;")[1]) for item \
                             in str(attr_str).split("|") \
                             if item != "nan" and ";;" in item)

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
            
            # Convert all metadata columns to string type, to avoid
            # issues with missing values.
            df_metadata = df_metadata.astype(str)

    #-----------------------------------------------------------------#

    # If the user wants to save the metadata
    if save_metadata:

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


def get_read_counts(df_raw_counts: pd.DataFrame,
                    avg_mapped_read_length: pd.Series,
                    do_round: bool = True) -> pd.DataFrame:
    """Compute read counts from Recount3's raw counts when SAMPLES
    are ROWS and GENES are COLUMNS.

    Each row (sample) is divided by its corresponding average mapped
    read length.

    Parameters
    ----------
    df_raw_counts : :class:`pandas.DataFrame`
        A data frame of raw counts, with samples as rows and genes
        as columns. Row indices are sample IDs.
    
    avg_mapped_read_length : :class:`pandas.Series`
        Per-sample average mapped read length,
        indexed by sample IDs (matches ``df_raw_counts.index``).
    
    do_round : :class:`bool`, :obj:`True`
        If :obj:`True`, round to 0 decimals (banker's rounding,
        like R).

    Returns
    -------
    :class:`pandas.DataFrame`
        A data frame containing the read counts, with the
        same shape/index/columns as ``df_raw_counts``.
    """

    # Align the series to the data frame's index.
    aml = avg_mapped_read_length.reindex(df_raw_counts.index)

    # If there are any missing values
    if aml.isnull().any():

        # Get the sample IDs with missing values.
        missing = aml[aml.isnull()].index.tolist()

        # Raise an error.
        raise ValueError(
            "Average mapped read length missing for samples: " \
            f"{missing}")

    # Divide each row by its sample-specific length.
    df_read_counts = df_raw_counts.div(aml,
                                       axis = 0)

    # If rounding is requested
    if do_round:

        # Round to 0 decimals (banker's rounding, like R).
        df_read_counts = df_read_counts.round(0)

    # Return the output DataFrame.
    return df_read_counts