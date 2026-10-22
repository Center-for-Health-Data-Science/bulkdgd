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

# Set the module's public names.
__all__ = \
    ["get_gene_sums",
     "get_qc",
     "get_metadata",
     "get_read_counts",
     "get_geo_metadata",
     "get_ena_metadata",
     "get_run_selector"]


#######################################################################


# Import from the standard library.
import gzip
import io
import logging as log
import os
import time
from typing import Optional

# Import from third-party libraries.
import pandas as pd
import requests as rq

# Import from 'bulkdgd'.
from ..ioutil.tableio import save_table
from . import defaults


#######################################################################


# Get the module's logger.
logger = log.getLogger(__name__)


########################## PRIVATE FUNCTIONS ##########################


def _to_strings(df: pd.DataFrame) -> pd.DataFrame:
    """Convert a data frame's values to strings, with the missing
    values as ``"nan"``.

    Parameters
    ----------
    df : :class:`pandas.DataFrame`
        The data frame.

    Returns
    -------
    df : :class:`pandas.DataFrame`
        The data frame with all values as strings.
    """

    # Return the values as strings, with the missing values as 'nan'.
    return df.astype(object).fillna("nan").astype(str)


def _download_file(url: str,
                   file_path: str) -> None:
    """Download a file, writing it to its path only once it is
    complete.

    Parameters
    ----------
    url : :class:`str`
        The file's URL.

    file_path : :class:`str`
        The path to the file.
    """

    # Set the path to the partial file.
    part_path = f"{file_path}.part"

    # Get the file, failing on an HTTP error.
    response = rq.get(url,
                      stream = True,
                      timeout = defaults.REQUEST_TIMEOUT)
    response.raise_for_status()

    # Open the partial file.
    with open(part_path,
              "wb") as out:

        # For each chunk of the file
        for chunk in response.iter_content(\
                chunk_size = 1024 * 1024):

            # Write the chunk.
            out.write(chunk)

    # Move the complete file to its path.
    os.replace(part_path,
               file_path)


########################## PUBLIC FUNCTIONS ###########################


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

        # Get the supported releases as a string.
        releases_str = \
            ", ".join(map(str,
                          defaults.RECOUNT3_GENCODE_RELEASES))

        # Raise an error.
        errstr = \
            "The given Gencode release is not supported in " \
            f"Recount3. Supported releases are: {releases_str}."
        raise ValueError(errstr)

    #-----------------------------------------------------------------#

    # Set the name of the file that will contain the RNA-seq data.
    f_gene_sums_name = \
        defaults.RECOUNT3_GENE_SUMS_FILE.format(
            project_name.lower(),
            samples_category)

    #-----------------------------------------------------------------#

    # If no working directory was specified
    if wd is None:

        # Use the current working directory.
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

        # Read the data from the file.
        source = f_gene_sums_path

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

        # Read the data from the URL.
        source = gene_sums_url

        # If the user wants to save the original file
        if save_gene_sums:

            # Download the file.
            _download_file(url = gene_sums_url,
                           file_path = f_gene_sums_path)

            # Inform the user that the data were saved.
            infostr = \
                "The RNA-seq data were successfully saved " \
                f"in '{f_gene_sums_path}'."
            logger.info(infostr)

            # Read the data from the saved file.
            source = f_gene_sums_path

    #-----------------------------------------------------------------#

    # Read the data into a data frame, with the samples as rows.
    df_gene_sums = pd.read_csv(source,
                               sep = "\t",
                               skiprows = 2,
                               index_col = 0,
                               compression = "gzip",
                               low_memory = False).T

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

    save_qc : :class:`bool`, :obj:`True`
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

        # Use the current working directory.
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

        # Read the data from the file.
        source = f_qc_path

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

        # Read the data from the URL.
        source = qc_url

        # If the user wants to save the original file
        if save_qc:

            # Download the file.
            _download_file(url = qc_url,
                           file_path = f_qc_path)

            # Inform the user that the data were saved.
            infostr = \
                "The QC metadata were successfully saved " \
                f"in '{f_qc_path}'."
            logger.info(infostr)

            # Read the data from the saved file.
            source = f_qc_path

    #-----------------------------------------------------------------#

    # Read the data into a data frame.
    df_qc = pd.read_csv(source,
                        sep = "\t",
                        compression = "gzip",
                        low_memory = False,
                        index_col = "external_id")

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
        If :obj:`True`, save the metadata, with the attributes split
        into columns, in the working directory.

    wd : :class:`str`, optional
        The working directory where the metadata file will be saved,
        if ``save_metadata`` is :obj:`True`.

        If not specified, it will be the current working directory.

    Returns
    -------
    df_metadata : :class:`pandas.DataFrame`
        A data frame containing the metadata for the samples associated
        with the given category.

    Notes
    -----
    The returned metadata also have the ``"recount3_project_name"``
    and ``"recount3_samples_category"`` columns, containing
    ``project_name`` and ``samples_category``.
    """

    # Set the name of the file that will contain the metadata.
    f_metadata_name = \
        defaults.RECOUNT3_METADATA_FILE.format(project_name.lower(),
                                               samples_category)

    #-----------------------------------------------------------------#

    # If no working directory was specified
    if wd is None:

        # Use the current working directory.
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

        # Get the file, failing on an HTTP error.
        response = rq.get(metadata_url,
                          timeout = defaults.REQUEST_TIMEOUT)
        response.raise_for_status()

        # Read the file's content into a data frame.
        df_metadata = pd.read_csv(io.BytesIO(response.content),
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
                """Parse a string of attributes.

                Parameters
                ----------
                attr_str : :class:`str`
                    The attributes, as ``"name;;value"`` items
                    separated by ``"|"``.

                Returns
                -------
                attrs : :class:`dict`
                    The attributes' names mapped to their values.
                """

                # Return the attributes.
                return dict((item.split(";;")[0].replace(" ", "_"), \
                             item.split(";;")[1]) for item \
                             in str(attr_str).split("|") \
                             if item != "nan" and ";;" in item)

            # Parse the attributes into a data frame, one column per
            # attribute.
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

            # Get the attributes already present as columns in the
            # metadata.
            attrs_to_drop = \
                df_attrs.columns[\
                    [col_name in df_metadata.columns \
                     for col_name in df_attrs.columns]]

            # Drop them from the data frame of attributes.
            df_attrs = df_attrs.drop(labels = attrs_to_drop,
                                     axis = 1)

            # Add the attributes' columns to the metadata.
            df_metadata = df_metadata.join(df_attrs)

            # Drop the original column from the data frame containing
            # the metadata.
            df_metadata = df_metadata.drop(labels = [column_attrs],
                                           axis = 1)

            # Convert all the columns to strings (missing values become
            # 'nan').
            df_metadata = _to_strings(df_metadata)

    #-----------------------------------------------------------------#

    # If the user wants to save the metadata
    if save_metadata:

        # Write the data frame to a partial file.
        save_table(df_metadata,
                   f"{f_metadata_path}.part",
                   sep = "\t",
                   compression = "gzip")

        # Move the complete file to its path.
        os.replace(f"{f_metadata_path}.part",
                   f_metadata_path)

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
    """Compute read counts from Recount3's raw counts by dividing each
    sample's counts by its average mapped read length.

    Parameters
    ----------
    df_raw_counts : :class:`pandas.DataFrame`
        A data frame of raw counts, with samples as rows and genes
        as columns. Row indices are sample IDs.

    avg_mapped_read_length : :class:`pandas.Series`
        The samples' average mapped read lengths, indexed like the
        rows of ``df_raw_counts``.

    do_round : :class:`bool`, :obj:`True`
        If :obj:`True`, round to 0 decimals (banker's rounding,
        like R).

    Returns
    -------
    df_read_counts : :class:`pandas.DataFrame`
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

    # Return the read counts.
    return df_read_counts


def get_geo_metadata(geo_code: str,
                     wd: Optional[str] = None) -> pd.DataFrame:
    """Get the samples' metadata from a GEO series' SOFT file.

    Parameters
    ----------
    geo_code : :class:`str`
        The GEO series' code (for instance, ``"GSE123456"``).

    wd : :class:`str`, optional
        The directory where the SOFT file is saved and, if already
        there, read from. If not passed, the file is not saved.

    Returns
    -------
    df_geo : :class:`pandas.DataFrame`
        A data frame with one row per sample and the samples'
        attributes as columns.
    """

    # Get the URL of the SOFT file.
    geo_url = \
        defaults.GEO_SOFT_URL.format(geo_code[:-3],
                                     geo_code,
                                     geo_code)

    #-----------------------------------------------------------------#

    # If a directory was passed
    if wd is not None:

        # Create it if it does not exist.
        os.makedirs(wd,
                    exist_ok = True)

        # Get the path to the SOFT file.
        geo_path = \
            os.path.join(wd,
                         defaults.GEO_SOFT_FILE.format(geo_code))

        # If the file is not there yet
        if not os.path.exists(geo_path):

            # Download the file.
            _download_file(url = geo_url,
                           file_path = geo_path)

            # Inform the user that the file was saved.
            infostr = f"The GEO SOFT file was saved in '{geo_path}'."
            logger.info(infostr)

        # Otherwise
        else:

            # Inform the user that the file will be read.
            infostr = \
                f"'{geo_path}' already exists. The GEO metadata " \
                "will be read from this file."
            logger.info(infostr)

        # Open the file.
        f_geo = open(geo_path,
                     "rb")

    #-----------------------------------------------------------------#

    # Otherwise
    else:

        # Get the file, failing on an HTTP error.
        response = rq.get(geo_url,
                          stream = True,
                          timeout = defaults.REQUEST_TIMEOUT)
        response.raise_for_status()

        # Read the file from the response's stream.
        f_geo = response.raw

    #-----------------------------------------------------------------#

    # Initialize the list of samples.
    samples = []

    # Initialize the sample being parsed.
    current = None

    # Open the compressed file.
    with gzip.GzipFile(fileobj = f_geo,
                       mode = "rb") as f:

        # For each line
        for line in f:

            # Decode it and remove the trailing newline.
            line = line.decode("utf-8").rstrip("\n")

            # If a new sample starts
            if line.startswith("^SAMPLE"):

                # If a sample was being parsed
                if current is not None:

                    # Add it to the list of samples.
                    samples.append(current)

                # Start the new sample with its GEO accession.
                current = \
                    {"Sample_geo_accession" : \
                        line.split(" = ",
                                   1)[1]}

            # If the line holds one of the sample's characteristics
            elif line.startswith("!Sample_characteristics") \
            and current is not None:

                # Get the characteristic.
                _, rest = line[1:].split(" = ",
                                         1)

                # If the characteristic is labeled
                if ": " in rest:

                    # Split it into label and value at the last ': '.
                    subkey, subval = rest.rsplit(": ",
                                                 1)

                    # Add the characteristic.
                    current[subkey.strip()] = subval.strip()

                # Otherwise
                else:

                    # Add it to the unlabeled characteristics.
                    current.setdefault("characteristics",
                                       []).append(rest.strip())

            # If the line holds another field of the sample
            elif line.startswith("!Sample_") and current is not None:

                # Get the field's name and value.
                key, val = line[1:].split(" = ",
                                          1)

                # If the field is a characteristic (handled above)
                if key.startswith("Sample_characteristics"):

                    # Go to the next line.
                    continue

                # If the field was already found
                if key in current:

                    # Append the value to the existing one.
                    current[key] += "; " + val

                # Otherwise
                else:

                    # Add the field.
                    current[key] = val

        # If a sample was being parsed
        if current is not None:

            # Add it to the list of samples.
            samples.append(current)

    # If the file was opened from the directory
    if wd is not None:

        # Close it.
        f_geo.close()

    #-----------------------------------------------------------------#

    # Build the data frame.
    df_geo = pd.DataFrame(samples)

    # Make the values strings, leaving the missing ones missing.
    df_geo = df_geo.astype(str).where(df_geo.notna())

    # Put the GEO accession first.
    columns = \
        ["Sample_geo_accession"] + \
        [c for c in df_geo.columns if c != "Sample_geo_accession"]

    # Return the data frame.
    return df_geo[columns]


def get_ena_metadata(sra_code: str,
                     wd: Optional[str] = None) -> pd.DataFrame:
    """Get each run's experiment, BioSample and library details for
    an SRA study from ENA.

    Parameters
    ----------
    sra_code : :class:`str`
        The SRA study's code (for instance, ``"SRP030628"``).

    wd : :class:`str`, optional
        The directory where ENA's report is saved and, if already
        there, read from. If not passed, the report is not saved.

    Returns
    -------
    df_ena : :class:`pandas.DataFrame`
        A data frame indexed by run accession, with the fields in
        ``defaults.ENA_FILEREPORT_FIELDS`` as columns. It is empty if
        ENA has nothing for the study or could not be reached.
    """

    # Get the URL of the report.
    ena_url = \
        defaults.ENA_FILEREPORT_URL.format(\
            sra_code,
            ",".join(defaults.ENA_FILEREPORT_FIELDS))

    # Build the empty result.
    df_empty = \
        pd.DataFrame(\
            columns = defaults.ENA_FILEREPORT_FIELDS).set_index(\
                "run_accession",
                drop = False)

    #-----------------------------------------------------------------#

    # Try to get the report.
    try:

        # If a directory was passed
        if wd is not None:

            # Create it if it does not exist.
            os.makedirs(wd,
                        exist_ok = True)

            # Get the path to the report.
            ena_path = \
                os.path.join(\
                    wd,
                    defaults.ENA_FILEREPORT_FILE.format(sra_code))

            # If the report is not there yet
            if not os.path.exists(ena_path):

                # Download the report.
                _download_file(url = ena_url,
                               file_path = ena_path)

                # Pause between requests to ENA.
                time.sleep(defaults.ENA_REQUEST_PAUSE)

            # Read the report.
            with open(ena_path,
                      "r") as f:
                text = f.read()

        # Otherwise
        else:

            # Get the report, failing on an HTTP error.
            response = rq.get(ena_url,
                              timeout = defaults.REQUEST_TIMEOUT)
            response.raise_for_status()

            # Take its text.
            text = response.text

            # Pause between requests to ENA.
            time.sleep(defaults.ENA_REQUEST_PAUSE)

    # If something went wrong
    except Exception as e:

        # Warn the user.
        warnstr = \
            "It was not possible to get the ENA report for " \
            f"'{sra_code}'. Error: {e}"
        logger.warning(warnstr)

        # Return the empty result.
        return df_empty

    #-----------------------------------------------------------------#

    # If the report is empty or has a header only
    if not text.strip() or len(text.strip().splitlines()) < 2:

        # Return the empty result.
        return df_empty

    #-----------------------------------------------------------------#

    # Parse the report, with all values as strings.
    df_ena = pd.read_csv(io.StringIO(text),
                         sep = "\t",
                         header = 0,
                         low_memory = False,
                         dtype = str,
                         keep_default_na = False)

    # Return it, indexed by run accession.
    return df_ena.set_index("run_accession",
                            drop = False)


def get_run_selector(sra_code: str,
                     wd: Optional[str] = None) -> pd.DataFrame:
    """Get an SRA study's Run Selector table.

    Parameters
    ----------
    sra_code : :class:`str`
        The SRA study's code (for instance, ``"SRP030628"``).

    wd : :class:`str`, optional
        The directory where the table is saved and, if already there,
        read from. If not passed, the table is not saved.

    Returns
    -------
    df_runs : :class:`pandas.DataFrame`
        The table, with one row per run and all values as strings.
    """

    # Get the URL of the table.
    runs_url = defaults.SRA_RUN_SELECTOR_URL.format(sra_code)

    #-----------------------------------------------------------------#

    # If a directory was passed
    if wd is not None:

        # Create it if it does not exist.
        os.makedirs(wd,
                    exist_ok = True)

        # Get the path to the table.
        runs_path = \
            os.path.join(\
                wd,
                defaults.SRA_RUN_SELECTOR_FILE.format(sra_code))

        # If the table is not there yet
        if not os.path.exists(runs_path):

            # Download the table.
            _download_file(url = runs_url,
                           file_path = runs_path)

        # Read the table.
        with open(runs_path,
                  "r") as f:
            text = f.read()

    # Otherwise
    else:

        # Get the table, failing on an HTTP error.
        response = rq.get(runs_url,
                          timeout = defaults.REQUEST_TIMEOUT)
        response.raise_for_status()

        # Take its text.
        text = response.text

    #-----------------------------------------------------------------#

    # Parse the table, with all values as strings.
    df_runs = pd.read_csv(io.StringIO(text),
                          sep = ",",
                          header = 0,
                          low_memory = False,
                          dtype = str,
                          keep_default_na = False)

    # Return the table.
    return df_runs
