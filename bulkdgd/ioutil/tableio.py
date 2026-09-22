#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-

#    tableio.py
#
#    Utilities to read and write tables as Parquet or text.
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
__doc__ = "Utilities to read and write tables as Parquet or text."


#######################################################################


# Import from third-party libraries.
import pandas as pd


#######################################################################


# Set the extensions of Parquet files.
PARQUET_EXTENSIONS = (".parquet", ".pq")

# Set the default extension of new tables.
DEFAULT_TABLE_EXT = ".parquet"

# Set the extensions to try, in order, when reading a table of unknown
# format.
READ_EXTENSIONS = PARQUET_EXTENSIONS + (".csv",)


#######################################################################


def is_parquet(file_path):
    """Return whether a file is a Parquet file, by its extension.

    Parameters
    ----------
    file_path : :class:`str`
        The file's path.

    Returns
    -------
    is_parquet : :class:`bool`
        Whether the file is a Parquet file.
    """

    # Return whether the file has a Parquet extension.
    return str(file_path).lower().endswith(PARQUET_EXTENSIONS)


#---------------------------------------------------------------------#


def save_table(df,
               file_path,
               sep = ",",
               index = True,
               header = True,
               compression = "infer"):
    """Write a table as Parquet or as text, by the path's extension.

    Parameters
    ----------
    df : :class:`pandas.DataFrame`
        The table to write.

    file_path : :class:`str`
        The output file ('.parquet' or '.pq' for Parquet, delimited
        text otherwise).

    sep : :class:`str`, ``","``
        The column separator (text only).

    index : :class:`bool`, ``True``
        Whether to write the index.

    header : :class:`bool`, ``True``
        Whether to write the column names (text only; Parquet always
        writes them).

    compression : :class:`str`, ``"infer"``
        The compression of the text file (``"infer"`` takes it from
        the path's extension).
    """

    # If the file is a Parquet file
    if is_parquet(file_path):

        # Write the table (the column names are always written).
        df.to_parquet(file_path,
                      engine = "pyarrow",
                      compression = "snappy",
                      index = index)

    # Otherwise
    else:

        # Write the table.
        df.to_csv(file_path,
                  sep = sep,
                  index = index,
                  header = header,
                  compression = compression)


#---------------------------------------------------------------------#


def load_table(file_path,
               sep = ",",
               index_col = None):
    """Read a table written by :func:`save_table`.

    Parameters
    ----------
    file_path : :class:`str`
        The input file.

    sep : :class:`str`, ``","``
        The column separator (text only).

    index_col : :class:`int` or :class:`str`, optional
        The column to use as index (text only).

    Returns
    -------
    df : :class:`pandas.DataFrame`
        The table.
    """

    # If the file is a Parquet file
    if is_parquet(file_path):

        # Return the table.
        return pd.read_parquet(file_path)

    # Return the table.
    return pd.read_csv(file_path,
                       sep = sep,
                       index_col = index_col)


#---------------------------------------------------------------------#


def table_name(stem,
               ext = None):
    """Get the file name of a new table.

    Parameters
    ----------
    stem : :class:`str`
        The file name without extension.

    ext : :class:`str`, optional
        The extension (``DEFAULT_TABLE_EXT`` if not given).

    Returns
    -------
    name : :class:`str`
        The file name.
    """

    # Return the file name.
    return f"{stem}{DEFAULT_TABLE_EXT if ext is None else ext}"


#---------------------------------------------------------------------#


def resolve_name(stem,
                 available):
    """Get which of a stem's possible file names is available,
    trying the extensions in ``READ_EXTENSIONS`` in order.

    Parameters
    ----------
    stem : :class:`str`
        The file name without extension.

    available : any container
        The available file names (e.g., a directory listing or a set
        of archive members' names).

    Returns
    -------
    name : :class:`str` or :obj:`None`
        The file name, or :obj:`None` if none is available.
    """

    # For each extension
    for ext in READ_EXTENSIONS:

        # Get the file name.
        name = f"{stem}{ext}"

        # If the file name is available
        if name in available:

            # Return it.
            return name

    # Return None if no file name is available.
    return None
