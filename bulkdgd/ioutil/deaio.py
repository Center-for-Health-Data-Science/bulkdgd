#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-

#    deaio.py
#
#    Read a model's per-sample differential expression whether it is
#    still loose on disk or already packed into the directory's
#    'dea.zip'.
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
    "Utilities to read a model's per-sample differential expression, " \
    "loose on disk or packed into the directory's 'dea.zip'."


#######################################################################


# Import from the standard library.
import io
import os
import zipfile

# Import from third-party libraries.
import pandas as pd

# Import from the package.
from .tableio import is_parquet, resolve_name, READ_EXTENSIONS


#######################################################################


# Set the name of a packed directory's archive.
DEA_ZIP_NAME = "dea.zip"

# Set the prefix of the per-sample files.
DEA_PREFIX = "dea_"


#######################################################################


# Set the cache of open archives, by path (each archive is parsed
# once, not once per sample).
_ZIP_CACHE = {}

# Set the cache of each open archive's members' names.
_NAMES_CACHE = {}

# Set the process the caches were built in.
_CACHE_PID = None


#######################################################################


def _get_zip_path(dea_dir: str) -> str:
    """Get the path to the archive a packed directory would use.

    Parameters
    ----------
    dea_dir : :class:`str`
        The directory containing the differential expression analysis'
        results.

    Returns
    -------
    zip_path : :class:`str`
        The path to the archive.
    """

    # Return the path to the archive.
    return os.path.join(dea_dir, DEA_ZIP_NAME)


#---------------------------------------------------------------------#


def _get_archive(zip_path: str) -> tuple[zipfile.ZipFile, set]:
    """Get the open archive for a path, and its members' names, both
    cached.

    Parameters
    ----------
    zip_path : :class:`str`
        The path to the archive.

    Returns
    -------
    archive : :class:`zipfile.ZipFile`
        The open archive.

    names : :class:`set`
        The names of the archive's members.
    """

    # Use the process the caches were built in.
    global _CACHE_PID

    # Get the current process.
    pid = os.getpid()

    # If the process changed (a worker must not read through file
    # descriptors inherited from its parent)
    if _CACHE_PID != pid:

        # Empty the caches.
        _ZIP_CACHE.clear()
        _NAMES_CACHE.clear()

        # Record the process the caches now belong to.
        _CACHE_PID = pid

    #-----------------------------------------------------------------#

    # Get the archive, if it was already opened.
    archive = _ZIP_CACHE.get(zip_path)

    # If the archive was not opened yet
    if archive is None:

        # Open it.
        archive = zipfile.ZipFile(zip_path)

        # Cache it, together with its members' names.
        _ZIP_CACHE[zip_path] = archive
        _NAMES_CACHE[zip_path] = set(archive.namelist())

    #-----------------------------------------------------------------#

    # Return the archive and its members' names.
    return archive, _NAMES_CACHE[zip_path]


#---------------------------------------------------------------------#


def forget_archive(zip_path: str) -> None:
    """Close an archive cached for reading and drop it from the
    cache, so the next read opens it again.

    Parameters
    ----------
    zip_path : :class:`str`
        The path to the archive.
    """

    # Take the archive out of the cache, if it is there.
    archive = _ZIP_CACHE.pop(zip_path, None)

    # Drop its members' names.
    _NAMES_CACHE.pop(zip_path, None)

    # If it was cached
    if archive is not None:

        # Close it.
        archive.close()


#---------------------------------------------------------------------#


def has_sample(dea_dir: str,
               sample: str,
               prefix: str = DEA_PREFIX) -> bool:
    """Return whether a sample's differential expression is present,
    packed or loose.

    Parameters
    ----------
    dea_dir : :class:`str`
        The directory containing the differential expression analysis'
        results.

    sample : :class:`str`
        The sample's name.

    prefix : :class:`str`, ``"dea_"``
        The prefix the per-sample files are named with.

    Returns
    -------
    has_sample : :class:`bool`
        Whether the sample is present.
    """

    # Get the path to the archive.
    zip_path = _get_zip_path(dea_dir)

    #-----------------------------------------------------------------#

    # If the directory is packed
    if os.path.exists(zip_path):

        # Get the archive's members' names.
        _, names = _get_archive(zip_path)

        # Return whether the sample is present, in either format.
        return resolve_name(f"{prefix}{sample}", names) is not None

    #-----------------------------------------------------------------#

    # Get the files in the directory, if it exists.
    listing = os.listdir(dea_dir) if os.path.isdir(dea_dir) else []

    # Return whether the sample is present, in either format.
    return resolve_name(f"{prefix}{sample}", listing) is not None


#---------------------------------------------------------------------#


def read_dea(dea_dir: str,
             sample: str,
             prefix: str = DEA_PREFIX,
             **read_csv_kwargs) -> pd.DataFrame:
    """Read one sample's differential expression, from the archive if
    the directory is packed and from the loose file otherwise.

    Parameters
    ----------
    dea_dir : :class:`str`
        The directory containing the differential expression analysis'
        results.

    sample : :class:`str`
        The sample's name.

    prefix : :class:`str`, ``"dea_"``
        The prefix the per-sample files are named with.

    **read_csv_kwargs
        The keyword arguments to be passed to
        :func:`pandas.read_csv`.

    Returns
    -------
    df_dea : :class:`pandas.DataFrame` or :obj:`None`
        The sample's statistics, or :obj:`None` if the sample is not
        present.
    """

    # Get the path to the archive.
    zip_path = _get_zip_path(dea_dir)

    #-----------------------------------------------------------------#

    # If the directory is packed
    if os.path.exists(zip_path):

        # Get the archive and its members' names.
        archive, names = _get_archive(zip_path)

        # Get the sample's member, in whichever format it was stored.
        member = resolve_name(f"{prefix}{sample}", names)

        # If the sample is not in the archive
        if member is None:

            # Return nothing.
            return None

        # Open the member.
        with archive.open(member) as handle:

            # Read it in full (the handle is not seekable).
            data = io.BytesIO(handle.read())

            # Return the parsed data.
            return (pd.read_parquet(data) if is_parquet(member)
                    else pd.read_csv(data, **read_csv_kwargs))

    #-----------------------------------------------------------------#

    # Get the files in the directory, if it exists.
    listing = os.listdir(dea_dir) if os.path.isdir(dea_dir) else []

    # Get the sample's file, in whichever format it was stored.
    member = resolve_name(f"{prefix}{sample}", listing)

    # If the file is not there
    if member is None:

        # Return nothing.
        return None

    # Get the path to the loose file.
    path = os.path.join(dea_dir, member)

    # Read the file.
    return (pd.read_parquet(path) if is_parquet(path)
            else pd.read_csv(path, **read_csv_kwargs))


#---------------------------------------------------------------------#


def list_samples(dea_dir: str,
                 prefix: str = DEA_PREFIX) -> list:
    """List the samples a directory holds, packed or loose.

    Parameters
    ----------
    dea_dir : :class:`str`
        The directory containing the differential expression analysis'
        results.

    prefix : :class:`str`, ``"dea_"``
        The prefix the per-sample files are named with.

    Returns
    -------
    samples : :class:`list`
        The samples' names, sorted.
    """

    # Get the path to the archive.
    zip_path = _get_zip_path(dea_dir)

    #-----------------------------------------------------------------#

    # If the directory is packed
    if os.path.exists(zip_path):

        # Get the archive's members' names.
        _, names = _get_archive(zip_path)

    # Otherwise, if the directory exists
    elif os.path.isdir(dea_dir):

        # Get the names of the files it contains.
        names = os.listdir(dea_dir)

    # Otherwise
    else:

        # There are no samples.
        return []

    #-----------------------------------------------------------------#

    # Initialize the set of samples' names (a directory may hold both
    # formats).
    out = set()

    # For each name
    for name in names:

        # If the name does not start with the prefix
        if not name.startswith(prefix):

            # Skip it.
            continue

        # For each supported extension
        for ext in READ_EXTENSIONS:

            # If the name has the extension
            if name.endswith(ext):

                # Add the part between the prefix and the extension.
                out.add(name[len(prefix):-len(ext)])

                # Stop looking.
                break

    # Return the samples' names, sorted.
    return sorted(out)
