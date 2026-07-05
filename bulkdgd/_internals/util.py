#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-

#    util.py
#
#    Miscellanea utilities.
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
__doc__ = "Miscellanea utilities."


#######################################################################


# Import from the standard library.
from collections.abc import Callable
import copy
import logging as log
import os
import re
from typing import Optional


#######################################################################


# Get the module's logger.
logger = log.getLogger(__name__)


#######################################################################


def uniquify_file_path(file_path: str) -> str:
    """If ``file_path`` exists, number it uniquely.

    Parameters
    ----------
    file_path : :class:`str`
        The file path.

    Returns
    -------
    unique_file_path : :class:`str`
        A unique file path generated from the original file path.
    """
    
    # Get the file's name and extension.
    file_name, file_ext = os.path.splitext(file_path)

    # Set the counter to 1.
    counter = 1

    # If the file already exists
    while os.path.exists(file_path):

        # Set the path to the new unique file.
        file_path = file_name + "_" + str(counter) + file_ext

        # Update the counter.
        counter += 1

    # Return the new path.
    return file_path


def load_list(list_file: str) -> list[str]:
    """Load a list of newline-separated entities from a plain text
    file.

    Parameters
    ----------
    list_file : :class:`str`
        The plain text file containing the entities of interest.

    Returns
    -------
    list_entities : :class:`list`
        The list of entities.
    """

    # Return the list of entities from the file (exclude blank
    # and comment lines).
    return \
        [line.rstrip("\n") for line in open(list_file, "r") \
         if (not line.startswith("#") \
             and not re.match(r"^\s*$", line))]


def recursive_map_dict(d: dict[str, object],
                       func: Callable,
                       keys: \
                        Optional[list[str] | set[str] | \
                                 tuple[str, ...]] = None) -> \
                        dict[str, object]:
    """Recursively traverse a (possibly nested) dictionary mapping a
    function to the dictionary's leaf values (the function substitutes
    the values with the return value of the function applied to those
    values).

    Parameters
    ----------
    d : :class:`dict`
        The input dictionary.

    func : any callable
        A callable taking as keyword arguments the values of a
        dictionary and returning a single value.

    keys : :class:`list` or :class:`set` or :class:`tuple`, optional
        A list of specific keys on whose items the mapping should be
        performed.

        This means that all values associated with keys different
        from those in the list will not be affected.

        If :const:`None`, all keys and associated values will be
        considered.
    
    Returns
    -------
    new_d : :class:`dict`
        The new dictionary.
    """

    # Define the recursion.
    def recurse(d,
                func,
                keys):

        # If the current object is a dictionary
        if isinstance(d, dict):
            
            # Get the keys of the items on which the mapping will be
            # performed. If no keys are passed, all keys in the
            # dictionary will be considered.
            sel_keys = keys if keys else d.keys()

            # For each key, value pair in the dictionary
            for k, v in list(d.items()):

                # If the value is a dictionary
                if isinstance(v, dict):

                    # If the key is in the selected keys
                    if k in sel_keys:

                        # Substitute the value with the return value
                        # of 'func' applied to it.
                        d[k] = func(**v)
                    
                    # Otherwise
                    else:

                        # Recursively check the sub-dictionaries
                        # in the current dictionary.
                        recurse(d = v,
                                func = func,
                                keys = sel_keys)

    #-----------------------------------------------------------------#

    # Create a copy of the input dictionary.
    new_d = copy.deepcopy(d)

    #-----------------------------------------------------------------#

    # Recurse through the new dictionary.
    recurse(d = new_d,
            func = func,
            keys = keys)

    #-----------------------------------------------------------------#

    # Return the new dictionary.
    return new_d


def recursive_add(d: dict[str, object],
                  d2: dict[str, object],
                  keys: list[str] | set[str] | tuple[str, ...]) -> \
                    dict[str, object]:
    """Recursively add all elements from a (possibly nested) dictionary 
    to another (possibly nested) dictionary in specific places.

    Parameters
    ----------
    d : :class:`dict`
        The input dictionary.

    d2 : :class:`dict`
        The dictionary whose elements should be added to the input
        dictionary.

    keys : :class:`list` or :class:`set` or :class:`tuple`
        The keys corresponding to the places where the key, value
        pairs contained in ``d2`` will be added to the input
        dictionary.

    Returns
    -------
    new_d : :class:`dict`
        The updated dictionary.
    """

    # Define the recursion.
    def recurse(d,
                d2,
                keys):

        # If first dictionary is in fact a dictionary
        if isinstance(d, dict):

            # For each key in the first dictionary
            for key in d:
                
                # If the key is among the selected keys
                if key in keys:
                    
                    # If the associated value is a dictionary and the
                    # second dictionary is in fact a dictionary
                    if isinstance(d[key], dict) \
                    and isinstance(d2, dict):

                        # For each key, value pair in the second
                        # dictionary
                        for k, v in d2.items():

                            # If the key is not among the keys in the
                            # value associated with 'key' in the
                            # fist dictionary
                            if k not in d[key]:

                                # Add the key and associated value to
                                # the first dictionary.
                                d[key][k] = v
                    

                # If the value associated with they key is a dictionary
                if isinstance(d[key], dict):

                    # Recurse through the dictionary.
                    recurse(d = d[key],
                            d2 = d2,
                            keys = keys)

    #-----------------------------------------------------------------#

    # Create a copy of the input dictionary.
    new_d = copy.deepcopy(d)

    #-----------------------------------------------------------------#

    # Recurse through the copy of the dictionary.
    recurse(d = new_d,
            d2 = d2,
            keys = keys)

    #-----------------------------------------------------------------#

    # Return the modified dictionary.
    return new_d


def recursive_add_items(d: dict[str, object],
                        paths2values: dict[tuple[str, ...], object]) \
                            -> dict[str, object]:
    """Recursively add a new value to the key at the end of a
    each "key path" in a (possibly nested) dictionary.

    Parameters
    ----------
    d : :class:`dict`
        The input dictionary.

    paths2values : :class:`dict`
        A dictionary mapping "key paths" to values. Each "key path" is
        a tuple of keys leading to the key to which the value should be
        added. The value associated with each "key path" is the value
        to be added to the key at the end of the "key path".

    Returns
    -------
    new_d : :class:`dict`
        The updated dictionary.
    """

    # Define the recursion.
    def recurse(d,
                key_path,
                value):

        # If the key path is empty
        if not key_path:
            
            # Return
            return

        # Get the first key in the path.
        key = key_path[0]

        # If the key is the last key in the path
        if len(key_path) == 1:
            
            # If the key is not in the dictionary
            if key not in d:

                # Add the key and associated value to the dictionary.
                d[key] = value

        # Otherwise
        else:
            
            # If the key is not in the dictionary
            if key not in d:
                
                # Create a new dictionary if the key does not exist
                d[key] = {}
            
            # If the value associated with the key is not a dictionary
            elif not isinstance(d[key], dict):
                
                # Raise an error
                errstr = \
                    "It was not possible to traverse into key " \
                    f"'{key}' because the associated value is not " \
                    " a dictionary."
                raise ValueError(errstr)

            # Recurse into the next level.
            recurse(d = d[key],
                    key_path = key_path[1:],
                    value = value)

    #-----------------------------------------------------------------#

    # Create a copy of the input dictionary.
    new_d = copy.deepcopy(d)

    #-----------------------------------------------------------------#

    # For each key path and associated value in the input dictionary
    for key_path, value in paths2values.items():

        # Recurse through the copy of the dictionary.
        recurse(d = new_d,
                key_path = key_path,
                value = value)

    #-----------------------------------------------------------------#

    # Return the modified dictionary.
    return new_d


def recursive_get(d: dict[str, object],
                  key_path: tuple[str, ...] | list[str] | set[str]) \
                    -> object:
    """Recursively get an item from a (possibly nested) dictionary 
    given the item's ``key path``.

    Parameters
    ----------
    d : :class:`dict`
        The input dictionary.

    key_path : :class:`list` or :class:`set` or :class:`tuple`
        The "key path" leading to the item of interest.

    Returns
    -------
    item : any object
        The item of interest.
    """

    # Define the recursion.
    def recurse(d,
                key_path):

        # If the key  path is empty
        if not key_path:

            # Return None
            return None
        
        # Get the first key in the path.
        key = key_path[0]

        # If the key is in the dictionary
        if key in d:

            # If this is the last key in the path
            if len(key_path) == 1:

                # Return the value.
                return d[key]

            # Otherwise, if the value associated with the current key
            # is a dictionary
            elif isinstance(d[key], dict):

                # Continue the recursion with the next level of the
                # dictionary.
                return recurse(d = d[key],
                               key_path = key_path[1:])

        # If the key is not found, return None.
        return None

    #-----------------------------------------------------------------#

    # Return the result of the recursion.
    return recurse(d = d,
                   key_path = key_path)


def recursive_merge_dicts(*dicts: dict[str, object]) -> \
        dict[str, object]:
    """Recursively merge several (possibly nested) dictionaries.

    Parameters
    ----------
    *dicts : multiple :class:`dict`
        The dictionaries to be merged.

    Returns
    -------
    merged : :class:`dict`
        A dictionary representing the result of the merging.
    """

    # Define a recursive function to merge two dictionaries at a time.
    def merge_two_dicts(d1,
                        d2):

        # For each key, value pair in the second dictionary
        for k, v in d2.items():

            # If the key is in the first dictionary and the associated
            # value in both dictionaries is another dictionary
            if k in d1 \
            and isinstance(d1[k], dict) and isinstance(v, dict):

                # Recursively merge the associated values.
                d1[k] = merge_two_dicts(d1[k], v)

            # Otherwise
            else:

                # The value associated to the key in the first
                # dictionary will be the value associated to the key
                # in the second dictionary.
                d1[k] = v

        # Return the updated first dictionary.
        return d1

    #-----------------------------------------------------------------#

    # Initialize an empty dictionary to store the result of the
    # merging.
    merged = {}

    #-----------------------------------------------------------------#

    # For each provided dictionary
    for d in dicts:

        # Recursively merge it with the current result.
        merged = merge_two_dicts(merged, d)

    #-----------------------------------------------------------------#

    # Return the result of the merging.
    return merged


def kwargs_to_dict(kwargs: dict[str, object]) -> dict[str, object]:
    """Convert a dictionary of keyword arguments to a nested
    dictionary.

    Parameters
    ----------
    kwargs : :class:`dict`
        The dictionary of keyword arguments.

    Returns
    -------
    d : :class:`dict`
        The final nested dictionary.
    """

    # Create an empty dictionary.
    d = {}

    # For each key-value pair in the dictionary of keyword arguments
    for key, value in kwargs.items():

        # Split the key into parts.
        parts = key.split("_")

        # Initialize the nested dictionary.
        d_init = d

        # For each part in the key
        for part in parts[:-1]:

            # If the part is not in the dictionary, add it.
            d_init = d_init.setdefault(part, {})
        
        # Add the value to the dictionary.
        d_init[parts[-1]] = value
    
    # Return the dictionary.
    return d