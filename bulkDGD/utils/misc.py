#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-

#    misc.py
#
#    Miscellanea utility functions.
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
__doc__ = "Miscellanea utility functions."


# Standard library
import logging as log
import os
import re
# Third-party packages
import numpy as np
# bulkDGD
from . import _util


# Get the module's logger
logger = log.getLogger(__name__)


#------------------------- Helper functions --------------------------#


def get_list(list_file):
    """Get a list of entities from a plain text file.

    Parameters
    ----------
    list_file : ``str``
        File containing the list of entities of interest.

    Returns
    -------
    ``list``
        List of entities of interest.
    """

    # Return the list of entities from the file (exclude blank
    # and comment lines)
    return \
        [l.rstrip("\n") for l in open(list_file, "r") \
         if (not l.startswith("#") and not re.match(r"^\s*$", l))]


def filter_2d_array(array,
                    min_value = None,
                    max_value = None,
                    min_inclusive = True,
                    max_inclusive = True,
                    flatten = False):
    """Filter a two-dimensional array.

    Parameters
    ----------
    array : ``numpy.ndarray``
        Input array.

    min_value : ``float``, optional
        Minimum value to be kept in the array. If not
        passed, all values will be kept.

    max_value : ``float``, optional
        Maximum value to be kept in the array. If not
        passed, all values will be kept.

    min_inclusive : ``bool``, optional, default: ``True``
        Whether to keep values greater than ``min_value``
        or equal to ``min_value`` (if ``True``) or only
        values strictly greater than ``min_value``
        (if ``False``).

    max_inclusive : ``bool``, optional, default: ``True``
        Whether to keep values lower than ``max_value``
        or equal to ``max_value`` (if ``True``) or only
        values strictly lower than ``max_value``
        (if ``False``).

    flatten : `bool`, optional, default: `False`
        Whether to flatten the array.

    Returns
    -------
    ``numpy.ndarray``
        Filtered (and, possibly, flattened) array.
    """

    # If the user requested flattening of the array
    if flatten:

        # If the array is multi-dimensional
        if array.ndim > 1:

            # Inform the user that the array will be flattened
            # for plotting purposes
            logger.info(\
                f"A multi-dimensional array was passed " \
                f"(ndim = {array.ndim}). The array will be " \
                f"flattened for plotting.")

            # Flatten the array
            array = array.flatten()

        # Otherwise
        elif array.ndim == 1:

            # Warn the user that the array is already
            # one-dimensional
            logger.info(\
                f"'flatten' was set to 'True', but the array " \
                f"is already one-dimensional. Therefore, no " \
                f"flattening will be performed.")

    # Inform the user about how many values there are in
    # the array
    logger.info(\
        f"There are {len(array)} values in the array.")

    # Remove NaN values from the array
    array = array[~np.isnan(array)]

    # Inform the user of the removal of NaN values
    logger.info(\
        f"There are {len(array)} values in the array after " \
        f"removing NaN values.")
    
    # If a minimum value was specified
    if min_value is not None:

        # If the minimum value should be included
        if min_inclusive:

            # Filter with >=
            array = array[array >= min_value]

            # Inform the user about the filtering
            logger.info(\
                f"There are {len(array)} values in the array " \
                f"after keeping only values >= {min_value}.")

        # Otherwise
        else:

            # Filter with >
            array = array[array > min_value]

            # Inform the user about the filtering
            logger.info(\
                f"There are {len(array)} values in the array " \
                f"after keeping only values > {min_value}.")

    # If a maximum value was specified
    if max_value is not None:

        # If the maximum value should be included
        if max_inclusive:

            # Filter with <=
            array = array[array <= max_value]

            # Inform the user about the filtering
            logger.info(\
                f"There are {len(array)} values in the array " \
                f"after keeping only values <= {max_value}.")

        # Otherwise
        else:

            # Filter with <
            array = array[array < max_value]

            # Inform the user about the filtering
            logger.info(\
                f"There are {len(array)} values in the array " \
                f"after keeping only values < {max_value}.")

    # Return the filtered (and possibly flattened) array
    return array


#--------------------------- Configuration ---------------------------#


def get_config_rep(config_file):
    """Get the configuration used to find the representations
    and the gradients of the corresponding decoder outputs with
    respect to them.

    Parameters
    ----------
    config_file : ``str``
        YAML configuration file.

    Returns
    -------
    ``dict``
        Dictionary containing the configuration.
    """

    # Load the configuration
    config = _util.load_config(config_file = config_file)

    # Check the configuration against the template
    config = _util.check_config_against_template(\
                config = config,
                template = _util.CONFIG_TEMPLATE)

    # Get the absolute path to the configuration file
    config_file_path = os.path.abspath(config_file)

    # Split the path into its 'head' (path to the file without
    # the file name) and its 'tail' (the file name)
    path_head, path_tail = os.path.split(config_file_path)

    # Get the correct path for the PyTorch files containing the
    # parameters of the training decoder, the Gaussian mixture
    # model, and the representations'a layer (they may be given as
    # relative paths with respect to the location of the configuration
    # file, which may or may not correspond to the location from
    # where this function is called)
    config["dec"]["pth_file"] = \
        os.path.normpath(os.path.join(path_head,
                                      config["dec"]["pth_file"]))
    config["gmm"]["pth_file"] = \
        os.path.normpath(os.path.join(path_head,
                                      config["gmm"]["pth_file"]))
    config["rep_layer"]["pth_file"] = \
        os.path.normpath(os.path.join(path_head,
                                      config["rep_layer"]["pth_file"]))

    # Return the configuration
    return config


def get_config_plot(config_file):
    """Get a configuration file for a plot.

    Parameters
    ----------
    config_file : ``str``
        YAML configuration file.

    Returns
    -------
    ``dict``
        Dictionary containing the configuration.
    """

    # Load the configuration
    config = _util.load_config(config_file = config_file)

    # Create a copy of the configuration
    new_config = dict(config)
    
    # Substitute the font properties definitions
    # with the corresponding FontProperties instances
    _util.recursive_substitute_dict_with_func(\
        d = new_config,
        func = fm.FontProperties,
        keys = {"fontproperties"})

    # Return the new configuration
    return new_config
