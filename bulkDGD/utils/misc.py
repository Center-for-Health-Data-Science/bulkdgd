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
import matplotlib.font_manager as fm
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


#--------------------------- Configuration ---------------------------#


def get_config_model(config_file):
    """Get the configuration specifying the DGD model's parameters
    and the files containing the trained model.

    Parameters
    ----------
    config_file : ``str``
        The YAML configuration file.

    Returns
    -------
    ``dict``
        A dictionary containing the configuration.
    """

    # Load the configuration
    config = _util.load_config(config_file = config_file)

    # Check the configuration against the template
    config = _util.check_config_against_template(\
                config = config,
                template = _util.CONFIG_MODEL_TEMPLATE)

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


def get_config_rep(config_file):
    """Get the configuration file containing the options for data
    loading and optimization to find the best representations.

    Parameters
    ----------
    config_file : ``str``
        The YAML configuration file.

    Returns
    -------
    ``dict``
        A dictionary containing the configuration.
    """

    # Load the configuration
    config = _util.load_config(config_file = config_file)

    # Check the configuration against the template
    config = _util.check_config_against_template(\
                config = config,
                template = _util.CONFIG_REP_TEMPLATE)

    # Return the configuration
    return config


def get_config_plot(config_file):
    """Get a configuration file for a plot.

    Parameters
    ----------
    config_file : ``str``
        A YAML configuration file.

    Returns
    -------
    ``dict``
        A dictionary containing the configuration.
    """

    # Load the configuration
    config = _util.load_config(config_file = config_file)
    
    # Substitute the font properties definitions
    # with the corresponding FontProperties instances
    new_config = _util.recursive_map_dict(\
        d = config,
        func = fm.FontProperties,
        keys = {"fontproperties", "prop", "title_fontproperties"})

    # Return the new configuration
    return new_config