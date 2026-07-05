#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-

#    configio.py
#
#    Utilities to load and save configurations.
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
__doc__ = "Utilities to load and save configurations."


#######################################################################


# Import from the standard library.
import logging as log
import os
from typing import Optional

# Import from third-party libraries.
import yaml

# Import from 'bulkdgd'.
from bulkdgd import defaults
from bulkdgd.core._util import (
    parse_config_model,
    parse_config_train,
    parse_config_rep)
from bulkdgd.plotting._util import parse_config_plot


#######################################################################


# Get the module's logger.
logger = log.getLogger(__name__)


#######################################################################


# Set a mapping between the type of configuration and the function to
# parse it.
type2parsefunc = \
   {"model" : parse_config_model,
    "training" : parse_config_train,
    "representations" : parse_config_rep,
    "plotting" : parse_config_plot}


#######################################################################


def _load_config(config_file: str,
                 config_type: str) -> dict[str, object]:
    """Load a configuration from a YAML configuration file.

    Parameters
    ----------
    config_file : :class:`str`
        The YAML configuration file.

    config_type : :class:`str`
        The type of configuration to load from the file.

    Returns
    -------
    config : :class:`dict`
        The configuration.
    """

    # Get the name of the configuration file.
    config_file_name = \
        os.path.splitext(os.path.basename(config_file))[0]

    #-----------------------------------------------------------------#

    # If the configuration file is a name without extension
    if config_file == config_file_name:

        # Assume it is a configuration file in the directory storing
        # configuration files for the given type of configuration.
        config_file = os.path.join(defaults.CONFIG_DIRS[config_type],
                                   config_file_name + ".yaml")

    # Otherwise
    else:
        
        # Assume it is a file name/file path.
        config_file = os.path.abspath(config_file)

    #-----------------------------------------------------------------#

    # Load the configuration from the file.
    config = yaml.safe_load(open(config_file, "r"))

    #-----------------------------------------------------------------#

    # If the configuration type is "model", "representations",
    # "training", or "plotting"
    if config_type in \
        ["model", "representations", "training", "plotting"]:

        # Parse and check the configuration.
        if config_type == "model":
            config_validated, errors, warnings = \
                type2parsefunc[config_type](config = config,
                                            path = config_file)
        else:
            config_validated, errors, warnings = \
                type2parsefunc[config_type](config = config)

        # If there are errors in the configuration
        if errors:

            # Raise an exception.
            errstr = \
                f"The configiration loaded from '{config_file}' is  " \
                "not valid. Errors: " + "|".join(errors)
            raise ValueError(errstr)
        
        # If there are warnings in the configuration
        if warnings:

            # Log the warnings.
            warnstr = \
                f"The configuration loaded from '{config_file}' has " \
                "warnings. Warnings: " + "|".join(warnings)
            logger.warning(warnstr)

    # If the configuration type is "genes" or
    # "dimensionality_reduction"
    elif config_type in ["genes", "dimensionality_reduction"]:

        # Do not check the configuration, as there is no function to
        # do it so far.
        config_validated = config

    #-----------------------------------------------------------------#

    # Return the (possibly validated) configuration.
    return config_validated


#######################################################################


def load_config_model(config_file: Optional[str]) -> dict[str, object]:
    """Load the configuration specifying the DGD model's parameters
    and, possibly, the path to the files containing the trained model
    from a YAML file.

    Parameters
    ----------
    config_file : :class:`str`, optional
        The YAML configuration file. If no file is provided, the
        default configuration file "model_tgmm.yaml" in the directory
        storing configuration files for the model will be used.

    Returns
    -------
    config : :class:`dict`
        A dictionary containing the configuration.
    """

    # If no configuration file is provided
    if config_file is None:

        # Use the default configuration file.
        config_file = \
            os.path.join(defaults.CONFIG_DIRS["model"],
                        "model_tgmm.yaml")

    # Load and check the configuration.
    return _load_config(config_file = config_file,
                        config_type = "model")


def load_config_rep(config_file: Optional[str]) -> dict[str, object]:
    """Load the configuration containing the options for the
    optimization round(s) to find the best representations for a
    set of samples from a YAML file.

    Parameters
    ----------
    config_file : :class:`str`, optional
        The YAML configuration file. If no file is provided, the
        default configuration file "two_opt.yaml" in the directory
        storing configuration files for the representations will be
        used.

    Returns
    -------
    config : :class:`dict`
        A dictionary containing the configuration.
    """

    # If no configuration file is provided
    if config_file is None:

        # Use the default configuration file.
        config_file = \
            os.path.join(defaults.CONFIG_DIRS["representations"],
                        "two_opt.yaml")

    # Load and check the configuration.
    return _load_config(config_file = config_file,
                        config_type = "representations")


def load_config_train(config_file: Optional[str]) -> dict[str, object]:
    """Load the configuration containing the options for training the
    model from a YAML file.

    Parameters
    ----------
    config_file : :class:`str`, optional
        The YAML configuration file. If no file is provided, the
        default configuration file "training_tgmm.yaml" in the
        directory storing configuration files for the training will
        be used.

    Returns
    -------
    config : :class:`dict`
        A dictionary containing the configuration.
    """

    # If no configuration file is provided
    if config_file is None:

        # Use the default configuration file.
        config_file = \
            os.path.join(defaults.CONFIG_DIRS["training"],
                        "training_tgmm.yaml")

    # Load the configuration from the file.
    return _load_config(config_file = config_file,
                        config_type = "training")


def load_config_plot(config_file: str) -> dict[str, object]:
    """Load a configuration for a plot from a YAML file.

    Parameters
    ----------
    config_file : :class:`str`
        A YAML configuration file.

    Returns
    -------
    config : :class:`dict`
        A dictionary containing the configuration.
    """

    # Load the configuration from the file.
    return _load_config(config_file = config_file,
                        config_type = "plotting")


def load_config_genes(config_file: str) -> dict[str, object]:
    """Load the configuration for creating a new list of genes from a
    YAML file.

    Parameters
    ----------
    config_file : :class:`str`
        A YAML configuration file.

    Returns
    -------
    config : :class:`dict`
        A dictionary containing the configuration.
    """

    # Load and check the configuration.
    return _load_config(config_file = config_file,
                        config_type = "genes")


def load_config_dim_red(config_file: str) -> dict[str, object]:
    """Load the configuration for performing a dimensionality
    reduction analysis.

    Parameters
    ----------
    config_file : :class:`str`
        A YAML configuration file.

    Returns
    -------
    config : :class:`dict`
        A dictionary containing the configuration.
    """

    # Load and check the configuration.
    return _load_config(config_file = config_file,
                        config_type = "dimensionality_reduction")
