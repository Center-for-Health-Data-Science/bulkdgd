#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-

#    configio.py
#
#    Utilities to load and save configurations.
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
__doc__ = "Utilities to load and save configurations."


#######################################################################


# Import from the standard library.
import logging as log
import os
# Import from third-party packages.
import matplotlib.font_manager as fm
import yaml
# Import from 'bulkDGD'.
from bulkDGD import defaults, util


#######################################################################


# Get the module's logger.
logger = log.getLogger(__name__)


########################## PRIVATE FUNCTIONS ##########################


def _load_config(config_file,
                 config_type):
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

    # Return the configuration.
    return config


########################## PUBLIC FUNCTIONS ###########################


def load_config_model(config_file):
    """Load the configuration specifying the DGD model's parameters
    and, possibly, the path to the files containing the trained model
    from a YAML file.

    Parameters
    ----------
    config_file : :class:`str`
        The YAML configuration file.

    Returns
    -------
    config : :class:`dict`
        A dictionary containing the configuration.
    """

    # Load the configuration from the file.
    config = _load_config(config_file = config_file,
                          config_type = "model")

    #-----------------------------------------------------------------#

    # Split the path into its 'head' (path to the file without the
    # file's name) and its 'tail' (the file's name).
    path_head, path_tail = os.path.split(config_file)

    #-----------------------------------------------------------------#

    # Check the configuration.
    config = util.check_config_model(config = config)

    #-----------------------------------------------------------------#

    # For each model's component
    for comp in \
        ("gmm_pth_file", "dec_pth_file", "genes_txt_file"):

        # Get the PyTorch file containing the trained component of the
        # model.
        comp_file = config.get(comp)

        # If the file is not None
        if comp_file is not None:

            # If the default file should be used
            if comp_file == "default":

                # Get the path to the default file.
                config[comp] = \
                    os.path.normpath(\
                        defaults.DATA_FILES_MODEL[comp.split("_")[0]])

            # Otherwise
            else:

                # Get the path to the file.
                config[comp] = \
                    os.path.normpath(os.path.join(path_head,
                                                  comp_file))
        # Otherwise
        else:

            # If the component is the list of genes
            if comp == "genes_txt_file":

                # Raise an exception.
                errstr = \
                    f"A '{comp}' must be specified in the " \
                    "model's configuration."
                raise ValueError(errstr)

    #-----------------------------------------------------------------#

    # Return the configuration.
    return config


def load_config_rep(config_file):
    """Load the configuration containing the options for the
    optimization round(s) to find the best representations for a
    set of samples from a YAML file.

    Parameters
    ----------
    config_file : :class:`str`
        The YAML configuration file.

    Returns
    -------
    config : :class:`dict`
        A dictionary containing the configuration.
    """

    # Load the configuration from the file.
    config = _load_config(config_file = config_file,
                          config_type = "representations")

    #-----------------------------------------------------------------#

    # Check the configuration.
    config = util.check_config_rep(config = config)

    #-----------------------------------------------------------------#

    # Return the configuration.
    return config


def load_config_train(config_file):
    """Load the configuration containing the options for training the
    model from a YAML file.

    Parameters
    ----------
    config_file : :class:`str`
        The YAML configuration file.

    Returns
    -------
    config : :class:`dict`
        A dictionary containing the configuration.
    """

    # Load the configuration from the file.
    config = _load_config(config_file = config_file,
                          config_type = "training")

    #-----------------------------------------------------------------#

    # Check the configuration.
    config = util.check_config_train(config = config)

    #-----------------------------------------------------------------#

    # Return the configuration.
    return config


def load_config_plot(config_file):
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
    config = _load_config(config_file = config_file,
                          config_type = "plotting")

    #-----------------------------------------------------------------#

    # Check the configuration.
    config = util.check_config_plot(config = config)

    #-----------------------------------------------------------------#
    
    # Substitute the font properties definitions with the corresponding
    # 'FontProperties' instances.
    new_config = util.recursive_map_dict(\
        d = config,
        func = fm.FontProperties,
        keys = {"fontproperties", "prop", "title_fontproperties"})

    #-----------------------------------------------------------------#

    # Return the new configuration.
    return new_config


def load_config_genes(config_file):
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

    # Load the configuration from the file.
    config = _load_config(config_file = config_file,
                          config_type = "genes")

    #-----------------------------------------------------------------#

    # Check the configuration.
    config = util.check_config_genes(config = config)

    #-----------------------------------------------------------------#

    # Return the new configuration.
    return config


def load_config_dim_red(config_file):
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

    # Load the configuration from the file.
    config = _load_config(config_file = config_file,
                          config_type = "dimensionality_reduction")

    #-----------------------------------------------------------------#

    # Check the configuration.
    config = util.check_config_dim_red(config = config)

    #-----------------------------------------------------------------#

    # Return the new configuration.
    return config
