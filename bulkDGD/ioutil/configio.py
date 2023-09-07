#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-

#    configio.py
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


# Standard library
import copy
import logging as log
import os
# Third-party packages
import matplotlib.font_manager as fm
import yaml
# bulkDGD
from . import defaults
from . import _util


# Get the module's logger
logger = log.getLogger(__name__)


#------------------------- Private constants -------------------------#


# Template to check the model's configuration file against
_CONFIG_MODEL_TEMPLATE = \
    {# Options for the Gaussian mixture model
     "gmm_pth_file" : str,
     "dim" : int,
     "n_comp" : int,
     "cm_type" : str,
     "means_prior_name" : str,
     "means_prior_options" : None,
     "weights_prior_name" : str,
     "weights_prior_options" : None,
     "log_var_prior_name" : str,
     "log_var_prior_options" : None,

     # Options for the decoder
     "dec_pth_file" : str,
     "n_units_hidden_layers" : list,
     "r_init" : int,
     "activation_output" : str,
    
    # Genes
    "genes_txt_file" : str}


#------------------------- Public functions --------------------------#


def load_config_model(config_file):
    """Load the configuration specifying the DGD model's parameters
    and, possibly, the files containing the trained model from a
    YAML file.

    Parameters
    ----------
    config_file : ``str``
        The YAML configuration file.

    Returns
    -------
    config : ``dict``
        A dictionary containing the configuration.
    """


    #-------------------- Load the configuration ---------------------#


    # Get the name of the configuration file
    config_file_name = os.path.basename(config_file).rstrip(".yaml")

    # If the configuration file is a name without extension
    if config_file == config_file_name:
        
        # Assume it is a configuration file in the directory
        # storing configuration files for running protocols
        config_file = os.path.join(defaults.CONFIG_MODEL_DIR,
                                   config_file_name + ".yaml")

    # Otherwise
    else:
        
        # Assume it is a file name/file path
        config_file = os.path.abspath(config_file)

    # Load the configuration from the file
    config = yaml.safe_load(open(config_file, "r"))

    # Split the path into its 'head' (path to the file without
    # the file name) and its 'tail' (the file name)
    path_head, path_tail = os.path.split(config_file)


    #------------------ Check against the template -------------------#


    # Check the configuration against the template
    config = _util._check_config_against_template(\
                config = config,
                template = _CONFIG_MODEL_TEMPLATE)


    #--------------- Get the file with the trained GMM ---------------#


    # Get the PyTorch file containing the trained Gaussian
    # mixture model
    gmm_pth_file = config["gmm_pth_file"]

    # If the default file should be used
    if gmm_pth_file == "default":

        # Get the path to the default file
        config["gmm_pth_file"] = \
            os.path.normpath(defaults.GMM_FILE)

    # Otherwise
    else:

        # Get the path to the file
        config["gmm_pth_file"] = \
            os.path.normpath(os.path.join(path_head,
                                          gmm_pth_file))


    #------------- Get the file with the trained decoder -------------#


    # Get the PyTorch file containing the trained decoder
    dec_pth_file = config["dec_pth_file"]

    # If the default file should be used
    if dec_pth_file == "default":

        # Get the path to the default file
        config["dec_pth_file"] = \
            os.path.normpath(defaults.DEC_FILE)

    # Otherwise
    else:

        # Get the path to the file
        config["dec_pth_file"] = \
            os.path.normpath(os.path.join(path_head,
                                          dec_pth_file))


    #------------------ Get the file with the genes ------------------#


    # Get the .txt file containing the genes
    genes_txt_file = config["genes_txt_file"]

    # If the default file should be used
    if genes_txt_file == "default":

        # Get the path to the default file
        config["genes_txt_file"] = \
            os.path.normpath(defaults.GENES_FILE)

    # Otherwise
    else:

        # Get the path to the file
        config["genes_txt_file"] = \
            os.path.normpath(os.path.join(path_head,
                                          genes_txt_file))

    # Return the configuration
    return config


def load_config_rep(config_file):
    """Load the configuration containing the options for data
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

    # Get the name of the configuration file
    config_file_name = os.path.basename(config_file).rstrip(".yaml")

    # If the configuration file is a name without extension
    if config_file == config_file_name:
        
        # Assume it is a configuration file in the directory
        # storing configuration files for running protocols
        config_file = os.path.join(defaults.CONFIG_REP_DIR,
                                   config_file_name + ".yaml")

    # Otherwise
    else:
        
        # Assume it is a file name/file path
        config_file = os.path.abspath(config_file)

    # Load the configuration from the file
    config = yaml.safe_load(open(config_file, "r"))

    # Split the path into its 'head' (path to the file without
    # the file name) and its 'tail' (the file name)
    path_head, path_tail = os.path.split(config_file)

    # Return the configuration
    return config


def load_config_plot(config_file):
    """Load a configuration for a plot.

    Parameters
    ----------
    config_file : ``str``
        A YAML configuration file.

    Returns
    -------
    ``dict``
        A dictionary containing the configuration.
    """

    # Get the name of the configuration file
    config_file_name = os.path.basename(config_file).rstrip(".yaml")

    # If the configuration file is a name without extension
    if config_file == config_file_name:
        
        # Assume it is a configuration file in the directory
        # storing configuration files for running protocols
        config_file = os.path.join(defaults.CONFIG_PLOT_DIR,
                                   config_file_name + ".yaml")

    # Otherwise
    else:
        
        # Assume it is a file name/file path
        config_file = os.path.abspath(config_file)

    # Load the configuration from the file
    config = yaml.safe_load(open(config_file, "r"))

    # Split the path into its 'head' (path to the file without
    # the file name) and its 'tail' (the file name)
    path_head, path_tail = os.path.split(config_file)
    
    # Substitute the font properties definitions
    # with the corresponding FontProperties instances
    new_config = _util._recursive_map_dict(\
        d = config,
        func = fm.FontProperties,
        keys = {"fontproperties", "prop", "title_fontproperties"})

    # Return the new configuration
    return new_config