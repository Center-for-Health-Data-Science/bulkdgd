#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-

#    _util.py
#
#    Internal utility functions. Not part of the public API.
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
__doc__ = \
    "Internal utility functions. Not part of the public API."
# Package name
pkg_name = "bulkDGD"


# Standard library
import logging as log
import os
from pkg_resources import resource_filename, Requirement
# Third-party packages
import yaml


# Get the module's logger
logger = log.getLogger(__name__)


#----------------------------- Constants -----------------------------#


# Template to check the model's configuration file against
CONFIG_MODEL_TEMPLATE = \
    {"dim_latent" : int,
     
     "dec" : \
        {"pth_file" : str,
         "options" : \
            {"n_neurons_hidden1" : int,
             "n_neurons_hidden2" : int,
             "n_neurons_out" : int,
             "r_init" : int,
             "scaling_type" : str
             },
        },
     
     "gmm" : \
        {"pth_file" : str,
         "options" : \
            {"n_comp" : int,
             "cm_type" : str,
             "logbeta_params" : list,
             "alpha" : int},
         "mean_prior" : \
            {"type" : str,
             "options" : None},
        },
     
     "rep_layer" : \
        {"pth_file" : str,
         "options" : \
            {"n_samples" : int},
        },
    }


# Template to check the representations' configuration file
# against
CONFIG_REP_TEMPLATE = \
    {"data" : \
        {"batch_size" : int,
         "shuffle" : bool},
     
     "optimization" : \
        {"opt1" : \
            {"epochs" : int,
             "type" : str,
             "options" : None},
         "opt2" : \
            {"epochs" : int,
             "type" : str,
             "options" : None},
        },
    }


#------------------------- Helper functions --------------------------#


def get_abspath(path):
    """Given a path, return its absolute path. Return
    None if the path given is None.

    Parameters
    ----------
    path : ``str``
        Path to a file/directory.

    Returns
    -------
    ``str``
        Absolute path to the file/directory.
    """

    return os.path.abspath(path) if path is not None else path


#--------------------------- Configuration ---------------------------#


def load_config(config_file):
    """Load a configuration from a YAML file.

    Parameters
    ----------
    config_file : ``str``
        YAML configuration file.

    Returns
    -------
    ``dict``
        Dictionary containing the configuration.
    """

    # Get the name of the configuration file
    config_file_name = os.path.basename(config_file).rstrip(".yaml")

    # If the configuration file is a name without extension
    if config_file == config_file_name:
        
        # Assume it is a configuration file in the directory
        # storing configuration files for running protocols
        config_file = os.path.join(CONFIG_REP_DIR,
                                   config_file_name + ".yaml")

    # Otherwise
    else:
        
        # Assume it is a file name/file path
        config_file = get_abspath(config_file)

    # Load the configuration from the file
    config = yaml.safe_load(open(config_file, "r"))

    # Return the configuration
    return config


def check_config_against_template(config,
                                  template):
    """Check the configuration against the configuration's
    template.

    Parameters
    ----------
    config : ``dict``
        Configuration loaded from the file provided by the
        user.

    template : ``template``
        Template of how the configuration should be structures.

    Returns
    -------
    ``dict``
        The configuration loaded from the file provided by
        the user, if all checks were successful.
    """

    # The recursive step
    def recursive_step(config,
                       template,
                       key):

        # If both the current configuration dictionary and the
        # template dictionary are dictionaries (they are
        # sub-dictionaries of the original ones in the
        # recursive calls)
        if (isinstance(config, dict) \
        and isinstance(template, dict)):

            # Get the fields (= keys) in the configuration
            config_fields = set(config.keys())
            
            # Get the fields (= keys) in the template
            template_fields = set(template.keys())
            
            # Get the fields (= keys) unique to the configuration
            unique_config_fields = \
                config_fields - template_fields

            # Get the fields (= keys) unique to the template
            unique_template_fields = \
                template_fields - config_fields

            # If any unique field was found in the configuration
            if len(unique_config_fields) != 0:

                # Warn the user and raise an exception
                fields = \
                    ", ".join([f"'{f}'" for f \
                               in unique_config_fields])
                errstr = \
                    f"Unrecognized field(s) in configuration: " \
                    f"{fields}."
                logger.error(errstr)
                raise KeyError(errstr)

            # If any field was found in the template but not
            # in the configuration
            if len(unique_template_fields) != 0:

                # Warn the user and raise an exception
                fields = \
                    ", ".join([f"'{f}'" for f \
                               in unique_template_fields])
                errstr = \
                    f"Missing field(s) in configuration: " \
                    f"{fields}."
                logger.error(errstr)
                raise KeyError(errstr)

            # For each key, value pair in the configuration
            for k, val_config in config.items():

                # If the element corresponds to a dictionary
                # of options that can vary according to the
                # "type" of something (i.e., the options for
                # the optimizer used in optimization steps)
                if k == "options" and "type" in list(config.keys()):
                    continue

                # Get the corresponding value in the template
                # (we are sure that there will be a value for
                # the given key, because we checked the equality
                # of all keys between configuration and template
                # earlier)
                val_template = template[k]

                # Recursively check the values
                recursive_step(config = val_config,
                               template = val_template,
                               key = k)

        # If the configuration is a dictionary, but the
        # template is not
        elif (isinstance(config, dict) \
        and not isinstance(template, dict)):

            # Warn the user and raise an exception
            errstr = \
                f"The configuration contains sub-fields " \
                f"for the field {str(template)}, which is " \
                f"not supposed to have sub-fields."
            logger.error(errstr)
            raise TypeError(errstr)

        # If the template is a dictionary, but the
        # configuration is not
        elif (not isinstance(config, dict) \
        and isinstance(template, dict)):

            # Warn the user and raise an exception
            errstr = \
                f"The configuration does not contain " \
                f"sub-fields for the field {str(config)}, " \
                f"while it is supposed to have the following " \
                f"sub-fields: {str(template)}."
            raise TypeError(errstr)

        # If both the configuration and the template are
        # not dictionaries (we are in a "leaf" value, not
        # a key of a nested dictionary)
        elif (not isinstance(config, dict) \
        and not isinstance(template, dict)):

            # If the type of value found in the configuration
            # does not match the type set in the template
            if not isinstance(config, template):

                # Warn the user and raise an exception
                errstr = \
                    f"'{key}' must be an instance of " \
                    f"{str(template)}, not {str(type(config))} " \
                    f"('{config}')."
                raise TypeError(errstr)

        # Return the dictionary
        return config

    # Call the recursive step and return the result
    return recursive_step(config = config,
                          template = template,
                          key = None)
