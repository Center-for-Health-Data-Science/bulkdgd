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
import copy
import logging as log
import os
# Import from third-party packages.
import matplotlib.font_manager as fm
import yaml
# Import from 'bulkDGD'.
from . import defaults
from . import _util


#######################################################################


# Get the module's logger.
logger = log.getLogger(__name__)


########################## PRIVATE CONSTANTS ##########################


# Set the template against which to check the model's configuration.
_CONFIG_MODEL_TEMPLATE = \
    {# Set the options for the Gaussian mixture model.
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

     # Set the options for the decoder.
     "dec_pth_file" : str,
     "n_units_hidden_layers" : list,
     "r_init" : int,
     "activation_output" : str,
    
    # Set the file containing the genes included in the model.
    "genes_txt_file" : str}


# Set the templates against which to check the configuration for
# the different types of optimization schemes that can be used
# when finding the representations for new samples.
_CONFIG_REP_TEMPLATE = \
    {# Set the template for the 'one_opt' scheme.
     "one_opt" : \
        {# Set the number of representations to get per component
         # of the Gaussian mixture model per sample.
         "n_rep_per_comp" : int,

         # Set the options to configure the data loader.
         "data_loader" : None,

         # Set the optimization scheme.
         "scheme" : str,

         # Set the options for the optimization.
         "opt" : \
            {# Set the number of epochs.
             "epochs" : int,
            # Set the optimizer
             "optimizer" : \
                {"name" : str,
                 "options" : None}}},

    # Set the template for the 'two_opt' scheme.
    "two_opt" : \
        {# Set the number of representations to get per component
         # of the Gaussian mixture model per sample.
         "n_rep_per_comp" : int,

         # Set the options to configure the data loader.
         "data_loader" : None,

         # Set the optimization scheme.
         "scheme" : str,

         # Set the options for the first optimization.
         "opt1" : \
            {# Set the number of epochs.
             "epochs" : int,
            # Set the optimizer
             "optimizer" : \
                {"name" : str,
                 "options" : None}},
         
         # Set the options for the second optimization.
         "opt2" : \
            {# Set the number of epochs.
             "epochs" : int,
            # Set the optimizer
             "optimizer" : \
                {"name" : str,
                 "options" : None}}}}


# Set the template against which to check the training configuration.
_CONFIG_TRAIN_TEMPLATE = \
    {# Set the number of epochs.
     "epochs" : int,

     # Set the options to configure the data loader.
     "data_loader" : None,

     # Set the options to train the Gaussian mixture model.
     "gmm" : \
        {"optimizer" : \
            {"name" : str,
             "options" : None}},
     
     # Set the options to train the decoder.
     "dec" : \
        {"optimizer" : \
            {"name" : str,
             "options" : None}},

    # Set the options to optimize the representations.
    "rep" : \
        {"optimizer" : \
            {"name" : str,
             "options" : None}},}


########################## PUBLIC FUNCTIONS ###########################


def load_config_model(config_file):
    """Load the configuration specifying the DGD model's parameters
    and, possibly, the path to the files containing the trained model
    from a YAML file.

    Parameters
    ----------
    config_file : ``str``
        The YAML configuration file.

    Returns
    -------
    config : ``dict``
        A dictionary containing the configuration.
    """

    # Set the fields that can be missing from the configuration,
    # so that they do not raise an exception if they are not found.
    ignore_if_missing = \
        {"gmm_pth_file", "dec_pth_file"}

    # Set the fields that can vary in the configuration, so that they
    # do not raise an exception if they are different than expected.
    ignore_if_varying = \
        {"means_prior_options", "weights_prior_options",
         "log_var_prior_options"}

    #-----------------------------------------------------------------#

    # Get the name of the configuration file.
    config_file_name = os.path.basename(config_file).rstrip(".yaml")

    #-----------------------------------------------------------------#

    # If the configuration file is a name without extension
    if config_file == config_file_name:
        
        # Assume it is a configuration file in the directory storing
        # configuration files for the model.
        config_file = os.path.join(defaults.CONFIG_MODEL_DIR,
                                   config_file_name + ".yaml")

    # Otherwise
    else:
        
        # Assume it is a file name/file path.
        config_file = os.path.abspath(config_file)

    #-----------------------------------------------------------------#

    # Load the configuration from the file.
    config = yaml.safe_load(open(config_file, "r"))

    #-----------------------------------------------------------------#

    # Split the path into its 'head' (path to the file without the
    # file's name) and its 'tail' (the file's name).
    path_head, path_tail = os.path.split(config_file)

    #-----------------------------------------------------------------#

    # Check the configuration against the template.
    config = _util.check_config_against_template(\
                config = config,
                template = _CONFIG_MODEL_TEMPLATE,
                ignore_if_missing = ignore_if_missing,
                ignore_if_varying = ignore_if_varying)

    #-----------------------------------------------------------------#

    # Get the PyTorch file containing the trained Gaussian mixture
    # model.
    gmm_pth_file = config.get("gmm_pth_file")

    # If the file is not None
    if gmm_pth_file is not None:

        # If the default file should be used
        if gmm_pth_file == "default":

            # Get the path to the default file.
            config["gmm_pth_file"] = \
                os.path.normpath(defaults.GMM_FILE)

        # Otherwise
        else:

            # Get the path to the file.
            config["gmm_pth_file"] = \
                os.path.normpath(os.path.join(path_head,
                                              gmm_pth_file))

    #-----------------------------------------------------------------#

    # Get the PyTorch file containing the trained decoder.
    dec_pth_file = config.get("dec_pth_file")

    # If the file is not None
    if dec_pth_file is not None:

        # If the default file should be used
        if dec_pth_file == "default":

            # Get the path to the default file.
            config["dec_pth_file"] = \
                os.path.normpath(defaults.DEC_FILE)

        # Otherwise
        else:

            # Get the path to the file.
            config["dec_pth_file"] = \
                os.path.normpath(os.path.join(path_head,
                                              dec_pth_file))

    #-----------------------------------------------------------------#

    # Get the plain text file containing the genes
    genes_txt_file = config.get("genes_txt_file")

    # If the file is not None
    if genes_txt_file is not None:

        # If the default file should be used
        if genes_txt_file == "default":

            # Get the path to the default file.
            config["genes_txt_file"] = \
                os.path.normpath(defaults.GENES_FILE)

        # Otherwise
        else:

            # Get the path to the file.
            config["genes_txt_file"] = \
                os.path.normpath(os.path.join(path_head,
                                              genes_txt_file))

    # Otherwise
    else:

        # Raise an exception.
        errstr = \
            "A 'genes_txt_file' must be specified in the " \
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
    config_file : ``str``
        The YAML configuration file.

    Returns
    -------
    config : ``dict``
        A dictionary containing the configuration.
    """

    # Set the names of the supported optimization schemes.
    optimization_schemes = ["one_opt", "two_opt"]

    #-----------------------------------------------------------------#

    # Get the name of the configuration file.
    config_file_name = os.path.basename(config_file).rstrip(".yaml")

    #-----------------------------------------------------------------#

    # If the configuration file is a name without extension
    if config_file == config_file_name:
        
        # Assume it is a configuration file in the directory storing
        # configuration files for the optimization rounds.
        config_file = os.path.join(defaults.CONFIG_REP_DIR,
                                   config_file_name + ".yaml")

    # Otherwise
    else:
        
        # Assume it is a file name/file path.
        config_file = os.path.abspath(config_file)

    #-----------------------------------------------------------------#

    # Load the configuration from the file.
    config = yaml.safe_load(open(config_file, "r"))

    #-----------------------------------------------------------------#

    # Get the optimization scheme from the configuration.
    opt_scheme = config.get("scheme")

    # If no scheme was defined
    if not opt_scheme:

        # Raise an error.
        errstr = \
            "The configuration must include a 'scheme' key " \
            "associated with the name of the optimization " \
            "scheme the configuration refers to."
        raise KeyError(errstr)

    # If the optimization scheme is not supported
    if opt_scheme not in optimization_schemes:

        # Raise an error.
        supported_schemes = \
            ", ".join([f"'{s}'" for s in optimization_schemes])
        errstr = \
            f"Unsupported optimization scheme '{opt_scheme}'. " \
            "Supported optimization schemes are: " \
            f"'{supported_schemes}."
        raise ValueError(errstr)

    #-----------------------------------------------------------------#

    # Check the configuration against the template.
    config = _util.check_config_against_template(\
                config = config,
                template = _CONFIG_REP_TEMPLATE[opt_scheme],
                ignore_if_missing = set(),
                ignore_if_varying = set())

    #-----------------------------------------------------------------#

    # Return the configuration.
    return config


def load_config_train(config_file):
    """Load the configuration containing the options for training the
    model from a YAML file.

    Parameters
    ----------
    config_file : ``str``
        The YAML configuration file.

    Returns
    -------
    config : ``dict``
        A dictionary containing the configuration.
    """

    # Get the name of the configuration file.
    config_file_name = os.path.basename(config_file).rstrip(".yaml")

    #-----------------------------------------------------------------#

    # If the configuration file is a name without extension
    if config_file == config_file_name:
        
        # Assume it is a configuration file in the directory storing
        # configuration files for training the model.
        config_file = os.path.join(defaults.CONFIG_TRAIN_DIR,
                                   config_file_name + ".yaml")

    # Otherwise
    else:
        
        # Assume it is a file name/file path.
        config_file = os.path.abspath(config_file)

    #-----------------------------------------------------------------#

    # Load the configuration from the file.
    config = yaml.safe_load(open(config_file, "r"))

    #-----------------------------------------------------------------#

    # Check the configuration against the template.
    config = _util.check_config_against_template(\
                config = config,
                template = _CONFIG_TRAIN_TEMPLATE,
                ignore_if_missing = set(),
                ignore_if_varying = set())

    #-----------------------------------------------------------------#

    # Return the configuration.
    return config


def load_config_plot(config_file):
    """Load a configuration for a plot from a YAML file.

    Parameters
    ----------
    config_file : ``str``
        A YAML configuration file.

    Returns
    -------
    config : ``dict``
        A dictionary containing the configuration.
    """

    # Get the name of the configuration file.
    config_file_name = os.path.basename(config_file).rstrip(".yaml")

    #-----------------------------------------------------------------#

    # If the configuration file is a name without extension
    if config_file == config_file_name:
        
        # Assume it is a configuration file in the directory storing
        # configuration files for plotting.
        config_file = os.path.join(defaults.CONFIG_PLOT_DIR,
                                   config_file_name + ".yaml")

    # Otherwise
    else:
        
        # Assume it is a file name/file path.
        config_file = os.path.abspath(config_file)

    #-----------------------------------------------------------------#

    # Load the configuration from the file.
    config = yaml.safe_load(open(config_file, "r"))

    #-----------------------------------------------------------------#
    
    # Substitute the font properties definitions with the corresponding
    # 'FontProperties' instances.
    new_config = _util.recursive_map_dict(\
        d = config,
        func = fm.FontProperties,
        keys = {"fontproperties", "prop", "title_fontproperties"})

    #-----------------------------------------------------------------#

    # Return the new configuration.
    return new_config


def load_config_genes(config_file):
    """Load a configuration for creating a new list of genes from a
    YAML file.

    Parameters
    ----------
    config_file : ``str``
        A YAML configuration file.

    Returns
    -------
    config : ``dict``
        A dictionary containing the configuration.
    """

    # Get the name of the configuration file.
    config_file_name = os.path.basename(config_file).rstrip(".yaml")

    #-----------------------------------------------------------------#

    # If the configuration file is a name without extension
    if config_file == config_file_name:
        
        # Assume it is a configuration file in the directory storing
        # configuration files for plotting.
        config_file = os.path.join(defaults.CONFIG_GENES_DIR,
                                   config_file_name + ".yaml")

    # Otherwise
    else:
        
        # Assume it is a file name/file path.
        config_file = os.path.abspath(config_file)

    #-----------------------------------------------------------------#

    # Load the configuration from the file.
    config = yaml.safe_load(open(config_file, "r"))

    #-----------------------------------------------------------------#
    
    # Substitute the font properties definitions with the corresponding
    # 'FontProperties' instances.
    new_config = _util.recursive_map_dict(\
        d = config,
        func = fm.FontProperties,
        keys = {"fontproperties", "prop", "title_fontproperties"})

    #-----------------------------------------------------------------#

    # Return the new configuration.
    return new_config
