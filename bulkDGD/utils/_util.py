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
import copy
import logging as log
import os
from pkg_resources import resource_filename, Requirement
# Third-party packages
import matplotlib.pyplot as plt
import numpy as np
import yaml


# Get the module's logger
logger = log.getLogger(__name__)


#----------------------------- Constants -----------------------------#


# Template to check the model's configuration file against
CONFIG_MODEL_TEMPLATE = \
    {# Options for the Gaussian mixture model
     "gmm" : \
        {"pth_file" : str,
         "options" : \
            {"dim" : int,
             "n_comp" : int,
             "cm_type" : str,
             "log_var_params" : list,
             "alpha" : int},
         "mean_prior" : \
            {"type" : str,
             "options" : None},
        },

     # Options for the decoder
     "dec" : \
        {"pth_file" : str,
         "options" : \
            {"n_neurons_input" : int,
             "n_neurons_hidden1" : int,
             "n_neurons_hidden2" : int,
             "n_neurons_output" : int,
             "r_init" : int,
             "activation_output" : str,
            },
        },
    }


# Template to check the representations' configuration file
# against
CONFIG_REP_TEMPLATE = \
    {"data" : \
        {"batch_size" : int,
         "shuffle" : bool},
     
     "rep" : \
        {"n_rep_per_comp" : int,
         "opt1" : \
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
        The path to a file/directory.

    Returns
    -------
    ``str``
        The absolute path to the file/directory.
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


def recursive_map_dict(d,
                       func,
                       keys = None):
    """Recursively traverse a dictionary mapping a function to the
    dictionary's leaf values (= substituting the values
    which the return value of the function applied to those
    values).

    Parameters
    ----------
    d : ``dict``
        The input dictionary.

    func : any callable
        Callable taking as inputs the leaf values of the dictionary
        and returning a value which will take the dictionary's
        place.

    keys : ``list``, ``set``, optional
        List of specific keys on whose items the mapping
        should be performed. This means that all values associated
        with keys different from those in the list will not be
        affected. If ``None``, all keys and associated values
        will be considered.
    
    Returns
    -------
    ``dict``
        The new dictionary.
    """

    # Define the recursive step
    def recursive_step(d,
                       func,
                       keys):

        # If the current object is a dictionary
        if isinstance(d, dict):
            
            # Get the keys of items on which the maping will be
            # performed. If no keys are passed, all keys
            # in the dictionary will be considered.
            sel_keys = keys if keys else d.keys()

            # For each key, value pair in the dictionary
            for k, v in list(d.items()):

                # If the value is a dictionary
                if isinstance(v, dict):

                    # If the key is in the selected keys
                    if k in sel_keys:

                        # Substitute the value with the return value
                        # of 'func' applied to it
                        d[k] = func(**v)
                    
                    # Otherwise
                    else:

                        # Recursively check the sub-dictionaries
                        # of the current dictionary
                        recursive_step(d = v,
                                       func = func,
                                       keys = sel_keys)



    # Create a copy of the input dictionary
    new_d = copy.deepcopy(d)

    # Add the "key path" and its value to either the
    # input dictionary or the new dictionary
    recursive_step(d = new_d,
                   func = func,
                   keys = keys)

    # Return the new dictionary
    return new_d


#----------------------------- Plotting ------------------------------#


def get_ticks_positions(values,
                        item,
                        config):
    """Generate the positions that the ticks
    will have on a plot axis/colorbar/etc.

    This original code for this function was originally
    developed by Valentina Sora for the RosettaDDGPrediction
    package.
    
    The original function can be found at:

    https://github.com/ELELAB/RosettaDDGPrediction/
    blob/master/RosettaDDGPrediction/plotting.py

    Parameters
    ----------
    values : {``list``, ``numpy.ndarray``}
        The values from which the ticks' positions should be set.

    item : ``str``
        Name of the item of the plot you are setting the ticks'
        positions for (e.g., ``"x-axis"``, ``"y-axis"``, or
        ``"colorbar"``).

    config : ``dict``
        Configuration for the interval that the ticks' positions
        should cover.

    Returns
    -------
    ``numpy.ndarray``
        Array containing the ticks' positions.
    """

    # Get the top configuration
    config = config.get("interval")

    # If there is no configuration for the interval
    if config is None:

        # Raise an error
        errstr = \
            f"No 'interval' section was found in the " \
            f"configuration for the {item}."
        raise KeyError(errstr)
    
    # Get the configurations
    int_type = config.get("type")
    rtn = config.get("round_to_nearest")
    top = config.get("top")
    bottom = config.get("bottom")
    steps = config.get("steps")
    spacing = config.get("spacing")
    ciz = config.get("center_in_zero")

    # Inform the user that we are now setting the ticks' interval
    infostr = \
        f"Now setting the interval for the plot's {item}'s ticks..."
    logger.info(infostr)


    #--------------------------- Rounding ---------------------------#


    # If no rounding was specified
    if rtn is None:

        # If the interval is discrete
        if int_type == "discrete":

            # Default to rounding to the nearest 1
            rtn = 1

        # If the interval is continuous
        elif int_type == "continuous":
        
            # Default to rounding to the nearest 0.5
            rtn = 0.5

        # Inform the user about the rounding value
        infostr = \
            f"Since 'round_to_nearest' is not " \
            f"defined and 'type' is '{int_type}', " \
            f"the rounding will be set to the nearest " \
            f"{rtn}."
        logger.info(infostr)

    # Otherwise
    else:

        # Inform the user about the chosen rounding value
        infostr = \
            f"The user set rounding (up and down) " \
            f"to the nearest {rtn} " \
            f"('round_to_nearest' = {rtn})."
        logger.info(infostr)


    #--------------------------- Top value ---------------------------#


    # If the maximum of the ticks interval was not specified
    if top is None:
        
        # If the interval is discrete
        if int_type == "discrete":
            
            # The default top value will be the
            # maximum of the values provided
            top = int(np.ceil(max(values)))

            # Inform the user about the top value
            infostr = \
                f"Since 'top' is not defined and " \
                f"'type' is '{int_type}', 'top' " \
                f"will be the maximum of all values " \
                f"found ({top})."
            logger.info(infostr)
        
        # If the interval is continuous
        elif int_type == "continuous":
            
            # The default top value will be the
            # rounded-up maximum of the values
            top = \
                np.ceil(max(values)*(1/rtn)) / (1/rtn)

            # Inform the user about the top value
            infostr = \
                f"Since 'top' is not defined and " \
                f"'type' is '{int_type}', 'top' " \
                f"will be the maximum of all values " \
                f"found, rounded up to the nearest " \
                f"{rtn} ({top})."
            logger.info(infostr)

    # Otherwise
    else:

        # Inform the user about the chosen top value
        infostr = \
            f"The user set the top value to {top} " \
            f"('top' = {top})."
        logger.info(infostr)


    #------------------------- Bottom value --------------------------#


    # If the minimum of the ticks interval was not specified
    if bottom is None:
        
        # If the interval is discrete
        if int_type == "discrete":
            
            # The default bottom value is the
            # minimim of the values provided
            bottom = int(min(values))

            # Inform the user about the bottom value
            infostr = \
                f"Since 'bottom' is not defined and " \
                f"'type' is '{int_type}', 'bottom' " \
                f"will be the minimum of all values " \
                f"found ({bottom})."
            logger.info(infostr)
        
        # If the interval is continuous
        elif int_type == "continuous":
            
            # The default bottom value is the rounded
            # down minimum of the values
            bottom = \
                np.floor(min(values)*(1/rtn)) / (1/rtn)

            # Inform the user about the bottom value
            infostr = \
                f"Since 'bottom' is not defined and " \
                f"'type' is '{int_type}', 'bottom' " \
                f"will be the minimum of all values " \
                f"found, rounded down to the nearest " \
                f"{rtn} ({bottom})."
            logger.info(infostr)

    # Otherwise
    else:

        # Inform the user about the chosen top value
        infostr = \
            f"The user set the bottom value to {bottom} " \
            f"('bottom' = {bottom})."
        logger.info(infostr)


    # If the two extremes of the interval coincide
    if top == bottom:
        
        # Return only one value
        return np.array([bottom])


    #----------------------------- Steps -----------------------------# 


    # If the number of steps the interval should have
    # was not specified
    if steps is None:

        # A default of 10 steps will be set
        steps = 10

        # Inform the user about the steps if in info mode
        infostr = \
            f"Since the number of steps the interval should have " \
            f"is not defined, 'steps' will be '10'."
        logger.info(infostr)

    # Otherwise
    else:

        # Inform the user about the chosen top value
        infostr = \
            f"The user set the number of steps the interval " \
            f"should have to {steps} ('steps' = {steps})."
        logger.info(infostr)


    #---------------------------- Spacing ----------------------------#


    # If the interval spacing was not specified
    if spacing is None:
        
        # If the interval is discrete
        if int_type == "discrete":

            # The default spacing is the one between two steps,
            # rounded up
            spacing = \
                int(np.ceil(np.linspace(bottom,
                                        top,
                                        steps,
                                        retstep = True)[1]))

            # Inform the user about the spacing
            infostr = \
                f"Since the spacing between the ticks is not " \
                f"defined, 'spacing' will be the value " \
                f"guaranteeing an equipartition of the interval " \
                f"between {bottom} and {top} in {steps} " \
                f"number of steps, rounded up to the nearest 1 " \
                f"({spacing})."
            logger.info(infostr)

        
        # If the interval is continuous
        elif int_type == "continuous":
            
            # The default spacing is the one between two steps,
            # rounded up
            spacing = np.linspace(bottom,
                                  top,
                                  steps,
                                  retstep = True)[1]

            spacing = np.ceil(spacing*(1/rtn)) / (1/rtn)

            # Inform the user about the spacing
            infostr = \
                f"Since the spacing between the ticks is not " \
                f"defined, 'spacing' will be the value " \
                f"guaranteeing an equipartition of the interval " \
                f"between {bottom} and {top} in {steps} " \
                f"number of steps ({spacing})."
            logger.info(infostr)


    #------------------------ Center in zero -------------------------#


    # If the interval should be centered in zero
    if ciz:
        
        # Get the highest absolute value
        absval = \
            np.ceil(top) if top > bottom else np.floor(bottom)
        
        # Top and bottom will be opposite numbers with
        # absolute value equal to absval
        top, bottom = absval, -absval

        # Get an evenly-spaced interval between the bottom
        # and top value
        interval = np.linspace(bottom, top, steps)

        # Inform the user about the change in the interval's
        # extreme
        infostr = \
            f"Since the user requested a ticks' interval centered " \
            f"in zero, the interval will be now between {top} " \
            f"and {bottom} with {steps} number of steps: " \
            f"{', '.join([str(i) for i in interval.tolist()])}."
        logger.info(infostr)
        
        # Return the interval
        return interval

    # Get the interval
    interval = np.arange(bottom, top + spacing, spacing)

    # Inform the user about the interval that will be used
    infostr = \
        f"The ticks' interval will be between {bottom} and {top} " \
        f"with a spacing of {spacing}: " \
        f"{', '.join([str(i) for i in interval.tolist()])}."
    logger.info(infostr)

    # Return the interval
    return interval


def set_axis(ax,
             axis,
             config,
             ticks = None,
             tick_labels = None):
    """Set up the x- or y-axis after generating a plot.

    Parameters
    ----------
    ax : ``matplotlib.axes.Axes``
        Axes instance.

    axis : ``str``, {``"x"``, ``"y"``}
        Whether the axis to be set is the x- or the y-axis.

    config : ``dict``
        Configuration for setting the axis.

    ticks : ``list``, optional
        List of ticks' positions. If it is not passed, the ticks
        will be those already present on the axis (automatically
        determined by matplotlib when generating the plot).

    tick_labels : ``list``, optional
        List of ticks' labels. If not passed, the ticks' labels
        will represent the ticks' positions.

    Returns
    -------
    ``matplotlib.axes.Axes``
        Axes instance. 
    """


    #----------------------------- Axes ------------------------------#


    # If the axis to be set is the x-axis
    if axis == "x":

        # Get the corresponding methods
        plot_ticks = plt.xticks
        set_label = ax.set_xlabel
        set_ticks = ax.set_xticks
        set_ticklabels = ax.set_xticklabels
        get_ticklabels = ax.get_xticklabels

        # Get the corresponding spine
        spine = "bottom"

    # If the axis to be set is the y-axis
    elif axis == "y":

        # Get the corresponding methods
        plot_ticks = plt.yticks
        set_label = ax.set_ylabel
        set_ticks = ax.set_yticks
        set_ticklabels = ax.set_yticklabels
        get_ticklabels = ax.get_yticklabels

        # Get the corresponding spine
        spine = "left"

    # If there is an axis label's configuration
    if config.get("label"):
        
        # Set the axis label
        set_label(**config["label"])        


    #----------------------------- Ticks -----------------------------#

    
    # If no ticks' positions were passed
    if ticks is None:

        # Default to the tick locations already present
        ticks = plot_ticks()[0]

    # If there are any ticks on the axis
    if ticks != []:      
        
        # Set the axis boundaries
        ax.spines[spine].set_bounds(ticks[0],
                                    ticks[-1])

    # If a configuration for the tick parameters was provided
    if config.get("tick_params"):
        
        # Apply the configuration to the ticks
        ax.tick_params(axis = axis,
                       **config["tick_params"])

    # Set the ticks
    set_ticks(ticks = ticks)


    #------------------------- Ticks' labels -------------------------#

    
    # If no ticks' labels were passed
    if tick_labels is None:
        
        # Default to the string representations
        # of the ticks' positions
        tick_labels = [str(t) for t in ticks]

    # Get the configuration for ticks' labels
    tick_labels_config = config.get("ticklabels", {})
    
    # Set the ticks' labels
    set_ticklabels(labels = tick_labels,
                   **tick_labels_config)

    # Return the axis
    return ax


def set_legend(ax,
               config):
    """Set a legend for the current plot.
    """

    # Get the legend's handles and labels
    handles, labels = ax.get_legend_handles_labels()
    
    # Draw the legend
    ax.legend(handles = handles,
              labels = labels,
              bbox_transform = plt.gcf().transFigure,
              **config)

    # Retutn the ax
    return ax