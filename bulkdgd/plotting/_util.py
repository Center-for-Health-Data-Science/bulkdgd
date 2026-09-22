#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-

#    _util.py
#
#    Private utilities for plotting.
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
__doc__ = "Private utilities for plotting."


#######################################################################


# Import from the standard library.
import copy
import itertools
import logging as log
import math
import os
from typing import Optional

# Import from third-party libraries.
import matplotlib
from matplotlib.cm import ScalarMappable
import matplotlib.colors as mcolors
import matplotlib.font_manager as fm
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Import from 'bulkdgd'.
from bulkdgd import _internals
from . import _templates


#######################################################################


# Get the module's logger.
logger = log.getLogger(__name__)


#######################################################################


# Get all palettes available in Seaborn.
SEABORN_PALETTES = \
    set(sns.palettes.SEABORN_PALETTES.keys()).union(\
        {"cubehelix", "dark", "colorblind", "husl", "hls",
         "muted", "pastel", "bright"})

# Get all color maps available in Matplotlib.
MATPLOTLIB_CMAPS = set(plt.colormaps())


#######################################################################


def recurse_config_plot(
        config: dict[str, object],
        template: dict[str, object],
        general_fontproperties: Optional[dict[str, object]] = None,
        key_path: tuple[str, ...] = ()):
    """Recursively check a plot's configuration against a template.

    Parameters
    ----------
    config : :class:`dict`
        The configuration, or an option's value.

    template : :class:`dict`
        The section of the template matching ``config``.

    general_fontproperties : :class:`dict`, optional
        The general font properties.

    key_path : :class:`tuple`, ``()``
        The keys leading to ``config`` in the configuration.

    Returns
    -------
    config : :class:`dict`
        The pruned configuration, or the option's value.

    errors : :class:`list`
        The list of errors found in the configuration.

    warnings : :class:`list`
        The list of warnings found in the configuration.
    """

    # If no general font properties were passed
    if general_fontproperties is None:

        # Initialize them as an empty dictionary.
        general_fontproperties = {}

    # Initialize an empty list to store the errors.
    errors = []

    # Initialize an empty list to store the warnings.
    warnings = []

    # If the configuration is not a dictionary but the template is
    if not isinstance(config, dict) and isinstance(template, dict):

        # If the template describes a section, not an option
        if "dtypes" not in template or "help" not in template:

            # Get the section's name.
            section_name = key_path[-1] if key_path else "<root>"

            # Set the error message.
            errstr = \
                f"Section '{section_name}' must be a dictionary."

            # Append the error message to the list of errors.
            errors.append(errstr)

            # Return the configuration, the errors, and the warnings.
            return config, errors, warnings

        # Get the supported data types, including None.
        dtypes = template["dtypes"] + (None.__class__,)

        # Initialize an empty list to store the errors about lists.
        list_errors = []

        # Set a flag to check whether a supported data type is found.
        dtype_found = False

        # For each supported data type
        for dtype in dtypes:

            # If the data type is a list
            if isinstance(dtype, list):

                # If the value is not a list of the expected length
                if not isinstance(config, list) \
                or len(config) != len(dtype):

                    # Set the error message.
                    err_msg = \
                        f"Expected {len(dtype)} values " \
                        f"for option '{key_path[-1]}'."

                    # Append the error message to the list of
                    # errors about lists.
                    list_errors.append(err_msg)

                    # Continue to the next supported data type.
                    continue

                # Set a flag to check whether all values match.
                values_match = True

                # For each value in the list
                for i, v in enumerate(config):

                    # If the value is not of the expected type
                    if not isinstance(v, dtype[i]):

                        # Set the error message.
                        errstr = \
                            f"Element {i} of option " \
                            f"'{key_path[-1]}' must be of type " \
                            f"{dtype[i]}."

                        # Append the error message to the
                        # list of errors about lists.
                        list_errors.append(errstr)

                        # Set the flag to False.
                        values_match = False

                # If all values are of the expected types
                if values_match:

                    # Set the flag to True.
                    dtype_found = True

                    # Break the loop.
                    break

            # Otherwise
            else:

                # If the value is of a supported type
                if isinstance(config, dtype):

                    # Set the flag to True.
                    dtype_found = True

                    # Break the loop.
                    break

        #-------------------------------------------------------------#

        # If no supported data type was found
        if not dtype_found:

            # Add the errors about lists to the list of errors.
            errors.extend(list_errors)

            # Get the supported data types as a string.
            dtypes_str = \
                ", ".join([f"'{d}'" for d in dtypes])

            # Get the help message for the option.
            help_msg = template["help"]

            # Set the error message.
            errstr = \
                f"Option '{key_path[-1]}' must be of either of " \
                f"these types: {dtypes_str}. The option " \
                "defines: " \
                f"{help_msg[0].lower() + help_msg[1:]}"

            # Append the error message to the list of errors.
            errors.append(errstr)

            # Return the configuration, the errors, and the warnings.
            return config, errors, warnings

        #-------------------------------------------------------------#

        # Return the configuration, the errors, and the warnings.
        return config, errors, warnings

    #-----------------------------------------------------------------#

    # If there is a configuration for the general font properties
    if "general_fontproperties" in config:

        # Set the general font properties.
        general_fontproperties.update(config["general_fontproperties"])

    #-----------------------------------------------------------------#

    # Initialize an empty dictionary to store the pruned configuration.
    pruned_dict = {}

    # For each key in the configuration
    for key in config:

        # If the key is in the template
        if key in template:

            # Get the value of the key in the configuration.
            actual_value = config[key]

            # Get the value of the key in the template.
            template_value = template[key]

            #---------------------------------------------------------#

            # If the key indicates that the section is a font
            # properties section
            if key in ("prop", "fontproperties",
                       "title_fontproperties"):

                # If the value is a dictionary
                if isinstance(actual_value, dict):

                    # Get the general font properties, if any.
                    opts = dict(general_fontproperties)

                    # Update the general font properties with the
                    # specific ones.
                    opts.update(actual_value)

                    # For each option in the dictionary
                    for opt, opt_val in actual_value.items():

                        # If the option has conflicts
                        if "conflicts" in template[key][opt]:

                            # Get conflicting options.
                            conflicts = \
                                set(opts.keys()).intersection(\
                                    template[key][opt]["conflicts"])

                            # If the option conflicts with any other
                            # option
                            if conflicts:

                                # Get the priority of the option.
                                has_priority = \
                                    template[key][opt]["has_priority"]

                                # If the option does not have priority
                                if not has_priority:

                                    # Remove the option from the
                                    # dictionary.
                                    opts.pop(opt)

                                    # Get the conflicting options as
                                    # a string.
                                    conflicts_str = \
                                        ", ".join([f"'{c}'" \
                                                   for c in conflicts])

                                    # Warn the user that the option
                                    # will be ignored.
                                    warn_msg = \
                                        f"Option '{opt}' conflicts " \
                                        "with option(s) " \
                                        f"{conflicts_str}, and has " \
                                        "lower priority. Therefore, " \
                                        "it will be ignored."
                                    warnings.append(warn_msg)

                    # Get the font properties.
                    fp = fm.FontProperties(**opts)

                    # Set the font properties.
                    pruned_dict[key] = fp

                #-----------------------------------------------------#

                # Otherwise
                else:

                    # Set the error message.
                    errstr = \
                        "All 'prop'/'fontproperties'/" \
                        "'title_fontproperties' sections must " \
                        "be dictionaries."

                    # Append the error message to the list of
                    # errors.
                    errors.append(errstr)

                    # Continue to the next key.
                    continue

            #---------------------------------------------------------#

            # Otherwise
            else:

                # If the template's value is a dictionary
                if isinstance(template_value, dict):

                    # Use the key as the new key by default.
                    new_key = key

                    # If the key is a 'boxplot' option
                    if len(key_path) > 0 and key_path[-1] == "boxplot":

                        # If the key is a 'meanprops' option
                        if key.startswith("meanprops_"):

                            # Get the meanline option.
                            meanline = config.get("meanline", False)

                            # If the properties do not match the
                            # 'meanline' setting
                            if not (meanline \
                                    and key == "meanprops_line") \
                            and not (not meanline \
                                     and key == "meanprops_marker"):

                                # Skip the key.
                                continue

                            # Otherwise
                            else:

                                # Set the new key to 'meanprops'.
                                new_key = "meanprops"

                    #-------------------------------------------------#

                    # Recurse into the nested dictionaries.
                    pruned_value, nested_errors, nested_warnings = \
                        recurse_config_plot(\
                            config = actual_value,
                            template = template_value,
                            general_fontproperties = \
                                general_fontproperties,
                            key_path = key_path + (key,))

                    # Add the nested errors to the list of errors.
                    errors.extend(nested_errors)

                    # Add the nested warnings to the list of warnings.
                    warnings.extend(nested_warnings)

                    # Update the pruned dictionary.
                    pruned_dict[new_key] = pruned_value

        #-------------------------------------------------------------#

        # Otherwise
        else:

            # Warn the user that the key will be ignored.
            warn_msg = \
                f"Option '{key}' is not recognized and will be " \
                "ignored."
            warnings.append(warn_msg)

    #-----------------------------------------------------------------#

    # Return the pruned dictionary and the lists of errors and
    # warnings.
    return pruned_dict, errors, warnings


#######################################################################


def parse_config_plot(config: dict[str, object]) -> \
        tuple[dict[str, object],
              list[str],
              list[str]]:
    """Parse and check the configuration passed for a plot.

    Parameters
    ----------
    config : :class:`dict`
        The configuration.

    Returns
    -------
    config : :class:`dict`
        The parsed configuration.

    errors : :class:`list`
        The list of errors found in the configuration.

    warnings : :class:`list`
        The list of warnings found in the configuration.
    """

    # Check the configuration.
    config, errors, warnings = recurse_config_plot(\
                        config = config,
                        template = _templates.CONFIG_PLOT_TEMPLATE)

    #-----------------------------------------------------------------#

    # Return the configuration, the errors, and the warnings.
    return config, errors, warnings


def check_config_plot(config: dict[str, object]) -> \
        tuple[dict[str, object],
              list[str]]:
    """Check the configuration passed for a plot.

    Parameters
    ----------
    config : :class:`dict`
        The configuration.

    Returns
    -------
    config : :class:`dict`
        The parsed configuration.

    errors : :class:`list`
        The list of errors found in the configuration.
    """

    # Check the configuration.
    config, errors, warnings = parse_config_plot(config = config)

    # For each warning
    for warn_msg in warnings:

        # Log it.
        logger.warning(warn_msg)

    # Return the configuration and the errors.
    return config, errors


#######################################################################


def split_text_by_length(text: object,
                         max_length: int) -> str:
    """Split a text into pieces of at most a given length.

    Parameters
    ----------
    text : :class:`str`
        The text to split.

    max_length : :class:`int`
        The maximum length of a piece of text.

    Returns
    -------
    result : :class:`str`
        The text split into pieces of at most ``max_length``
        characters.
    """

    # Split the text into words.
    words = text.split()

    #-----------------------------------------------------------------#

    # Initialize an empty list to store the pieces.
    pieces = []

    # Initialize an empty string to store the current piece.
    current_piece = ""

    #-----------------------------------------------------------------#

    # For each word in the text
    for word in words:

        # If there is a current piece
        if current_piece:

            # If the current piece plus the current word fit in the
            # maximum length
            if len(current_piece) + 1 + len(word) <= max_length:

                # Add the current word to the current piece.
                current_piece += " " + word

            # Otherwise
            else:

                # Add the current piece to the list of pieces.
                pieces.append(current_piece)

                # Start a new piece with the current word.
                current_piece = word

        #-------------------------------------------------------------#

        # Otherwise
        else:

            # If the current word fits in the maximum length
            if len(word) <= max_length:

                # Start a new piece with the current word.
                current_piece = word

            # Otherwise
            else:

                # Add the whole word to the list of pieces.
                pieces.append(word)

                # Start a new piece.
                current_piece = ""

    #-----------------------------------------------------------------#

    # If there is a current piece
    if current_piece:

        # Add it to the list of pieces.
        pieces.append(current_piece)

    #-----------------------------------------------------------------#

    # Join the pieces with newline characters.
    result = '\n'.join(pieces)

    #-----------------------------------------------------------------#

    # Return the result.
    return result


def get_formatted_ticklabels(
        ticklabels: tuple[np.ndarray, list[str]],
        fmt: str = "{:s}",
        max_length: int = 20) -> list[str]:
    """Return the ticks' labels, formatted according to a given format
    string.

    Parameters
    ----------
    ticklabels : :class:`numpy.ndarray` or :class:`list`
        An array or list of labels.

    fmt : :class:`str`, ``"{:s}"``
        The format string.

    max_length : :class:`int`, ``20``
        The maximum length of a tick's label. If a label exceeds this
        length, it will be split into pieces of at most this length.
        The pieces will be separated by newline characters.

    Returns
    -------
    ticklabels : :class:`list`
        A list with the formatted ticks' labels.
    """

    # Initialize an empty list to store the formatted ticks' labels.
    fmt_ticklabels = []

    #-----------------------------------------------------------------#

    # For each tick's label
    for ticklabel in ticklabels:

        # Format the label.
        fmt_ticklabel = fmt.format(ticklabel)

        #-------------------------------------------------------------#

        # If the label is a single 0
        if fmt_ticklabel == "0":

            # Add it to the list.
            fmt_ticklabels.append(fmt_ticklabel)

            # Go to the next one.
            continue

        #-------------------------------------------------------------#

        # If the label is a float
        if "." in fmt_ticklabel or "," in fmt_ticklabel:

            # Strip the label of trailing zeroes.
            fmt_ticklabel = fmt_ticklabel.rstrip("0")

        #-------------------------------------------------------------#

        # If the label now ends with a dot or a comma (it was an
        # integer, e.g., 1.0 or 3,00)
        if fmt_ticklabel.endswith(".") or fmt_ticklabel.endswith(","):

            # Remove the dot/comma.
            fmt_ticklabel = fmt_ticklabel.rstrip(".").rstrip(",")

        #-------------------------------------------------------------#

        # Split the label into lines of at most 'max_length'
        # characters.
        fmt_ticklabel = \
            split_text_by_length(text = fmt_ticklabel,
                                 max_length = max_length)

        #-------------------------------------------------------------#

        # Add the label to the list.
        fmt_ticklabels.append(fmt_ticklabel)

    #-----------------------------------------------------------------#

    # Return the labels.
    return fmt_ticklabels


def find_rectangular_grid(n: int) -> tuple[int, int]:
    """Find the 'squarest' two-dimensional grid for ``n`` items.

    If ``n`` is prime, the grid has one more cell than items.

    Parameters
    ----------
    n : :class:`int`
        The number of items.

    Returns
    -------
    nrows : :class:`int`
        The number of rows in the squarest grid.

    ncols : :class:`int`
        The number of columns in the squarest grid.
    """

    # If we have only one item
    if n == 1:

        # The grid will have one row and one column.
        return (1, 1)

    #-----------------------------------------------------------------#

    # If we have two items
    elif n == 2:

        # The grid will have one row and two columns.
        return (1, 2)

    #-----------------------------------------------------------------#

    # Initialize the number of rows and columns as one row and 'n'
    # columns.
    nrows, ncols = (1, n)

    #-----------------------------------------------------------------#

    # For each possible number ranging from 2 to the square root of
    # 'n', plus 1
    for i in range(2, int(math.sqrt(n)) + 1):

        # If the number is a factor of 'n'
        if n % i == 0:

            # Update the number of rows and columns in the grid.
            nrows, ncols = (i, n // i)

    #-----------------------------------------------------------------#

    # If there is only one row ('n' is prime)
    if nrows == 1:

        # For each possible number ranging from 2 to the square root
        # of 'n + 1', plus 1
        for i in range(2, int(math.sqrt(n+1)) + 1):

            # If the number is a factor of 'n + 1'
            if (n+1) % i == 0:

                # Update the number of rows and columns in the grid.
                nrows, ncols = (i, (n+1) // i)

    #-----------------------------------------------------------------#

    # Return the number of rows and columns.
    return (nrows, ncols)


def get_ticks_positions(values: tuple[np.ndarray, list[str]],
                        item: str,
                        config: dict[str, object]) -> np.ndarray:
    """Generate the positions of the ticks on a plot's axis or
    colorbar.

    Parameters
    ----------
    values : :class:`list` or :class:`numpy.ndarray`
        The values from which the ticks' positions should be set.

    item : :class:`str`
        The name of the item of the plot you are setting the ticks'
        positions for (e.g., ``"x-axis"``, ``"y-axis"``, or
        ``"colorbar"``).

    config : :class:`dict`
        The configuration for the interval that the ticks' positions
        should cover.

    Returns
    -------
    ticks_positions : :class:`numpy.ndarray`
        An array containing the ticks' positions.
    """

    # Get the configuration for the interval.
    config = config.get("interval", {})

    #-----------------------------------------------------------------#

    # Get the interval's options.
    int_type = config.get("type", "continuous")
    rtn = config.get("round_to_nearest")
    top = config.get("top")
    bottom = config.get("bottom")
    steps = config.get("steps")
    spacing = config.get("spacing")
    caz = config.get("center_around_zero")

    # Inform the user that the ticks' interval is being set.
    debugstr = \
        f"Now setting the interval for the plot's {item}'s ticks..."
    logger.debug(debugstr)

    #-----------------------------------------------------------------#

    # If no rounding was specified
    if rtn is None:

        # If the interval is discrete
        if int_type == "discrete":

            # Default to rounding to the nearest 1.
            rtn = 1

        # If the interval is continuous
        elif int_type == "continuous":

            # Default to rounding to the nearest 0.5.
            rtn = 0.5

        # Inform the user about the rounding value.
        debugstr = \
            "Since 'round_to_nearest' is not defined and 'type' " \
            f"is '{int_type}', the rounding will be set to the " \
            f"nearest {rtn}."
        logger.debug(debugstr)

    # Otherwise
    else:

        # Inform the user about the chosen rounding value.
        debugstr = \
            "The user set the rounding (up and down) to the nearest " \
            f"{rtn} ('round_to_nearest' = {rtn})."
        logger.debug(debugstr)

    #-----------------------------------------------------------------#

    # If the maximum of the ticks interval was not specified
    if top is None:

        # If the interval is discrete
        if int_type == "discrete":

            # Set the top value to the maximum value.
            top = int(np.ceil(max(values)))

            # Inform the user about the top value.
            debugstr = \
                "Since 'top' is not defined and 'type' is " \
                f"'{int_type}', 'top' will be the maximum of all " \
                f"values found, ({top})."
            logger.debug(debugstr)

        # If the interval is continuous
        elif int_type == "continuous":

            # Set the top value to the maximum value, rounded up.
            top = np.ceil(max(values)/rtn) * rtn

            # Inform the user about the top value.
            debugstr = \
                "Since 'top' is not defined and 'type' is " \
                f"'{int_type}', 'top' will be the maximum of all " \
                f"values found, rounded up to the nearest {rtn} " \
                f"({top})."
            logger.debug(debugstr)

    # Otherwise
    else:

        # Inform the user about the chosen top value.
        debugstr = \
            f"The user set the top value to {top} ('top' = {top})."
        logger.debug(debugstr)

    #-----------------------------------------------------------------#

    # If the minimum of the ticks interval was not specified
    if bottom is None:

        # If the interval is discrete
        if int_type == "discrete":

            # Set the bottom value to the minimum value.
            bottom = int(min(values))

            # Inform the user about the bottom value.
            debugstr = \
                f"Since 'bottom' is not defined and 'type' is " \
                f"'{int_type}', 'bottom' will be the minimum of all " \
                f"values found ({bottom})."
            logger.debug(debugstr)

        # If the interval is continuous
        elif int_type == "continuous":

            # Set the bottom value to the minimum value, rounded down.
            bottom = np.floor(min(values)/rtn) * rtn

            # Inform the user about the bottom value.
            debugstr = \
                "Since 'bottom' is not defined and 'type' is " \
                f"'{int_type}', 'bottom' will be the minimum of all " \
                f"values found, rounded down to the nearest {rtn} " \
                f"({bottom})."
            logger.debug(debugstr)

    # Otherwise
    else:

        # Inform the user about the chosen bottom value.
        debugstr = \
            f"The user set the bottom value to {bottom} ('bottom' " \
            f"= {bottom})."
        logger.debug(debugstr)

    #-----------------------------------------------------------------#

    # If the two extremes of the interval coincide
    if top == bottom:

        # Return only one value.
        return np.array([bottom])

    #-----------------------------------------------------------------#

    # If the number of steps in the interval was not specified
    if steps is None:

        # Set a default of 10 steps.
        steps = 10

        # Inform the user about the steps.
        debugstr = \
            "Since the number of steps in the interval is not " \
            "defined, 'steps' will be '10'."
        logger.debug(debugstr)

    # Otherwise
    else:

        # Inform the user about the chosen number of steps.
        debugstr = \
            "The user set the number of steps the interval " \
            f"should have to {steps} ('steps' = {steps})."
        logger.debug(debugstr)

    #-----------------------------------------------------------------#

    # If the interval spacing was not specified
    if spacing is None:

        # If the interval is discrete
        if int_type == "discrete":

            # Set the spacing between two steps, rounded up.
            spacing = \
                int(np.ceil(np.linspace(bottom,
                                        top,
                                        steps,
                                        retstep = True)[1]))

            # Inform the user about the spacing.
            debugstr = \
                "Since the spacing between the ticks is not " \
                "defined, 'spacing' will be the value " \
                "guaranteeing an equipartition of the interval " \
                f"between {bottom} and {top} in {steps} " \
                "number of steps, rounded up to the nearest 1 " \
                f"({spacing})."
            logger.debug(debugstr)

        # If the interval is continuous
        elif int_type == "continuous":

            # Get the spacing between two steps.
            spacing = np.linspace(bottom,
                                  top,
                                  steps,
                                  retstep = True)[1]

            # Round the spacing up to the nearest 'rtn'.
            spacing = np.ceil(spacing / rtn) * rtn

            # Inform the user about the spacing.
            debugstr = \
                "Since the spacing between the ticks is not " \
                "defined, 'spacing' will be the value " \
                "guaranteeing an equipartition of the interval " \
                f"between {bottom} and {top} in {steps} " \
                f"number of steps ({spacing})."
            logger.debug(debugstr)

    #-----------------------------------------------------------------#

    # If the interval should be centered in zero
    if caz:

        # Get the highest absolute value.
        absval = np.ceil(max(abs(top), abs(bottom)))

        # Set the top and bottom to +/- the highest absolute value.
        top, bottom = absval, -absval

        # Get an evenly-spaced interval between the bottom and top
        # values.
        interval = np.arange(bottom,
                             top + spacing,
                             spacing)

        # Inform the user about the change in the interval.
        debugstr = \
            "Since the user requested a ticks' interval centered " \
            f"in zero, the interval will be between {top} " \
            f"and {bottom} with {steps} number of steps: " \
            f"{', '.join([str(i) for i in interval.tolist()])}."
        logger.debug(debugstr)

        # Return the interval.
        return interval

    #-----------------------------------------------------------------#

    # Get the interval.
    interval = np.arange(bottom,
                         top + spacing,
                         spacing)

    # Inform the user about the interval that will be used.
    debugstr = \
        f"The ticks' interval will be between {bottom} and {top} " \
        f"with a spacing of {spacing}: " \
        f"{', '.join([str(i) for i in interval.tolist()])}."
    logger.debug(debugstr)

    #-----------------------------------------------------------------#

    # Return the interval.
    return interval


def get_colormap(cmap: str,
                 n_colors: Optional[int] = None) -> \
                    mcolors.Colormap | list:
    """Discretize a given colormap into a specific number of colors.

    Parameters
    ----------
    cmap : :class:`str`
        The name of a color map available in Seaborn or Matplotlib.

    n_colors : :class:`int`, optional
        The number of colors the palette should be discretized into.

    Returns
    -------
    cmap : :class:`matplotlib.colors.Colormap` or :class:`list`
        Either the color map's object or a list of colors constituting
        the color map.
    """

    # If the color map is available both in Seaborn and in Matplotlib
    if cmap in SEABORN_PALETTES and cmap in MATPLOTLIB_CMAPS:

        # Inform the user about it.
        infostr = \
            f"The '{cmap}' color map is available in both " \
            "Matplotlib and in Seaborn. The Seaborn " \
            "implementation will be used."
        logger.info(infostr)

        # Get the color map from Seaborn.
        cmap = sns.color_palette(cmap, as_cmap = True)

    #-----------------------------------------------------------------#

    # If the color map is available in Seaborn
    elif cmap in SEABORN_PALETTES:

        # Get the color map from Seaborn.
        cmap = sns.color_palette(cmap, as_cmap = True)

    #-----------------------------------------------------------------#

    # If the color map is available in Matplotlib
    elif cmap in MATPLOTLIB_CMAPS:

        # Get the color map from Matplotlib.
        cmap = plt.get_cmap(cmap)

    #-----------------------------------------------------------------#

    # Otherwise
    else:

        # Get the names of all the available color maps.
        available_cmaps = \
            ", ".join([f"'{c}'" for c \
                       in SEABORN_PALETTES.union(MATPLOTLIB_CMAPS)])

        # Raise an error.
        errstr = \
            f"Unrecognized '{cmap}' color map. Available color " \
            f"maps are: {available_cmaps}."
        raise ValueError(errstr)

    #-----------------------------------------------------------------#

    # If a discrete number of colors was specified
    if n_colors:

        # Tell whether the color map ends where it starts.
        cyclic = np.allclose(cmap(0.0),
                             cmap(1.0),
                             atol = 0.02)

        # Discretize the color map, not repeating the start of a
        # cyclic one.
        colors = cmap(np.linspace(0,
                                  1,
                                  n_colors,
                                  endpoint = not cyclic))

        # Return the list of colors.
        return [c.tolist() for c \
                in mcolors.ListedColormap(colors).colors]

    #-----------------------------------------------------------------#

    # Return the color map.
    return cmap


def get_colors(config: dict[str, object],
               len_data: int) -> list[str]:
    """Get the color(s) that will be used for a plot.

    Parameters
    ----------
    config : :class:`dict`
        The configuration for the plot.

    len_data : :class:`int`
        The number of data frames/categories that will be plotted.

    Returns
    -------
    colors_plot : :class:`list`
        The colors that will be used for the plot.
    """

    # Get the configuration for the colors to be used for the
    # sub-plots.
    config_colors = config.get("colors", {})

    #-----------------------------------------------------------------#

    # If no configuration for the colors was provided
    if not {"color", "colors", "cmap"} & config_colors.keys():

        # Get the colors that will be used for the sub-plots from the
        # default color map.
        colors_plot = get_colormap(cmap = "husl",
                                   n_colors = len_data)

    #-----------------------------------------------------------------#

    # If there is a single color specified in the configuration
    if "color" in config_colors:

        # Use the color for all sub-plots.
        colors_plot = [config_colors["color"]]

    #-----------------------------------------------------------------#

    # If there is a list of colors defined
    if "colors" in config_colors:

        # If the list of colors is not a list
        if not isinstance(config_colors["colors"], list):

            # Raise an error.
            errstr = \
                "The 'colors:colors' option must be a list of colors."
            raise TypeError(errstr)

        #-------------------------------------------------------------#

        # Get how many colors were provided.
        num_colors = len(config_colors["colors"])

        #-------------------------------------------------------------#

        # If there are fewer colors than the number of sub-plots
        if num_colors < len_data:

            # Warn the user that the colors will be re-used.
            warnstr = \
                f"{num_colors} colors were provided. The colors " \
                "will be re-used."
            logger.warning(warnstr)

            # Set a cycle over the colors.
            cycle_colors = itertools.cycle(config_colors["colors"])

            # Get the colors by cycling through the available ones.
            colors_plot = [next(cycle_colors) for _ in range(len_data)]

        #-------------------------------------------------------------#

        # If there are more colors than the number of sub-plots
        elif num_colors > len_data:

            # Warn the user that the extra colors will be ignored.
            warnstr = \
                f"{num_colors} colors were provided. The extra " \
                "colors will be ignored."
            logger.warning(warnstr)

            # Set the list of colors to the first 'len_data' colors.
            colors_plot = config_colors["colors"][:len_data]

        #-------------------------------------------------------------#

        # If the number of colors is the same as the number of
        # sub-plots
        else:

            # Use the colors as they are.
            colors_plot = config_colors["colors"]

    #-----------------------------------------------------------------#

    # If there is a color map defined
    if "cmap" in config_colors:

        # Get the colors that will be used for the sub-plots from the
        # color map.
        colors_plot = \
            get_colormap(cmap = config_colors["cmap"],
                         n_colors = len_data)

    #-----------------------------------------------------------------#

    # Return the colors.
    return colors_plot


#######################################################################


def set_figure(num_plots: int,
               config: Optional[dict[str, object]] = None) -> \
                tuple[matplotlib.figure.Figure, np.ndarray]:
    """Set up a figure and the sub-plots for a plot.

    Parameters
    ----------
    num_plots : :class:`int`
        The number of plots that will be generated.

    config : :class:`dict`, optional
        The configuration for the figure and the sub-plots.

    Returns
    -------
    fig : :class:`matplotlib.figure.Figure`
        The figure.

    sub_plots : :class:`numpy.ndarray`
        The sub-plots.
    """

    # If no configuration was passed
    if config is None:

        # Use an empty configuration.
        config = {}

    # Get the squarest grid for the sub-plots.
    nrows, ncols = find_rectangular_grid(n = num_plots)

    # Get the figure's size.
    fig_size = \
        config.get("sizeinches",
                   (ncols * 5, nrows * 5))

    # Generate the figure and subplots.
    fig, sub_plots = \
        plt.subplots(\
            nrows = nrows,
            ncols = ncols,
            figsize = fig_size)

    # If there is only one axis
    if not isinstance(sub_plots, np.ndarray):

        # Convert it to an array.
        sub_plots = np.array([sub_plots])

    #-----------------------------------------------------------------#

    # If there is a configuration to adjust the figure
    if "subplots" in config:

        # Adjust the sub-plots.
        plt.subplots_adjust(**config["subplots"])

    #-----------------------------------------------------------------#

    # Return the figure and the sub-plots.
    return fig, sub_plots


def set_title(title: str,
              sub_plot: matplotlib.axes.Axes,
              config: Optional[dict[str, object]] = None) -> \
                matplotlib.axes.Axes:
    """Set the title of a plot.

    Parameters
    ----------
    title : :class:`str`
        The title of the plot.

    sub_plot : :class:`matplotlib.axes.Axes`
        The sub-plot.

    config : :class:`dict`, optional
        The configuration for setting the title.

    Returns
    -------
    sub_plot : :class:`matplotlib.axes.Axes`
        The sub-plot.
    """

    # Create a copy of the configuration, if any.
    config = dict(config) if config is not None else {}

    # Get the maximum length of a title.
    max_length = config.pop("max_length", 20)

    # Split the title into lines of at most 'max_length' characters.
    title_fmt = \
        split_text_by_length(text = title,
                             max_length = max_length)

    #-----------------------------------------------------------------#

    # Set the sub-plot's title.
    sub_plot.set_title(label = title_fmt,
                       **config)

    #-----------------------------------------------------------------#

    # Return the sub-plot.
    return sub_plot


def set_extra_subplots(sub_plots: np.ndarray,
                       x_ticks: list,
                       y_ticks: list,
                       config: Optional[dict[str, object]] = None) \
                        -> None:
    """Set the unused slots for sub-plots in a multi-panel plot.

    Parameters
    ----------
    sub_plots : :class:`numpy.ndarray`
        The sub-plots.

    x_ticks : :class:`list`
        The ticks' positions for the x-axis.

    y_ticks : :class:`list`
        The ticks' positions for the y-axis.

    config : :class:`dict`, optional
        The configuration for the figure and the sub-plots.
    """

    # If no configuration was passed
    if config is None:

        # Use an empty configuration.
        config = {}

    # For each extra sub-plot in the figure
    for sub_plot in sub_plots:

        # Set the x-axis as in the other sub-plots.
        sub_plot = set_axis(sub_plot = sub_plot,
                            axis = "x",
                            config = config.get("xaxis", {}),
                            ticks = x_ticks)

        #-------------------------------------------------------------#

        # Set the y-axis as in the other sub-plots.
        sub_plot = set_axis(sub_plot = sub_plot,
                            axis = "y",
                            config = config.get("yaxis", {}),
                            ticks = y_ticks)

        #-------------------------------------------------------------#

        # Hide it.
        sub_plot.set_visible(False)


def set_axis(sub_plot: matplotlib.axes.Axes,
             axis: str,
             config: Optional[dict[str, object]] = None,
             label: Optional[str] = None,
             ticks: Optional[list[float]] = None,
             tick_labels: Optional[list[str]] = None,
             abs_values: bool = False,
             categorical: bool = False) -> matplotlib.axes.Axes:
    """Set up the x- or y-axis after generating a plot.

    Parameters
    ----------
    sub_plot : :class:`matplotlib.axes.Axes`
        The sub-plot.

    axis : :class:`str`, {``"x"``, ``"y"``}
        Whether the axis to be set is the x- or the y-axis.

    config : :class:`dict`, optional
        The configuration for setting the axis.

    label : :class:`str`, optional
        The axis' label. If it is provided, it overrides the label
        specified in the configuration.

    ticks : :class:`list`, optional
        A list of ticks' positions. If it is not passed, the ticks
        will be those already present on the axis (automatically
        determined by matplotlib when generating the plot).

    tick_labels : :class:`list`, optional
        A list of ticks' labels. If not passed, the ticks' labels
        will represent the ticks' positions.

    abs_values : :class:`bool`, optional
        Whether the ticks' positions should be absolute values. If
        ``True``, the ticks' positions will be the absolute values of
        the original positions.

    categorical : :class:`bool`, :obj:`False`
        Whether the axis is categorical.

    Returns
    -------
    sub_plot : :class:`matplotlib.axes.Axes`
        The sub-plot.
    """

    # Create a copy of the configuration, if any.
    config = copy.deepcopy(config) if config is not None else {}

    #-----------------------------------------------------------------#

    # If the axis to be set is the x-axis
    if axis == "x":

        # Get the corresponding methods.
        plot_ticks = plt.xticks
        set_label = sub_plot.set_xlabel
        set_lim = sub_plot.set_xlim
        set_ticks = sub_plot.set_xticks
        set_ticklabels = sub_plot.set_xticklabels
        get_ticklines = sub_plot.get_xticklines

        # Get the corresponding spine.
        spine = "bottom"

    # If the axis to be set is the y-axis
    elif axis == "y":

        # Get the corresponding methods.
        plot_ticks = plt.yticks
        set_label = sub_plot.set_ylabel
        set_lim = sub_plot.set_ylim
        set_ticks = sub_plot.set_yticks
        set_ticklabels = sub_plot.set_yticklabels
        get_ticklines = sub_plot.get_yticklines

        # Get the corresponding spine.
        spine = "left"

    #-----------------------------------------------------------------#

    # If there is a configuration for the spine
    if config.get("spine") is not None:

        # For each option for the spine
        for opt, val in config["spine"].items():

            # Set the spine's option.
            sub_plot.spines[spine].set(**{opt : val})

        # If no position was set for the spine
        if "position" not in config["spine"]:

            # Set the default for the spine's position.
            sub_plot.spines[spine].set_position(("outward", 5))

    #-----------------------------------------------------------------#

    # If there is an axis label's configuration
    if config.get("label") is not None:

        # Get the label.
        label = config["label"].pop(f"{axis}label", label)

        # If a label was provided
        if label is not None:

            # Get the maximum length of a label.
            max_length = config["label"].pop("max_length", 20)

            # Split the label into lines of at most 'max_length'
            # characters.
            fmt_label = \
                split_text_by_length(text = label,
                                     max_length = max_length)

            # Set the axis label.
            set_label(\
                **{f"{axis}label" : fmt_label, **config["label"]})

    #-----------------------------------------------------------------#

    # If no ticks' positions were passed
    if ticks is None:

        # Default to the tick locations already present.
        ticks = plot_ticks()[0]

    #-----------------------------------------------------------------#

    # Get the margin past the first and last ticks (half a category
    # for a categorical axis).
    margin = 0.5 if categorical else 0.0

    # If there are any ticks on the axis
    if len(ticks) > 0:

        # Set the axis boundaries.
        sub_plot.spines[spine].set_bounds(ticks[0] - margin,
                                          ticks[-1] + margin)

        # Set the axis limits.
        set_lim(ticks[0] - margin, ticks[-1] + margin)

    #-----------------------------------------------------------------#

    # Set the ticks.
    set_ticks(ticks = ticks)

    #-----------------------------------------------------------------#

    # If a configuration for the tick parameters was provided
    if config.get("tick_params") is not None:

        # Get the configuration for the tick parameters.
        config_tick_params = \
            {opt : val for opt, val in config["tick_params"].items() \
             if opt != "alpha"}

        # Apply the configuration to the ticks.
        sub_plot.tick_params(axis = axis,
                             **config_tick_params)

        # Get the alpha value, if provided.
        alpha = config["tick_params"].get("alpha", None)

        # If an alpha was provided
        if alpha is not None:

            # For each tick
            for tick in get_ticklines():

                # Set the alpha.
                tick.set_alpha(alpha)

    #-----------------------------------------------------------------#

    # Get the configuration for the ticks' labels.
    tick_labels_config = config.get("ticklabels", {})

    # Get the options for the ticks' labels.
    tick_labels_options = tick_labels_config.get("options", {})

    # If no ticks' labels were passed
    if tick_labels is None:

        # If the ticks' labels should be the absolute values of the
        # ticks
        if abs_values:

            # Get the absolute values of the ticks.
            tick_labels = np.abs(ticks)

        # Otherwise
        else:

            # Use the ticks' positions as labels.
            tick_labels = ticks

        # Get the format to be used for the tick labels.
        tick_labels_fmt = tick_labels_config.get("fmt", "{:.3f}")

        # Get the maximum length of a tick's label.
        max_length = tick_labels_config.get("max_length", 20)

        # Format the ticks' labels.
        tick_labels = \
            get_formatted_ticklabels(ticklabels = tick_labels,
                                     fmt = tick_labels_fmt,
                                     max_length = max_length)

    # Set the ticks' labels.
    set_ticklabels(labels = tick_labels,
                   **tick_labels_options)

    #-----------------------------------------------------------------#

    # Return the sub-plot.
    return sub_plot


def set_cbar_axis(cbar: matplotlib.colorbar.Colorbar,
                  config: Optional[dict[str, object]] = None,
                  label: Optional[str] = None,
                  ticks: Optional[list[float]] = None,
                  tick_labels: Optional[list[str]] = None) -> \
                    matplotlib.colorbar.Colorbar:
    """Set up the colorbar after generating a plot.

    Parameters
    ----------
    cbar : :class:`matplotlib.colorbar.Colorbar`
        The colorbar.

    config : :class:`dict`, optional
        The configuration for setting the colorbar's axis.

    label : :class:`str`, optional
        The colorbar's label. If it is provided, it overrides the
        label specified in the configuration.

    ticks : :class:`list`, optional
        A list of ticks' positions.

        If it is not passed, the ticks will be those already present
        on the colorbar's axis (automatically determined by Matplotlib
        when generating the plot).

    tick_labels : :class:`list`, optional
        A list of ticks' labels.

        If not passed, the ticks' labels will represent the ticks'
        positions.

    Returns
    -------
    cbar : :class:`matplotlib.colorbar.Colorbar`
        The colorbar.
    """

    # Create a copy of the configuration, if any.
    config = copy.deepcopy(config) if config is not None else {}

    #-----------------------------------------------------------------#

    # If there is a configuration for the spine
    if config.get("spine") is not None:

        # For each option for the spine
        for opt, val in config["spine"].items():

            # Set the outline's option.
            cbar.outline.set(**{opt : val})

    #-----------------------------------------------------------------#

    # If there is a configuration for the colorbar's label
    if config.get("label") is not None:

        # Get the label.
        label = config["label"].pop("label", label)

        # If a label was provided
        if label is not None:

            # Get the maximum length of a label.
            max_length = config["label"].pop("max_length", 20)

            # Split the label into lines of at most 'max_length'
            # characters.
            fmt_label = \
                split_text_by_length(text = label,
                                     max_length = max_length)

            # Set the colorbar's label.
            cbar.set_label(\
                **{"label" : fmt_label, **config["label"]})

    #-----------------------------------------------------------------#

    # If no ticks' positions were passed
    if ticks is None:

        # Default to the ticks' locations already present.
        ticks = cbar.get_ticks()

    # Set the ticks.
    cbar.set_ticks(ticks = ticks)

    #-----------------------------------------------------------------#

    # If a configuration for the tick parameters was provided
    if config.get("tick_params") is not None:

        # Get the configuration for the tick parameters.
        config_tick_params = \
            {opt : val for opt, val in config["tick_params"].items() \
             if opt != "alpha"}

        # Apply the configuration to the ticks.
        cbar.ax.tick_params(axis = "both",
                            **config_tick_params)

        # Get the alpha value, if provided.
        alpha = config["tick_params"].get("alpha", None)

        # If an alpha was provided
        if alpha is not None:

            # If the colorbar is vertical
            if cbar.orientation == "vertical":

                # For each tick
                for tick in cbar.ax.get_yticklines():

                    # Set the alpha.
                    tick.set_alpha(alpha)

            # If the colorbar is horizontal
            elif cbar.orientation == "horizontal":

                # For each tick
                for tick in cbar.ax.get_xticklines():

                    # Set the alpha.
                    tick.set_alpha(alpha)

    #-----------------------------------------------------------------#

    # Get the configuration for the ticks' labels.
    tick_labels_config = config.get("ticklabels", {})

    # Get the configuration for the ticks' labels' options.
    tick_labels_options = tick_labels_config.get("options", {})

    # If no ticks' labels were passed
    if tick_labels is None:

        # Get the format to be used for the tick labels.
        tick_labels_fmt = tick_labels_config.get("fmt", "{:.3f}")

        # Get the maximum length of a tick's label.
        max_length = tick_labels_config.get("max_length", 20)

        # Format the ticks' labels.
        tick_labels = \
            get_formatted_ticklabels(ticklabels = ticks,
                                     fmt = tick_labels_fmt,
                                     max_length = max_length)

    # Set the ticks' labels.
    cbar.set_ticklabels(ticklabels = tick_labels,
                        **tick_labels_options)

    #-----------------------------------------------------------------#

    # Return the colorbar.
    return cbar


def set_legend(sub_plot: matplotlib.axes.Axes,
               config: Optional[dict[str, object]] = None) -> \
                matplotlib.axes.Axes:
    """Set a legend for the current plot.

    Parameters
    ----------
    sub_plot : :class:`matplotlib.axes.Axes`
        The sub-plot to which the legend should be added.

    config : :class:`dict`, optional
        The configuration for the legend.

    Returns
    -------
    sub_plot : :class:`matplotlib.axes.Axes`
        The sub-plot.
    """

    # Create a copy of the configuration, if any.
    config = dict(config) if config is not None else {}

    #-----------------------------------------------------------------#

    # Set the title's text properties to None.
    title_text_properties = None

    # If there are text properties for the title
    if "title_text_properties" in config:

        # Get the title's text properties.
        title_text_properties = config.pop("title_text_properties")

    #-----------------------------------------------------------------#

    # Set the labels' text properties to None.
    label_text_properties = None

    # If there are text properties for the labels
    if "label_text_properties" in config:

        # Get the labels' text properties.
        label_text_properties = config.pop("label_text_properties")

    #-----------------------------------------------------------------#

    # Get the legend's handles and labels.
    handles, labels = sub_plot.get_legend_handles_labels()

    #-----------------------------------------------------------------#

    # If there are handles
    if handles:

        # Draw the legend.
        legend = \
            sub_plot.legend(handles = handles,
                            labels = labels,
                            bbox_transform = plt.gcf().transFigure,
                            **config)

    # Otherwise
    else:

        # Return the sub-plot.
        return sub_plot

    #-----------------------------------------------------------------#

    # If there are text properties for the title
    if title_text_properties is not None:

        # Set the title's text properties.
        legend.get_title().set(**title_text_properties)

    #-----------------------------------------------------------------#

    # If there are text properties for the labels
    if label_text_properties is not None:

        # For each label's text
        for text in legend.get_texts():

            # Set its text properties.
            text.set(**label_text_properties)

    #-----------------------------------------------------------------#

    # Return the sub-plot.
    return sub_plot


def set_text(fig: matplotlib.figure.Figure,
             config: Optional[dict[str, object]] = None) -> \
                matplotlib.figure.Figure:
    """Write text on a plot.

    Parameters
    ----------
    fig : :class:`matplotlib.figure.Figure`
        The figure containing the plot to which the text should be
        added.

    config : :class:`dict`, optional
        The configuration for the text.

    Returns
    -------
    fig : :class:`matplotlib.figure.Figure`
        The figure.
    """

    # If no configuration was passed
    if config is None:

        # Use an empty configuration.
        config = {}

    # Add the text.
    fig.text(**config)

    #-----------------------------------------------------------------#

    # Return the figure.
    return fig


#######################################################################


def generate_plots(dfs: list[pd.DataFrame],
                   output_file: str,
                   max_plots_per_output: int,
                   plot_type: str,
                   config: Optional[dict[str, object]] = None,
                   plot_func_kwargs: Optional[object] = None,
                   config_default: Optional[dict[str, object]] = None,
                   dfs_2: Optional[list[pd.DataFrame]] = None,
                   dfs_names: Optional[list[str]] = None,
                   categories: Optional[list[str]] = None,
                   kwargs: Optional[dict[str, object]] = None) -> None:
    """Generate one or multiple plots from a data frame or a list of
    data frames.

    Parameters
    ----------
    dfs : :class:`list`
        A list of data frames.

    output_file : :class:`str`
        The name of the output file.

    max_plots_per_output : :class:`int`
        The number of plots that should be written in each output
        file.

    plot_type : :class:`str`
        The type of plot that will be generated.

    config : :class:`dict`, optional
        The configuration for the plot.

    plot_func_kwargs : :class:`dict`, optional
        Additional keyword arguments to be passed to the plot
        function.

    config_default : :class:`dict`, optional
        The default configuration for the plot.

    dfs_2 : :class:`list`, optional
        A list of data frames paired with the data frames in ``dfs``.

    dfs_names : :class:`list`, optional
        A list of names for the data frames.

    categories : :class:`list`, optional
        A list of categories that will be plotted.

    kwargs : :class:`dict`, optional
        Additional keyword arguments.
    """

    # If no keyword arguments for the plot function were passed
    if plot_func_kwargs is None:

        # Use an empty dictionary.
        plot_func_kwargs = {}

    # If no keyword arguments were passed
    if kwargs is None:

        # Use an empty dictionary.
        kwargs = {}

    #-----------------------------------------------------------------#

    # Remove the keyword arguments that are not needed.
    kwargs = \
        {k : v for k, v in kwargs.items() \
         if k not in ["dfs", "output_file", "max_plots_per_output",
                      "plot_type", "config", "dfs_2", "dfs_names",
                      "categories"]}

    #-----------------------------------------------------------------#

    # Merge the default configuration, the configuration provided,
    # and the keyword arguments (later ones take priority).
    config = \
        _internals.recursive_merge_dicts(\
            config_default if config_default is not None else {},
            config if config is not None else {},
            kwargs)

    #-----------------------------------------------------------------#

    # Check the configuration.
    config, errors = check_config_plot(config = config)

    # If there are errors in the configuration
    if errors:

        # Raise an error.
        errstr = "The configuration is not valid. Errors: " + \
            " ".join(errors)
        raise ValueError(errstr)

    #-----------------------------------------------------------------#

    # Split the list of data frames into equally-sized chunks
    # with 'max_plots_per_output' data frames per chunk.
    dfs_chunks = \
        [dfs[i:i + max_plots_per_output] \
         for i in range(0,
                        len(dfs),
                        max_plots_per_output)]

    # If there are names for the data frames
    if dfs_names is not None:

        # Split the list of names into equally-sized chunks.
        dfs_names_chunks = \
            [dfs_names[i:i + max_plots_per_output] \
             for i in range(0,
                            len(dfs_names),
                            max_plots_per_output)]

    # Otherwise
    else:

        # Set the names to None.
        dfs_names_chunks = [None] * len(dfs_chunks)

    #-----------------------------------------------------------------#

    # Initialize an empty list to store the second set of data frames.
    dfs_2_chunks = []

    # If there is a second set of data frames
    if dfs_2 is not None:

        # If the two sets have different numbers of data frames
        if len(dfs_2) != len(dfs):

            # Raise an error.
            errstr = \
                "The number of data frames in the second set must " \
                "be the same as the number of data frames in the " \
                "first set."
            raise ValueError(errstr)

        # Split the second list of data frames into equally-sized
        # chunks with 'max_plots_per_output' data frames per chunk.
        dfs_2_chunks = \
            [dfs_2[i:i + max_plots_per_output] \
             for i in range(0,
                            len(dfs_2),
                            max_plots_per_output)]

    # Inform the user about how many output files will be written.
    infostr = f"{len(dfs_chunks)} output files will be generated."
    logger.info(infostr)

    # Use the same number of plots for all chunks.
    num_plots = max([len(chunk) for chunk in dfs_chunks])

    #-----------------------------------------------------------------#

    # If an output file was provided
    if output_file is not None:

        # Split the output file into prefix and extension.
        output_prefix, output_ext = os.path.splitext(output_file)

        # Get the output format.
        output_fmt = output_ext[1:]

    #-----------------------------------------------------------------#

    # For each chunk and associated output number
    for num_output, dfs_chunk in enumerate(dfs_chunks):

        # If we need to plot a single scatter plot
        if plot_type == "scatterplot":

            # Get the colors for the plot.
            colors_plot = get_colors(config = config,
                                     len_data = len(dfs_chunk))

            # Generate the scatter plots.
            plot_scatterplots(\
                dfs_chunk = dfs_chunk,
                num_plots = num_plots,
                num_output = num_output,
                dfs_names = dfs_names_chunks[num_output],
                colors = colors_plot,
                config = config,
                **plot_func_kwargs)

        #-------------------------------------------------------------#

        # If we need to plot single histograms
        elif plot_type == "histogram":

            # Generate the histograms.
            plot_histograms(\
                dfs_chunk = dfs_chunk,
                num_plots = num_plots,
                dfs_names = dfs_names_chunks[num_output],
                config = config)

        #-------------------------------------------------------------#

        # If we need to plot dual or overlapping histograms
        elif plot_type in ("histogram_bihist", "histogram_overlap"):

            # Generate the histograms.
            plot_histograms_dual(\
                dfs_chunk = dfs_chunk,
                dfs_2_chunk = dfs_2_chunks[num_output],
                num_plots = num_plots,
                dfs_names = dfs_names_chunks[num_output],
                categories = categories,
                plot_type = plot_type,
                config = config)

        #-------------------------------------------------------------#

        # If we need to plot a box plot or violin plot
        elif plot_type in ("boxplot", "violinplot"):

            # If there is a second set of data frames
            if dfs_2_chunks:

                # Get the second set of data frames.
                dfs_2_chunk = dfs_2_chunks[num_output]

            # Otherwise
            else:

                # Set it to None.
                dfs_2_chunk = None

            # Generate one box plot or violin plot per output.
            plot_box_violin(\
                num_plots = num_plots,
                num_output = num_output,
                dfs_chunk = dfs_chunk,
                dfs_2_chunk = dfs_2_chunk,
                dfs_names = dfs_names_chunks[num_output],
                categories = categories,
                config = config,
                plot_type = plot_type)

        #-------------------------------------------------------------#

        # If an output file was provided
        if output_file is not None:

            # If the number of output files is greater than 1
            if len(dfs_chunks) > 1:

                # Set the name of the output file.
                output_file = \
                    output_prefix + f"{num_output+1}.{output_fmt}"

            # Otherwise
            else:

                # Set the name of the output file.
                output_file = \
                    output_prefix + f".{output_fmt}"

            # Save the plot in the output file.
            plt.savefig(fname = output_file,
                        **config.get("output", {}))

        # Otherwise
        else:

            # Show the plot.
            plt.show()

        #-------------------------------------------------------------#

        # Close any figure that may be open.
        plt.close()


def plot_scatterplots(dfs_chunk: list[pd.DataFrame],
                      num_plots: int,
                      num_output: int,
                      columns: list[str],
                      colors: list[str],
                      config: Optional[dict[str, object]] = None,
                      groups_column: Optional[str] = None,
                      groups: Optional[list[str] | dict] = None,
                      plot_other_groups: Optional[bool] = None,
                      dfs_names: Optional[list[str]] = None) -> None:
    """Plot scatter plots.

    Parameters
    ----------
    dfs_chunk : :class:`list`
        A list of data frames.

    num_plots : :class:`int`
        The number of plots that will be generated.

    num_output : :class:`int`
        The number of the output file where the plots will be
        generated.

    columns : :class:`list`
        A list of the names of the two columns containing the data that
        should be plotted on the x and y axes.

    colors : :class:`list`
        A list of colors that will be used for the scatter plots.

    config : :class:`dict`, optional
        The configuration for the scatter plots.

    groups_column : :class:`str`, optional
        The name of the column containing the groups of the data
        points, if any.

    groups : :class:`list` or :class:`dict`, optional
        The names of the groups of data points that should be plotted,
        for all scatter plots (list) or for each scatter plot (dict).

    plot_other_groups : :class:`bool`, optional
        Whether data points not belonging to the groups of interest
        should also be plotted.

    dfs_names : :class:`list`, optional
        A list of names for the data frames.
    """

    # Create a copy of the original configuration, if any.
    config = copy.deepcopy(config) if config is not None else {}

    #-----------------------------------------------------------------#

    # Set the figure and the sub-plots.
    fig, sub_plots = set_figure(num_plots = num_plots,
                                config = config.get("figure", {}))

    #-----------------------------------------------------------------#

    # Initialize the plot's number.
    plot_num = 0

    # Keep the configuration shared by all plots.
    config_base = config

    #-----------------------------------------------------------------#

    # For each data frame
    for num_df, (df, sub_plot) in \
        enumerate(zip(dfs_chunk,
                      sub_plots.flatten()[:len(dfs_chunk)])):

        # Set the aspect ratio of the plot.
        sub_plot.set_box_aspect(1)

        #-------------------------------------------------------------#

        # Create a new copy of the configuration.
        config = copy.deepcopy(config_base)

        #-------------------------------------------------------------#

        # If the user provided the names of the data frames
        if dfs_names is not None:

            # Get the name of the current plot.
            plot_name = dfs_names[plot_num]

        # Otherwise
        else:

            # Use the plot's number as its name.
            plot_name = plot_num

        #-------------------------------------------------------------#

        # If the user provided the names of selected groups
        if groups is not None:

            # If the groups are a list
            if isinstance(groups, list):

                # Use them for all plots.
                plot_groups = groups

            # If there are groups for the current plot's name
            elif plot_name in groups:

                # Get them using the plot's name.
                plot_groups = groups[plot_name]

            # Otherwise
            else:

                # Get them using the plot's number.
                plot_groups = groups[plot_num]

        # Otherwise
        else:

            # Treat all groups the same.
            plot_groups = None

        #-------------------------------------------------------------#

        # Get the names of the columns containing the values of the
        # projections along the first and second dimension.
        c1_col, c2_col = columns[:2]

        #-------------------------------------------------------------#

        # If specific groups were defined
        if plot_groups is not None:

            # Get the data points belonging to other groups.
            df_2 = df[~df[groups_column].isin(plot_groups)]

            # Get the data points belonging to the groups of
            # interest.
            df = df[df[groups_column].isin(plot_groups)]

        # Otherwise
        else:

            # Set no data points for other groups.
            df_2 = None

        #-------------------------------------------------------------#

        # If the data points of the other groups should be plotted
        if df_2 is not None and plot_other_groups:

            # Plot them first, below the groups of interest.
            sub_plot = sns.scatterplot(\
                    x = c1_col,
                    y = c2_col,
                    data = df_2,
                    ax = sub_plot,
                    legend = False,
                    **config.get("scatterplot_2", {}))

        #-------------------------------------------------------------#

        # Get the scatter plot's options (the plot's color by default).
        config_scatter = \
            {"color" : colors[plot_num],
             **config.get("scatterplot", {})}

        # Generate the scatter plot.
        sub_plot = sns.scatterplot(\
                x = c1_col,
                y = c2_col,
                data = df,
                ax = sub_plot,
                legend = False,
                **config_scatter)

        #-------------------------------------------------------------#

        # Hide the top and right spine.
        for spine in ["top", "right"]:
            sub_plot.spines[spine].set_visible(False)

        #-------------------------------------------------------------#

        # If the user provided the names of the data frames
        if plot_name != plot_num:

            # Set the title.
            sub_plot = set_title(title = plot_name,
                                 sub_plot = sub_plot,
                                 config = config.get("title", {}))

        #-------------------------------------------------------------#

        # Get the configuration for the x-axis.
        config_x_axis = config.get("xaxis", {})

        # Get the positions of the ticks on the x-axis.
        x_ticks = get_ticks_positions(values = df[c1_col],
                                      item = "x-axis",
                                      config = config_x_axis)

        # Set the x-axis.
        sub_plot = set_axis(sub_plot = sub_plot,
                            axis = "x",
                            config = config_x_axis,
                            ticks = x_ticks)

        #-------------------------------------------------------------#

        # Get the configuration for the y-axis.
        config_y_axis = config.get("yaxis", {})

        # Get the positions of the ticks on the y-axis.
        y_ticks = get_ticks_positions(values = df[c2_col],
                                      item = "y-axis",
                                      config = config_y_axis)

        # Set the y-axis.
        sub_plot = set_axis(sub_plot = sub_plot,
                            axis = "y",
                            config = config_y_axis,
                            ticks = y_ticks)

        #-------------------------------------------------------------#

        # Update the plot's number.
        plot_num += 1

    #-----------------------------------------------------------------#

    # Set the extra sub-plots.
    set_extra_subplots(sub_plots = sub_plots.flat[len(dfs_chunk):],
                       config = config,
                       x_ticks = x_ticks,
                       y_ticks = y_ticks)


def plot_histograms(dfs_chunk: list[pd.DataFrame],
                    num_plots: int,
                    config: Optional[dict[str, object]] = None,
                    dfs_names: Optional[list[str]] = None) -> None:
    """Plot histograms.

    Parameters
    ----------
    dfs_chunk : :class:`list`
        A list of data frames.

    num_plots : :class:`int`
        The number of plots to generate.

    config : :class:`dict`, optional
        The configuration for the histograms.

    dfs_names : :class:`list`, optional
        A list of names for the data frames.
    """

    # Create a copy of the original configuration, if any.
    config = copy.deepcopy(config) if config is not None else {}

    #-----------------------------------------------------------------#

    # Set the figure and the sub-plots.
    fig, sub_plots = set_figure(num_plots = num_plots,
                                config = config.get("figure", {}))

    #-----------------------------------------------------------------#

    # Get whether the histograms are to be plotted as densities.
    plot_density = config.get("histogram", {}).pop("density", False)

    # Keep the configuration shared by all plots.
    config_base = config

    #-----------------------------------------------------------------#

    # For each data frame
    for num_df, (df, sub_plot) in \
        enumerate(zip(dfs_chunk,
                      sub_plots.flatten()[:len(dfs_chunk)])):

        # Set the aspect ratio of the plot.
        sub_plot.set_box_aspect(1)

        #-------------------------------------------------------------#

        # Create a new copy of the configuration.
        config = copy.deepcopy(config_base)

        # Get the configuration for the histogram.
        config_hist = config.get("histogram", {})

        #-------------------------------------------------------------#

        # If the user provided the names of the data frames
        if dfs_names is not None:

            # Get the title for the current plot.
            title = dfs_names[num_df]

            # If the title is not None
            if title is not None:

                # Set the title.
                sub_plot = set_title(title = title,
                                     sub_plot = sub_plot,
                                     config = config.get("title", {}))

        #-------------------------------------------------------------#

        # Take all values.
        x = df.values.flatten()

        #-------------------------------------------------------------#

        # Remove NaN values from the data.
        x = x[~np.isnan(x)]

        #-------------------------------------------------------------#

        # Get the minimum value found in the data.
        min_val = min(x)

        # Get the maximum value found in the data.
        max_val = max(x)

        #-------------------------------------------------------------#

        # Get the number of bins, if passed.
        num_bins = config_hist.pop("num_bins", None)

        # Get the width of the bins.
        bin_width = config_hist.pop("width", 0.25)

        # If the user passed the number of bins
        if num_bins is not None:

            # Get the bins.
            bins = np.linspace(min_val,
                               max_val,
                               num_bins + 1)

        # Otherwise
        else:

            # Get the bins.
            bins = np.arange(min_val,
                             max_val + bin_width,
                             bin_width)

        #-------------------------------------------------------------#

        # Generate the histogram.
        counts, bins_edges = np.histogram(x,
                                          bins = bins)

        #-------------------------------------------------------------#

        # Compute the bins' centers.
        bins_centers = (bins_edges[:-1] + bins_edges[1:]) / 2

        # Get the width of the bins.
        bins_width = np.diff(bins_edges)

        #-------------------------------------------------------------#

        # If the histograms are to be plotted as densities
        if plot_density:

            # Set the height of the bars for the histogram.
            height = counts / (np.sum(counts) * bins_width)

        # Otherwise
        else:

            # Set the height of the bars for the histogram.
            height = counts

        #-------------------------------------------------------------#

        # Get the configuration of the x-axis.
        config_x_axis = config.get("xaxis", {})

        # Get the positions of the ticks on the x-axis.
        x_ticks = get_ticks_positions(values = bins,
                                      item = "x-axis",
                                      config = config_x_axis)

        # Get the configuration of the y-axis.
        config_y_axis = config.get("yaxis", {})

        # Get the positions of the ticks on the y-axis.
        y_ticks = get_ticks_positions(values = height,
                                      item = "y-axis",
                                      config = config_y_axis)

        #-------------------------------------------------------------#

        # If there is a configuration for the colorbar
        if "colorbar" in config:

            # Get the color map to be used for the histogram.
            cmap = \
                plt.get_cmap(config["colorbar"].get(\
                    "cmap", "summer_r"))

            # Normalize the heights to the range [0, 1].
            norm = mcolors.Normalize(vmin = y_ticks.min(),
                                     vmax = y_ticks.max())

            # Get the color to be used for the histogram.
            color = cmap(norm(height))

        # Otherwise
        else:

            # Get the color to be used for the histogram.
            color = config_hist.pop("color", None)

        #-------------------------------------------------------------#

        # Plot the histogram.
        sub_plot.bar(x = bins_centers,
                     height = height,
                     width = bins_width,
                     align = "center",
                     color = color,
                     **config_hist)

        #-------------------------------------------------------------#

        # Hide the top and right spine.
        for spine in ["top", "right"]:
            sub_plot.spines[spine].set_visible(False)

        #-------------------------------------------------------------#

        # Set the x-axis.
        sub_plot = set_axis(sub_plot = sub_plot,
                            axis = "x",
                            config = config_x_axis,
                            ticks = x_ticks)

        # Set the y-axis.
        sub_plot = set_axis(sub_plot = sub_plot,
                            axis = "y",
                            config = config_y_axis,
                            ticks = y_ticks)

        #-------------------------------------------------------------#

        # Get the configuration for the colorbar.
        config_cbar = config.get("colorbar", {})

        # If a configuration for the colorbar was provided
        if config_cbar:

            # Create a 'ScalarMappable' object to use for the colorbar.
            sm = ScalarMappable(cmap = cmap,
                                norm = norm)

            # Set a dummy array for the colorbar to work.
            sm.set_array([])

            # Get the position of the sub-plot.
            pos = sub_plot.get_position()

            # Create a new sub-plot for the colorbar matching the
            # plot's height.
            cbar_sub_plot = \
                fig.add_axes([pos.x0 + pos.width + 0.01,
                              pos.y0,
                              pos.width * 0.04,
                              pos.height])

            # Add the colorbar.
            cbar = fig.colorbar(sm,
                                cax = cbar_sub_plot,
                                **config_cbar.get("options", {}))

            #---------------------------------------------------------#

            # Get user-defined ticks, if provided.
            cbar_ticks = config_cbar.get("ticks")

            # If no user-defined ticks were provided
            if cbar_ticks is None:

                # Get the positions of the ticks on the colorbar's axis.
                cbar_ticks = get_ticks_positions(values = y_ticks,
                                                 item = "cbar",
                                                 config = config_cbar)

            #---------------------------------------------------------#

            # Set the colorbar's axis.
            set_cbar_axis(cbar = cbar,
                          config = config_cbar,
                          ticks = cbar_ticks)

        #-------------------------------------------------------------#

        # Get the configuration for the vertical line to be drawn.
        config_vline = config.get("vline")

        # If a configuration for the line was provided
        if config_vline is not None:

            # Plot the vertical line.
            sub_plot.axvline(**config_vline)

    #-----------------------------------------------------------------#

    # Set the extra sub-plots.
    set_extra_subplots(sub_plots = sub_plots.flat[len(dfs_chunk):],
                       config = config,
                       x_ticks = x_ticks,
                       y_ticks = y_ticks)


def plot_histograms_dual(dfs_chunk: list[pd.DataFrame],
                         dfs_2_chunk: list[pd.DataFrame],
                         plot_type: str,
                         num_plots: int,
                         categories: list[str],
                         config: Optional[dict[str, object]] = None,
                         dfs_names: Optional[list[str]] = None) \
                            -> None:
    """Plot dual histograms.

    Parameters
    ----------
    dfs_chunk : :class:`list`
        A list of data frames.

    dfs_2_chunk : :class:`list`
        A list of data frames paired with the data frames in
        ``dfs_chunk``.

    plot_type : :class:`str`
        The type of plot to generate.

    num_plots : :class:`int`
        The number of plots to generate.

    categories : :class:`list`
        A list of names for the categories of data points represented
        by the two lists of data frames.

    config : :class:`dict`, optional
        The configuration for the histograms.

    dfs_names : :class:`list`, optional
        A list of names for the data frames.
    """

    # Create a copy of the original configuration, if any.
    config = copy.deepcopy(config) if config is not None else {}

    #-----------------------------------------------------------------#

    # Set the figure and the sub-plots.
    fig, sub_plots = set_figure(num_plots = num_plots,
                                config = config.get("figure", {}))

    #-----------------------------------------------------------------#

    # Get whether the histograms are to be plotted as densities.
    plot_density = config.get("histogram", {}).pop("density", False)

    #-----------------------------------------------------------------#

    # If the user passed the number of bins for the second histogram
    if "num_bins" in config.get("histogram_2", {}):

        # Remove the key from the configuration.
        config["histogram_2"].pop("num_bins")

    # If the user passed the width of the bins for the second
    # histogram
    if "width" in config.get("histogram_2", {}):

        # Remove the key from the configuration.
        config["histogram_2"].pop("width")

    # If the user passed the 'density' option for the second
    # histogram
    if "density" in config.get("histogram_2", {}):

        # Remove the key from the configuration.
        config["histogram_2"].pop("density")

    #-----------------------------------------------------------------#

    # Get the color to be used for the first histogram.
    color_1 = config.get("histogram", {}).pop("color", None)

    # Get the color to be used for the second histogram.
    color_2 = config.get("histogram_2", {}).pop("color", color_1)

    # Keep the configuration shared by all plots.
    config_base = config

    #-----------------------------------------------------------------#

    # For each data frame
    for num_df, (df, df_2, sub_plot) in \
        enumerate(zip(dfs_chunk,
                      dfs_2_chunk,
                      sub_plots.flatten()[:len(dfs_chunk)])):

        # Set the aspect ratio of the plot.
        sub_plot.set_box_aspect(1)

        #-------------------------------------------------------------#

        # Create a new copy of the configuration.
        config = copy.deepcopy(config_base)

        #-------------------------------------------------------------#

        # Get the configuration for the histograms.
        config_hist = config["histogram"]

        # Get the configuration for the paired histograms, if any.
        config_hist_2 = config.get("histogram_2", config_hist)

        #-------------------------------------------------------------#

        # If the user provided the names of the data frames
        if dfs_names is not None:

            # Get the title for the current plot.
            title = dfs_names[num_df]

            # If the title is not None
            if title is not None:

                # Set the title.
                sub_plot = set_title(title = title,
                                     sub_plot = sub_plot,
                                     config = config.get("title", {}))

        #-------------------------------------------------------------#

        # Take all values for the first set of data.
        x_1 = df.values.flatten()

        # Take all values for the second set of data.
        x_2 = df_2.values.flatten()

        #-------------------------------------------------------------#

        # Remove NaN values from the first set of data.
        x_1 = x_1[~np.isnan(x_1)]

        # Remove NaN values from the second set of data.
        x_2 = x_2[~np.isnan(x_2)]

        #-------------------------------------------------------------#

        # Get the minimum value found in the data.
        min_val = min(x_1.min(), x_2.min())

        # Get the maximum value found in the data.
        max_val = max(x_1.max(), x_2.max())

        #-------------------------------------------------------------#

        # Get the number of bins, if passed.
        num_bins = config_hist.pop("num_bins", None)

        # Get the width of each bin.
        bin_width = config_hist.pop("width", 0.25)

        # If the user passed the number of bins
        if num_bins is not None:

            # Get the bins.
            bins = np.linspace(min_val,
                               max_val,
                               num_bins + 1)

        # Otherwise
        else:

            # Get the bins.
            bins = np.arange(min_val,
                             max_val + bin_width,
                             bin_width)

        #-------------------------------------------------------------#

        # Get the histogram for the first set of data.
        counts_1, bins_edges = np.histogram(x_1,
                                            bins = bins,
                                            weights = None)

        # Get the histogram for the second set of data.
        counts_2, _ = np.histogram(x_2,
                                   bins = bins,
                                   weights = None)

        #-------------------------------------------------------------#

        # Get the width of the bins.
        bins_width = np.diff(bins_edges)

        # Get the bins' centers.
        bins_centers = (bins_edges[:-1] + bins_edges[1:]) / 2

        #-------------------------------------------------------------#

        # If the histograms are to be plotted as densities
        if plot_density:

            # Get the density for the first set of data.
            density_1 = counts_1 / (np.sum(counts_1) * bins_width)

            # Get the density for the second set of data.
            density_2 = counts_2 / (np.sum(counts_2) * bins_width)

            # Get the negative of the density for the second set of
            # data.
            density_2_negated = - density_2

        #-------------------------------------------------------------#

        # If the histograms are to be plotted on the opposite sides
        # of a shared x-axis
        if plot_type == "histogram_bihist":

            # If the histograms are to be plotted as densities
            if plot_density:

                # Set the height of the bars for the first histogram.
                height_1 = density_1

                # Set the height of the bars for the second histogram.
                height_2 = density_2_negated

            # Otherwise
            else:

                # Set the height of the bars for the first histogram.
                height_1 = counts_1

                # Set the height of the bars for the second histogram.
                height_2 = - counts_2

        #-------------------------------------------------------------#

        # If the histograms are to be plotted on the same side of a
        # shared x-axis
        elif plot_type == "histogram_overlap":

            # If the histograms are to be plotted as densities
            if plot_density:

                # Set the height of the bars for the first histogram.
                height_1 = density_1

                # Set the height of the bars for the second histogram.
                height_2 = density_2

            # Otherwise
            else:

                # Set the height of the bars for the first histogram.
                height_1 = counts_1

                # Set the height of the bars for the second histogram.
                height_2 = counts_2

        #-------------------------------------------------------------#

        # Get the minimum height.
        min_height = min(np.min(height_1), np.min(height_2))

        # Get the maximum height.
        max_height = max(np.max(height_1), np.max(height_2))

        # Get the range of heights.
        range_heights = np.array([min_height, max_height])

        #-------------------------------------------------------------#

        # Generate the first histogram.
        sub_plot.bar(x = bins_centers,
                     height = height_1,
                     width = bins_width,
                     align = "center",
                     color = color_1,
                     **config_hist)

        # Generate the second histogram.
        sub_plot.bar(x = bins_centers,
                     height = height_2,
                     width = bins_width,
                     align = "center",
                     color = color_2,
                     **config_hist_2)

        #-------------------------------------------------------------#

        # Hide the top and right spine.
        for spine in ["top", "right"]:
            sub_plot.spines[spine].set_visible(False)

        #-------------------------------------------------------------#

        # Get the configuration of the x-axis.
        config_x_axis = config.get("xaxis", {})

        # Get the positions of the ticks on the x-axis.
        x_ticks = get_ticks_positions(values = bins_edges,
                                      item = "x-axis",
                                      config = config_x_axis)

        # Set the x-axis.
        sub_plot = set_axis(sub_plot = sub_plot,
                            axis = "x",
                            config = config_x_axis,
                            ticks = x_ticks)

        #-------------------------------------------------------------#

        # Get the configuration of the y-axis.
        config_y_axis = config.get("yaxis", {})

        # Get the positions of the ticks on the y-axis.
        y_ticks = get_ticks_positions(values = range_heights,
                                      item = "y-axis",
                                      config = config_y_axis)

        # If the histogram is a bi-histogram
        if plot_type == "histogram_bihist":

            # Use the absolute values as the ticks' labels.
            abs_values = True

        # If the histograms overlap
        elif plot_type == "histogram_overlap":

            # Use the values as the ticks' labels.
            abs_values = False

        # Set the y-axis.
        sub_plot = set_axis(sub_plot = sub_plot,
                            axis = "y",
                            config = config_y_axis,
                            ticks = y_ticks,
                            abs_values = abs_values)

        #-------------------------------------------------------------#

        # Get the configuration for the vertical line to be drawn.
        config_vline = config.get("vline")

        # If a configuration for the line was provided
        if config_vline is not None:

            # Plot the vertical line.
            sub_plot.axvline(**config_vline)

    #-----------------------------------------------------------------#

    # Get the configuration for the legend.
    config_legend = config.get("legend", {})

    # Create custom legend handles.
    legend_handles = \
        [Patch(facecolor = col, label = cat) for col, cat \
         in zip((color_1, color_2), categories)]

    # Add a single legend to the figure.
    fig.legend(handles = legend_handles,
               **config_legend)

    #-----------------------------------------------------------------#

    # Set the extra sub-plots.
    set_extra_subplots(sub_plots = sub_plots.flat[len(dfs_chunk):],
                       config = config,
                       x_ticks = x_ticks,
                       y_ticks = y_ticks)


def plot_box_violin(plot_type: str,
                    dfs_chunk: list[pd.DataFrame],
                    dfs_2_chunk: list[pd.DataFrame] | None,
                    num_plots: int,
                    num_output: int,
                    categories: list[str],
                    config: Optional[dict[str, object]] = None,
                    dfs_names: Optional[list[str]] = None) -> None:
    """Plot distributions as box or violin plots.

    Parameters
    ----------
    plot_type : :class:`str`, {``"boxplot"``, ``"violinplot"``}
        The type of plot to generate.

    dfs_chunk : :class:`list`
        A list of data frames.

    dfs_2_chunk : :class:`list` or :obj:`None`
        A list of data frames paired with the data frames in
        ``dfs_chunk``.

    num_plots : :class:`int`
        The number of plots to generate.

    num_output : :class:`int`
        The number of the output being generated.

    categories : :class:`list`
        A list of names for the categories of data points represented
        by the two lists of data frames.

    config : :class:`dict`, optional
        The configuration for the plot.

    dfs_names : :class:`list`, optional
        A list of names for the data frames.
    """

    # Create a copy of the original configuration, if any.
    config = copy.deepcopy(config) if config is not None else {}

    # Get whether the data is paired.
    paired = dfs_2_chunk is not None

    #-----------------------------------------------------------------#

    # Get the configuration for the plot.
    config_plot = config.get(plot_type, {})

    #-----------------------------------------------------------------#

    # If the plot type is a box plot
    if plot_type == "boxplot":

        # Get the function to be used for plotting.
        plot_func = sns.boxplot

    # If the plot type is a violin plot
    elif plot_type == "violinplot":

        # Get the function to be used for plotting.
        plot_func = sns.violinplot

        # If the data is paired
        if paired:

            # Add the 'split' option to the configuration.
            config_plot["split"] = True

    # Otherwise
    else:

        # Raise an error.
        errstr = \
            "The 'plot_type' parameter must be 'boxplot' or " \
            "'violinplot'."
        raise ValueError(errstr)

    #-----------------------------------------------------------------#

    # Generate the figure and subplots.
    _, sub_plot = \
        plt.subplots(\
            nrows = 1,
            ncols = 1,
            figsize = \
                config.get("figure", {}).get("sizeinches", None))

    #-----------------------------------------------------------------#

    # Initialize an empty list to store the x values.
    x = []

    # Initialize an empty list to store the y values.
    y = []

    # Initialize an empty list to store the hue values.
    hue = []

    #-----------------------------------------------------------------#

    # For each data frame
    for i, df in enumerate(dfs_chunk):

        # Get the values in the first data frame.
        values_1 = df.values.flatten().tolist()

        # Add the values to the list.
        y.extend(values_1)

        # Get the name of the current data frame.
        name = \
            dfs_names[i] \
            if dfs_names is not None else i

        # Add the names to the list.
        x.extend([name] * len(values_1))

        #-------------------------------------------------------------#

        # If there is a second set of data frames
        if dfs_2_chunk:

            # Get the second data frame.
            df2 = dfs_2_chunk[i]

            # Get the values in the second data frame.
            values_2 = df2.values.flatten().tolist()

            # Add the values to the list.
            y.extend(values_2)

            # Add the name of the current data frame to the list.
            x.extend([name] * len(values_2))

            # Add the hue values for the first data frame to the list.
            hue.extend([categories[0]] * len(values_1))

            # Add the hue values for the second data frame to the list.
            hue.extend([categories[1]] * len(values_2))

            # Mark the data as paired.
            paired = True

        # Otherwise
        else:

            # Add the hue values for the only data frame to the list.
            hue.extend([name] * len(values_1))

            # Mark the data as not paired.
            paired = False

    #-----------------------------------------------------------------#

    # Create a data frame.
    data = pd.DataFrame({"x": x, "y": y})

    # If there are hue values
    if hue:

        # Add the hue values to the data frame.
        data["hue"] = hue

        # Set the column name for the hue values.
        hue = "hue"

    # Otherwise
    else:

        # Set the hue to None.
        hue = None

    #-----------------------------------------------------------------#

    # Generate the plot.
    plot_func(data = data,
              x = "x",
              y = "y",
              hue = hue,
              ax = sub_plot,
              **config_plot)

    #-----------------------------------------------------------------#

    # Hide the top and right spine.
    for spine in ["top", "right"]:
        sub_plot.spines[spine].set_visible(False)

    #-----------------------------------------------------------------#

    # Get the configuration of the x-axis.
    config_x_axis = config.get("xaxis", {})

    # Set the x-axis (one tick per category).
    sub_plot = set_axis(sub_plot = sub_plot,
                        axis = "x",
                        config = config_x_axis,
                        ticks = sub_plot.get_xticks(),
                        tick_labels = sub_plot.get_xticklabels(),
                        categorical = True)

    #-----------------------------------------------------------------#

    # Get the configuration of the y-axis.
    config_y_axis = config.get("yaxis", {})

    # Get the positions of the ticks on the y-axis.
    y_ticks = get_ticks_positions(values = sub_plot.get_yticks(),
                                  item = "y-axis",
                                  config = config_y_axis)

    # Set the y-axis.
    sub_plot = set_axis(sub_plot = sub_plot,
                        axis = "y",
                        config = config_y_axis,
                        ticks = y_ticks)

    #-----------------------------------------------------------------#

    # Get the configuration for the legend.
    config_legend = config.get("legend", {})

    # Set the legend.
    sub_plot = set_legend(sub_plot = sub_plot,
                          config = config_legend)

    #-----------------------------------------------------------------#

    # Get the configuration for the horizontal line to be drawn.
    config_hline = config.get("hline", {})

    # If a configuration for the line was provided
    if config_hline:

        # Plot the horizontal line.
        sub_plot.axhline(**config_hline)


def plot_lineplot(data: np.ndarray,
                  x: str,
                  y: str,
                  hue: str,
                  sub_plot: Optional[matplotlib.axes.Axes] = None,
                  ax: Optional[matplotlib.axes.Axes] = None,
                  config: Optional[dict[str, object]] = None) -> \
                    matplotlib.axes.Axes:
    """Plot a line plot.

    Parameters
    ----------
    data : :class:`pandas.DataFrame`
        The data to be plotted.

    x : :class:`str`
        The name of the column containing the x-values.

    y : :class:`str`
        The name of the column containing the y-values.

    hue : :class:`str`
        The name of the column containing the hue values.

    sub_plot : :class:`matplotlib.axes.Axes`, optional
        The sub-plot where the line plot will be drawn.

    ax : :class:`matplotlib.axes.Axes`, optional
        An alias for ``sub_plot``.

    config : :class:`dict`, optional
        The configuration for the line plot.

    Returns
    -------
    sub_plot : :class:`matplotlib.axes.Axes`
        The sub-plot.
    """

    # If no sub-plot was passed
    if sub_plot is None:

        # Use the 'ax' argument.
        sub_plot = ax

    # If there is still no sub-plot
    if sub_plot is None:

        # Raise an error.
        raise ValueError("A valid subplot axis must be provided.")

    # Create a copy of the original configuration, if any.
    config = copy.deepcopy(config) if config is not None else {}

    #-----------------------------------------------------------------#

    # Generate the plot.
    sns.lineplot(data = data,
                 x = x,
                 y = y,
                 hue = hue,
                 ax = sub_plot,
                 **config.get("lineplot", {}))

    #-----------------------------------------------------------------#

    # Get the title configuration.
    config_title = config.get("title", {})

    # If there is a label
    if config_title.get("label") is not None:

        # Get the title.
        title = config_title.pop("label", "")

        # Get the maximum length of a title.
        max_length = config_title.pop("max_length", 20)

        # Split the title into lines of at most 'max_length'
        # characters.
        title_fmt = \
            split_text_by_length(text = title,
                                 max_length = max_length)

        # Set the sub-plot's title.
        sub_plot.set_title(label = title_fmt,
                           **config.get("title", {}))

    #-----------------------------------------------------------------#

    # Hide the top and right spine.
    for spine in ["top", "right"]:
        sub_plot.spines[spine].set_visible(False)

    #-----------------------------------------------------------------#

    # Get the configuration of the x-axis.
    config_x_axis = config.get("xaxis", {})

    # Get the positions of the ticks on the x-axis.
    x_ticks = get_ticks_positions(values = data[x].values,
                                  item = "x-axis",
                                  config = config_x_axis)

    # Set the x-axis.
    sub_plot = set_axis(sub_plot = sub_plot,
                        axis = "x",
                        config = config_x_axis,
                        ticks = x_ticks)

    #-----------------------------------------------------------------#

    # Get the configuration of the y-axis.
    config_y_axis = config.get("yaxis", {})

    # Get the positions of the ticks on the y-axis.
    y_ticks = get_ticks_positions(values = data[y].values,
                                  item = "y-axis",
                                  config = config_y_axis)

    # Set the y-axis.
    sub_plot = set_axis(sub_plot = sub_plot,
                        axis = "y",
                        config = config_y_axis,
                        ticks = y_ticks)

    #-----------------------------------------------------------------#

    # Get the configuration for the legend.
    config_legend = config.get("legend", {})

    # Set the legend.
    sub_plot = set_legend(sub_plot = sub_plot,
                          config = config_legend)

    #-----------------------------------------------------------------#

    # Return the sub-plot.
    return sub_plot


def plot_enrichplot(data: np.ndarray,
                    x: str,
                    y_1: str,
                    y_2: str,
                    y_3: str,
                    hue: str,
                    config: Optional[dict[str, object]] = None) -> \
                        np.ndarray:
    """Plot the enrichment scores as a bar plot, a violin plot, and
    a strip plot.

    Parameters
    ----------
    data : :class:`pandas.DataFrame`
        The data to be plotted.

    x : :class:`str`
        The name of the column containing the x-values.

    y_1 : :class:`str`
        The name of the column containing the y-values for the bar
        plot.

    y_2 : :class:`str`
        The name of the column containing the y-values for the
        violin plot.

    y_3 : :class:`str`
        The name of the column containing the y-values for the
        strip plot.

    hue : :class:`str`
        The name of the column containing the hue values.

    config : :class:`dict`, optional
        The configuration for the plot.

    Returns
    -------
    sub_plots : :class:`numpy.ndarray`
        The sub-plots.
    """

    # Create a copy of the original configuration, if any.
    config = copy.deepcopy(config) if config is not None else {}

    #-----------------------------------------------------------------#

    # Get the configuration for the figure.
    config_figure = config.get("figure", {})

    # Create the figure and the sub-plots.
    fig, (sub_plot_1, sub_plot_2, sub_plot_3) = \
        plt.subplots(nrows = 3,
                     ncols = 1,
                     sharex = True,
                     figsize = config_figure.get("sizeinches", None))

    # If a configuration for the sub-plots is provided
    if "subplots" in config_figure:

        # Adjust the sub-plots.
        plt.subplots_adjust(**config_figure["subplots"])

    #-----------------------------------------------------------------#

    # If a palette for all three plots is provided
    if "general_palette" in config:

        # For each plot-specific section
        for section in ["barplot", "violinplot", "stripplot"]:

            # If the section is not in the configuration
            if section not in config:

                # Add the section to the configuration.
                config[section] = {}

            # If there is no palette in the configuration for the
            # section
            if "palette" not in config[section]:

                # Add the palette to the configuration.
                config[section]["palette"] = config["general_palette"]

            # Otherwise
            else:

                # Warn the user that the section's palette will be
                # used.
                warnstr = \
                    f"The 'palette' option in the '{section}' " \
                    "section will be used instead of the 'general_" \
                    "palette' option."
                logger.warning(warnstr)

            # If there are both a 'facecolor' and a 'color' in the
            # section
            if ("facecolor" in config[section] \
                and "color" in config[section]):

                # Warn the user that the specific color will be used.
                warnstr = \
                    f"The 'facecolor' option in the '{section}' " \
                    "section will be used instead of the " \
                    "'general_palette' option."
                logger.warning(warnstr)

                # Remove the 'color' option from the section.
                config[section].pop("color")

            # If there is no 'facecolor' in the section and a 'color'
            # is present
            elif ("facecolor" not in config[section] \
                  and "color" in config[section]):

                # Warn the user that the specific color will be used.
                warnstr = \
                    f"The 'color' option in the '{section}' " \
                    "section will be used instead of the " \
                    "'general_palette' option."
                logger.warning(warnstr)

            # If there is no 'color' in the section and a 'facecolor'
            # is present
            elif ("facecolor" in config[section] \
                  and "color" not in config[section]):

                # Warn the user that the specific color will be used.
                warnstr = \
                    f"The 'facecolor' option in the '{section}' " \
                    "section will be used instead of the " \
                    "'general_palette' option."
                logger.warning(warnstr)

    #-----------------------------------------------------------------#

    # Generate the bar plot for the genes of interest.
    sns.barplot(data = data,
                x = x,
                y = y_1,
                hue = hue,
                ax = sub_plot_1,
                **config.get("barplot", {}))

    #-----------------------------------------------------------------#

    # Get the configuration for the violin plot (no inner by default).
    config_violin = {"inner" : None, **config.get("violinplot", {})}

    # Generate the violin plot for the significant genes.
    sns.violinplot(data = data,
                   x = x,
                   y = y_2,
                   ax = sub_plot_2,
                   hue = hue,
                   **config_violin)

    #-----------------------------------------------------------------#

    # Generate the strip plot for the enrichment scores.
    sns.stripplot(data = data,
                  x = x,
                  y = y_3,
                  ax = sub_plot_3,
                  hue = hue,
                  **config.get("stripplot", {}))

    #-----------------------------------------------------------------#

    # Get the configuration for the title.
    config_title = config.get("title", {})

    # If there is a label
    if config_title.get("label") is not None:

        # Get the title.
        title = config_title.pop("label", "")

        # Get the maximum length of a title.
        max_length = config_title.pop("max_length", 20)

        # Split the title into lines of at most 'max_length'
        # characters.
        title_fmt = \
            split_text_by_length(text = title,
                                 max_length = max_length)

        # Set the figure's title.
        fig.suptitle(label = title_fmt,
                     **config.get("title", {}))

    #-----------------------------------------------------------------#

    # For the first two sub-plots
    for sub_plot in [sub_plot_1, sub_plot_2]:

        # Remove the ticks on the x-axis.
        sub_plot.tick_params(axis = "x",
                             direction = "out",
                             length = 0)

    #-----------------------------------------------------------------#

    # Get the configuration of the x-axis.
    config_x_axis = config.get("xaxis", {})

    # Set the x-axis (one tick per category).
    sub_plot_3 = set_axis(sub_plot = sub_plot_3,
                          axis = "x",
                          config = config_x_axis,
                          ticks = sub_plot_3.get_xticks(),
                          tick_labels = sub_plot_3.get_xticklabels(),
                          categorical = True)

    #-----------------------------------------------------------------#

    # For each sub-plot, the name of the column containing values
    # plotted on its y-axis, and the name of the corresponding plot
    for sub_plot, y, name in \
        zip([sub_plot_1, sub_plot_2, sub_plot_3],
            [y_1, y_2, y_3],
            ["barplot", "violinplot", "stripplot"]):

        # Hide the top and right spine.
        for spine in ["top", "right"]:
            sub_plot.spines[spine].set_visible(False)

        #-------------------------------------------------------------#

        # Get the configuration of the y-axis (the plot's options
        # override the general ones).
        config_y_axis = \
            _internals.recursive_merge_dicts(\
                config.get("yaxis", {}),
                config.get(f"yaxis_{name}", {}))

        # Get the positions of the ticks on the y-axis.
        y_ticks = get_ticks_positions(values = data[y].values,
                                      item = "y-axis",
                                      config = config_y_axis)

        # Set the y-axis.
        set_axis(sub_plot = sub_plot,
                 axis = "y",
                 config = config_y_axis,
                 ticks = y_ticks)

        #-------------------------------------------------------------#

        # Get the configuration for the horizontal line to be drawn
        # (the plot's options override the general ones).
        config_hline = \
            _internals.recursive_merge_dicts(\
                config.get("hline", {}),
                config.get(f"hline_{name}", {}))

        # If a configuration for the line was provided
        if config_hline:

            # Plot the horizontal line.
            sub_plot.axhline(**config_hline)

    #-----------------------------------------------------------------#

    # Return the sub-plots.
    return np.array([sub_plot_1, sub_plot_2, sub_plot_3])
