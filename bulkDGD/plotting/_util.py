#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-

#    _util.py
#
#    Private utilities for plotting.
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
__doc__ = "Private utilities for plotting."


####################################################################### 


# Import from the standard library.
import logging as log
import math
# Import from third-party packages.
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
# Import from 'bulkDGD'.
from bulkDGD import util


#######################################################################


# Get the module's logger.
logger = log.getLogger(__name__)


########################## PRIVATE CONSTANTS ##########################


# Set a mapping between the sections that we can encounter in a
# configuration and the name of the item they refer to.
SECTIONS2ITEMS = \
    {# Plot elements.
     "plot:xaxis" : "x-axis",
     "plot:yaxis" : "y-axis",
     "plot:legend" : "legend",
     "plot:title" : "title",
     "plot:text" : "text",
     # Plot types.
     "plot:other_groups" : "scatter plot for 'other' groups",
     "plot:scatterplot" : "scatter plot",
     "plot:lineplot" : "line plot",
     "plot:histogram" : "histogram"}

########################## PUBLIC CONSTANTS ###########################


# Get all palettes available in Seaborn.
SEABORN_PALETTES = \
    set(sns.palettes.SEABORN_PALETTES.keys()).union(\
        {"cubehelix", "dark", "colorblind", "husl", "hls", 
         "muted", "pastel", "bright"})

# Get all color maps available in Matplotlib.
MATPLOTLIB_CMAPS = set(plt.colormaps())


########################## PUBLIC FUNCTIONS ###########################


def get_formatted_ticklabels(ticklabels,
                             fmt = "{:s}"):
    """Return the ticks' labels, formatted according to a given format
    string.

    Parameters
    ----------
    ticklabels : :class:`numpy.ndarray` or :class:`list`
        An array or list of labels.

    fmt: :class:`str`
        The format string.

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

        # Strip the label of trailing zeroes.
        fmt_ticklabel = fmt_ticklabel.rstrip("0")

        # If the label now ends with a dot or a comma (because it was
        # an integer expressed as 1.0, 3,00, etc., and we removed all
        # trailing zeroes)
        if fmt_ticklabel.endswith(".") or fmt_ticklabel.endswith(","):

            # Remove the dot/comma.
            fmt_ticklabel = fmt_ticklabel.rstrip(".").rstrip(",")

        # Add the label to the list.
        fmt_ticklabels.append(fmt_ticklabel)

    #-----------------------------------------------------------------#

    # Return the labels.
    return fmt_ticklabels


def find_rectangular_grid(n):
    """Given an array of ``n`` items, find the best way to arrange
    them in the 'squarest' possible two-dimensional grid (namely, the
    grid where the difference between the two dimensions is minimal).

    Allow for blank 'cells' in the grid, so that the number of cells
    in the grid may exceed the number of items to avoid making a grid
    with only one row in case the number of items is a prime number.

    Parameters
    ----------
    n : :class:`numpy.ndarray`
        The array of items.

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

    # If we ended up with only one row (because we did not find any
    # factors, meaning that 'n' is prime)
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


def get_ticks_positions(values,
                        item,
                        config):
    """Generate the positions that the ticks will have on a plot's
    axis/colorbar/etc.

    This original code for this function was originally developed
    by Valentina Sora for the RosettaDDGPrediction package.
    
    The original function can be found at:

    https://github.com/ELELAB/RosettaDDGPrediction/
    blob/master/RosettaDDGPrediction/plotting.py

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

    # Get the top configuration.
    config = config.get("interval", {})

    #-----------------------------------------------------------------#
    
    # Get the configurations.
    int_type = config.get("type", "continuous")
    rtn = config.get("round_to_nearest")
    top = config.get("top")
    bottom = config.get("bottom")
    steps = config.get("steps")
    spacing = config.get("spacing")
    ciz = config.get("center_in_zero")

    # Inform the user that we are now setting the ticks' interval.
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
            
            # The default top value will be the maximum of the values
            # provided.
            top = int(np.ceil(max(values)))

            # Inform the user about the top value.
            debugstr = \
                "Since 'top' is not defined and 'type' is " \
                f"'{int_type}', 'top' will be the maximum of all " \
                f"values found, ({top})."
            logger.debug(debugstr)
        
        # If the interval is continuous
        elif int_type == "continuous":
            
            # The default top value will be the maximum of the values
            # provided, rounded up.
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
            
            # The default bottom value is the minimum of the values
            # provided.
            bottom = int(min(values))

            # Inform the user about the bottom value.
            debugstr = \
                f"Since 'bottom' is not defined and 'type' is " \
                f"'{int_type}', 'bottom' will be the minimum of all " \
                f"values found ({bottom})."
            logger.debug(debugstr)
        
        # If the interval is continuous
        elif int_type == "continuous":
            
            # The default bottom value is the minimum of the values
            # provided, rounded down.
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

        # Inform the user about the chosen top value
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

        # A default of 10 steps will be set.
        steps = 10

        # Inform the user about the steps.
        debugstr = \
            "Since the number of steps the interval should have " \
            "is not defined, 'steps' will be '10'."
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

            # The default spacing will be the one between two steps,
            # rounded up.
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
            
            # The default spacing will be the one between two steps,
            # rounded up.
            spacing = np.linspace(bottom,
                                  top,
                                  steps,
                                  retstep = True)[1]

            # Get the spacing by rounding up the spacing obtained
            # obtained above.
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
    if ciz:
        
        # Get the highest absolute value.
        absval = \
            np.ceil(top) if top > bottom else np.floor(bottom)
        
        # The top and bottom values will be opposite numbers with
        # absolute value equal to the highest absolute value found.
        top, bottom = absval, -absval

        # Get an evenly-spaced interval between the bottom and top
        # value.
        interval = np.linspace(bottom, top, steps)

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
    interval = np.arange(bottom, top + spacing, spacing)

    # Inform the user about the interval that will be used.
    debugstr = \
        f"The ticks' interval will be between {bottom} and {top} " \
        f"with a spacing of {spacing}: " \
        f"{', '.join([str(i) for i in interval.tolist()])}."
    logger.debug(debugstr)

    #-----------------------------------------------------------------#

    # Return the interval.
    return interval


def set_axis(ax,
             axis,
             config,
             ticks = None,
             tick_labels = None):
    """Set up the x- or y-axis after generating a plot.

    Parameters
    ----------
    ax : :class:`matplotlib.axes.Axes`
        An Axes instance.

    axis : :class:`str`, {``"x"``, ``"y"``}
        Whether the axis to be set is the x- or the y-axis.

    config : :class:`dict`
        The configuration for setting the axis.

    ticks : :class:`list`, optional
        A list of ticks' positions. If it is not passed, the ticks
        will be those already present on the axis (automatically
        determined by matplotlib when generating the plot).

    tick_labels : :class:`list`, optional
        A list of ticks' labels. If not passed, the ticks' labels
        will represent the ticks' positions.

    Returns
    -------
    ax : :class:`matplotlib.axes.Axes`
        An Axes instance. 
    """

    # If the axis to be set is the x-axis
    if axis == "x":

        # Get the corresponding methods.
        plot_ticks = plt.xticks
        set_label = ax.set_xlabel
        set_ticks = ax.set_xticks
        set_ticklabels = ax.set_xticklabels

        # Get the corresponding spine.
        spine = "bottom"

    # If the axis to be set is the y-axis
    elif axis == "y":

        # Get the corresponding methods.
        plot_ticks = plt.yticks
        set_label = ax.set_ylabel
        set_ticks = ax.set_yticks
        set_ticklabels = ax.set_yticklabels

        # Get the corresponding spine.
        spine = "left"

    #-----------------------------------------------------------------#

    # If there is a configuration for the spine
    if config.get("spine"):

        # If there is a configuration for the spine's position
        if config["spine"].get("position"):

            # Set the spine's position.
            ax.spines[spine].set_position(config["spine"]["position"])

        # Otherwise
        else:

            # Default it to 5.
            ax.spine[spine].set_position(("outward", 5))

    #-----------------------------------------------------------------#

    # If there is an axis label's configuration
    if config.get("label"):
        
        # Set the axis label.
        set_label(**config["label"])        

    #-----------------------------------------------------------------#
    
    # If no ticks' positions were passed
    if ticks is None:

        # Default to the tick locations already present.
        ticks = plot_ticks()[0]

    #-----------------------------------------------------------------#

    # If there are any ticks on the axis
    if len(ticks) > 0:      
        
        # Set the axis boundaries.
        ax.spines[spine].set_bounds(ticks[0],
                                    ticks[-1])

    #-----------------------------------------------------------------#

    # If a configuration for the tick parameters was provided
    if config.get("tick_params"):
        
        # Apply the configuration to the ticks.
        ax.tick_params(axis = axis,
                       **config["tick_params"])

    #-----------------------------------------------------------------#

    # Set the ticks.
    set_ticks(ticks = ticks)

    #-----------------------------------------------------------------#
    
    # Get the configuration for the ticks' labels.
    tick_labels_config = config.get("ticklabels", {})

    # Get the options for the ticks' labels.
    tick_labels_options = tick_labels_config.get("options", {})
    
    # If no ticks' labels were passed
    if tick_labels is None:

        # Get the format to be used for the tick labels.
        tick_labels_fmt = tick_labels_config.get("fmt", "{:.3f}")
        
        # Default to the string representations of the ticks'
        # positions.
        tick_labels = \
            get_formatted_ticklabels(ticklabels = ticks,
                                     fmt = tick_labels_fmt)
    
    # Set the ticks' labels.
    set_ticklabels(labels = tick_labels,
                   **tick_labels_options)

    #-----------------------------------------------------------------#

    # Return the axis.
    return ax


def set_cbar_axis(cbar,
                  config,
                  ticks = None,
                  tick_labels = None):
    """Set up the colorbar after generating a plot.

    Parameters
    ----------
    cbar : :class:`matplotlib.colorbar.Colorbar`
        The colorbar.

    config : :class:`dict`
        The configuration for setting the colorbar's axis.

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

    # If there is a configuration for the colorbar's label.
    if config.get("label"):
        
        # Set the colorbar's label.
        cbar.set_label(**config["label"])        

    #-----------------------------------------------------------------#

    # If no ticks' positions were passed
    if ticks is None:

        # Default to the ticks' locations already present.
        ticks = cbar.get_ticks()

    # Set the ticks.
    cbar.set_ticks(ticks = ticks)

    #-----------------------------------------------------------------#

    # Get the configuration for the ticks' labels.
    tick_labels_config = config.get("ticklabels", {})

    # Get the configuration for the ticks' labels' options.
    tick_labels_options = tick_labels_config.get("options", {})

    # If no ticks' labels were passed
    if tick_labels is None:

        # Get the format to be used for the tick labels.
        tick_labels_fmt = tick_labels_config.get("fmt", "{:.3f}")
        
        # Default to the string representations of the ticks'
        # positions.
        tick_labels = \
            get_formatted_ticklabels(ticklabels = ticks,
                                     fmt = tick_labels_fmt)
    
    # Set the ticks' labels.
    cbar.set_ticklabels(ticklabels = tick_labels,
                        **tick_labels_options)

    #-----------------------------------------------------------------#

    # Return the colorbar.
    return cbar


def set_legend(ax,
               config):
    """Set a legend for the current plot.

    Parameters
    ----------
    config : ``dict``
        The configuration for the legend.

    Returns
    -------
    ax : ``matplotlib.axes.Axes``
        An Axes instance. 
    """

    # Get the legend's handles and labels.
    handles, labels = ax.get_legend_handles_labels()

    #-----------------------------------------------------------------#
    
    # If there are handles
    if handles:

        # Draw the legend.
        ax.legend(handles = handles,
                  labels = labels,
                  bbox_transform = plt.gcf().transFigure,
                  **config)

    #-----------------------------------------------------------------#

    # Return the ax.
    return ax

def set_text(fig,
             config):
    """Write text on a plot.

    Parameters
    ----------
    fig : :class:`matplotlib.figure.Figure`
        The figure containing the plot to which the text should be
        added.

    config : :class:`dict`
        The configuration for the text.
    """

    # Add the text.
    fig.text(**config)

    #-----------------------------------------------------------------#

    # Return the figure.
    return fig


def get_colormap(cmap,
                 n_colors = None):
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

    # If the color map is available both in Seaborn and in Matplotlib.
    if cmap in SEABORN_PALETTES and cmap in MATPLOTLIB_CMAPS:

        # Inform the user about it.
        infostr = \
            f"The '{cmap}' color map is available in both " \
            "Matplotlib and in Seaborn. The Seaborn " \
            "implementation will be used."
        logger.info(infostr)

        # Get the color map from Seaborn.
        cmap = sns.color_palette(cmap, as_cmap = True)

    # If the color map is available in Seaborn.
    elif cmap in SEABORN_PALETTES:

        # Get the color map from Seaborn.
        cmap = sns.color_palette(cmap, as_cmap = True)

    # If the color map is available in Matplotlib.
    elif cmap in MATPLOTLIB_CMAPS:

        # Get the color map from Matplotlib.
        cmap = plt.get_cmap(cmap)

    # Otherwise
    else:

        # Get the names of all the available color maps.
        available_cmaps = \
            ", ".join([f"'{c}'" for c \
                       in SEABORN_PALETTES.union(MATPLOTLIB_CMAPS)])
        # Raise an error
        errstr = \
            f"Unrecognized '{cmap}' color map. Available color " \
            f"maps are: {available_cmaps}."
        raise ValueError(errstr)

    #-----------------------------------------------------------------#
    
    # If a discrete number of colors was specified.
    if n_colors:
        
        # Discretize the color map.
        colors = cmap(np.linspace(0, 1, n_colors))
        
        # Return the list of colors.
        return [c.tolist() for c \
                in mcolors.ListedColormap(colors).colors]

    #-----------------------------------------------------------------#

    # Return the color map.
    return cmap


def check_custom_configs(config,
                         sections,
                         sections_needed = [],
                         multiple_plots = False):
    """Check the configuration passed for a plot.

    Parameters
    ----------
    config : :class:`dict`
        The configuration.

    sections : :class:`list`
        The sections that we may find in the configuration passed.

    sections_needed : :class:`list`, ``[]``
        The sections that need to be present in the configuration
        passed.

    multiple_plots : :class:`bool`, ``False``
        Whether the configuration refers to a figure that will contain
        multiple plots.
    """

    # For each section needed in the configuration
    for section in sections_needed:

        # Get the 'key path' to the section in the configuration.
        key_path = section.split(":")

        # If the section is absent from the configuration
        if util.recursive_get(d = config,
                              key_path = key_path) is None:

            # Raise an error.
            errstr = \
                f"The '{section}' section must be present in the " \
                "configuration."
            raise KeyError(errstr)

    #-----------------------------------------------------------------#

    # For each section that we may encounter in the configuration
    for possible_section, item_name in SECTIONS2ITEMS.items():

        # If the section is among those that could be in the current
        # configuration
        if possible_section in sections:

            # Get the 'key path' to the section in the configuration.
            key_path = possible_section.split(":")

            # If the section is present in the configuration
            if util.recursive_get(d = config,
                                  key_path = key_path) is not None:


                # If there will be multiple plots on the same figure
                if multiple_plots:
                    
                    # Inform the user that the custom configuration
                    # will be used.
                    infostr = \
                        "A custom configuration will be used for " \
                        f"the {item_name} in each plot."

                # Otherwise
                else:
                    
                    # Inform the user that the custom configuration
                    # will be used.
                    infostr = \
                        "A custom configuration will be used for " \
                        f"the {item_name} in the plot."

                # Inform the user about the custom configuration used.
                logger.info(infostr)
