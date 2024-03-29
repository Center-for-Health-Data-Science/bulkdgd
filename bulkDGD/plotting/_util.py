#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-

#    _util.py
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
import logging as log
# Third-party packages
import matplotlib.pyplot as plt
import numpy as np


# Get the module's logger
logger = log.getLogger(__name__)


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
        The name of the item of the plot you are setting the ticks'
        positions for (e.g., ``"x-axis"``, ``"y-axis"``, or
        ``"colorbar"``).

    config : ``dict``
        The configuration for the interval that the ticks'
        positions should cover.

    Returns
    -------
    ticks_positions : ``numpy.ndarray``
        An array containing the ticks' positions.
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

    #-----------------------------------------------------------------#
    
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

    #-----------------------------------------------------------------#

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

    #-----------------------------------------------------------------#

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

    #-----------------------------------------------------------------#

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

    #-----------------------------------------------------------------#

    # If the two extremes of the interval coincide
    if top == bottom:
        
        # Return only one value
        return np.array([bottom])

    #-----------------------------------------------------------------#

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

    #-----------------------------------------------------------------#

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

            # Get the spacing by rounding up the spacing
            # obtained above
            spacing = np.ceil(spacing*(1/rtn)) / (1/rtn)

            # Inform the user about the spacing
            infostr = \
                f"Since the spacing between the ticks is not " \
                f"defined, 'spacing' will be the value " \
                f"guaranteeing an equipartition of the interval " \
                f"between {bottom} and {top} in {steps} " \
                f"number of steps ({spacing})."
            logger.info(infostr)

    #-----------------------------------------------------------------#

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

    #-----------------------------------------------------------------#

    # Get the interval
    interval = np.arange(bottom, top + spacing, spacing)

    # Inform the user about the interval that will be used
    infostr = \
        f"The ticks' interval will be between {bottom} and {top} " \
        f"with a spacing of {spacing}: " \
        f"{', '.join([str(i) for i in interval.tolist()])}."
    logger.info(infostr)

    #-----------------------------------------------------------------#

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
        An Axes instance.

    axis : ``str``, {``"x"``, ``"y"``}
        Whether the axis to be set is the x- or the y-axis.

    config : ``dict``
        The configuration for setting the axis.

    ticks : ``list``, optional
        A list of ticks' positions. If it is not passed, the ticks
        will be those already present on the axis (automatically
        determined by matplotlib when generating the plot).

    tick_labels : ``list``, optional
        A list of ticks' labels. If not passed, the ticks' labels
        will represent the ticks' positions.

    Returns
    -------
    ax : ``matplotlib.axes.Axes``
        An Axes instance. 
    """

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

    #-----------------------------------------------------------------#

    # If there is an axis label's configuration
    if config.get("label"):
        
        # Set the axis label
        set_label(**config["label"])        

    #-----------------------------------------------------------------#
    
    # If no ticks' positions were passed
    if ticks is None:

        # Default to the tick locations already present
        ticks = plot_ticks()[0]

    #-----------------------------------------------------------------#

    # If there are any ticks on the axis
    if len(ticks) > 0:      
        
        # Set the axis boundaries
        ax.spines[spine].set_bounds(ticks[0],
                                    ticks[-1])

    #-----------------------------------------------------------------#

    # If a configuration for the tick parameters was provided
    if config.get("tick_params"):
        
        # Apply the configuration to the ticks
        ax.tick_params(axis = axis,
                       **config["tick_params"])

    #-----------------------------------------------------------------#

    # Set the ticks
    set_ticks(ticks = ticks)

    #-----------------------------------------------------------------#
    
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

    #-----------------------------------------------------------------#

    # Return the axis
    return ax


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

    # Get the legend's handles and labels
    handles, labels = ax.get_legend_handles_labels()

    #-----------------------------------------------------------------#
    
    # If there are handles
    if handles:

        # Draw the legend
        ax.legend(handles = handles,
                  labels = labels,
                  bbox_transform = plt.gcf().transFigure,
                  **config)

    #-----------------------------------------------------------------#

    # Retutn the ax
    return ax