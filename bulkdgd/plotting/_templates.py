#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-

#    _templates.py
#
#    Templates for the different configurations.
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
__doc__ = "Templates for the different configurations."


#######################################################################


# Import from 'bulkdgd'.
from . import _defaults


#######################################################################


# Set the template for the plotting configurations.
CONFIG_PLOT_TEMPLATE = {

    # Set the options for the output.
    "output" : _defaults.OUTPUT_OPTIONS,

    #-----------------------------------------------------------------#

    # Set the options for the figure.
    "figure" : _defaults.FIGURE_OPTIONS,

    #-----------------------------------------------------------------#

    # Set the font properties to be used throughout the plot.
    "general_fontproperties" : _defaults.FONT_PROPERTIES_OPTIONS,

    # Set the palette to be used throughout the plot (if the plot
    # has multiple panels.
    "general_palette" : \
        
       {# Set the supported data types.
        "dtypes" : (str, list),
        # Set a help string.
        "help" : \
            "The palette to use for all panels for the plot. It can " \
            "be either the name of a colormap, or a list of named " 
            "colors/RGB tuples. More details about how colors can " \
            "be specified can be found at: " \
            f"{_defaults.LINKS['colors']}.",
       },

    #-----------------------------------------------------------------#

    # Set the options if the plot is a histogram.
    "histogram" : _defaults.HISTOGRAM_OPTIONS,

    # Set the options for the second histogram if the plot is a dual or
    # overlapping histogram.
    "histogram_2" : _defaults.HISTOGRAM_OPTIONS,

    #-----------------------------------------------------------------#

    # Set the options if the plot is a bar plot.
    "barplot" : _defaults.BARPLOT_OPTIONS,

    #-----------------------------------------------------------------#

    # Set the options if the plot is a line plot.
    "lineplot" : _defaults.LINEPLOT_OPTIONS,

    #-----------------------------------------------------------------#

    # Set the options if the plot is a scatter plot.
    "scatterplot" : _defaults.SCATTERPLOT_OPTIONS,

    # Set the options if the plot is a secondary scatter plot.
    "scatterplot_2" : _defaults.SCATTERPLOT_OPTIONS,

    #-----------------------------------------------------------------#

    # Set the options if the plot is a boxplot.
    "boxplot" : _defaults.BOXPLOT_OPTIONS,

    #-----------------------------------------------------------------#

    # Set the options if the plot is a violin plot.
    "violinplot" : _defaults.VIOLINPLOT_OPTIONS,

    #-----------------------------------------------------------------#

    # Set the options if the plot is a strip plot.
    "stripplot" : _defaults.STRIPPLOT_OPTIONS,

    #-----------------------------------------------------------------#

    # Set the options for the title.
    "title" : _defaults.TITLE_OPTIONS,

    #-----------------------------------------------------------------#

    # Set the options for the x-axis.
    "xaxis" : _defaults.X_AXIS_OPTIONS,

    #-----------------------------------------------------------------#
    
    # Set the options for the y-axis.
    "yaxis" : _defaults.Y_AXIS_OPTIONS,

    # Set the options for the y-axis of a bar plot in a multi-panel
    # plot.
    "yaxis_barplot" : _defaults.Y_AXIS_OPTIONS,

    # Set the options for the y-axis of a violin plot in a multi-panel
    # plot.
    "yaxis_violinplot" : _defaults.Y_AXIS_OPTIONS,

    # Set the options for the y-axis of a strip plot in a multi-panel
    # plot.
    "yaxis_stripplot" : _defaults.Y_AXIS_OPTIONS,

    #-----------------------------------------------------------------#

    # Set the options for the legend.
    "legend" : _defaults.LEGEND_OPTIONS,

    #-----------------------------------------------------------------#

    # Set the options for the colorbar.
    "colorbar" : _defaults.COLORBAR_OPTIONS,

    #-----------------------------------------------------------------#

    # Set the options for the vertical line to be plotted.
    "vline" : _defaults.VLINE_OPTIONS,

    #-----------------------------------------------------------------#

    # Set the options for the horizontal line to be plotted.
    "hline" : _defaults.HLINE_OPTIONS,

    # Set the options for the horizontal line to be plotted on a bar
    # plot in a multi-panel plot.
    "hline_barplot" : _defaults.HLINE_OPTIONS,

    # Set the options for the horizontal line to be plotted on a violin
    # plot in a multi-panel plot.
    "hline_violinplot" : _defaults.HLINE_OPTIONS,

    # Set the options for the horizontal line to be plotted on a strip
    # plot in a multi-panel plot.
    "hline_stripplot" : _defaults.HLINE_OPTIONS,

    #-----------------------------------------------------------------#

    # Set the options for the text to be plotted.
    "text" : _defaults.TEXT_OPTIONS,

    #-----------------------------------------------------------------#

    }