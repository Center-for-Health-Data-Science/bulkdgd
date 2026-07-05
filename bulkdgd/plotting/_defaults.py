#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-

#    _defaults.py
#
#    Private default values.
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
__doc__ = "Private default values."


#######################################################################


# Set the links where the documentation of specific options is located.
LINKS = {

    # Set the link where to find the available markers.
    "markers" : \
        "https://matplotlib.org/stable/api/markers_api.html#" \
        "module-matplotlib.markers",
    
    #-----------------------------------------------------------------#

    # Set the link where to find the available line styles for the
    # markers.
    "fillstyles" : \
        "https://matplotlib.org/stable/api/_as_gen/" \
        "matplotlib.markers.MarkerStyle.html#" \
        "matplotlib.markers.MarkerStyle",
    
    #-----------------------------------------------------------------#

    # Set the link where to find the available cap styles for the
    # markers.
    "capstyles" : \
        "https://matplotlib.org/stable/api/_enums_api.html#" 
        "matplotlib._enums.CapStyle",
    
    #-----------------------------------------------------------------#

    # Set the link where to find the available join styles for the
    # markers.
    "joinstyles" : \
        "https://matplotlib.org/stable/api/_enums_api.html#" \
        "matplotlib._enums.JoinStyle",
    
    #-----------------------------------------------------------------#
    
    # Set the link where to find the available ways to specify colors.
    "colors" : \
        "https://matplotlib.org/stable/tutorials/colors/colors.html",
    
    #-----------------------------------------------------------------#
    
    # Set the link where to find the available backends.
    "backends" : \
        "https://matplotlib.org/stable/users/explain/figure" \
        "/backends.html#the-builtin-backends",
    
    #-----------------------------------------------------------------#

   }


#######################################################################


# Set the options for the interval to be displayed on an axis.
INTERVAL_OPTIONS = {\

    # The type of interval displayed on the axis.
    "type" : \
    
       {# Set the supported data types.
        "dtypes" : (str,),
        # Set a help string.
        "help" : \
            "The type of interval displayed on the axis. It can " \
            "be either 'discrete' or 'continuous'.",
       },
    
    #-----------------------------------------------------------------#
    
    # Round all ticks' positions to the nearest X number.
    "round_to_nearest" : \
    
       {# Set the supported data types.
        "dtypes" : (int, float),
        # Set a help string.
        "help" : \
            "Round all ticks' positions to the nearest " \
            "'round_to_nearest' number. For instance, setting it to " \
            "1 would round to the nearest integer, while setting " \
            "it to 0.5 would round to the nearest half an " \
            "integer. If the interval is 'discrete', it is set to " \
            "1 by default. If the interval is 'continuous', it is " \
            "set to 0.5 by default.",
       },

    #-----------------------------------------------------------------#

    # The highest value displayed on the axis.
    "top" : \
    
       {# Set the supported data types.
        "dtypes" : (int, float),
        # Set a help string.
        "help" : \
            "The highest value displayed on the axis. It is " \
            "automatically inferred from the plot if not passed " \
            "(and appropriately rounded if the 'round_to_nearest' " \
            "option is passed)",
       },
    
    #-----------------------------------------------------------------#

    # The lowest value displayed on the axis.
    "bottom" : \
    
       {# Set the supported data types.
        "dtypes" : (int, float),
        # Set a help string.
        "help" : \
            "The lowest value displayed on the axis. It is " \
            "automatically inferred from the plot if not passed " \
            "(and appropriately rounded if the 'round_to_nearest' " \
            "option is passed)",
       },
    
    #-----------------------------------------------------------------#

    # The number of steps (ticks) displayed on the axis.
    "steps" : \
    
       {# Set the supported data types.
        "dtypes" : (int,),
        # Set a help string.
        "help" : \
            "The number of steps (ticks) displayed on the axis. " \
            "This number includes the most extreme ticks. It is " \
            "set to 10 by default.",
       },
    
    #-----------------------------------------------------------------#

    # The spacing between the ticks on the axis.
    "spacing" : \
    
       {# Set the supported data types.
        "dtypes" : (int, float),
        # Set a help string.
        "help" : \
            "The spacing between the ticks on the axis. If not " \
            "passed, it is automatically computed from the 'top', " \
            "'bottom', and 'steps' values.",
       },
    
    #-----------------------------------------------------------------#

    # Whether to center the interval around 0.
    "center_around_zero" : \
    
       {# Set the supported data types.
        "dtypes" : (bool,),
        # Set a help string.
        "help" : \
            "Whether to center the interval around 0. If 'True', " \
            "the highest absolute value between 'top' and 'bottom' " \
            "(plus 'spacing') will be taken as the most extreme " \
            "value of the interval in the positive and negative " \
            "directions.",
       },
    
    #-----------------------------------------------------------------#
    
    }


#######################################################################


# Set the supported options to set the font properties for text
# elements.
FONT_PROPERTIES_OPTIONS = {

    # The name of the font family.
    "family" : \
       
       {# Set the supported data types.
        "dtypes" : (str, list),
        # Set the conflicting options.
        "conflicts" : ["fname"],
        # Set whether the option has priority over the conflicting
        # ones.
        "has_priority" : False,
        # Set a help string.
        "help" : "The name of the font family.",
       },
    
    #-----------------------------------------------------------------#
    
    # The size of the font.
    "size" : \
       
       {# Set the supported data types.
        "dtypes" : (int, float, str),
        # Set a help string.
        "help" : "The size of the font.",
       },

    #-----------------------------------------------------------------#
    
    # The weight of the font.
    "weight" : \
       
       {# Set the supported data types.
        "dtypes" : (str, int, float),
        # Set the conflicting options.
        "conflicts" : ["fname"],
        # Set whether the option has priority over the conflicting
        # ones.
        "has_priority" : False,
        # Set a help string.
        "help" : "The weight of the font.",
       },

    #-----------------------------------------------------------------#
    
    # The style of the font.
    "style" : \
       
       {# Set the supported data types.
        "dtypes" : (str,),
        # Set the conflicting options.
        "conflicts" : ["fname"],
        # Set whether the option has priority over the conflicting
        # ones.
        "has_priority" : False,
        # Set a help string.
        "help" : "The style of the font.",
       },

    #-----------------------------------------------------------------#
    
    # The variant of the font.
    "variant" : \
       
       {# Set the supported data types.
        "dtypes" : (str,),
        # Set the conflicting options.
        "conflicts" : ["fname"],
        # Set whether the option has priority over the conflicting
        # ones.
        "has_priority" : False,
        # Set a help string.
        "help" : "The variant of the font.",
       },

    #-----------------------------------------------------------------#
    
    # The stretch of the font.
    "stretch" : \
       
       {# Set the supported data types.
        "dtypes" : (str, float),
        # Set a help string.
        "help" : "The stretch of the font.",
       },

    #-----------------------------------------------------------------#
    
    # The font family for the math text.
    "math_fontfamily" : \
       
       {# Set the supported data types.
        "dtypes" : (str,),
        # Set a help string.
        "help" : "The font family for math text.",
       },

    #-----------------------------------------------------------------#
    
    # The path to the font file.
    "fname" : \
       
       {# Set the supported data types.
        "dtypes" : (str,),
        # Set the conflicting options.
        "conflicts" : \
            ["family", "weight", "style", "variant", "stretch"],
        # Set whether the option has priority over the conflicting
        # ones.
        "has_priority" : True,
        # Set a help string.
        "help" : "The path to the font file.",
       },
    
    #-----------------------------------------------------------------#

    }


#######################################################################


# Set the supported options for the figure.
FIGURE_OPTIONS = {

    # The options to adjust the sub-plots in the figure.
    "subplots" : {

        # The position of the left edge of the sub-plots as a
        # fraction of the figure's width.
        "left" : \
        
           {# Set the supported data types.
            "dtypes" : (int, float),
            # Set a help string.
            "help" : \
                "The position of the left edge of the sub-plots " \
                "as a fraction of the figure's width.",
           },
        
        #-------------------------------------------------------------#

        # The position of the right edge of the sub-plots as a
        # fraction of the figure's width.
        "right" : \
        
           {# Set the supported data types.
            "dtypes" : (int, float),
            # Set a help string.
            "help" : \
                "The position of the right edge of the sub-plots " \
                "as a fraction of the figure's width.",
           },
        
        #-------------------------------------------------------------#

        # The position of the top edge of the sub-plots as a
        # fraction of the figure's height.
        "top" : \
        
           {# Set the supported data types.
            "dtypes" : (int, float),
            # Set a help string.
            "help" : \
                "The position of the top edge of the sub-plots " \
                "as a fraction of the figure's height.",
           },

        #-------------------------------------------------------------#

        # The position of the bottom edge of the sub-plots as a
        # fraction of the figure's height.
        "bottom" : \
        
           {# Set the supported data types.
            "dtypes" : (int, float),
            # Set a help string.
            "help" : \
                "The position of the bottom edge of the sub-plots " \
                "as a fraction of the figure's height.",
           },
        
        #-------------------------------------------------------------#

        # The height of the padding between sub-plots, as a fraction
        # of the average height of the y-axis.
        "hspace" : \
            
           {# Set the supported data types.
            "dtypes" : (int, float),
            # Set a help string.
            "help" : \
                "The height of the padding between sub-plots as a " \
                "fraction of the average height of the y-axis.",
           },
        
        #-------------------------------------------------------------#

        # The width of the padding between sub-plots as a fraction
        # of the average width of the x-axis.
        "wspace" : \
        
           {# Set the supported data types.
            "dtypes" : (int, float),
            # Set a help string.
            "help" : \
                "The width of the padding between sub-plots as a " \
                "fraction of the average width of the x-axis.",
           },
    
        },
    
    #-----------------------------------------------------------------#
    
    # The figure's size in inches (width, height).
    "sizeinches" :     

       {# Set the supported data types.
        "dtypes" : ([(int, float), (int, float)],),
        # Set a help string.
        "help" : "The figure's size in inches (width, height).",
       },
    
    #-----------------------------------------------------------------#
    
    }


#######################################################################


# Set the supported options for line elements.
LINE_OPTIONS = {

    # The color of the line elements.
    "color" : \
    
       {# Set the supported data types.
        "dtypes" : (str, tuple),
        # Set a help string.
        "help" : \
            "The color of the {:s}. More details about how colors " \
            f"can be specified can be found at: {LINKS['colors']}.",
       },

    #-----------------------------------------------------------------#
    
    # The alpha blending value for the colors of the line elements.
    "alpha" : \
      
       {# Set the supported data types.
        "dtypes" : (int, float),
        # Set a help string.
        "help" : \
            "The alpha blending value for the color of the {:s}. " \
            "It must be between 0 (transparent) and 1 (opaque).",
       },

    #-----------------------------------------------------------------#
   
    # The width of the line elements in points.
    "linewidth" : \
    
       {# Set the supported data types.
        "dtypes" : (int, float),
        # Set a help string.
        "help" : "The width of the {:s} in points.",
       },

    #-----------------------------------------------------------------#

    # The style of the line elements.
    "linestyle" : \
    
       {# Set the supported data types.
        "dtypes" : (str, tuple),
        # Set a help string.
        "help" : \
            "The style of the {:s}. It can be: '-' or 'solid' " \
            "(solid line), '--' or 'dashed' (dashed line). '-.' " \
            "or 'dashdot' (dash-dotted line), ':' or 'dotted' " \
            "(dotted line), or a tuple containing as the first " \
            "element the offset to draw a dashed line ('float') " \
            "and as the second element a tuple with a sequence of " \
            "floats of even length describing the length of " \
            "dashes and spaces in ink points ('float', 'float'). " \
            "For instance, '(4, 1, 2, 1)' represents a sequence " \
            "of 4-point and 2-point dashes separated by 1-point " \
            "spaces.",
       },

    #-----------------------------------------------------------------#

    # How to draw the end caps if the line elements are solid.
    "solid_capstyle" : \
    
       {# Set the supported data types.
        "dtypes" : (str,),
        # Set a help string.
        "help" :    
            "How to draw the end caps of the {:s} if solid. It " \
            "can be: 'butt', 'projecting', or 'round'. A visual " \
            "representation of the different cap styles is " \
            f"available at: {LINKS['capstyles']}.",
       },

    #-----------------------------------------------------------------#
    
    # How to join line segments if the line elements are solid.
    "solid_joinstyle" : \
    
       {# Set the supported data types.
        "dtypes" : (str,),
        # Set a help string.
        "help" : \
            "How to join line segments of the {:s} if solid. It " \
            "can be: 'miter', 'round', or 'bevel'. A visual " \
            "representation of the different choices is available " \
            f"at: {LINKS['joinstyles']}.",
       },

    #-----------------------------------------------------------------#
    
    # How to draw the end caps if the line elements are dashed.
    "dash_capstyle" : \
    
       {# Set the supported data types.
        "dtypes" : (str,),
        # Set a help string.
        "help" :    
            "How to draw the end caps of the {:s} if dashed. It " \
            "can be: 'butt', 'projecting', or 'round'. A visual " \
            "representation of the different cap styles is " \
            f"available at: {LINKS['capstyles']}.",
       },

    #-----------------------------------------------------------------#
    
    # How to join line segments if the line elements are dashed.
    "dash_joinstyle" : \
    
       {# Set the supported data types.
        "dtypes" : (str,),
        # Set a help string.
        "help" : \
            "How to join line segments of the {:s} if dashed. It " \
            "can be: 'miter', 'round', or 'bevel'. A visual " \
            "representation of the different choices is available " \
            f"at: {LINKS['joinstyles']}.",
       },

    #-----------------------------------------------------------------#
    
    # How the data points are connected.
    "drawstyle" : \
    
       {# Set the supported data types.
        "dtypes" : (str,),
        # Set a help string.
        "help" : \
            "How the data points in the {:s} are connected. The " \
            "available options are: 'default' (the points are " \
            "connected with straight lines), 'steps-pre' (the  " \
            "points are connected with horizontal lines with " \
            "vertical steps, and the step is at the beginning of " \
            "the line segment), 'steps-mid' (the points are " \
            "connected with horizontal lines with vertical steps, " \
            "and the step is halfway  between the points), " \
            "'steps-post' (the points are connected with horizontal " \
            "ines with vertical steps, and the step is at the end "
            "of the line segment).",
       },

    #-----------------------------------------------------------------#

    # The color of the gaps between dashes if the line elements are
    # dashed.
    "gapcolor" : \
    
       {# Set the supported data types.
        "dtypes" : (str, tuple),
        # Set a help string.
        "help" : \
            "The color of the gaps between the dashes forming the " \
            "{:s}, if dashed. More details about how colors can be " \
            f"specified can be found at: {LINKS['colors']}.",
       },
    
    #-----------------------------------------------------------------#

    # The style of the marker elements.
    "marker" : \
        
       {# Set the supported data types.
        "dtypes" : (str,),
        # Set a help string.
        "help" : \
            "The style of the {:s}. All available styles " \
            f"can be found at: {LINKS['markers']}.",
       },

    #-----------------------------------------------------------------#
    
    # The frequency of the marker elements.
    "markevery" : \
        
       {# Set the supported data types.
        "dtypes" : (int, float),
        # Set a help string.
        "help" : \
            "If 'int', only every 'markevery'-th {:s} will be " \
            "plotted. If 'float', markers will be spaced along " \
            "the lines at approximately equal visual distances.",
       },

    #-----------------------------------------------------------------#
    
    # The size of the marker elements.
    "markersize" : \
    
       {# Set the supported data types.
        "dtypes" : (int, float),
        # Set a help string.
        "help" : "The size of the {:s} in points.",
       },

    #-----------------------------------------------------------------#
    
    # The width of the edges of the marker elements in points.
    "markeredgewidth" : \
    
       {# Set the supported data types.
        "dtypes" : (int, float),
        # Set a help string.
        "help" : "The width of the edges of the {:s} in points.",
       },

    #-----------------------------------------------------------------#

    # How the marker elements are filled.
    "fillstyle" : \
    
       {# Set the supported data types.
        "dtypes" : (str,),

        # Set a help string.
        "help" : \
            "How the {:s} are filled. Different styles produce " \
            "fully-filled markers, half-filled markers, or markers " \
            "with no fill. The complete list of fill styles can be " \
            f"found at: {LINKS['fillstyles']}.",
       },

    #-----------------------------------------------------------------#

    # The color of the edges of the marker elements.
    "markeredgecolor" : \
    
       {# Set the supported data types.
        "dtypes" : (str, tuple),
        # Set a help string.
        "help" : \
            "The color of the edges of the {:s}. More details about " \
            "how colors can be specified can be found at: " \
            f"{LINKS['colors']}.",
       },

    #-----------------------------------------------------------------#
    
    # The color of the filling of the marker elements.
    "markerfacecolor" : \
    
       {# Set the supported data types.
        "dtypes" : (str, tuple),
        # Set a help string.
        "help" : \
            "The color of the filling of the {:s}. More details " \
            "about how colors can be specified can be found at: " \
            f"{LINKS['colors']}.",
       },

    #-----------------------------------------------------------------#
    
    # The alternate color for the filling of the marker elements.
    "markerfacecoloralt" : \
    
       {# Set the supported data types.
        "dtypes" : (str, tuple),
        # Set a help string.
        "help" : \
            "The color to use for the second half of the {:s} " \
            "when plotting half-filled markers. More details " \
            "about how colors can be specified can be found at: " \
            f"{LINKS['colors']}. Please see the complete list of " \
            f"fill styles (at {LINKS['fillstyles']}) for more " \
            "information about how this color is applied.",
       },
    
    #-----------------------------------------------------------------#

    }


#######################################################################


# Set the supported options for text.
TEXT_OPTIONS = {

    # The alpha blending value, between 0 (transparent) and 1
    # (opaque).
    "alpha" : \

       {# Set the supported data types.
        "dtypes" : (int, float),
        # Set a help string.
        "help" : \
            "The alpha blending value for the {:s} color. It must " \
            "be between 0 (transparent) and 1 (opaque).",
       },
    
    #-----------------------------------------------------------------#
    
    # The background color of the text.
    "backgroundcolor" : \
    
       {# Set the supported data types.
        "dtypes" : (str, tuple),
        # Set a help string.
        "help" : \
            "The background color of the text. More details about " \
            "how colors can be specified can be found at: " \
            f"{LINKS['colors']}.",
       },
    
    #-----------------------------------------------------------------#
    
    # The color of the text.
    "color" : \
    
       {# Set the supported data types.
        "dtypes" : (str, tuple),
        # Set a help string.
        "help" : \
            "The color of the text. More details about how colors " \
            f"may be specified can be found at: {LINKS['colors']}.",
       },
    
    #-----------------------------------------------------------------#
    
    # The horizontal alignment of the text.
    "horizontalalignment" : \
    
       {# Set the supported data types.
        "dtypes" : (str,),
        # Set a help string.
        "help" : \
            "The horizontal alignment of the text. It can be " \
            "'left', 'right', or 'center'.",
       },
    
    #-----------------------------------------------------------------#
    
    # The vertical alignment of the text.
    "verticalalignment" : \
    
       {# Set the supported data types.
        "dtypes" : (str,),
        # Set a help string.
        "help" : \
            "The vertical alignment of the text. It can be " \
            "'baseline', 'bottom', 'center', 'center_baseline', or " \
            "'top'",
       },
    
    #-----------------------------------------------------------------#
    
    # The text alignment for multiline texts.
    "multialignment" : \
    
       {# Set the supported data types.
        "dtypes" : (str,),
        # Set a help string.
        "help" : \
            "The text alignment for multiline texts. It can be " \
            "'left', 'right', or 'center'.",
       },
    
    #-----------------------------------------------------------------#
    
    # The position of the text in x-y coordinates.
    "position" : \
    
       {# Set the supported data types.
        "dtypes" : ([(int, float), (int, float)],),
        # Set a help string.
        "help" : \
            "A tuple of 'float' indicating the position of the " \
            "text in x-y coordinates.",
       },
    
    #-----------------------------------------------------------------#
    
    # The text's rotation angle.
    "rotation" : \
    
       {# Set the supported data types.
        "dtypes" : (int, float),
        # Set a help string.
        "help" :    
            "The text's rotation angle (in degrees) in a " \
            "counterclockwise direction.",
       },
    
    #-----------------------------------------------------------------#
    
    # The text's rotation mode.
    "rotation_mode" : \
    
       {# Set the supported data types.
        "dtypes" : (str,),
        # Set a help string.
        "help" : \
            "The text's rotation mode. If 'default', the text is " \
            "first rotated and then aligned according to its " \
            "horizontal and vertical alignments. If 'anchor', " \
            "the text is first aligned and then rotated.",
       },
    
    #-----------------------------------------------------------------#
    
    # The line spacing as a multiple of the font size.
    "linespacing" : \
    
       {# Set the supported data types.
        "dtypes" : (int, float),
        # Set a help string.
        "help" : \
            "The line spacing as a multiple of the font size. " \
            "By default, it is '1.2'.",
       },
    
    #-----------------------------------------------------------------#
    
    # Whether to force parsing the text as mathematical text.
    "parse_math" : \
    
       {# Set the supported data types.
        "dtypes" : (bool,),
        # Set a help string.
        "help" : \
            "Whether to force parsing the text as mathematical text.",
       },
    
    #-----------------------------------------------------------------#
    
    # Whether to render text using TeX.
    "usetex" : \
    
       {# Set the supported data types.
        "dtypes" : (bool,),
        # Set a help string.
        "help" : \
            "Whether to render text using TeX. It is 'False' by " \
            "default.",
       },
    
    #-----------------------------------------------------------------#

    }


#######################################################################


# Set the supported options for patches.
PATCH_OPTIONS = {

    # The alpha blending value, between 0 (transparent) and 1
    # (opaque).
    "alpha" : \

       {# Set the supported data types.
        "dtypes" : (int, float),
        # Set a help string.
        "help" : \
            "The alpha blending value for the {:s}, between 0 " \
            "(transparent) and 1 (opaque).",
       },
    
    #-----------------------------------------------------------------#

    # The cap style used for the patches.
    "capstyle" : \

       {# Set the supported data types.
        "dtypes" : (str,),
        # Set a help string.
        "help" : \
            "How to draw the end caps of the edges of the {:s}. It " \
            "can be: 'butt', 'projecting', or 'round'. A visual " \
            "representation of the different cap styles is " \
            f"available at: {LINKS['capstyles']}.",
        },

    #-----------------------------------------------------------------#

    # The join style used for the patches.
    "joinstyle" : \

       {# Set the supported data types.
        "dtypes" : (str,),
        # Set a help string.
        "help" : \
            "How to join line segments composing the edges of the " \
            "{:s}. It can be: 'miter', 'round', or 'bevel'. A " 
            "visual representation of the different choices is " \
            f"available at: {LINKS['joinstyles']}.",
       },

    #-----------------------------------------------------------------#

    # The color of the patches.
    "color" : \
       
       {# Set the supported data types.
        "dtypes" : (str, tuple),
        # Set a help string.
        "help" : \
            "The color of the {:s}. More details about how colors " \
            f"can be specified can be found at: {LINKS['colors']}.",
       },

    #-----------------------------------------------------------------#

    # Whether the patch is filled.
    "fill" : \
        
       {# Set the supported data types.
        "dtypes" : (bool,),
        # Set a help string.
        "help" : "Whether the {:s} are filled.",
       },
    
    #-----------------------------------------------------------------#

    # The color of the edges of the patches.
    "edgecolor" : \

       {# Set the supported data types.
        "dtypes" : (str, tuple),
        # Set a help string.
        "help" : \
            "The color of the edges of the {:s}. More details about " \
            "how colors can be specified can be found at: " \
            f"{LINKS['colors']}.",
       },
    
    #-----------------------------------------------------------------#

    # The color of the faces of the patches.
    "facecolor" : \

       {# Set the supported data types.
        "dtypes" : (str, tuple),
        # Set a help string.
        "help" : \
            "The color of the filling of the {:s}, if they are " \
            "filled. More details about how colors can be specified " \
            f"can be found at: {LINKS['colors']}.",
       },
    
    #-----------------------------------------------------------------#

    # The line style of the patch.
    "linestyle" : \
        
       {# Set the supported data types.
        "dtypes" : (str, tuple),
        # Set a help string.
        "help" : \
            "The style of the edges of the {:s}. It can be: '-' or " \
            "'solid' (solid line), '--' or 'dashed' (dashed line). " \
            "'-.' or 'dashdot' (dash-dotted line), ':' or 'dotted' " \
            "(dotted line), or a tuple containing as the first " \
            "element the offset to draw a dashed line ('float') and " \
            "as the second element a tuple with a sequence of " \
            "floats of even length describing the length of dashes " \
            "and spaces in ink points ('float', 'float'). For " \
            "instance, '(4, 1, 2, 1)' represents a sequence of " \
            "4-point and 2-point dashes separated by 1-point spaces.",
       },
    
    #-----------------------------------------------------------------#
    
    # The line width of the patch.
    "linewidth" : \
        
       {# Set the supported data types.
        "dtypes" : (int, float),
        # Set a help string.
        "help" : "The width of the edges of the {:s} in points.",
       },
    
    #-----------------------------------------------------------------#
    
    # The hatch pattern.
    "hatch" : \
        
       {# Set the supported data types.
        "dtypes" : (str,),
        # Set a help string.
        "help" : \
            "The hatch pattern for the filling of the {:s}, if " \
            "they are filled.",
       },
    
    #-----------------------------------------------------------------#

    }


#######################################################################


# Set the supported options for collections.
COLLECTION_OPTIONS = {

    # The color of the filling of the collections.
    "facecolors" : \
    
       {# Set the supported data types.
        "dtypes" : (str, tuple),
        # Set a help string.
        "help" : \
            "The color of the filling of the {:s}. More details " \
            "about how colors can be specified can be found at: " \
            f"{LINKS['colors']}.",
       },
    
    #-----------------------------------------------------------------#

    # The color of the edges of the collections.
    "edgecolors" : \
    
       {# Set the supported data types.
        "dtypes" : (str, tuple),
        # Set a help string.
        "help" : \
            "The color of the edges of the {:s}. More details " \
            "about how colors can be specified can be found at: " \
            f"{LINKS['colors']}.",
       },

    #-----------------------------------------------------------------#

    # The alpha blending value for the collections.
    "alpha" : \

       {# Set the supported data types.
        "dtypes" : (int, float),
        # Set a help string.
        "help" : \
            "The alpha blending value for the {:s}. It must be " \
            "between 0 (transparent) and 1 (opaque).",
       },

    #-----------------------------------------------------------------#
    
    # The width of the edges of the collections in points.
    "linewidths" : \
    
       {# Set the supported data types.
        "dtypes" : (int, float),
        # Set a help string.
        "help" : "The width of the edges of the {:s} in points.",
       },

    #-----------------------------------------------------------------#

    # The style of the edges of the collections.
    "linestyles" : \

       {# Set the supported data types.
        "dtypes" : (str, tuple),
        # Set a help string.
        "help" : \
            "The style of the edges of the {:s}. It can be: '-' or " \
            "'solid' (solid line), '--' or 'dashed' (dashed line). " \
            "'-.' or 'dashdot' (dash-dotted line), ':' or 'dotted' " \
            "(dotted line), or a tuple containing as the first " \
            "element the offset to draw a dashed line ('float') and " \
            "as the second element a tuple with a sequence of " \
            "floats of even length describing the length of dashes " \
            "and spaces in ink points ('float', 'float'). For " \
            "instance, '(4, 1, 2, 1)' represents a sequence of " \
            "4-point and 2-point dashes separated by 1-point spaces.",
       },
    
    #-----------------------------------------------------------------#
    
    # How to draw the end caps of the edges of the collections.
    "capstyle" : \

       {# Set the supported data types.
        "dtypes" : (str,),
        # Set a help string.
        "help" : \
            "How to draw the end caps of the edges of the {:s}. It " \
            "can be: 'butt', 'projecting',  or 'round'. A visual " \
            "representation of the different cap styles is " \
            f"available at: {LINKS['capstyles']}.",
       },

    #-----------------------------------------------------------------#

    # How to join the line segments composing the edges of the
    # collections.
    "joinstyle" : \

       {# Set the supported data types.
        "dtypes" : (str,),
        # Set a help string.
        "help" : \
            "How to join the segments composing the edges of " \
            "the {:s}. It can be: 'miter', 'round', or 'bevel'. A " \
            "visual representation of the different choices is " \
            f"available at: {LINKS['joinstyles']}.",
       },

    #-----------------------------------------------------------------#

    # The hatching pattern of the filling of the collections.
    "hatch" : \
    
       {# Set the supported data types.
        "dtypes" : (str,),
        # Set a help string.
        "help" : "The hatching pattern of the filling of the {:s}.",
       },

    #-----------------------------------------------------------------#

    }

#######################################################################


# Set the supported options for legends.
LEGEND_OPTIONS = {
    
    # The position of the legend.
    "loc" : \
        
       {# Set the supported data types.
        "dtypes" : (str, [(int, float), (int, float)]),
        # Set a help string.
        "help" : \
            "The location of the legend. The options 'upper " \
            "left', 'upper right', 'lower left', and 'lower " \
            "right' place the legend at the corresponding corners " \
            "of the plot area. In contrast, the options 'upper " \
            "center', 'lower center', 'center left', and 'center " \
            "right' place the legend at the center of the " \
            "corresponding edge of the plot area. 'center' places " \
            "the legend at the center of the plot area while 'best' " \
            "places the legend at the location with the minimum " \
            "overlap with the other elements of the plot. If a " \
            "tuple of 'float' is passed, they are interpreted as " \
            "the coordinates of the lower-left corner of the " \
            "legend in the coordinate system formed by the axes.",
       },
    
    #-----------------------------------------------------------------#
    
    # The box that is used to position the legend in conjunction with
    # 'loc'.
    "bbox_to_anchor" : \
        
       {# Set the supported data types.
        "dtypes" : \
            ([(int, float), (int, float)], 
             [(int, float), (int, float), (int, float), (int, float)]),
        # Set a help string.
        "help" : \
            "The box that, together with 'loc', is used to place " \
            "the legend. This argument allows for an arbitrary " \
            "legend placement using coordinates in the coordinate " \
            "system formed by the plot's axes. If it is a tuple " \
            "of two 'float', they will interpreted as the x and y " \
            "coordinates where the corner of the legend specified " \
            "with 'loc' is placed. If it is a tuple of four " \
            "'float', it specifies the box that the legend is " \
            "placed in (x coordinate, y coordinate, width, height).",
       },
    
    #-----------------------------------------------------------------#
    
    # Whether the legend should be drawn on a frame.
    "frameon" : \
        
       {# Set the supported data types.
        "dtypes" : (bool,),
        # Set a help string.
        "help" : \
            "Whether the legend has a frame. It is 'True' by default.",
       },
    
    #-----------------------------------------------------------------#

    # The alpha value for the frame.
    "framealpha" : \
        
       {# Set the supported data types.
        "dtypes" : (int, float),
        # Set a help string.
        "help" : \
            "The alpha blending value for the legend's frame. " \
            "It must be between 0 (transparent) and 1 (opaque).",
       },

    #-----------------------------------------------------------------#

    # The color of the legend's background.
    "facecolor" : \
        
       {# Set the supported data types.
        "dtypes" : (str, tuple),
        # Set a help string.
        "help" : \
            "The legend's background color. More details about how " \
            "colors can be specified can be found at: " \
            f"{LINKS['colors']}. If set to 'inherit', it will be " \
            "the same color as the plot's background.",
       },
    
    #-----------------------------------------------------------------#
    
    # The color of the legend's edges.
    "edgecolor" : \
    
       {# Set the supported data types.
        "dtypes" : (str, tuple),
        # Set a help string.
        "help" : \
            "The legend's edges' color. More details about how " \
            "colors can be specified can be found at: " \
            f"{LINKS['colors']} If set to 'inherit', it will be " \
            "the same color as the plot's background.",
       },
    
    #-----------------------------------------------------------------#

    # Whether the legend's frame has round edges.
    "fancybox" : \
        
       {# Set the supported data types.
        "dtypes" : (bool,),
        # Set a help string.
        "help" : \
            "Whether the legend's frame has round edges. It is " \
            "'False' by default.",
       },
    
    #-----------------------------------------------------------------#

    # The shadow behind the legend.
    "shadow" : \
       
       {# Set the supported data types.
        "dtypes" : (bool,),
        # Set a help string.
        "help" : "Whether to draw a shadow behind the legend.",
       },

    #-----------------------------------------------------------------#

    # The pad between the axes and the legend's border.
    "borderaxespad" : \
    
       {# Set the supported data types.
        "dtypes" : (int, float),
        # Set a help string.
        "help" : \
            "The pad between the axes and the legend's border in " \
            "font-size units. It is set to 0.5 by default.",
       },

    #-----------------------------------------------------------------#

    # The legend title.
    "title" : \
        
       {# Set the supported data types.
        "dtypes" : (str,),
        # Set a help string.
        "help" : "The legend title.",
       },
    
    #-----------------------------------------------------------------#
    
    # The font properties of the legend title.
    "title_fontproperties" : FONT_PROPERTIES_OPTIONS,

    #-----------------------------------------------------------------#

    # The number of columns in the legend.
    "ncols" : \
    
       {# Set the supported data types.
        "dtypes" : (int,),
        # Set a help string.
        "help" : \
            "The number of columns in the legend. By default, this " \
            "is set to 1.",
       },

    #-----------------------------------------------------------------#

    # The space between columns.
    "columnspacing" : \
        
       {# Set the supported data types.
        "dtypes" : (int, float),
        # Set a help string.
        "help" : "The space between columns in font-size units.",
       },

    #-----------------------------------------------------------------#

    # The relative size of legend markers compared to the
    # originally drawn ones.
    "markerscale" : \
        
       {# Set the supported data types.
        "dtypes" : (int, float),
        # Set a help string.
        "help" : \
            "The relative size of the legend markers with respect " \
            "to the markers on the plot. By default, it is set to " \
            "1.0.",
       },
    
    #-----------------------------------------------------------------#
    
    # Whether the legend marker is placed to the left of the legend
    # label.
    "markerfirst" : \
        
       {# Set the supported data types.
        "dtypes" : (bool,),
        # Set a help string.
        "help" : \
            "Whether, for each legend entry, the legend marker is " \
            "placed to the left of the legend label instead of to " \
            "the right of it. It is 'True' by default. ",
       },
    
    #-----------------------------------------------------------------#

    # Whether the legend labels are displayed in reverse order
    # with respect to the input.
    "reverse" : \
    
       {# Set the supported data types.
        "dtypes" : (bool,),
        # Set a help string.
        "help" : \
            "Whether the legend labels are displayed in reverse " \
            "order with respect to the input. It is 'False' by " \
            "default.",
       },
    
    #-----------------------------------------------------------------#

    # The vertical space between the legend entries, in font-size
    # units.
    "labelspacing" : \
        
       {# Set the supported data types.
        "dtypes" : (int, float),
        # Set a help string.
        "help" : \
            "The vertical space between the legend entries in " \
            "font-size units. It is set to 0.5 by default.",
       },
    
    #-----------------------------------------------------------------#

    # The font properties of the legend labels.
    "prop" : FONT_PROPERTIES_OPTIONS,

    #-----------------------------------------------------------------#

    # The alignment of the legend title and entries.
    "alignment": \
        
       {# Set the supported data types.
        "dtypes" : (str,),
        # Set a help string.
        "help" : \
            "The alignment of the legend title and entries. It can " \
            "be either 'center', 'left', or 'right'. It is set to " \
            "'center' by default.",
       },
    
    #-----------------------------------------------------------------#

    # The fractional whitespace inside the legend border in
    # font-size units.
    "borderpad" : \
        
       {# Set the supported data types.
        "dtypes" : (int, float),
        # Set a help string.
        "help" : \
            "The fractional whitespace inside the legend border in " \
            "font-size units. It is set to 0.4 by default.",
       },
    
    #-----------------------------------------------------------------#

    # The length of the legend handles in font-size units.
    "handlelength" : \
        
       {# Set the supported data types.
        "dtypes" : (int, float),
        # Set a help string.
        "help" : \
            "The length of the legend handles in font-size units. " \
            "It is set to 2.0 by default.",
       },
    
    #-----------------------------------------------------------------#
    
    # The height of the legend handles in font-size units.
    "handleheight" : \
        
       {# Set the supported data types.
        "dtypes" : (int, float),
        # Set a help string.
        "help" : \
            "The height of the legend handles in font-size units. " \
            "It is set to 0.7 by default.",
       },
    
    #-----------------------------------------------------------------#

    # The pad between each legend handle and corresponding text in
    # font-size units.
    "handletextpad" : \
        
       {# Set the supported data types.
        "dtypes" : (int, float),
        # Set a help string.
        "help" : \
            "The pad between each legend handle and corresponding " \
            "text in font-size units. It is set to 0.8 by default.",
       },
    
    #-----------------------------------------------------------------#

    # Set the text options for the legend title.
    "title_text_properties" : TEXT_OPTIONS,

    #-----------------------------------------------------------------#

    # Set the text options for the legend labels.
    "label_text_properties" : TEXT_OPTIONS,

    #-----------------------------------------------------------------#

    }


#######################################################################


# Set the supported options for plots' titles.
TITLE_OPTIONS = {

    # The title.
    "label" : \
        
       {# Set the supported data types.
        "dtypes" : (str,),
        # Set a help string.
        "help" : "The title.",
       },
    
    #-----------------------------------------------------------------#

    # The position of the title.
    "loc" : \
        
       {# Set the supported data types.
        "dtypes" : (str,),
        # Set a help string.
        "help" : \
            "The position of the title. It can be either 'left', " \
            "'right', or 'center'.",
       },
    
    #-----------------------------------------------------------------#

    # The offset of the title from the top of the plot area in points.
    "pad" : \
        
       {# Set the supported data types.
        "dtypes" : (int, float),
        # Set a help string.
        "help" : \
            "The offset of the title from the top of the plot area " \
            "in points.",
       },

    #-----------------------------------------------------------------#

    # Font properties options.
    "fontproperties" : FONT_PROPERTIES_OPTIONS,
    
    #-----------------------------------------------------------------#
    
    # Other options for the title's text.
    **TEXT_OPTIONS,

    #-----------------------------------------------------------------#

    }


#######################################################################


# Set the supported options for the x-axis.
X_AXIS_OPTIONS = {

    # The options for the axis' spine.
    "spine" : \

       {k : {"dtypes" : v["dtypes"],
             "help" : v["help"].format("spine")} \
        for k, v in LINE_OPTIONS.items() \
        if k in ["linewidth", "linestyle", "solid_capstyle",
                 "dash_capstyle", "dash_joinstyle", "color",
                 "gapcolor", "alpha"]},

    #-----------------------------------------------------------------#

    # The options for the axis' label.
    "label" : {

        # The label text for the x-axis.
        "xlabel" : \
            
           {# Set the supported data types.
            "dtypes" : (str,),
            # Set a help string.
            "help" : "The label text for the x-axis.",
           },

       # The padding around the label.
        "labelpad" : \
           
           {# Set the supported data types.
            "dtypes" : (int, float),
            # Set a help string.
            "help" : "The padding around the label.",
           },

        # The alignment of the label text.
        "loc" : \
           
           {# Set the supported data types.
            "dtypes" : (str,),
            # Set a help string.
            "help" : \
                "The alignment of the label text. It can be 'left', " \
                "'right', or 'center'.",
           },

        # Font properties options.
        "fontproperties" : FONT_PROPERTIES_OPTIONS,
        
        # Other supported options.
        **{k : v for k, v in TEXT_OPTIONS.items() \
           if k != "horizontalalignment"},

        },

    #-----------------------------------------------------------------#

    # The options for the x-axis' ticks' labels.
    "ticklabels" : {

        # The format for the x-axis' ticks' labels.
        "fmt" : \
        
           {# Set the supported data types.
            "dtypes" : (str,),
            # Set a help string.
            "help" : \
                "The format for the x-axis' ticks' labels.",
           },

        # Other options for the x-axis' ticks' labels.
        "options" : {
        
            # The list of labels for the x-ticks.
            "labels" : \
            
               {# Set the supported data types.
                "dtypes" : (list,),
                # Set a help string.
                "help" : "The list of labels for the x-axis' ticks.",
               },

            # Font properties options.
            "fontproperties" : FONT_PROPERTIES_OPTIONS,
            
            # Other options for the labels' text.
            **TEXT_OPTIONS,
            
            },

        },

    #-----------------------------------------------------------------#

    # The options for the interval displayed on the axis.
    "interval" : INTERVAL_OPTIONS,

    #-----------------------------------------------------------------#

    }

#######################################################################


# Set the supported options for the y-axis.
Y_AXIS_OPTIONS = {

    # The options for the axis' spine.
    "spine" : \

       {k : {"dtypes" : v["dtypes"],
             "help" : v["help"].format("spine")} \
        for k, v in LINE_OPTIONS.items() \
        if k in ["linewidth", "linestyle", "solid_capstyle",
                 "dash_capstyle", "dash_joinstyle", "color",
                 "gapcolor", "alpha"]},

    #-----------------------------------------------------------------#


    # The options for the axis' label.
    "label" : {
        
        # The label text for the y-axis.
        "ylabel" : \

            {# Set the supported data types.
            "dtypes" : (str,),
            # Set a help string.
            "help" : "The label text for the y-axis.",
            }, 
        
        # The padding around the label.
        "labelpad" : \

           {# Set the supported data types.
            "dtypes" : (int, float),
            # Set a help string.
            "help" : "The padding around the label.",
           },

        # The alignment of the label text.
        "loc" : \

           {# Set the supported data types.
            "dtypes" : (str,),
            # Set a help string.
            "help" : \
                "The alignment of the label text. It can be " \
                "'bottom', 'top', or 'center'.",
           },

        # Font properties options.
        "fontproperties" : FONT_PROPERTIES_OPTIONS,
        
        # Other supported options.
        **{k : v for k, v in TEXT_OPTIONS.items() \
           if k != "verticalalignment"},

        },

    #-----------------------------------------------------------------#

    # The options for the y-axis ticks' labels.
    "ticklabels" : {

        # The format for the y-axis' ticks' labels.
        "fmt" : \
        
           {# Set the supported data types.
            "dtypes" : (str,),
            # Set a help string.
            "help" : \
                "The format for the y-axis' ticks' labels.",
           },

        # Other options for the y-axis' ticks' labels.
        "options" : {
        
            # The list of labels for the y-ticks.
            "labels" : \
            
               {# Set the supported data types.
                "dtypes" : (list,),
                # Set a help string.
                "help" : "The list of labels for the y-axis' ticks.",
               },

            # Font properties options.
            "fontproperties" : FONT_PROPERTIES_OPTIONS,
            
            # Other options for the labels' text.
            **TEXT_OPTIONS,
            
            },

        },
    
    #-----------------------------------------------------------------#

    # The options for the interval displayed on the axis.
    "interval" : INTERVAL_OPTIONS,

    #-----------------------------------------------------------------#

    }


#######################################################################


# Set the supported options for the color bar's axis.
COLORBAR_AXIS_OPTIONS = {

    # The options for the axis' label.
    "label" : {
        
        # The label for the color bar's axis.
        "label" : \
        
           {# Set the supported data types.
            "dtypes" : (str,),
            # Set a help string.
            "help" : "The label for the color bar's axis.",
           },
        

       # The padding around the label.
        "labelpad" : \
        
           {# Set the supported data types.
            "dtypes" : (int, float),
            # Set a help string.
            "help" : "The padding around the label.",
           },
        

        # The alignment of the label text.
        "loc" : \

           {# Set the supported data types.
            "dtypes" : (str,),
            # Set a help string.
            "help" : "The alignment of the label text.",
           },
        
        # Font properties options.
        "fontproperties" : FONT_PROPERTIES_OPTIONS,

        # Other supported options.
        **TEXT_OPTIONS,

        },

    #-----------------------------------------------------------------#

    # The options for the color bar's axis' ticks' labels.
    "ticklabels" : {

        # The format for the color bar's axis' ticks' labels.
        "fmt" : \
        
           {# Set the supported data types.
            "dtypes" : (str,),
            # Set a help string.
            "help" : \
                "The format for the color bar's' ticks' labels.",
           },
        
        #-------------------------------------------------------------#

        # Other options for the color bar's axis' ticks' labels.
        "options" : {
        
            # The list of labels for the color bar's axis' ticks.
            "labels" : \
            
               {# Set the supported data types.
                "dtypes" : (list,),
                # Set a help string.
                "help" : \
                    "The list of labels for the color bar's' axis' " \
                    "ticks.",
               },

            # Font properties options.
            "fontproperties" : FONT_PROPERTIES_OPTIONS,
            
            # Other options for the labels' text.
            **TEXT_OPTIONS,
            
            },

        },
    
    #-----------------------------------------------------------------#

    # The options for the interval displayed on the axis.
    "interval" : INTERVAL_OPTIONS,

    #-----------------------------------------------------------------#

    }


#######################################################################


# Set the supported options for drawing a vertical line on a plot.
VLINE_OPTIONS = {

    # The x-coordinate of the vertical line.
    "x" : \
        
       {# Set the supported data types.
        "dtypes" : (int, float),
        # Set a help string.
        "help" : "The x-coordinate of the vertical line.",
       },
    
    #-----------------------------------------------------------------#

    # The starting point of the vertical line.
    "ymin" : \
        
       {# Set the supported data types.
        "dtypes" : (int, float),
        # Set a help string.
        "help" : \
            "The starting point of the vertical line. It should be " \
            "between 0 and 1, with 0 being the bottom of the plot " \
            "and 1 the top.",
       },
    
    #-----------------------------------------------------------------#

    # The ending point of the vertical line.
    "ymax" : \
        
        {# Set the supported data types.
        "dtypes" : (int, float),
        # Set a help string.
        "help" : \
            "The ending point of the vertical line. It should be " \
            "between 0 and 1, with 0 being the bottom of the plot " \
            "and 1 the top.",
        },

    #-----------------------------------------------------------------#

    # Additional options for the line.
    **{k : {"dtypes" : v["dtypes"],
             "help" : v["help"].format("vertical line")} \
        for k, v in LINE_OPTIONS.items() \
        if k in ["linewidth", "linestyle", "solid_capstyle",
                 "dash_capstyle", "dash_joinstyle", "color",
                 "gapcolor", "alpha"]},

    #-----------------------------------------------------------------#

    }


#######################################################################


# Set the supported options for drawing a horizontal line on a plot.
HLINE_OPTIONS = {

    # The y-coordinate of the horizontal line.
    "y" : \
        
       {# Set the supported data types.
        "dtypes" : (int, float),
        # Set a help string.
        "help" : "The y-coordinate of the horizontal line.",
       },
    
    #-----------------------------------------------------------------#

    # The starting point of the horizontal line.
    "xmin" : \
        
       {# Set the supported data types.
        "dtypes" : (int, float),
        # Set a help string.
        "help" : \
            "The starting point of the horizontal line. It should " \
            "be between 0 and 1, with 0 being the leftmost part of " \
            "the plot and 1 the rightmost.",
       },
    
    #-----------------------------------------------------------------#

    # The ending point of the horizontal line.
    "ymax" : \
        
       {# Set the supported data types.
        "dtypes" : (int, float),
        # Set a help string.
        "help" : \
            "The ending point of the horizontal line. It should be " \
            "between 0 and 1, with 0 being the leftmost part of the " \
            "plot and 1 the rightmost.",
        },

    #-----------------------------------------------------------------#

    # Additional options for the line.
    **{k : {"dtypes" : v["dtypes"],
             "help" : v["help"].format("horizontal line")} \
        for k, v in LINE_OPTIONS.items() \
        if k in ["linewidth", "linestyle", "solid_capstyle",
                 "dash_capstyle", "dash_joinstyle", "color",
                 "gapcolor", "alpha"]},

    #-----------------------------------------------------------------#

    }


#######################################################################


# Set the supported options for a colorbar.
COLORBAR_OPTIONS = {

    # The color map used for the color bar.
    "cmap" : \
    
       {# Set the supported data types.
        "dtypes" : (str,),
        # Set a help string.
        "help" : "The color map used for the color bar.",
       },
    
    #-----------------------------------------------------------------#

    # Other options for the color bar.
    "options" : {

        # The location of the color bar with respect to the plot area
        # where the color bar is created.
        "location" : \
        
           {# Set the supported data types.
            "dtypes" : (str,),
            # Set a help string.
            "help" : \
                "The location of the color bar with respect to the " \
                "plot area where the color bar is created. It can " \
                "be 'left', 'right', 'top', or 'bottom'. If the " \
                "location is not set, it will be determined by the " \
                "color bar's 'orientation' if set or default to " \
                "'right' if the color bar's orientation is not " \
                "specified.",
           },
        
        #-------------------------------------------------------------#

        # The orientation of the color bar.
        "orientation" : \
        
           {# Set the supported data types.
            "dtypes" : (str,),
            # Set a help string.
            "help" : \
                "The orientation of the color bar. It can be " \
                "'vertical' or 'horizontal'. If the orientation is " \
                "not set, it will be determined by the color bar's " \
                "'location' if set or default to 'vertical' if the " \
                "color bar's location is not specified.",
           },

        #-------------------------------------------------------------#
        
        # The fraction of the plot's area used for the color bar.
        "fraction" : \
        
           {# Set the supported data types.
            "dtypes" : (int, float),
            # Set a help string.
            "help" : \
                "The fraction of the plot's area used for the color " \
                "bar.",
           },
        
        #-------------------------------------------------------------#

        # The fraction by which to multiply the size of the color bar.
        "shrink" : \
            
           {# Set the supported data types.
            "dtypes" : (int, float),
            # Set a help string.
            "help" : \
                "The fraction by which to multiply the size of the " \
                "color bar. It is set to '1.0' by default.",
           },
        
        #-------------------------------------------------------------#

        # The ratio between the long and short dimensions of the color
        # bar.
        "aspect" : \
        
           {# Set the supported data types.
            "dtypes" : (int, float),
            # Set a help string.
            "help" : \
                "The ratio between the long and short dimensions of " \
                "the color bar. It is set to '20' by default.",
           },
        
        #-------------------------------------------------------------#

        # The fraction of the original plot area between the color bar
        # and the plot.
        "pad" : \
        
           {# Set the supported data types.
            "dtypes" : (int, float),
            # Set a help string.
            "help" : \
                "The fraction of the original plot area between the " \
                "color bar and the plot. It is set to '0.05' by " \
                "default if the color bar is vertical and to '0.15' " \
                "if horizontal.",
           },
            
        #-------------------------------------------------------------#

        # Extend the color bar for out-of-range values.
        "extend" : \
        
           {# Set the supported data types.
            "dtypes" : (str,),
            # Set a help string.
            "help" : \
                "Extend the color bar for out-of-range values. " \
                "'neither' means no extensions, 'both' means both " \
                "ends are extended, and 'min' and 'max' mean that " \
                "only the lower or upper end of the color bar is " \
                "extended.",
           },
        
        #-------------------------------------------------------------#

        # The length of the extensions.
        "extendfrac" : \
        
           {# Set the supported data types.
            "dtypes" : (int, float),
            # Set a help string.
            "help" : \
                "The length of the extensions. If 'auto', the " \
                "length of the color bar's extensions is " \
                "automatically determined. If 'float', it is the " \
                "length of both extensions as a fraction of the " \
                "length of the color bar's interior. If a 'tuple' " \
                "of two 'float', it contains the length of the " \
                "lower and upper extension as a fraction of the " \
                "length of the color bar's interior.",
           },
        
        #-------------------------------------------------------------#

        # Whether the color bar's extensions are rectangular (as
        # opposed to the default triangular ones).
        "extendrect" : \
        
           {# Set the supported data types.
            "dtypes" : (bool,),
            # Set a help string.
            "help" : \
                "Whether the color bar's extensions are rectangular " \
                "(as opposed to the default triangular ones).",
           },
        
        #-------------------------------------------------------------#

        # A format string representing the format of the ticks' labels.
        "format" : \
        
           {# Set the supported data types.
            "dtypes" : (str,),
            # Set a help string.
            "help" : \
                "A format string representing the format of the " \
                "ticks' labels.",
           },

        },
        
    #-----------------------------------------------------------------#

    # Options for the color bar's axis.
    **COLORBAR_AXIS_OPTIONS,

    #-----------------------------------------------------------------#

    } 


#######################################################################


# Set the supported options for histograms.
HISTOGRAM_OPTIONS = {

    # The number of bins in the histogram.
    "num_bins" : \
    
       {# Set the supported data types.
        "dtypes" : (int,),
        # Set a help string.
        "help" : "The number of bins in the histogram.",
       },
    
    #-----------------------------------------------------------------#
    
    # Whether to draw a probability density function.
    "density" : \
    
       {# Set the supported data types.
        "dtypes" : (bool,),
        # Set a help string.
        "help" : \
            "Whether to draw a probability density function. It is " \
            "'False' by default.",
       },

    #-----------------------------------------------------------------#
    
    # Other options.
    **{k : {"dtypes" : v["dtypes"], 
            "help" : v["help"].format("histogram bins")} \
        for k, v in PATCH_OPTIONS.items() \
        if k in ["color", "alpha", "fill", "edgecolor", "linewidth",
                 "linestyle", "capstyle", "joinstyle"]},
    
    #-----------------------------------------------------------------#

    }


#######################################################################


# Set the supported options for bar plots.
BARPLOT_OPTIONS = {

    # The palette.
    "palette" : \

       {# Set the supported data types.
        "dtypes" : (str, list),
        # Set a help string.
        "help" : \
            "The palette used to color the bars. It can be " \
            "either the name of a color map or a list of colors. " \
            "More details about how colors can be specified can be " \
            f"found at: {LINKS['colors']}.",
        },

    #-----------------------------------------------------------------#

    # The proportion of the original colors' saturation used when
    # coloring the bars. 
    "saturation" : \
    
       {# Set the supported data types.
        "dtypes" : (int, float),
        # Set a help string.
        "help" : \
            "The proportion of the original colors' saturation " \
            "used when coloring the bars.",
       },
    
    #-----------------------------------------------------------------#
    
    # Other options.
    **{k : {"dtypes" : v["dtypes"], 
            "help" : v["help"].format("bars")} \
        for k, v in PATCH_OPTIONS.items() \
        if k in ["fill", "edgecolor", "alpha", "linewidth",
                 "linestyle", "capstyle", "joinstyle"]},

    }


#######################################################################


# Set the supported options for line plots.
LINEPLOT_OPTIONS = {

    # The palette.
    "palette" : \

       {# Set the supported data types.
        "dtypes" : (str, list),
        # Set a help string.
        "help" : \
            "The palette used to color the lines. It can be " \
            "either the name of a color map or a list of colors. " \
            "More details about how colors can be specified can be " \
            f"found at: {LINKS['colors']}.",
        },

    #-----------------------------------------------------------------#

    # Additional options for the lines.
    **{k : {"dtypes" : v["dtypes"],
            "help" : v["help"].format("lines")} \
        for k, v in LINE_OPTIONS.items()},

    #-----------------------------------------------------------------#

    }


#######################################################################


# Set the supported options for scatter plots.
SCATTERPLOT_OPTIONS = {

    # The palette used to color the markers.
    "palette" : \

       {# Set the supported data types.
        "dtypes" : (str, list),
        # Set a help string.
        "help" : \
            "The palette used to color the markers. It can be " \
            "either the name of a color map or a list of colors. " \
            "More details about how colors can be specified can be " \
            f"found at: {LINKS['colors']}.",
        },

    #-----------------------------------------------------------------#

    # The style of the markers.
    "marker" : \

       {# Set the supported data types.
        "dtypes" : (str,),
        # Set a help string.
        "help" : \
            "The style of the markers. All available marker styles " \
            f"can be found at: {LINKS['markers']}.",
       },
    
    #-----------------------------------------------------------------#

    # The size of the markers in points**2.
    "s" : \
    
       {# Set the supported data types.
        "dtypes" : (int, float),
        # Set a help string.
        "help" : \
            "The marker size in points**2. Typographic points " \
            "are 1/72 inches.",
       },
    
    #-----------------------------------------------------------------#

    # Other options.
    **{k : {"dtypes" : v["dtypes"], 
            "help" : v["help"].format("markers")} \
        for k, v in COLLECTION_OPTIONS.items() \
        if k in ["edgecolors", "alpha", "linewidths", "linestyles",
                 "capstyle", "joinstyle"]},
    
    #-----------------------------------------------------------------#

    }


#######################################################################


# Set the supported options for boxplots.
BOXPLOT_OPTIONS = {

    # The color of the boxes.
    "color" : \
        
       {# Set the supported data types.
        "dtypes" : (str, tuple),
        # Set a help string.
        "help" : \
            "The color of the boxes. More details about how colors " \
            f"can be specified can be found at: {LINKS['colors']}.",
       },
    
    #-----------------------------------------------------------------#

    # The palette used to color the boxes.
    "palette" : \

       {# Set the supported data types.
        "dtypes" : (str, list),
        # Set a help string.
        "help" : \
            "The palette used to color the boxes. It can be " \
            "either the name of a color map or a list of colors. " \
            "More details about how colors can be specified can be " \
            f"found at: {LINKS['colors']}.",
       },
    
    #-----------------------------------------------------------------#

    # The proportion of the original colors' saturation used when
    # coloring the boxes. 
    "saturation" : \
    
       {# Set the supported data types.
        "dtypes" : (int, float),
        # Set a help string.
        "help" : \
            "The proportion of the original colors' saturation " \
            "used when coloring the boxes.",
       },
    
    #-----------------------------------------------------------------#
    
    # Whether to draw solid boxes.
    "fill" : \
    
       {# Set the supported data types.
        "dtypes" : (bool,),
        # Set a help string.
        "help" : \
            "Whether to draw solid boxes. If 'False', only the " \
            "boxes' edges will be drawn. It is 'True' by default.",
       },

    #-----------------------------------------------------------------#

    # Whether to draw notched boxes instead of rectangular ones.
    "notch" : \
    
       {# Set the supported data types.
        "dtypes" : (bool,),
        # Set a help string.
        "help" : \
            "Whether to draw notched boxes instead of rectangular " \
            "ones. The notches represent the confidence interval " \
            "(CI) around the median.",
       },
    
    #-----------------------------------------------------------------#

    # Whether to bootstrap when calculating the confidence interval.
    "bootstrap" : \
    
       {# Set the supported data types.
        "dtypes" : (int,),
        # Set a help string.
        "help" : \
            "If passed, 'bootstrap' specifies the number of times " \
            "to bootstrap the median to determine its 95% " \
            "confidence interval for notched boxes. If not passed, " \
            "the median is not bootstrapped, and the notches are " \
            "calculated using a Gaussian-based asymptotic " \
            "approximation.",
       },
    
    #-----------------------------------------------------------------#
    
    # The position of the whiskers.
    "whis" : \
    
       {# Set the supported data types.
        "dtypes" : (float, tuple),
        # Set a help string.
        "help" : \
            "The position of the whiskers. If 'float', the lower " \
            "whisker is at the lowest data point above " \
            "'Q1 - whis*(Q3-Q1)', and the higher one is at the " \
            "highest data point below 'Q3 + whis*(Q3-Q1)', where " \
            "'Q1' and 'Q3' are the first and third quantiles, " \
            "respectively. If a 'tuple' of 'float', they indicate " \
            "the percentiles at which to draw the whiskers. It is " \
            "set to 1.5 by default.",
       },
    
    #-----------------------------------------------------------------#

    # The width of the boxes.
    "widths" : \
    
       {# Set the supported data types.
        "dtypes" : (int, float),
        # Set a help string.
        "help" : \
            "The width of the boxes. It is set to '0.5 by default.",
       },
    
    #-----------------------------------------------------------------#

    # Whether to show the boxes.
    "showbox" : \
    
       {# Set the supported data types.
        "dtypes" : (bool,),
        # Set a help string.
        "help" : \
            "Whether to show the boxes. It is 'True' by default.",
       },

    #-----------------------------------------------------------------#

    # Whether to show the arithmetic means.
    "showmeans" : \
    
       {# Set the supported data types.
        "dtypes" : (bool,),
        # Set a help string.
        "help" : \
            "Whether to show the arithmetic means. It is 'False' " \
            "by default.",
       },
    
    #-----------------------------------------------------------------#

    # Whether to show the caps at the end of the whiskers.
    "showcaps" : \
    
       {# Set the supported data types.
        "dtypes" : (bool,),
        # Set a help string.
        "help" : \
            "Whether to show the caps at the end of the whiskers. " \
            "It is 'True' by default.",
       },
    
    #-----------------------------------------------------------------#

    # Whether to show the fliers (the outliers beyond the caps).
    "showfliers" : \
    
       {# Set the supported data types.
        "dtypes" : (bool,),
        # Set a help string.
        "help" : \
            "Whether to show the fliers (the outliers beyond the " \
            "caps). It is 'True' by default.",
       },

    #-----------------------------------------------------------------#

    # Whether to render the means as lines spanning the entire width
    # of the boxes.
    "meanline" : \
    
       {# Set the supported data types.
        "dtypes" : (bool,),
        # Set a help string.
        "help" : \
            "Whether to render the means as lines spanning the " \
            "entire width of the boxes. It is 'False' by default.",
       },
    
    #-----------------------------------------------------------------#

    # The options for the boxes.
    "boxprops" : \
        
       {k : {"dtypes" : v["dtypes"], 
              "help" : v["help"].format("boxes")} \
        for k, v in PATCH_OPTIONS.items() \
        if k in ["facecolor", "edgecolor", "linewidth", "hatch"]},

    #-----------------------------------------------------------------#

    # The options for the medians.
    "medianprops" : \
    
       {k : {"dtypes" : v["dtypes"],
             "help" : v["help"].format(\
                "lines representing the medians")} \
        for k, v in LINE_OPTIONS.items() \
        if k in ["linewidth", "linestyle", "solid_capstyle",
                 "dash_capstyle", "dash_joinstyle", "color",
                 "gapcolor", "alpha"]},
    
    #-----------------------------------------------------------------#
    
    # The options for the means, if 'meanline' is set to 'True'.
    "meanprops_line" : \
        
       {k : {"dtypes" : v["dtypes"],
             "help" : v["help"].format(\
                "lines representing the means")} \
        for k, v in LINE_OPTIONS.items() \
        if k in ["linewidth", "linestyle", "solid_capstyle",
                 "dash_capstyle", "dash_joinstyle", "color",
                 "gapcolor", "alpha"]},
    
    #-----------------------------------------------------------------#

    # The options for the means, if 'meanline' is set to 'False'.
    "meanprops_marker" : \
    
       {k : {"dtypes" : v["dtypes"],
             "help" : v["help"].format(\
                "markers representing the means")} \
        for k, v in LINE_OPTIONS.items() \
        if k in ["marker", "markerevery", "markersize",
                 "markeredgewidth", "fillstyle", "markeredgecolor",
                 "markerfacecolor", "markerfacecoloralt", "alpha"]},
    
    #-----------------------------------------------------------------#

    # The options for the whiskers.
    "whiskerprops" : \
    
       {k : {"dtypes" : v["dtypes"],
             "help" : v["help"].format(\
                "lines representing the whiskers")} \
        for k, v in LINE_OPTIONS.items() \
        if k in ["linewidth", "linestyle", "solid_capstyle",
                 "dash_capstyle", "dash_joinstyle", "color",
                 "gapcolor", "alpha"]},
    
    #-----------------------------------------------------------------#

    # The options for the caps.
    "capprops" : \
    
       {k : {"dtypes" : v["dtypes"],
             "help" : v["help"].format(\
                "lines representing the caps")} \
        for k, v in LINE_OPTIONS.items() \
        if k in ["linewidth", "linestyle", "solid_capstyle",
                 "dash_capstyle", "dash_joinstyle", "color",
                 "gapcolor", "alpha"]},
    
    #-----------------------------------------------------------------#

    # The options for the fliers.
    "flierprops" : \
    
       {k : {"dtypes" : v["dtypes"],
             "help" : v["help"].format(\
                "lines representing the fliers")} \
        for k, v in LINE_OPTIONS.items() \
        if k in ["marker", "markerevery", "markersize",
                 "markeredgewidth", "fillstyle", "markeredgecolor",
                 "markerfacecolor", "markerfacecoloralt", "alpha"]},
    
    #-----------------------------------------------------------------#

    }


#######################################################################


# Set the supported options for violin plots.
VIOLINPLOT_OPTIONS = {

    # The color of the violins.
    "color" : \

       {# Set the supported data types.
        "dtypes" : (str, tuple),
        # Set a help string.
        "help" : \
            "The color of the violins. More details about " \
            "how colors can be specified can be found at: " \
            f"{LINKS['colors']}.",
       },
    
    #-----------------------------------------------------------------#

    # The colors to use either for each side of a split violin
    # or for each violin in a group.
    "palette" : \

       {# Set the supported data types.
        "dtypes" : (str, list),
        # Set a help string.
        "help" : \
            "The colors for each side of a split violin or each " \
            "violin in a group. It can be either the name of a " \
            "color map or a list of colors. More details about how " \
            "colors can be specified can be found at: " \
            f"{LINKS['colors']}.",
       },
    
    #-----------------------------------------------------------------#
    
    # The proportion of the original saturation to draw fill colors
    # in.
    "saturation" : \
        
       {# Set the supported data types.
        "dtypes" : (float),
        # Set a help string.
        "help" : \
            "The proportion of the original colors' saturation used " \
            "when coloring the violins.",
       },
    
    #-----------------------------------------------------------------#
    
    # Whether to draw a filled violin.
    "fill" : \
        
       {# Set the supported data types.
        "dtypes" : (bool,),
        # Set a help string.
        "help" : \
            "Whether to draw a filled violin. If 'False', only the " \
            "violin's edges will be drawn.",
       },
    
    #-----------------------------------------------------------------#
    
    # The representation of the data in the violin's interior.
    "inner" : \
        
       {# Set the supported data types.
        "dtypes" : (str, None.__class__),
        # Set a help string.
        "help" : \
            "The data representation in the violin's interior. " \
            "It can be either 'box' (draws a miniature boxplot), " \
            "'quartile' (draws the quartiles of the distribution), " \
            "'point' (draws the individual data points), or 'stick' " \
            "(draws the individual data points as sticks). If " 
            "'None', no representation is drawn.",
       },
    
    #-----------------------------------------------------------------#
    
    # The width allotted to each violin on the x-axis.
    "width" : \
        
       {# Set the supported data types.
        "dtypes" : (int, float),
        # Set a help string.
        "help" : "The width allotted to each violin on the x-axis.",
       },
    
    #-----------------------------------------------------------------#

    # Shrink the violins on the x-axis by this factor to add a gap.
    "gap" : \
        
       {# Set the supported data types.
        "dtypes" : (int, float),
        # Set a help string.
        "help" : \
            "Shrink the violins on the x-axis by this factor to " \
            "add a gap between them.",
       },

    #-----------------------------------------------------------------#
    
    # The distance, in units of bandwidth, to extend the density
    # past the extreme data points.
    "cut" : \
        
       {# Set the supported data types.
        "dtypes" : (int, float),
        # Set a help string.
        "help" : \
            "The distance, in units of bandwidth, to extend the " \
            "density past the extreme data points. Set it to 0 to " 
            "limit the violin's range within the range of the " \
            "observed data.",
       },

    #-----------------------------------------------------------------#

    # The number of points in the discrete grid used to evaluate
    # the KDE.
    "gridsize" : \
    
       {# Set the supported data types.
        "dtypes" : (int,),
        # Set a help string.
        "help" : \
            "The number of points in the discrete grid used to " \
            "evaluate the KDE.",
       },
    
    #-----------------------------------------------------------------#
    
    # Either the name of a reference rule or the scale factor to
    # use when computing the kernel bandwidth.
    "bw_method" : \
        
       {# Set the supported data types.
        "dtypes" : (str, int, float),
        # Set a help string.
        "help" : \
            "Either the name of a reference rule ('scott' or " \
            "'silverman') the scale factor to use when computing " \
            "the kernel bandwidth.",
       },
    
    #-----------------------------------------------------------------#
    
    # The factor that scales the bandwidth to use more or less
    # smoothing.
    "bw_adjust" : \
        
       {# Set the supported data types.
        "dtypes" : (int, float),
        # Set a help string.
        "help" : \
            "The factor that scales the bandwidth to use more " \
            "or less smoothing.",
       },
    
    #-----------------------------------------------------------------#

    # The method that normalizes each density to determine the
    # violins' width.
    "density_norm" : \
        
       {# Set the supported data types.
        "dtypes" : (str,),
        # Set a help string.
        "help" : \
            "The method that normalizes each density to determine " \
            "the violins' width. If 'area', each violin will have " \
            "the same area. If 'count', the width will be " \
            "proportional to the number of observations. If " \
            "'width', each violin will have the same width.",
        },
    
    #-----------------------------------------------------------------#
    
    # Whether to normalize the density across all violins.
    "common_norm" : \
        
       {# Set the supported data types.
        "dtypes" : (bool,),
        # Set a help string.
        "help" : \
            "Whether to normalize the density across all violins. " \
            "It is 'False' by default.",
       },
    
    #-----------------------------------------------------------------#

    # Other options.
    **{k : {"dtypes" : v["dtypes"], 
            "help" : v["help"].format("violins")} \
        for k, v in COLLECTION_OPTIONS.items() \
        if k in ["edgecolors", "linewidths", "linestyles",
                 "capstyle", "joinstyle"]},
    
    #-----------------------------------------------------------------#

    }


#######################################################################


# Set the supported options for strip plots.
STRIPPLOT_OPTIONS = {

    # The color of the points.
    "color" : \

       {# Set the supported data types.
        "dtypes" : (str, tuple),
        # Set a help string.
        "help" : \
            "The color of the points. More details about " \
            "how colors can be specified can be found at: " \
            f"{LINKS['colors']}.",
       },
    
    #-----------------------------------------------------------------#

    # The colors to use either for each side of a split violin
    # or for each violin in a group.
    "palette" : \

       {# Set the supported data types.
        "dtypes" : (str, list),
        # Set a help string.
        "help" : \
            "The colors for each side of a split violin or each " \
            "violin in a group. It can be either the name of a " \
            "color map or a list of colors. More details about how " \
            "colors can be specified can be found at: " \
            f"{LINKS['colors']}.",
       },
    
    #-----------------------------------------------------------------#
    
    # The width allotted to each violin on the x-axis.
    "width" : \
        
       {# Set the supported data types.
        "dtypes" : (int, float),
        # Set a help string.
        "help" : "The width allotted to each violin on the x-axis.",
       },
    

    #-----------------------------------------------------------------#
    
    # The width of the violin's edges.
    "linewidth" : \
        
       {# Set the supported data types.
        "dtypes" : (int, float),
        # Set a help string.
        "help" : "The width of the violin's edges in points.",
       },

    #-----------------------------------------------------------------#

    # The color to use for line elements.
    "linecolor" : \
        
       {# Set the supported data types.
        "dtypes" : (str, tuple),
        # Set a help string.
        "help" : \
            "The color for line elements. More details about how " \
            "colors can be specified can be found at: " \
            f"{LINKS['colors']}.",
       },
    

    
    #-----------------------------------------------------------------#

    # Other options.
    **{k : {"dtypes" : v["dtypes"], 
            "help" : v["help"].format("markers")} \
        for k, v in COLLECTION_OPTIONS.items() \
        if k in ["linewidth", "edgecolor", "linestyle",
                 "capstyle", "joinstyle"]},
    
    #-----------------------------------------------------------------#

    }


#######################################################################


# Set the supported options for the output.
OUTPUT_OPTIONS = {

    # Whether the figure's background is transparent.
    "transparent" : \
    
       {# Set the supported data types.
        "dtypes" : (bool,),
        # Set a help string.
        "help" : \
            "Whether the figure's background is transparent. It is " \
            "'False' by default.",
       },

    #-----------------------------------------------------------------#

    # The DPI (dots per inch) of the output file.
    "dpi" : \
    
       {# Set the supported data types.
        "dtypes" : (int,),
        # Set a help string.
        "help" : "The DPI (dots per inch) of the output file.",
       },

    #-----------------------------------------------------------------#

    # The figure's bounding box.
    "bbox_inches" : \
    
       {# Set the supported data types.
        "dtypes" : (str,),
        # Set a help string.
        "help" : \
            "The figure's bounding box. It can be 'tight' or " \
            "'standard'.",
       },
    
    #-----------------------------------------------------------------#

    # The color of the figure's background.
    "facecolor" : \
        
       {# Set the supported data types.
        "dtypes" : (str, tuple),
        # Set a help string.
        "help" : \
            "The color of the figure's background. More details " \
            "about how colors can be specified can be found at: " \
            f"{LINKS['colors']}.",
       },
    
    #-----------------------------------------------------------------#

    # The color of the edges of the figure.
    "edgecolor" : \

       {# Set the supported data types.
        "dtypes" : (str, tuple),
        # Set a help string.
        "help" : \
            "The color of the edges of the figure. More details " \
            "about how colors can be specified can be found at: " \
            f"{LINKS['colors']}.",
       },
    
    #-----------------------------------------------------------------#

    # The orientation of the figure.
    "orientation" : \
    
       {# Set the supported data types.
        "dtypes" : (str,),
        # Set a help string.
        "help" : \
            "The orientation of the figure. It can be 'landscape' " \
            "or 'portrait'.",
       },
    
    #-----------------------------------------------------------------#
    
    # The paper type for the figure.
    "papertype" : \
    
       {# Set the supported data types.
        "dtypes" : (str,),
        # Set a help string.
        "help" : \
            "The paper type for the figure. It can be 'letter', " \
            "'legal', 'executive', 'ledger', 'a0' through 'a10', " \
            "or 'b0' through 'b10'. It is only supported for " \
            "PostScript outputs.",
       },
    
    #-----------------------------------------------------------------#
    
    # The padding around the figure in inches.
    "pad_inches" : \
    
       {# Set the supported data types.
        "dtypes" : (int, float),
        # Set a help string.
        "help" : "The padding around the figure in inches.",
       },
    
    #-----------------------------------------------------------------#

    # The backend to use.
    "backend" : \
    
       {# Set the supported data types.
        "dtypes" : (str,),
        # Set a help string.
        "help" : \
            "The backend to use. All available backends " \
            f"can be found at: {LINKS['backends']}.",
       },
    
    #-----------------------------------------------------------------#

    }


#######################################################################


# Set a mapping between the sections that we can encounter in a
# configuration and the name of the item they refer to.
SECTIONS2ITEMS = {
    
    # Plot elements.
    "figure" : "figure",
    "title" : "title",
    "xaxis" : "x-axis",
    "yaxis" : "y-axis",
    "yaxis_barplot" : "y-axis of the bar plot",
    "yaxis_violinplot" : "y-axis of the violin plot",
    "yaxis_stripplot" : "y-axis of the strip plot",
    "legend" : "legend",
    "colorbar" : "color bar",
    "vline" : "vertical line",
    "hline" : "horizontal line",
    "hline_barplot" : "horizontal line for the bar plot",
    "hline_violinplot" : "horizontal line for the violin plot",
    "hline_stripplot" : "horizontal line for the strip plot",
    "text" : "text",

    # Plot types.
    "histogram" : "histogram",
    "histogram_2" : "second histogram",
    "lineplot" : "line plot",
    "scatterplot" : "scatter plot",
    "scatterplot_2" : "scatter plot for 'other' groups",
    "boxplot" : "box plot",
    "violinplot" : "violin plot",
    "stripplot" : "strip plot",
    
    # Output options.
    "output" : "output",
    }

