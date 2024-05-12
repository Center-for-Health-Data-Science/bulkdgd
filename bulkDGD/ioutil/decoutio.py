#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-

#    decoutio.py
#
#    Utilities to load and save the decoder's outputs.
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
__doc__ = "Utilities to load and save the decoder's outputs."


#######################################################################


# Import from the standard library.
import logging as log
# Import from third-party packages.
import pandas as pd


#######################################################################


# Get the module's logger.
logger = log.getLogger(__name__)


#######################################################################


def load_decoder_outputs(csv_file,
                         sep = ",",
                         split = True):
    """Load the decoder's outputs from a CSV file.

    Parameters
    ----------
    csv_file : ``str``
        A CSV file containing a data frame with the decoder's outputs.

        Each row should represent the decoder's output for a given
        representation, while each column should contain either the
        values of the output or additional information about it.

    sep : ``str``, ``","``
        The column separator in the input CSV file.

    split : ``bool``, ``True``
        Whether to split the input data frame into two data frames,
        one with only the columns containing the decoder's outputs
        and the other containing only the columns with additional
        information, if any were found.

    Returns
    -------    
    df_data : ``pandas.DataFrame``
        A data frame containing the decoder's outputs.

        Here, each row represents the decoder's output for a given
        representation. and the columns contain the values
        of the output.

        If ``split`` is ``False``, this data frame will also contain
        the columns with additional information about the output, if
        any were found.

    df_other_data : ``pandas.DataFrame``
        A data frame containing additional information about the
        decoder's outputs found in the input data frame.

        Here, each row represents the decoder's output for a given
        representations and the columns contain additional
        information provided in the input data frame.

        If ``split`` is ``False``, only ``df_data`` is returned.
    """

    # Load the data frame with the decoder's outputs.
    df = pd.read_csv(csv_file,
                     sep = sep,
                     index_col = 0,
                     header = 0)

    #-----------------------------------------------------------------#

    # Get the names of the columns containing the decoder's outputs
    # for the genes.
    dec_out_columns = \
        [col for col in df.columns if col.startswith("ENSG")]

    # Inform the user about how many columns were found containing
    # the decoder's outputs.
    infostr = \
        f"{len(dec_out_columns)} column(s) containing the " \
        "decoder's outputs was (were) found in the input data frame."
    logger.info(infostr)

    #-----------------------------------------------------------------#

    # If the user requested splitting the data frame
    if split:

        # Get the names of the other columns.
        other_columns = \
            [col for col in df.columns if col not in dec_out_columns]

        # If additional columns were found
        if other_columns:

            # Inform the user of the other columns found.
            infostr = \
                f"{len(other_columns)} column(s) containing " \
                "additional information was (were) found in the " \
                f"input data frame : {', '.join(other_columns)}."
            logger.info(infostr)

        # Return a data frame with the decoder's outputs and another
        # one with the extra information.
        return df[dec_out_columns], df[other_columns]

    #-----------------------------------------------------------------#
    
    # Otherwise
    else:

        # Return the full data frame.
        return df


def save_decoder_outputs(df,
                         csv_file,
                         sep = ","):
    """Save the decoder's outputs to a CSV file.

    Parameters
    ----------
    df : ``pandas.DataFrame``
        A data frame containing the decoder's outputs.

    csv_file : ``str``
        The output CSV file.

    sep : ``str``, ``","``
        The column separator in the output CSV file.
    """

    # Save the decoder's outputs.
    df.to_csv(csv_file,
              sep = sep,
              index = True,
              header = True)
