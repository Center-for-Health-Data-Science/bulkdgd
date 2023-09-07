#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-

#    repio.py
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
import pandas as pd


# Get the module's logger
logger = log.getLogger(__name__)


def load_representations(csv_file,
                         sep = ",",
                         split = True):
    """Load the representations from a CSV file.

    Parameters
    ----------
    csv_file : ``str``
        A CSV file containing a data frame with the representations.

        Each row should contain a representation. The columns should
        contain the representation's values along the latent
        space's dimensions and additional informations about
        the representations, if present (for instance, the loss
        associated with each representation).

    sep : ``str``, ``","``
        The column separator in the input CSV file.

    split : ``bool``, ``True``
        Whether to split the input data frame into two data frames,
        one with only the columns containing the representations'
        values along the latent space's dimensions and the other
        containing only the columns with additional information
        about the representations, if any were found.

    Returns
    -------
    df_data : ``pandas.DataFrame``
        A data frame containing the representations'
        values along the latent space's dimensions.

        Here, each row contains a representation and the columns
        contain the representations' values along the latent
        space's dimensions.

        If ``split`` is ``False``, this data frame will
        also contain the columns containing additional
        information about the representations, if any were found.

    df_other_data : ``pandas.DataFrame``
        A data frame containing the additional information about
        the representations found in the input data frame.

        Here, each row contains a representation and the columns
        contain additional information about the representations
        provided in the input data frame.

        If ``split`` is ``False``, only ``df_data`` is returned.
    """


    #---------------------- Load the data frame ----------------------#


    # Load the data frame with the representations
    df = pd.read_csv(csv_file,
                     sep = sep,
                     index_col = 0,
                     header = 0)


    #--------------------- Split the data frame ----------------------#


    # If the user requested splitting the data frame
    if split:

        # Get the names of the columns containing the values of
        # the representations along the latent space's dimensions
        latent_dims_columns = \
            [col for col in df.columns \
             if col.startswith("latent_dim_")]

        # Get the names of the other columns
        other_columns = \
            [col for col in df.columns \
             if col not in latent_dims_columns]

        # If additional columns were found
        if other_columns:

            # Inform the user of the other columns found
            infostr = \
                f"{len(other_columns)} column(s) containing " \
                "additional information (not values of the " \
                "representations along the latent space's " \
                "dimensions ) was (were) found the input data " \
                f"frame: {', '.join(other_columns)}."
            logger.info(infostr)

        # Return a data frame with the representations' values and
        # another with the extra information
        return df[latent_dims_columns], df[other_columns]


    #------------------ Do not split the data frame ------------------#


    # Otherwise
    else:

        # Return the full data frame
        return df


def save_representations(df,
                         csv_file,
                         sep = ","):
    """Save the representations to a CSV file.

    Parameters
    ----------
    df : ``pandas.DataFrame``
        A data frame containing the representations.

    csv_file : ``str``
        The output CSV file.

    sep : ``str``, ``","``
        The column separator in the output CSV file.
    """

    # Save the representations
    df.to_csv(csv_file,
              sep = sep,
              index = True,
              header = True)