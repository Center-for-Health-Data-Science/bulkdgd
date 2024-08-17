#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-

#    lossio.py
#
#    Utilities to load and save loss-related data.
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
__doc__ = "Utilities to load and save loss-related data."


#######################################################################


# Import from the standard library.
import logging as log
# Import from the third-party packages.
import pandas as pd


#######################################################################


# Get the module's logger.
logger = log.getLogger(__name__)


#######################################################################


def load_loss(csv_file,
              sep = ","):
    """Load the loss(es) from a CSV file.

    Parameters
    ----------
    csv_file : :class:`str`
        A CSV file containing the loss(es).

    sep : :class:`str`, ``","``
        The column separator in the input CSV file.

    Returns
    -------
    df_time : :class:`pandas.DataFrame`
        A data frame containing the loss(es).
    """

    # Load the data frame.
    df_loss = pd.read_csv(csv_file,
                          sep = sep,
                          index_col = False,
                          header = 0)

    # Return the data frame.
    return df_loss


def save_loss(df,
              csv_file,
              sep = ","):
    """Save the loss(es) in a CSV file.

    Parameters
    ----------
    df : :class:`pandas.DataFrame`
        A data frame containing the time data.

    csv_file : :class:`str`
        The output CSV file.

    sep : :class:`str`, ``","``
        The column separator in the output CSV file.
    """

    # Save the loss(es).
    df.to_csv(csv_file,
              sep = sep,
              index = False,
              header = True)
