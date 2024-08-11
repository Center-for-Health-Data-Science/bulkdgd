#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-

#    reduction.py
#
#    Utilities to perform dimensionality reduction.
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
__doc__ = "Utilities to perform dimensionality reduction."


#######################################################################


# Import from the standard library.
import logging as log
# Import from third-party packages.
import pandas as pd
from sklearn.decomposition import PCA


#######################################################################


# Get the module's logger.
logger = log.getLogger(__name__)


#######################################################################


def perform_2d_pca(df_rep,
                   pc_columns = ["PC1", "PC2"],
                   pca = None):
    """Perform a 2D principal component analysis (PCA) on a set of
    representations.

    Parameters
    ----------
    df_rep : :class:`pandas.DataFrame`
        A data frame containing the representations.

        The rows of the data frame should represent the samples, while
        the columns should represent the dimensions of the space
        where the representations live.

    pc_columns : :class:`list`, ``["PC1", "PC2"]``
        A list with the names of the two columns that will contain
        the values of the first two principal components.

    pca : :class:`sklearn.decomposition.PCA`, optional
        An already fitted PCA model onto which the data contained
        in ``df_rep`` should be projected. 

    Returns
    -------
    pca : :class:`sklearn.decomposition.PCA`
        The fitted PCA model.

    df_pca: :class:`pandas.DataFrame`
        A data frame containing the results of the PCA.

        The rows will contain the representations, while the columns
        will contain the values of each representation's projection
        along the principal components.
    """

    # Get the representations' values.
    rep_values = df_rep.values

    # Get the representations' names/IDs.
    rep_names = df_rep.index.tolist()

    #-----------------------------------------------------------------#

    # If a fitted PCA model was not provided
    if pca is None:

        # Set up the PCA.
        pca = PCA(n_components = 2)

        # Fit the model.
        pca.fit(rep_values)

    # Otherwise
    else:

        # If the number of components is not 2
        if pca.components_.shape[0] != 2:

            # Raise an error.
            errstr = \
                "The provided PCA model ('pca') must have 2 " \
                f"components, but {pca.components_.shape[0]} " \
                "components were found."
            raise ValueError(errstr)

    #-----------------------------------------------------------------#
    
    # Fit the model and apply the dimensionality reduction.
    projected = pca.fit_transform(rep_values)

    # Create a data frame containing the projected points.
    df_projected = pd.DataFrame(projected,
                                columns = pc_columns,
                                index = rep_names)

    #-----------------------------------------------------------------#

    # Return the data frame.
    return df_projected
