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
import re
# Import from third-party packages.
import pandas as pd
from sklearn import decomposition
from sklearn import manifold


#######################################################################


# Get the module's logger.
logger = log.getLogger(__name__)


########################## PRIVATE CONSTANTS ##########################


# Set a mapping between the dimensionality reduction model's names
# and the corresponding classes.
_modname2modclass = \
    {# PCA
     "pca" : decomposition.PCA,
     # KPCA
     "kpca" : decomposition.KernelPCA,
     # MDS
     "mds" : manifold.MDS,
     # t-SNE
     "tsne" : manifold.TSNE}


########################## PRIVATE FUNCTIONS ##########################


def _perform_dim_red(df,
                     mod_fitted,
                     mod_class,
                     mod_options,
                     input_columns,
                     keep_unused_columns,
                     output_columns_prefix):
    """Perform a dimensionality reduction analysis.

    Parameters
    ----------
    df_rep : :class:`pandas.DataFrame`
        A data frame containing the representations.

    mod_fitted : :obj:`sklearn` model
        An already-fitted model on which to project the new data
        points.
    
    mod_class : :obj:`sklearn` model class
        The class of the model that needs to be built.

        It should be :obj:`None` if ``mod_fitted`` is passed.

    mod_options : :class:`dict`
        A dictionary of options to initialize a model from
        ``mod_class``.

        It should be :obj:`None` if ``mod_fitted`` is passed.

    input_columns : :class:`str` or :class:`list` or :obj:`None`
        Either a list containing the names of the columns whose
        contents should be used for the analysis or a string
        representing a pattern that the columns of interest should fit.

        By default, all columns of the input data frame are used for
        the analysis.

    keep_unused_columns : :class:`bool`
        Whether to append the unused columns to the output data frame.

    output_columns_prefix : :class:`str`
        A string representing the prefix used for the columns of
        the output data frame.

    Returns
    -------
    pca : :class:`sklearn.decomposition.PCA`
        The fitted PCA model.

    df_pca: :class:`pandas.DataFrame`
        A data frame containing the results of the PCA.
    """

    # If the user specified the number of components
    if "n_components" in mod_options:

        # Get the number of components.
        num_comp = mod_options["n_components"]

        # If the number of components is lower than the number of
        # samples
        if num_comp > len(df):

            # Raise an error.
            errstr = \
                "The number of components must be higher than or " \
                "equal to the number of samples."
            raise ValueError(errstr)

    #-----------------------------------------------------------------#

    # If the input columns are defined by a string
    if isinstance(input_columns, str):

        # Get all columns in the data frame whose name matches the
        # string.
        input_columns = \
            list(filter(re.compile(input_columns).match, df.columns))

    # Get the columns to be used for the analysis.
    df_data = df.loc[:, input_columns]

    # Get the extra columns.
    df_extra = \
        df.loc[:, [col for col in df.columns \
                   if col not in input_columns]]

    #-----------------------------------------------------------------#

    # Get the data points' values.
    data_values = df_data.values

    # Get the data points' names/IDs.
    data_names = df_data.index.tolist()

    #-----------------------------------------------------------------#

    # If a fitted model was not provided
    if mod_fitted is None:

        # Set up the model.
        mod = mod_class(**mod_options)

        # Fit the model.
        mod.fit(data_values)

    #-----------------------------------------------------------------#
    
    # Fit the model and apply the dimensionality reduction.
    projected = mod.fit_transform(data_values)

    #-----------------------------------------------------------------#

    # Set the names of the columns for the data frame containing the
    # results of the analysis.
    columns = \
        [f"{output_columns_prefix}{i+1}" \
         for i in range(projected.shape[1])]

    #-----------------------------------------------------------------#

    # Create a data frame containing the projected points.
    df_projected = pd.DataFrame(projected,
                                columns = columns,
                                index = data_names)

    #-----------------------------------------------------------------#

    # If we need to add the extra columns
    if keep_unused_columns:

        # Add the extra columns.
        df_projected = pd.concat([df_projected, df_extra],
                                  axis = 1)

    #-----------------------------------------------------------------#

    # Return the data frame and the model.
    return df_projected, mod


########################## PUBLIC FUNCTIONS ########################### 


def perform_pca(df,
                fitted_model = None,
                options = None,
                input_columns = None,
                keep_unused_columns = True,
                output_columns_prefix = "C"):
    """Perform a principal component analysis (PCA) on a set of
    data points.

    Parameters
    ----------
    df : :class:`pandas.DataFrame`
        A data frame containing the data points.

        The rows of the data frame should represent the different data
        points, while the columns should represent the dimensions of
        the space where the data points live.

    fitted_model : :class:`sklearn.decomposition.PCA`, optional
        An already fitted model onto which the data points
        should be projected. 

    options : :class:`dict`, optional
        A dictionary containing the options used when performing
        the analysis.

        The available options are those that can be used to initialize
        a :class:`sklearn.decomposition.PCA` instance.

    input_columns : :class:`str` or :class:`list`, optional
        Either a list containing the names of the columns whose
        contents should be used for the analysis or a string
        representing a pattern that the columns of interest should fit.

        By default, all columns of the input data frame are used for
        the analysis.

    keep_unused_columns : :class:`bool`, :obj:`True`
        Whether to append the unused columns to the output data frame.

    output_columns_prefix : :class:`str`, ``"C"``
        A string representing the prefix used for the columns of
        the output data frame.

    Returns
    -------
    df_results : :class:`pandas.DataFrame`
        A data frame containing the results of the analysis.

        The rows will contain the data points, while the columns
        will contain the values of each data point's projection along
        the dimensions of the projection space.

    pca : :class:`sklearn.decomposition.PCA`
        The fitted model.
    """

    # Return the results of the dimensionality reduction.
    return _perform_dim_red(\
                df = df,
                mod_fitted = fitted_model,
                mod_class = _modname2modclass["pca"],
                mod_options = options,
                input_columns = input_columns,
                keep_unused_columns = keep_unused_columns,
                output_columns_prefix = output_columns_prefix)


def perform_kpca(df,
                 fitted_model = None,
                 options = None,
                 input_columns = None,
                 keep_unused_columns = True,
                 output_columns_prefix = "C"):
    """Perform a kernel principal component analysis (KPCA) on a set of
    data points.

    Parameters
    ----------
    df : :class:`pandas.DataFrame`
        A data frame containing the data points.

        The rows of the data frame should represent the different data
        points, while the columns should represent the dimensions of
        the space where the data points live.

    fitted_model : :class:`sklearn.decomposition.KernelPCA`, optional
        An already fitted model onto which the data points
        should be projected. 

    options : :class:`dict`, optional
        A dictionary containing the options used when performing
        the analysis.
        
        The available options are those that can be used to initialize
        a :class:`sklearn.decomposition.KernelPCA` instance.

    input_columns : :class:`str` or :class:`list`, optional
        Either a list containing the names of the columns whose
        contents should be used for the analysis or a string
        representing a pattern that the columns of interest should fit.

        By default, all columns of the input data frame are used for
        the analysis.

    keep_unused_columns : :class:`bool`, :obj:`True`
        Whether to append the unused columns to the output data frame.

    output_columns_prefix : :class:`str`, ``"C"``
        A string representing the prefix used for the columns of
        the output data frame.

    Returns
    -------
    df_results : :class:`pandas.DataFrame`
        A data frame containing the results of the analysis.

        The rows will contain the data points, while the columns
        will contain the values of each data point's projection along
        the dimensions of the projection space.

    pca : :class:`sklearn.decomposition.KernelPCA`
        The fitted model.
    """

    # Return the results of the dimensionality reduction.
    return _perform_dim_red(\
                df = df,
                mod_fitted = fitted_model,
                mod_class = _modname2modclass["kpca"],
                mod_options = options,
                input_columns = input_columns,
                keep_unused_columns = keep_unused_columns,
                output_columns_prefix = output_columns_prefix)


def perform_mds(df,
                fitted_model = None,
                options = None,
                input_columns = None,
                keep_unused_columns = True,
                output_columns_prefix = "C"):
    """Perform a multidimensional scaling (MDS) on a set of data
    points.

    Parameters
    ----------
    df : :class:`pandas.DataFrame`
        A data frame containing the data points.

        The rows of the data frame should represent the different data
        points, while the columns should represent the dimensions of
        the space where the data points live.

    fitted_model : :class:`sklearn.manifold.MDS`, optional
        An already fitted model onto which the data points
        should be projected. 

    options : :class:`dict`, optional
        A dictionary containing the options used when performing
        the analysis.

        The available options are those that can be used to initialize
        a :class:`sklearn.manifold.MDS` instance.

    input_columns : :class:`str` or :class:`list`, optional
        Either a list containing the names of the columns whose
        contents should be used for the analysis or a string
        representing a pattern that the columns of interest should fit.

        By default, all columns of the input data frame are used for
        the analysis.

    keep_unused_columns : :class:`bool`, :obj:`True`
        Whether to append the unused columns to the output data frame.

    output_columns_prefix : :class:`str`, ``"C"``
        A string representing the prefix used for the columns of
        the output data frame.

    Returns
    -------
    df_results : :class:`pandas.DataFrame`
        A data frame containing the results of the analysis.

        The rows will contain the data points, while the columns
        will contain the values of each data point's projection along
        the dimensions of the projection space.

    mds : :class:`sklearn.manifold.MDS`
        The fitted model.
    """

    # Return the results of the dimensionality reduction.
    return _perform_dim_red(\
                df = df,
                mod_fitted = fitted_model,
                mod_class = _modname2modclass["mds"],
                mod_options = options,
                input_columns = input_columns,
                keep_unused_columns = keep_unused_columns,
                output_columns_prefix = output_columns_prefix)


def perform_tsne(df,
                 fitted_model = None,
                 options = None,
                 input_columns = None,
                 keep_unused_columns = True,
                 output_columns_prefix = "C"):
    """Perform a t-distributed stochastic neighbor embedding (t-SNE) on
    a set of data points.

    Parameters
    ----------
    df : :class:`pandas.DataFrame`
        A data frame containing the data points.

        The rows of the data frame should represent the different data
        points, while the columns should represent the dimensions of
        the space where the data points live.

    fitted_model : :class:`sklearn.manifold.TSNE`, optional
        An already fitted model onto which the data points
        should be projected. 

    options : :class:`dict`, optional
        A dictionary containing the options used when performing
        the analysis.

        The available options are those that can be used to initialize
        a :class:`sklearn.manifold.TSNE` instance.

    input_columns : :class:`str` or :class:`list`, optional
        Either a list containing the names of the columns whose
        contents should be used for the analysis or a string
        representing a pattern that the columns of interest should fit.

        By default, all columns of the input data frame are used for
        the analysis.

    keep_unused_columns : :class:`bool`, :obj:`True`
        Whether to append the unused columns to the output data frame.

    output_columns_prefix : :class:`str`, ``"C"``
        A string representing the prefix used for the columns of
        the output data frame.

    Returns
    -------
    df_results : :class:`pandas.DataFrame`
        A data frame containing the results of the analysis.

        The rows will contain the data points, while the columns
        will contain the values of each data point's projection along
        the dimensions of the projection space.

    tsne : :class:`sklearn.manifold.TSNE`
        The fitted model.
    """

    # If the perplexity is not defined and the number of samples is
    # less than 30
    if "perplexity" not in options and len(df) <= 30:

        # Set the new perplexity.
        perplexity = float(len(df) - 1)

        # Set it to one unit less than the number of samples.
        options["perplexity"] = perplexity

        # Warn the user that the perplexity was set.
        warnstr = \
            "The TSNE 'perplexity' was not defined, and " \
            "scikit-learn's default is 30.0, which is " \
            "less than the number of samples in the input " \
            "data frame. For this reason, the 'perplexity' was " \
            f"set to {perplexity}."
        logger.warning(warnstr)

    # Return the results of the dimensionality reduction.
    return _perform_dim_red(\
                df = df,
                mod_fitted = fitted_model,
                mod_class = _modname2modclass["tsne"],
                mod_options = options,
                input_columns = input_columns,
                keep_unused_columns = keep_unused_columns,
                output_columns_prefix = output_columns_prefix)

