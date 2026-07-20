#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-

#    reduction.py
#
#    Utilities to perform dimensionality reduction.
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
__doc__ = "Utilities to perform dimensionality reduction."


#######################################################################


# Import from the standard library.
import logging as log
import re
from typing import Optional

# Import from third-party libraries.
import numpy as np
import pandas as pd
from sklearn import decomposition
from sklearn import manifold
import umap


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
     "tsne" : manifold.TSNE,
     # UMAP
     "umap" : umap.UMAP}


########################## PRIVATE FUNCTIONS ##########################


def _perform_dim_red(df: pd.DataFrame,
                     mod_fitted: Optional[object],
                     mod_class: type[object],
                     mod_options: Optional[dict[str, object]],
                     input_columns: Optional[str | list[str]],
                     keep_unused_columns: bool,
                     output_columns_prefix: str,
                     replace_nan: Optional[int | float] = None,
                     replace_inf: Optional[int | float] = None,
                     replace_ninf: Optional[int | float] = None) -> \
                        tuple[pd.DataFrame, object]:
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
    
    replace_nan : :obj:`int` or :obj:`float`, optional
        A value to replace NaNs with.
    
    replace_inf : :obj:`int` or :obj:`float`, optional
        A value to replace positive infinities with.
    
    replace_ninf : :obj:`int` or :obj:`float`, optional
        A value to replace negative infinities with.

    Returns
    -------
    df_projected: :class:`pandas.DataFrame`
        A data frame containing the results of the dimensionality
        reduction.

    mod : :obj:`sklearn` model class
        The fitted dimensionality reduction model.
    """

    # Create a copy of the options.
    mod_options = dict(mod_options) if mod_options is not None else {}

    #-----------------------------------------------------------------#

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

    # If the user specified a number to replace positive infinities
    # with
    if replace_inf is not None:

        # Replace the positive infinities.
        df = df.replace([np.inf], replace_inf)
    
    #-----------------------------------------------------------------#

    # If the user specified a number to replace negative infinities
    # with
    if replace_ninf is not None:

        # Replace the negative infinities.
        df = df.replace([-np.inf], replace_ninf)

    #-----------------------------------------------------------------#

    # If the user specified a number to replace NaNs with
    if replace_nan is not None:

        # Replace the NaNs.
        df = df.fillna(replace_nan)

    #-----------------------------------------------------------------#

    # If no input columns were specified, use all the columns.
    if input_columns is None:
        input_columns = df.columns.tolist()

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

        # Fit the model and apply the dimensionality reduction.
        projected = mod.fit_transform(data_values)

    # Otherwise, use the already-fitted model to project the data.
    else:

        mod = mod_fitted
        projected = mod.transform(data_values)

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


def perform_pca(df: pd.DataFrame,
                fitted_model: \
                    Optional[decomposition.PCA] = None,
                options: Optional[dict[str, object]] = None,
                input_columns: Optional[str | list[str]] = None,
                keep_unused_columns: bool = True,
                output_columns_prefix: str = "C",
                replace_nan: Optional[int | float] = None,
                replace_inf: Optional[int | float] = None,
                replace_ninf: Optional[int | float] = None) -> \
                    tuple[pd.DataFrame,
                          decomposition.PCA]:
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

    replace_nan : :obj:`int` or :obj:`float`, optional
        A value to replace NaNs with.
    
    replace_inf : :obj:`int` or :obj:`float`, optional
        A value to replace positive infinities with.
    
    replace_ninf : :obj:`int` or :obj:`float`, optional
        A value to replace negative infinities with.

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
                output_columns_prefix = output_columns_prefix,
                replace_nan = replace_nan,
                replace_inf = replace_inf,
                replace_ninf = replace_ninf)


def perform_kpca(df: pd.DataFrame,
                 fitted_model: \
                          Optional[decomposition.KernelPCA] = None,
                 options: Optional[dict[str, object]] = None,
                 input_columns: Optional[str | list[str]] = None,
                 keep_unused_columns: bool = True,
                 output_columns_prefix: str = "C",
                 replace_nan: Optional[int | float] = None,
                 replace_inf: Optional[int | float] = None,
                 replace_ninf: Optional[int | float] = None) -> \
                    tuple[pd.DataFrame,
                                  decomposition.KernelPCA]:
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

    replace_nan : :obj:`int` or :obj:`float`, optional
        A value to replace NaNs with.
    
    replace_inf : :obj:`int` or :obj:`float`, optional
        A value to replace positive infinities with.
    
    replace_ninf : :obj:`int` or :obj:`float`, optional
        A value to replace negative infinities with.

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
                output_columns_prefix = output_columns_prefix,
                replace_nan = replace_nan,
                replace_inf = replace_inf,
                replace_ninf = replace_ninf)


def perform_mds(df: pd.DataFrame,
                fitted_model: Optional[manifold.MDS] = None,
                options: Optional[dict[str, object]] = None,
                input_columns: Optional[str | list[str]] = None,
                keep_unused_columns: bool = True,
                output_columns_prefix: str = "C",
                replace_nan: Optional[int | float] = None,
                replace_inf: Optional[int | float] = None,
                replace_ninf: Optional[int | float] = None) -> \
                    tuple[pd.DataFrame, manifold.MDS]:
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

    replace_nan : :obj:`int` or :obj:`float`, optional
        A value to replace NaNs with.
    
    replace_inf : :obj:`int` or :obj:`float`, optional
        A value to replace positive infinities with.
    
    replace_ninf : :obj:`int` or :obj:`float`, optional
        A value to replace negative infinities with.

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
                output_columns_prefix = output_columns_prefix,
                replace_nan = replace_nan,
                replace_inf = replace_inf,
                replace_ninf = replace_ninf)


def perform_tsne(df: pd.DataFrame,
                      fitted_model: Optional[manifold.TSNE] = None,
                 options: Optional[dict[str, object]] = None,
                 input_columns: Optional[str | list[str]] = None,
                 keep_unused_columns: bool = True,
                 output_columns_prefix: str = "C",
                 replace_nan: Optional[int | float] = None,
                 replace_inf: Optional[int | float] = None,
                 replace_ninf: Optional[int | float] = None) -> \
                          tuple[pd.DataFrame, manifold.TSNE]:
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

    replace_nan : :obj:`int` or :obj:`float`, optional
        A value to replace NaNs with.
    
    replace_inf : :obj:`int` or :obj:`float`, optional
        A value to replace positive infinities with.
    
    replace_ninf : :obj:`int` or :obj:`float`, optional
        A value to replace negative infinities with.

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

    # Create a copy of the options.
    options = dict(options) if options is not None else {}

    #-----------------------------------------------------------------#

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
    
    #-----------------------------------------------------------------#

    # Return the results of the dimensionality reduction.
    return _perform_dim_red(\
                df = df,
                mod_fitted = fitted_model,
                mod_class = _modname2modclass["tsne"],
                mod_options = options,
                input_columns = input_columns,
                keep_unused_columns = keep_unused_columns,
                output_columns_prefix = output_columns_prefix,
                replace_nan = replace_nan,
                replace_inf = replace_inf,
                replace_ninf = replace_ninf)


def perform_umap(df: pd.DataFrame,
                 fitted_model: Optional[umap.UMAP] = None,
                 options: Optional[dict[str, object]] = None,
                 input_columns: Optional[str | list[str]] = None,
                 keep_unused_columns: bool = True,
                 output_columns_prefix: str = "C",
                 replace_nan: Optional[int | float] = None,
                 replace_inf: Optional[int | float] = None,
                 replace_ninf: Optional[int | float] = None) -> \
                    tuple[pd.DataFrame, umap.UMAP]:
    """Perform a uniform manifold approximation and projection on a
    set of data points.

    Parameters
    ----------
    df : :class:`pandas.DataFrame`
        A data frame containing the data points.

        The rows of the data frame should represent the different data
        points, while the columns should represent the dimensions of
        the space where the data points live.

    fitted_model : :class:`umap.UMAP`, optional
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

    replace_nan : :obj:`int` or :obj:`float`, optional
        A value to replace NaNs with.
    
    replace_inf : :obj:`int` or :obj:`float`, optional
        A value to replace positive infinities with.
    
    replace_ninf : :obj:`int` or :obj:`float`, optional
        A value to replace negative infinities with.

    Returns
    -------
    df_results : :class:`pandas.DataFrame`
        A data frame containing the results of the analysis.

        The rows will contain the data points, while the columns
        will contain the values of each data point's projection along
        the dimensions of the projection space.

    mds : :class:`umap.UMAP`
        The fitted model.
    """

    # Return the results of the dimensionality reduction.
    return _perform_dim_red(\
                df = df,
                mod_fitted = fitted_model,
                mod_class = _modname2modclass["umap"],
                mod_options = options,
                input_columns = input_columns,
                keep_unused_columns = keep_unused_columns,
                output_columns_prefix = output_columns_prefix,
                replace_nan = replace_nan,
                replace_inf = replace_inf,
                replace_ninf = replace_ninf)

