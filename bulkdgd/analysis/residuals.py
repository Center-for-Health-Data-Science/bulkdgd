#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-

#    residuals.py
#
#    Utilities to compute vectors of residuals.
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
__doc__ = "Utilities to compute vectors of residuals."


#######################################################################


# Import from the standard library.
import logging as log
from typing import Optional

# Import from third-party libraries.
import numpy as np
import pandas as pd
from scipy.stats import nbinom, norm, poisson

# Import from 'bulkdgd'.
from . import _util


#######################################################################


# Get the module's logger.
logger = log.getLogger(__name__)


#######################################################################


def _get_scaling_factors(obs,
                         scaling_factor = "mean",
                         config_model = None):
    """Get the factors the samples' predicted means are multiplied by
    to become counts.

    Parameters
    ----------
    obs : :class:`numpy.ndarray`
        The observed counts, with samples as rows and genes as
        columns.

    scaling_factor : :class:`str`, {``"mean"``, ``"median"``}, \
        ``"mean"``
        The scaling factor. It must be the one the model was trained
        with.

    config_model : :class:`dict`, optional
        The model's configuration. Its ``scaling_factor``, if set,
        overrides ``scaling_factor``.

    Returns
    -------
    factors : :class:`numpy.ndarray`
        The scaling factor of each sample, as a column vector.
    """

    # If the model's configuration was passed
    if config_model is not None:

        # Use the model's scaling factor, if set.
        scaling_factor = \
            config_model.get("scaling_factor", scaling_factor) \
            if hasattr(config_model, "get") else scaling_factor

    # If the scaling factor is the median
    if scaling_factor == "median":

        # Return the median of each sample's counts.
        return np.median(obs,
                         axis = 1,
                         keepdims = True)

    # If the scaling factor is the mean
    if scaling_factor == "mean":

        # Return the mean of each sample's counts.
        return obs.mean(axis = 1,
                        keepdims = True)

    # Raise an error.
    raise ValueError(
        f"Unrecognized scaling factor '{scaling_factor}' - it must "
        f"be 'mean' or 'median'.")


def get_residuals(obs_counts: pd.Series,
                  pred_means: pd.Series,
                  r_values: Optional[pd.Series] = None,
                  sample_name: Optional[str] = None,
                  scaling_factor: str = "mean",
                  config_model: Optional[dict] = None) -> pd.Series:
    """Calculate the residuals of a sample's observed gene counts
    under the distributions predicted for them.

    Parameters
    ----------
    obs_counts : :class:`pandas.Series`
        The observed gene counts in a single sample.

        This is a series whose index contains either the genes'
        Ensembl IDs or names of fields containing additional
        information about the sample.

    pred_means : :class:`pandas.Series`
        The predicted means of the distributions modelling
        the genes' counts in a single sample.

        This is a series whose index contains either the genes'
        Ensembl IDs or names of fields containing additional
        information about the sample.

        These are the scaled means the model outputs.

    r_values : :class:`pandas.Series`, optional
        The predicted r-values of the negative binomial distributions
        modelling the genes' counts in a single sample, if the genes'
        counts were modelled using negative binomial distributions.

        This is a series whose index contains either the genes'
        Ensembl IDs or names of fields containing additional
        information about the sample.

        If ``r_values`` is not provided, it is assumed that the genes'
        counts were modelled using Poisson distributions.

    sample_name : :class:`str`, optional
        The name of the sample under consideration. It is used as
        name for the :class:`pandas.Series` returned.

        If not passed, the series will be unnamed.

    scaling_factor : :class:`str`, {``"mean"``, ``"median"``}, \
        ``"mean"``
        The scaling factor. It must be the one the model was trained
        with.

    config_model : :class:`dict`, optional
        The model's configuration. Its ``scaling_factor``, if set,
        overrides ``scaling_factor``.

    Returns
    -------
    series_residuals : :class:`pandas.Series`
        A series containing the residuals for all genes.
    """

    # Get the genes, checking that all inputs list them in the same
    # order.
    genes_obs = \
        _util.get_aligned_genes(obs_counts = obs_counts,
                                pred_means = pred_means,
                                r_values = r_values)

    #-----------------------------------------------------------------#

    # If the r-values were passed
    if r_values is not None:

        # Get the genes' r-values, as an array.
        r_values = pd.to_numeric(r_values.loc[genes_obs]).values

    #-----------------------------------------------------------------#

    # Get the genes' observed counts, as an array.
    obs_counts = pd.to_numeric(obs_counts.loc[genes_obs]).values

    #-----------------------------------------------------------------#

    # Get the genes' predicted means, in the same order.
    pred_means = pd.to_numeric(pred_means.loc[genes_obs]).values

    #-----------------------------------------------------------------#

    # Get the sample's scaling factor (the model's own, if
    # 'config_model' is passed).
    obs_counts_scale = \
        _get_scaling_factors(
            np.asarray(obs_counts, dtype = np.float64).reshape(1, -1),
            scaling_factor = scaling_factor,
            config_model = config_model)[0, 0]

    #-----------------------------------------------------------------#

    # Rescale the predicted means by the scaling factor.
    pred_means = pred_means * obs_counts_scale

    #-----------------------------------------------------------------#

    # Create an empty list to store the residuals.
    residuals = []

    # For each gene's observed count, predicted mean count, and r-value
    for i, (obs_count_gene_i, pred_mean_gene_i) \
        in enumerate(zip(obs_counts, pred_means)):

        #-------------------------------------------------------------#

        # If negative binomial distributions were used to model the
        # genes' counts
        if r_values is not None:

            # Get the r-value for the current gene.
            r_value_gene_i = r_values[i]

            # Get the probability 'q' of a failure from the mean 'm'
            # and the r-value: m = r*q/(1-q), so q = m/(m+r).
            p_i = pred_mean_gene_i / \
                  (pred_mean_gene_i + r_value_gene_i)

            #---------------------------------------------------------#

            # Get the value of the negative binomial's CDF (SciPy's 'p'
            # is the probability of a success, 1 - q).
            cdf_nb_value = \
                nbinom.cdf(k = obs_count_gene_i,
                           n = r_value_gene_i,
                           p = 1 - p_i)

        #-------------------------------------------------------------#

        # If Poisson distributions were used to model the genes' counts
        else:

            # Get the value of the cumulative Poisson distribution.
            cdf_nb_value = \
                poisson.cdf(k = obs_count_gene_i,
                            mu = pred_mean_gene_i)

        #-------------------------------------------------------------#

        # Get the residual as the standard normal's quantile at the CDF
        # value.
        residual_gene_i = norm.ppf(q = cdf_nb_value)

        #-------------------------------------------------------------#

        # Save the residual for the current gene.
        residuals.append(residual_gene_i)

    #-----------------------------------------------------------------#

    # Create a series to store the residuals.
    series_residuals = pd.Series(residuals)

    # Set the index of the series equal to the genes' names.
    series_residuals.index = genes_obs

    # Set the name of the index to 'gene_id'.
    series_residuals = series_residuals.rename_axis("gene_id")

    # If a name was passed for the sample
    if sample_name:

        # Set the name of the series equal to the sample's name.
        series_residuals.name = sample_name

    #-----------------------------------------------------------------#

    # Return the series with the residuals.
    return series_residuals


#######################################################################


def get_residuals_df(df_obs_counts: pd.DataFrame,
                     df_pred_means: pd.DataFrame,
                     df_r_values: Optional[pd.DataFrame] = None,
                     clip: float = 0.0,
                     scaling_factor: str = "mean",
                     config_model: Optional[dict] = None) -> \
                        pd.DataFrame:
    """Calculate the residuals of many samples at once, as the
    standard normal's quantile at each observed count's CDF value.

    Parameters
    ----------
    df_obs_counts : :class:`pandas.DataFrame`
        The observed counts. Samples are rows and genes are columns.

    df_pred_means : :class:`pandas.DataFrame`
        The predicted scaled means, as the model outputs them.

    df_r_values : :class:`pandas.DataFrame`, optional
        The predicted r-values, if the genes' counts were modelled with
        negative binomial distributions (Poisson otherwise).

    clip : :class:`float`, ``0.0``
        How far from 0 and 1 to clip the CDF values before taking the
        inverse normal. With ``0.0``, CDF values of 0 or 1 give
        infinite residuals; with ``1e-12``, residuals of about 7.

    scaling_factor : :class:`str`, {``"mean"``, ``"median"``}, \
        ``"mean"``
        The scaling factor. It must be the one the model was trained
        with.

    config_model : :class:`dict`, optional
        The model's configuration. Its ``scaling_factor``, if set,
        overrides ``scaling_factor``.

    Returns
    -------
    df_residuals : :class:`pandas.DataFrame`
        The residuals. Samples are rows and genes are columns, as they
        are in the inputs.
    """

    # Get the genes, checking that all inputs list them in the same
    # order.
    genes = \
        _util.get_aligned_genes(
            obs_counts = df_obs_counts.columns.to_series(),
            pred_means = df_pred_means.columns.to_series(),
            r_values = None if df_r_values is None \
                       else df_r_values.columns.to_series())

    # Get the samples the counts and the predictions have in common.
    samples = df_obs_counts.index.intersection(df_pred_means.index)

    # Get the observed counts.
    obs = df_obs_counts.loc[samples, genes].to_numpy(dtype = np.float64)

    # Get the predicted means.
    means = df_pred_means.loc[samples, genes].to_numpy(
        dtype = np.float64)

    #-----------------------------------------------------------------#

    # Rescale the predicted means by each sample's scaling factor.
    means = \
        means * _get_scaling_factors(obs,
                                     scaling_factor = scaling_factor,
                                     config_model = config_model)

    #-----------------------------------------------------------------#

    # If the genes' counts were modelled with negative binomials
    if df_r_values is not None:

        # Get the predicted r-values.
        r = df_r_values.loc[samples, genes].to_numpy(
            dtype = np.float64)

        # Clip the r-values away from 0.
        r = np.clip(r,
                    1e-8,
                    None)

        # Get the CDF values (SciPy's 'p' is the probability of a
        # success).
        cdf = nbinom.cdf(k = obs,
                         n = r,
                         p = r / (r + means))

    # Otherwise
    else:

        # Get the CDF values.
        cdf = poisson.cdf(k = obs,
                          mu = means)

    #-----------------------------------------------------------------#

    # Clip the CDF values.
    cdf = np.clip(cdf,
                  clip,
                  1.0 - clip)

    # Return the residuals.
    return pd.DataFrame(norm.ppf(cdf),
                        index = samples,
                        columns = genes)


def get_significant_genes(
        series_residuals: pd.Series,
        res_pos_threshold: int | float = 1,
        res_neg_threshold: int | float = -1) -> pd.Series:
    """Get the genes that are significant at a given significance
    level.

    Parameters
    ----------
    series_residuals : :class:`pandas.Series`
        A series containing the residuals for all genes in a single
        sample.

        The series' index contains the genes' Ensembl IDs, and its
        values are the residuals.

    res_pos_threshold : :class:`int` or :class:`float`, ``1``
        The threshold above which a gene is considered significantly
        up-regulated.

    res_neg_threshold : :class:`int` or :class:`float`, ``-1``
        The threshold below which a gene is considered significantly
        down-regulated.

    Returns
    -------
    series_significant_genes : :class:`pandas.Series`
        The residuals of the significant genes.
    """

    # Get the genes satisfying all the conditions.
    series_significant_genes = \
        series_residuals[(series_residuals >= res_pos_threshold) | \
                         (series_residuals <= res_neg_threshold)]

    #-----------------------------------------------------------------#

    # Return the series.
    return series_significant_genes


def get_genes_by_residual_threshold(
        df_res: pd.DataFrame,
        le_than: int | float = -1,
        ge_than: int | float = 1,
        sort_genes: bool = False,
        ascending: bool = False) -> \
            tuple[pd.DataFrame, pd.DataFrame,
                  pd.DataFrame, pd.DataFrame]:
    """Get the genes whose residuals fall in specified intervals and
    are common to a certain number of samples.

    Parameters
    ----------
    df_res : :class:`pandas.DataFrame`
        A data frame containing the residual vectors for a set of
        samples.

    le_than : :class:`int` or :class:`float`, ``-1``
        The threshold at or below which a residual meets the first
        condition.

    ge_than : :class:`int` or :class:`float`, ``1``
        The threshold at or above which a residual meets the second
        condition.

    sort_genes : :class:`bool`, ``False``
        Whether to sort the output data frames by the number of
        samples.

    ascending : :class:`bool`, ``False``
        Whether to sort in ascending order.

    Returns
    -------
    df_samples_count_le : :class:`pandas.DataFrame`
        A data frame containing each gene with the count of samples
        where the gene's residual value is lower than or equal to
        ``le_than``.

        The data frame's rows are identified by each gene's ID (as
        provided in the input data frame) and the data frame includes
        one column:

        - 'n_samples': the number of samples where the gene's residual
          meets the condition.

    df_genes_distribution_le : :class:`pandas.DataFrame`
        A data frame displaying the distribution of genes having a
        residual value lower than or equal to ``le_than`` across
        different sample counts.

        Each row represents the number of samples, and the columns are:

        - 'n_genes': the number of genes that meet the condition in
          exactly that many samples.
        - 'genes': a period-separated string listing the genes that
          meet the condition in exactly that many samples.

    df_samples_count_ge : :class:`pandas.DataFrame`
        A data frame containing each gene with the count of samples
        where the gene's residual value is greater than or equal to
        ``ge_than``.

        The data frame's rows are identified by each gene's ID (as
        provided in the input data frame) and the data frame includes
        one column:

        - 'n_samples': the number of samples where the gene's residual
          meets the condition.

    df_genes_distribution_ge : :class:`pandas.DataFrame`
        A data frame displaying the distribution of genes having a
        residual value greater than or equal to ``ge_than`` across
        different sample counts.

        Each row represents the number of samples, and the columns are:

        - 'n_genes': the number of genes that meet the condition in
          exactly that many samples.
        - 'genes': a period-separated string listing the genes that
          meet the condition in exactly that many samples.
    """

    # Get the names of the cells containing residual values.
    df_res_data = \
        df_res.loc[:, [col for col in df_res.columns \
                       if col.startswith("ENSG")]]

    #-----------------------------------------------------------------#

    # Set the condition for which the residuals must be lower than or
    # equal to the specified value.
    condition_le = df_res_data <= le_than

    # Set the condition for which the residuals must be greater than
    # or equal to the specified value.
    condition_ge = df_res_data >= ge_than

    #-----------------------------------------------------------------#

    # Count the number of samples meeting the first condition.
    samples_count_le = condition_le.sum(axis = 0)

    # Count the number of samples meeting the second condition.
    samples_count_ge = condition_ge.sum(axis = 0)

    #-----------------------------------------------------------------#

    # Create a data frame containing genes and their count of samples
    # meeting the first condition.
    df_samples_count_le = \
        pd.DataFrame({"gene": df_res_data.columns,
                      "n_samples": samples_count_le})

    # Create a data frame containing genes and their count of samples
    # meeting the second condition.
    df_samples_count_ge = \
        pd.DataFrame({"gene": df_res_data.columns,
                      "n_samples": samples_count_ge})

    #-----------------------------------------------------------------#

    # If the genes must be sorted by the number of samples
    if sort_genes:

        # Sort the first data frame by the number of samples.
        df_samples_count_le = \
            df_samples_count_le.sort_values(\
                by = "n_samples",
                ascending = ascending)

        # Sort the second data frame by the number of samples.
        df_samples_count_ge = \
            df_samples_count_ge.sort_values(\
                by = "n_samples",
                ascending = ascending)

    #-----------------------------------------------------------------#

    # Get the total number of samples.
    n_all_samples = df_res_data.shape[0]

    # Create an empty dictionary to store how many (and which) genes
    # meet the first condition in each number of samples.
    distribution_data_le = {"n_genes": [], "genes": []}

    # Create an empty dictionary to store how many (and which) genes
    # meet the second condition in each number of samples.
    distribution_data_ge = {"n_genes": [], "genes": []}

    # For each possible number of samples in which the genes meet the
    # condition
    for i in range(n_all_samples + 1):

        # Find the genes that meet the first condition in the current
        # number of samples.
        genes_meeting_i_samples_le = \
            df_samples_count_le[\
                df_samples_count_le["n_samples"] == i]["gene"]

        # Find the genes that meet the second condition in the current
        # number of samples.
        genes_meeting_i_samples_ge = \
            df_samples_count_ge[\
                df_samples_count_ge["n_samples"] == i]["gene"]

        # Add the number of genes to the first dictionary.
        distribution_data_le["n_genes"].append(\
            len(genes_meeting_i_samples_le))

        # Add the number of genes to the second dictionary.
        distribution_data_ge["n_genes"].append(\
            len(genes_meeting_i_samples_ge))

        # Add the list of genes to the first dictionary.
        distribution_data_le["genes"].append(\
            ".".join(genes_meeting_i_samples_le))

        # Add the list of genes to the second dictionary.
        distribution_data_ge["genes"].append(\
            ".".join(genes_meeting_i_samples_ge))

    #-----------------------------------------------------------------#

    # Convert the first dictionary into a data frame.
    df_genes_distribution_le = \
        pd.DataFrame(distribution_data_le,
                     index = range(0, n_all_samples + 1))

    # Convert the second dictionary into a data frame.
    df_genes_distribution_ge = \
        pd.DataFrame(distribution_data_ge,
                     index = range(0, n_all_samples + 1))

    #-----------------------------------------------------------------#

    # Update the index of the first data frame.
    df_samples_count_le = df_samples_count_le.set_index("gene")

    # Update the index of the second data frame.
    df_samples_count_ge = df_samples_count_ge.set_index("gene")

    #-----------------------------------------------------------------#

    # If the genes must be sorted by the number of samples
    if sort_genes:

        # Sort the first data frame by the number of samples.
        df_genes_distribution_le = \
            df_genes_distribution_le.sort_index(ascending = ascending)

        # Sort the second data frame by the number of samples.
        df_genes_distribution_ge = \
            df_genes_distribution_ge.sort_index(ascending = ascending)

    #-----------------------------------------------------------------#

    # Return the data frames.
    return df_samples_count_le, df_genes_distribution_le, \
           df_samples_count_ge, df_genes_distribution_ge
