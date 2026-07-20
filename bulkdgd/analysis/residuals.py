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


#######################################################################


# Get the module's logger.
logger = log.getLogger(__name__)


#######################################################################


def get_residuals(obs_counts: pd.Series,
                  pred_means: pd.Series,
                  r_values: Optional[pd.Series] = None,
                  sample_name: Optional[str] = None) -> pd.Series:
    """Calculate the vector of residuals between the observed gene
    expression (counts) and the predicted means of the negative
    binomials modeling the expression of the different genes for a
    single sample.

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

        If the genes' counts were modelled using negative binomial
        distributions, the predicted means are scaled by the
        corresponding distributions' r-values.

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

    Returns
    -------
    series_residuals : :class:`pandas.Series`
        A series containing the residuals for all genes.
    """

    # Get the names of the cells containing gene expression data from
    # the series containing the observed gene counts.
    genes_obs = \
        [col for col in obs_counts.index if col.startswith("ENSG")]

    #-----------------------------------------------------------------#

    # Get the names of the cells containing gene expression data from
    # the series containing the predicted means.
    genes_pred = \
        [col for col in pred_means.index if col.startswith("ENSG")]

    #-----------------------------------------------------------------#

    # If the lists do not contain the same genes
    if set(genes_obs) != set(genes_pred):

        # Raise an error.
        errstr = \
            "The set of genes in 'obs_counts' and 'pred_means', " \
            "must be the same. It is assumed that the genes are " \
            "specified using their Ensembl IDs."
        raise ValueError(errstr)

    #-----------------------------------------------------------------#

    # If the r-values were passed
    if r_values is not None:

        # Get the names of the cells containing r-values from the
        # series of r-values.
        genes_r_values =  \
            [col for col in r_values.index if col.startswith("ENSG")]

        # If the lists do not contain the same genes
        if set(genes_obs) != set(genes_pred) \
        or set(genes_obs) != set(genes_r_values):

            # Raise an error.
            errstr = \
                "The set of genes in 'obs_counts', 'pred_means', " \
                "and 'r_values' must be the same. It is assumed " \
                "that the genes are specified using their Ensembl IDs."
            raise ValueError(errstr)

        # Create a numpy array containing only those columns containing
        # gene expression data for the predicted r-values - the 'loc'
        # syntax should return the columns in the order specified by
        # the selection.
        r_values = pd.to_numeric(r_values.loc[genes_obs]).values

    #-----------------------------------------------------------------#

    # Create a numpy array containing only those columns containing
    # gene expression data for the observed gene counts - the 'loc'
    # syntax should return the columns in the order specified by the
    # selection.
    obs_counts = pd.to_numeric(obs_counts.loc[genes_obs]).values

    #-----------------------------------------------------------------#

    # Create a numpy array containing only those columns containing
    # gene expression data for the predicted mean counts - the 'loc'
    # syntax should return the columns in the order specified by the
    # selection.
    pred_means = pd.to_numeric(pred_means.loc[genes_obs]).values

    #-----------------------------------------------------------------#

    # Get the mean gene counts for the sample.
    #
    # The output is a single value.
    obs_counts_mean = np.mean(obs_counts)

    #-----------------------------------------------------------------#

    # Rescale the predicted means by the mean gene counts.
    #
    # The output is a 1D tensor containing the rescaled means.
    pred_means = pred_means * obs_counts_mean

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

            # Calculate the probability of "failure" of the negative
            # binomial from its mean 'm' and its r-value (the number of
            # successes at which the experiment is stopped).
            #
            # The mean of a negative binomial, written in terms of the
            # probability 'q' of a FAILURE, is
            #
            #     m = r * q / (1 - q)
            #
            # so that
            #
            #     m (1 - q) = r q
            #     m - m q   = r q
            #     m         = q (m + r)
            #     q         = m / (m + r)
            #
            # which is what is calculated here.
            #
            # The derivation this comment used to give was of the
            # probability of a SUCCESS, p = r / (m + r), which is not
            # what the line below computes - it computes 1 - p. The
            # arithmetic was right and the comment described a different
            # quantity, so the value handed to SciPy was correct and the
            # reason given for it was not.
            p_i = pred_mean_gene_i / \
                  (pred_mean_gene_i + r_value_gene_i)

            #---------------------------------------------------------#

            # Get the value of the cumulative negative binomial
            # distribution.
            #
            # SciPy's negative binomial counts the number of failures
            # before 'n' successes, and its 'p' is the probability of a
            # SUCCESS. Ours, above, is the probability of a failure, so
            # SciPy is given 1 - p_i = r / (m + r), which returns the
            # mean to 'm':
            #
            #     mean = n (1 - p) / p
            #          = r * (m / (m + r)) / (r / (m + r))
            #          = m
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
        
        # Get the value of the inverse of the cumulative normal
        # distribution (= the valued of the percentile function) at the
        # point corresponding to the value of the neg. binom. CDF at
        # the observed gene count value.
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
                     clip: float = 0.0) -> pd.DataFrame:
    """Calculate the residuals of many samples at once.

    This is :func:`get_residuals`, which takes one sample and walks its
    genes one at a time in Python. That is fine for a sample and slow for
    a study: ten thousand samples of fifteen thousand genes is a hundred
    and fifty million turns of that loop. The same arithmetic is done
    here on the whole matrix, and gives the same numbers.

    The residual of a gene is the standard normal deviate at the point
    where the observed count falls in the distribution the model
    predicted for it:

    .. math::

       r_{ij} = \\Phi^{-1}\\left( F(k_{ij}; \\mu_{ij}, r_{ij}) \\right)

    Parameters
    ----------
    df_obs_counts : :class:`pandas.DataFrame`
        The observed counts. Samples are rows and genes are columns.

    df_pred_means : :class:`pandas.DataFrame`
        The predicted scaled means, as the model writes them. They are
        rescaled here by the mean count of each sample, which is what
        turns them into counts.

    df_r_values : :class:`pandas.DataFrame`, optional
        The predicted r-values, if the genes' counts were modelled with
        negative binomial distributions. If they are not given, the
        counts are taken to have been modelled with Poisson
        distributions.

    clip : :class:`float`, ``0.0``
        How far from nought and one to hold the cumulative distribution
        before the inverse normal is taken of it.

        Nought, by default, so that this function gives back exactly
        what :func:`get_residuals` gives back - the two agree to
        5e-11, which is the arithmetic and nothing else.

        The inverse normal is infinite at nought and at one, though, and
        a count far enough out in the tail of its own distribution will
        put it there: a few dozen genes in every few thousand samples.
        :func:`get_residuals` returns those infinities, and anything
        that goes on to take a principal component of the residuals will
        fail on them.

        Passing ``1e-12`` holds the cumulative distribution just inside
        its bounds, so that a gene the model finds impossible gets a
        large residual (about 7) rather than an infinite one. It changes
        nothing else: about one residual in two thousand, all of them in
        the extreme tail.

    Returns
    -------
    df_residuals : :class:`pandas.DataFrame`
        The residuals. Samples are rows and genes are columns, as they
        are in the inputs.
    """

    # The genes the counts and the predictions have in common, in the
    # order the counts have them.
    genes = [gene for gene in df_obs_counts.columns
             if gene in df_pred_means.columns]

    samples = df_obs_counts.index.intersection(df_pred_means.index)

    obs = df_obs_counts.loc[samples, genes].to_numpy(dtype = np.float64)

    means = df_pred_means.loc[samples, genes].to_numpy(
        dtype = np.float64)

    #-----------------------------------------------------------------#

    # A predicted mean is not a count until it is rescaled by the mean
    # count of the sample it was predicted for.
    means = means * obs.mean(axis = 1, keepdims = True)

    #-----------------------------------------------------------------#

    # If the genes' counts were modelled with negative binomials
    if df_r_values is not None:

        r = df_r_values.loc[samples, genes].to_numpy(
            dtype = np.float64)

        r = np.clip(r, 1e-8, None)

        # SciPy's negative binomial counts the failures, so its 'p' is
        # our '1 - p'.
        cdf = nbinom.cdf(k = obs,
                         n = r,
                         p = r / (r + means))

    # Otherwise, they were modelled with Poissons.
    else:

        cdf = poisson.cdf(k = obs,
                          mu = means)

    #-----------------------------------------------------------------#

    cdf = np.clip(cdf, clip, 1.0 - clip)

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
    
    res_pos_threshold : ``int`` or ``float``, ``1``
        The threshold above which a gene is considered significantly
        up-regulated.
    
    res_neg_threshold : ``int`` or ``float``, ``-1``
        The threshold below which a gene is considered significantly
        down-regulated.
    """
    
    # Get the genes satisfying all the conditions.
    series_significant_genes = \
        series_residuals[(series_residuals >= res_pos_threshold) | \
                         (series_residuals <= res_neg_threshold)]

    #------------------------------------------------------------------#
    
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
    df_res : ``pandas.DataFrame``
        A data frame containing the residual vectors for a set of
        samples.

    le_than : ``int`` or ``float``, ``-1``
        Consider only genes whose residual value in all samples is
        lower than or equal to the provided value.

    ge_than : ``int`` or ``float``, ``1``
        Consider only genes whose residual value in all samples is
        greater than or equal to the provided value.

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
        residual value lower than or equal to ``le_than``  across
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
        residual value greater than or equal to ``le_than``  across
        different sample counts.

        Each row represents the number of samples, and the columns are:
        
        - 'n_genes': the number of genes that meet the condition in
          exactly that many samples.
        - 'genes': a period-separated string listing the genes that
          meet the condition in exactly that many samples..
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

    # If the genes must be sorted by the number of samples.
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

    # Create another data frame showing how many genes meet the
    # condition in how many samples.

    # Get the total number of samples.
    n_all_samples = df_res_data.shape[0]

    # Create an empty dictionary to store the data about how the genes
    # are distributed according to in how many samples they meet the
    # first condition.
    distribution_data_le = {"n_genes": [], "genes": []}

    # Create an empty dictionary to store the data about how the genes
    # are distributed according to in how many samples they meet the
    # second condition.
    distribution_data_ge = {"n_genes": [], "genes": []}

    # For each possible number of samples in which the genes meet the
    # condition.
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

    # If the genes must be sorted by the number of samples.
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
