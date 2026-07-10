#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-

#    dea.py
#
#    Utilities to perform differential expression analysis (DEA).
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
__doc__ = \
	"Utilities to perform differential expression analysis (DEA)."


#######################################################################


# Import from the standard library.
import logging as log
from typing import Iterator, Optional

# Import from third-party libraries.
import numpy as np
import pandas as pd
from scipy.stats import nbinom, poisson
from statsmodels.stats.multitest import multipletests
import torch

# Import from 'bulkdgd'.
from . import _util


#######################################################################


# Get the module's logger.
logger = log.getLogger(__name__)


########################## PRIVATE FUNCTIONS ##########################


def _yield_p_values(obs_counts: torch.Tensor,
                    pred_means: torch.Tensor,
                    r_values: Optional[torch.Tensor] = None,
                    resolution: Optional[int] = None) -> \
                        Iterator[tuple[float, np.ndarray, np.ndarray]]:
    """For each gene, yield the p-value, the points at which the
    log-probability mass function was evaluated, and the values of
    the log-probability mass function at those points.

    Parameters
    ----------
    obs_counts : :class:`torch.Tensor`
        A one-dimensional tensor containing the observed counts
        for the genes.

    pred_means : :class:`torch.Tensor`
        A one-dimensional tensor containing the predicted scaled
        mean counts for the genes.

    r_values : :class:`torch.Tensor`, optional
        A one-dimensional tensor containing the r-values for the genes.

    resolution : :class:`int`, optional
        The resolution at which to perform the p-value calculation.

    Yields
    ------
    p_val : :class:`float`
        The calculated p-values for all genes.

    k : :class:`numpy.ndarray`
        A two-dimensional array containing the points at which the
        log-probability mass function was evaluated for each gene.

    pmf : :class:`numpy.ndarray`
        A two-dimensional array containing the values of the
        log-probability mass function evaluated at each ``k`` point
        for each gene.
    """

    # For each gene's observed count, predicted mean count, and r-value
    for i, (obs_count_gene_i, pred_mean_gene_i) \
        in enumerate(zip(obs_counts, pred_means)):

        #-------------------------------------------------------------#

        # If negative binomial distributions were used to model the
        # genes' counts
        if r_values is not None:

            # Get the r-value for the current gene.
            r_value_gene_i = r_values[i]

            # Calculate the probability of "success" from the r-value
            # (number of successes till the experiment is stopped) from
            # the mean of the negative binomial. This is a single
            # value, and is calculated from the mean 'm' of the
            # negative binomial as:
            #
            # m = r(1-p) / p
            # mp = r - rp
            # mp + rp = r
            # p(m+r) = r
            # p = r / (m + r)
            p_i = pred_mean_gene_i / \
                  (pred_mean_gene_i + r_value_gene_i)

            #---------------------------------------------------------#

            # Set the percent point function to be calculated.
            ppf_dist = nbinom

            # Set the options to calculate the percent point function.
            #
            # Since SciPy's negative binomial function is implemented
            # as a function of the number of failures, their 'p' is
            # equivalent to our '1-p' and their 'n' is our 'r'.
            ppf_options = \
                {"q" : 0.99999,
                 "n" : float(r_value_gene_i),
                 "p" : 1 - p_i}

        #-------------------------------------------------------------#

        # If Poisson distributions were used to model the genes' counts
        else:

            # Set the percent point function to be calculated.
            ppf_dist = poisson

            # Set the options to calculate the percent point function.
            ppf_options = \
                {"q" : 0.99999,
                 "mu" : float(pred_mean_gene_i)}

        #-------------------------------------------------------------#
        
        # Get the count value at which the value of the percent
        # point function (the inverse of the cumulative mass
        # function) is 0.99999.
        #
        # This corresponds to the value in the probability mass
        # function beyond which lies 0.00001 of the mass. This is a
        # single value.
        tail = float(ppf_dist.ppf(**ppf_options))

        # A negative binomial with a vanishingly small r-value (as can
        # happen for genes the decoder predicts as essentially
        # unexpressed) drives 'p' to (numerically) exactly 1, which is
        # a degenerate parameterization SciPy's 'nbinom.ppf' resolves
        # to NaN instead of a finite value -- even though the
        # distribution's mass is, in this limit, concentrated at 0
        # (consistent with the finite, non-NaN 'tail' SciPy itself
        # returns for the same gene at slightly less extreme
        # parameter values). Fall back to 0 in that case rather than
        # letting a single near-zero-dispersion gene crash the entire
        # sample's DEA.
        if not np.isfinite(tail):
            tail = 0.0

        #-------------------------------------------------------------#

        # If no resolution was passed
        if resolution is None:
            
            # We are going to evaluate the log-probability mass
            # function at steps of width 1 (exact calculation).
            #
            # The result 'k' is a 1D tensor whose length is equal to
            # 'tail' since we are taking steps of size 1 starting from
            # 0 and ending in 'tail'.
            k = np.arange(\
                    start = 0,
                    stop = tail,
                    step = 1)
        
        # Otherwise
        else:
            
            # We are going to evaluate the log-probability mass
            # function at steps of width 'resolution' (rounded).
            #
            # The result 'k' is a 1D tensor whose length is equal to
            # 'resolution' steps (rounded) between 0 and 'tail'.
            k = np.linspace(\
                    start = 0,
                    stop = int(tail),
                    num = int(resolution)).round()

        #-------------------------------------------------------------#

        # If negative binomial distributions were used to model the
        # genes' counts
        if r_values is not None:

            # Get the log-probability mass distribution to be used.
            log_prob_mass_dist = _util.log_prob_mass_nb

            # Get the options for the log-probability mass function to
            # be used when calculating the PMF.
            log_prob_mass_pmf_options = \
                {"k" : k,
                 "m" : pred_mean_gene_i,
                 "r" : r_value_gene_i}

            # Get the options for the log-probability mass function to
            # be used when calculating the value of the mass function
            # for the actual value of the count for gene 'i'.
            log_prob_mass_count_options = \
                {"k" : obs_count_gene_i,
                 "m" : pred_mean_gene_i,
                 "r" : r_value_gene_i}

        #-------------------------------------------------------------#

        # If Poisson distributions were used to model the genes' counts
        else:

            # Get the log-probability mass distribution to be used.
            log_prob_mass_dist = _util.log_prob_mass_poisson

            # Get the options for the log-probability mass function to
            # be used when calculating the PMF.
            log_prob_mass_pmf_options = \
                {"k" : k,
                 "m" : pred_mean_gene_i}

            # Get the options for the log-probability mass function to
            # be used when calculating the value of the mass function
            # for the actual value of the count for gene 'i'.
            log_prob_mass_count_options = \
                {"k" : obs_count_gene_i,
                 "m" : pred_mean_gene_i}

        #-------------------------------------------------------------#

        # Find the value of the log-probability mass function for
        # each point in the 'k' tensor.
        #
        # The output is a 1D tensor whose length is equal to
        # the length of 'k'.
        pmf = \
            log_prob_mass_dist(**log_prob_mass_pmf_options).astype(
                np.float64)

        #-------------------------------------------------------------#

        # Find the value of the log-probability mass function for the
        # actual value of the count for gene 'i', 'obs_count_gene_i'.
        #
        # The output is a single value.
        prob_obs_count_gene_i = \
            log_prob_mass_dist(**log_prob_mass_count_options).astype(
                np.float64)

        #-------------------------------------------------------------#

        # Find the probability that a point falls lower than the
        # observed count (= sum over all values of 'k' lower than
        # the value of the log-probability mass function at the actual
        # count value. Exponentiate it since for now we dealt with
        # log-probability masses, and we want the actual probability.
        #
        # The output is a single value.
        lower_probs = \
            np.exp(pmf[pmf <= prob_obs_count_gene_i]).sum()

        #-------------------------------------------------------------#

        # Get the total mass of the "discretized" probability mass
        # function we computed above.
        norm_const = np.exp(pmf).sum()

        #-------------------------------------------------------------#
        
        # Calculate the p-value as the ratio between the probability
        # mass associated to the event where a point falls lower than
        # the observed count and the total probability mass.
        p_val = lower_probs / norm_const

        #-------------------------------------------------------------#

        # Yield the p-value found for the current gene, the 'k' values
        # at which the log-probability mass was evaluated, and the
        # value of the log-probability mass at each value 'k' for the
        # gene.
        yield p_val, k, pmf


########################## PUBLIC FUNCTIONS ###########################


def get_p_values(obs_counts: pd.Series,
                 pred_means: pd.Series,
                 r_values: Optional[pd.Series] = None,
                 resolution: Optional[int] = None,
                 return_pmf_values: bool = False,
                 pseudocount: int = 1) -> \
                    tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """Given the observed gene counts in a single sample, and the
    predicted mean gene counts in a single sample, calculate the
    p-value associated with the predicted mean of each distribution
    modeling a gene's counts by comparing it to the actual gene count.

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

    resolution : :class:`int`, optional
        How accurate the calculation of the p-values should be.

        The ``resolution`` corresponds to the coarseness of the sum
        over the probability mass function of each distribution
        to compute the corresponding p-value.

        The higher the ``resolution``, the more accurate (and more
        computationally expensive) the calculation of the p-values
        will be.

        If not passed, the calculation will be exact.

    return_pmf_values : :class:`bool`, :obj:`False`
        Return the points at which the log-probability mass function
        was evaluated and the corresponding values of the log-
        probability mass function, together with the p-values.

        Set it to ``True`` only if you have a low resolution
        (for instance, ``1e3`` or lower) or a lot of RAM available
        since the arrays containing the points at which the log-
        probability mass function was evaluated and the corresponding
        values of the function will contain ``resolution``
        floating-point numbers for each gene.
    
    pseudocount : :class:`int`, ``1``
        A pseudocount to add to both the predicted means and observed
        counts to avoid artifacts.

    Returns
    -------
    p_values : :class:`pandas.Series`
        A series containing one p-value per gene.
    
    ks : :class:`pandas.DataFrame`
        A data frame containing the count values at which the log-
        probability mass function was evaluated to compute the
        p-values.

        The data frame has as many rows as the number of genes and as
        many columns as the number of count values.

        This is an empty data frame if ``return_pmf_values`` is
        ``False``.
    
    pmfs : :class:`numpy.ndarray`
        A data frame containing the value of the log-probability mass
        function for each count value at which it was evaluated.

        The data frame has as many rows as the number of genes and as
        many columns as the number of count values.

        This is an empty data frame if ``return_pmf_values`` is
        ``False``.
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

        # Create a tensor containing only those columns containing gene
        # expression data for the predicted r-values - the 'loc' syntax
        # should return the columns in the order specified by the
        # selection.
        r_values = pd.to_numeric(r_values.loc[genes_r_values]).values

    #-----------------------------------------------------------------#

    # Create a tensor containing only those columns containing gene
    # expression data for the observed gene counts - the 'loc' syntax
    # should return the columns in the order specified by the
    # selection.
    obs_counts = pd.to_numeric(obs_counts.loc[genes_obs]).values

    #-----------------------------------------------------------------#

    # Create a tensor containing only those columns containing gene
    # expression data for the predicted mean counts - the 'loc' syntax
    # should return the columns in the order specified by the
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

    # Add a pseudocount of 1 to the predicted means.
    pred_means = pred_means + pseudocount

    # Add a pseudocount of 1 to the observed counts.
    obs_counts = obs_counts + pseudocount

    #-----------------------------------------------------------------#

    # Yield the p-values computed per gene in the current sample, the
    # 'k' points at which the log-probability mass function was
    # calculated, and the values of the function at those points.
    results = _yield_p_values(obs_counts = obs_counts,
                              pred_means = pred_means,
                              r_values = r_values,
                              resolution = resolution)
    
    #-----------------------------------------------------------------#

    # Create an empty list of lists to store the final results.
    final_results = [[], [], []]

    # For each:
    # - p-value
    # - Associated 'k' points at which the log-probability mass
    #   function was evaluated.
    # - Associated values of the log-probability mass function
    #   evaluated at the 'k' points.
    for p_val, k, pmf in results:

        # Save the p-value to the final results.
        final_results[0].append(p_val)

        # If we need to return the points at which the log-probability
        # mass function was evaluated and the values of the function
        # itself       
        if return_pmf_values:

            # Add them to the final results.
            final_results[1].append(k)
            final_results[2].append(pmf)

    #-----------------------------------------------------------------#

    # Convert the list of p-values into a series.
    series_p_values = pd.Series(np.array(final_results[0]))

    # Set the index of the series equal to the genes' names.
    series_p_values.index = genes_obs

    # Set the series' name.
    series_p_values.name = "p_value"

    #-----------------------------------------------------------------#

    # If we saved the 'k' values
    if final_results[1]:

        # Convert the array of 'k' values into a data frame.
        df_ks = pd.DataFrame(np.stack(final_results[1]))

        # Set the index of the data frame equal to the genes' names.
        df_ks.index = genes_obs

    # Otherwise
    else:

        # Create an empty data frame.
        df_ks = pd.DataFrame()

    #-----------------------------------------------------------------#

    # If we saved the values of the log-probability mass function
    if final_results[2]:

        # Convert the array of values into a data frame.
        df_pmfs = pd.DataFrame(np.stack(final_results[2]))

        # Set the index of the data frame equal to the genes' names.
        df_pmfs.index = genes_obs

    # Otherwise
    else:

        # Create an empty data frame.
        df_pmfs = pd.DataFrame()

    #-----------------------------------------------------------------#
    
    # Return the series and the data frames.
    return series_p_values, df_ks, df_pmfs


def get_q_values(p_values: pd.Series,
                 alpha: float = 0.05,
                 method: str = "fdr_bh") -> \
                    tuple[pd.Series, pd.Series]:
    """Get the q-values associated with a set of p-values.

    The q-values are the p-values adjusted for the false discovery
    rate.

    Parameters
    ----------
    p_values : :class:`pandas.Series`
        The p-values.

    alpha : :class:`float`, ``0.05``
        The family-wise error rate for the calculation of the q-values.

    method : :class:`str`, ``"fdr_bh"``
        The method used to adjust the p-values. The available methods
        are listed in the documentation for
        ``statsmodels.stats.multitest.multipletests``.

    Returns
    -------
    q_values : :class:`pandas.Series`
        A series containing the q-values.

        The index of the series is equal to the index of the input
        series of p-values.

    rejected : :class:`pandas.Series`
        A series containing booleans indicating whether a p-value in
        the input data frame was rejected (``True``) or not
        (``False``).

        The index of the series is equal to the index of the input
        series of p-values.
    """

    # Get the genes' names from the index of the input series.
    genes = p_values.index.tolist()

    #-----------------------------------------------------------------#

    # Adjust the p-values.
    rejected, q_values, _, _ = multipletests(pvals = p_values.values,
                                             alpha = alpha,
                                             method = method)

    #-----------------------------------------------------------------#

    # Create a series for the q-values.
    series_q_values = pd.Series(q_values)

    # Set the index of the series.
    series_q_values.index = genes

    # Set the series' name.
    series_q_values.name = "q_value"

    #-----------------------------------------------------------------#

    # Create a Series for the boolean list
    series_rejected = pd.Series(rejected)

    # Set the index of the series.
    series_rejected.index = genes

    # Set the series' name.
    series_rejected.name = "is_p_value_rejected"

    #-----------------------------------------------------------------#

    # Return the q-values and the series indicating, for each p-value,
    # whether it was rejected or not.
    return series_q_values, series_rejected


def get_log2_fold_changes(obs_counts: pd.Series,
                          pred_means: pd.Series,
                          pseudocount: int = 1) -> pd.Series:
    """Get the log2-fold change of the expression of a set of genes.

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

    pseudocount : :class:`int`, ``1``
        A pseudocount to add to both the predicted means and observed
        counts to avoid artifacts.
    
    Returns
    -------
    log2_fold_changes : :class:`pandas.Series`
        The log2-fold change associated with each gene in the given
        sample.

        This is a series whose index correspond to the one of
        ``obs_counts`` and ``pred_means``.
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
            "The set of genes in 'obs_counts' and 'pred_means' " \
            "must be the same. It is assumed that the genes are " \
            "specified using their Ensembl IDs."
        raise ValueError(errstr)

    #-----------------------------------------------------------------#

    # Create a tensor containing only those columns containing gene
    # expression data for the observed gene counts - the 'loc' syntax
    # should return the columns in the order specified by the
    # selection.
    obs_counts = obs_counts.loc[genes_obs].astype("float64").values
 
    #-----------------------------------------------------------------#
    
    # Create a tensor containing only those columns containing gene
    # expression data for the predicted mean counts - the 'loc' syntax
    # should return the columns in the order specified by the
    # selection.
    pred_means = pred_means.loc[genes_obs].astype("float64").values

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

    # Get the log-fold change for each gene by dividing the predicted
    # mean count by the observed count. A small value is added
    # to ensure we do not divide by zero and avoid artifacts.
    log2_fold_changes = \
        np.log2((pred_means + pseudocount) / \
                (obs_counts + pseudocount))

    #-----------------------------------------------------------------#

    # Convert the tensor into a series.
    series_log2_fold_changes = pd.Series(log2_fold_changes)

    # Set the index of the series.
    series_log2_fold_changes.index = genes_obs

    # Set the series' name.
    series_log2_fold_changes.name = "log2_fold_change"

    #-----------------------------------------------------------------#

    # Return the series.
    return series_log2_fold_changes


def get_statistics(obs_counts: pd.Series,
                   pred_means: pd.Series,
                   r_values: Optional[pd.Series] = None,
                   sample_name: Optional[str] = None,
                   statistics: list[str] = \
                        ["p_values",
                         "q_values",
                         "log2_fold_changes"],
                   resolution: Optional[int] = None,
                   alpha: float = 0.05,
                   method: str = "fdr_bh",
                   pseudocount: int = 1) -> \
                        tuple[pd.DataFrame, Optional[str]]:
    """Compute p-values, q-values, and/or log2-fold changes.

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
        The name of the sample under consideration.

        It is returned together with the results of the analysis
        to facilitate the identification of the sample when running
        the analysis in parallel for multiple samples (i.e., launching
        the function in parallel on multiple samples).

    statistics : :class:`list`, \
        {``["p_values", "q_values", "log2_fold_changes"]``}
        The statistics to be computed. By default, all of them
        will be computed.

    resolution : :class:`int`, optional
        How accurate the calculation of the p-values should be.

        The ``resolution`` corresponds to the coarseness of the sum
        over the probability mass function of each distribution
        to compute the corresponding p-value.

        The higher the ``resolution``, the more accurate (and more
        computationally expensive) the calculation of the p-values
        will be.

        If not passed, the calculation will be exact.

    alpha : :class:`float`, ``0.05``
        The family-wise error rate for the calculation of the
        q-values (adjusted p-values).

    method : :class:`str`, ``"fdr_bh"``
        The method used to calculate the q-values (in other words, to
        adjust the p-values). The available methods are listed in the
        documentation for
        ``statsmodels.stats.multitest.multipletests``.
    
    pseudocount : :class:`int`, ``1``
        A pseudocount to add to both the predicted means and observed
        counts to avoid artifacts.

    Returns
    -------
    df_stats : :class:`pandas.DataFrame`
        A data frame whose rows represent the genes on which the DEA
        was performed, and whose columns contain the statistics
        computed (p-values, q_values, log2-fold changes). If not all
        statistics were computed, the columns corresponding to the
        missing ones will be empty.

    sample_name : :class:`str` or :obj:`None`
        The name of the sample under consideration.
    """

    # Set a list of the available statistics.
    AVAILABLE_STATISTICS = \
        ["p_values", "q_values", "log2_fold_changes"]

    #-----------------------------------------------------------------#
    
    # Initialize all the statistics to None.
    p_values = None
    q_values = None
    log2_fold_changes = None

    #-----------------------------------------------------------------#

    # If no statistics were selected
    if not statistics:

        # Format a string for the available statistics.
        available_stats_str = \
            ", ".join(f"'{s}'" for s in AVAILABLE_STATISTICS)

        # Raise an error.
        errstr = \
            "The 'statistics' list should contain at least one " \
            "element. Available statistics are: " \
            f"{available_stats_str}."
        raise ValueError(errstr)

    #-----------------------------------------------------------------#

    # If the user requested the calculation of p-values
    if "p_values" in statistics:

        # Calculate the p-values. Do not return the points at which
        # the log-probability mass function was evaluated or the 
        # value of the function at these points.
        p_values, ks, pmfs = \
            get_p_values(obs_counts = obs_counts,
                         pred_means = pred_means,
                         r_values = r_values,
                         resolution = resolution,
                         return_pmf_values = False,
                         pseudocount = pseudocount)

    #-----------------------------------------------------------------#

    # If the user requested the calculation of q-values
    if "q_values" in statistics:

        # If no p-values were calculated
        if p_values is None:

            # Calculate the p-values. Do not return the points at which
            # the log-probability mass function was evaluated or the 
            # value of the function at these points.
            p_values, ks, pmfs = \
                get_p_values(obs_counts = obs_counts,
                             pred_means = pred_means,
                             r_values = r_values,
                             resolution = resolution,
                             return_pmf_values = False)

        # Calculate the q-values.
        q_values, _ = \
            get_q_values(p_values = p_values,
                         alpha = alpha,
                         method = method)

    #-----------------------------------------------------------------#

    # If the user requested the calculation of fold changes
    if "log2_fold_changes" in statistics:

        # Calculate the fold changes.
        log2_fold_changes = \
            get_log2_fold_changes(obs_counts = obs_counts,
                                  pred_means = pred_means,
                                  pseudocount = pseudocount)

    #-----------------------------------------------------------------#

    # Get the results for the statistics that were computed.
    stats_results = \
        [stat if stat is not None else pd.Series()
         for stat in (p_values, q_values, log2_fold_changes)]

    #-----------------------------------------------------------------#

    # Create a data frame from the statistics computed.
    df_stats = pd.concat(stats_results,
                         axis = 1)

    #-----------------------------------------------------------------#

    # Return the data frame and the name of the sample.
    return df_stats, sample_name


def get_significant_genes(df_stats: pd.DataFrame,
                          p_val: float = 0.05,
                          q_val: float = 0.05,
                          log2_fold_change: float = 2) -> pd.DataFrame:
    """Get the genes that are significant at a given significance
    level.

    Parameters
    ----------
    df_stats : :class:`pandas.DataFrame`
        A data frame whose rows represent the genes on which the DEA
        was performed, and whose columns contain the statistics
        computed (p-values, q_values, log2-fold changes).

    p_val : :class:`float`, ``0.05``
        The p-value threshold to consider a gene as significant.
    
    q_val : :class:`float`, ``0.05``
        The q-value threshold to consider a gene as significant.
    
    log2_fold_change : :class:`float`, ``2``
        The log2-fold change threshold to consider a gene as
        significant. This value and its negative are used as the
        thresholds for the log2-fold change.
    
    Returns
    -------
    df_significant_genes : :class:`pandas.DataFrame`
        A data frame containing the genes that are significant at
        the given significance levels.
    """

    # Create a copy of the input data frame to avoid modifying it.
    df_stats = df_stats.copy()
    
    # Get the genes satisfying all the conditions.
    df_significant_genes = \
        df_stats[(df_stats["p_value"] <= p_val) & \
                 (df_stats["q_value"] <= q_val) & \
                 (df_stats["log2_fold_change"].abs() \
                    >= log2_fold_change)]
    
    #------------------------------------------------------------------#
    
    # Return the data frame.
    return df_significant_genes


def get_enrichment_scores(df_significant_genes: pd.DataFrame,
                          genes_sets: dict[str, list[str]],
                          genes_all: list[str]) -> pd.DataFrame:
    """Compute the enrichment scores for a set of genes of interest.

    Parameters
    ----------
    df_significant_genes : :class:`pandas.DataFrame`
        A data frame containing the genes that are significant at
        the given significance levels.

        The index of the data frame is equal to the genes' names.
    
    genes_sets : :class:`dict`
        A dictionary containing sets of genes of interest.
    
    genes_all : :class:`list`
        A list containing all the genes in the analysis.
    
    Returns
    -------
    df_e_scores : :class:`pandas.DataFrame`
        A data frame containing the enrichment scores for each sample.
    """

    # Initialize a list to store the enrichment scores for each
    # sample and the associated genes.
    e_scores = []
    
    #-----------------------------------------------------------------#

    # For each set of genes of interest
    for genes_set_name, genes_set in genes_sets.items():

        # Get the list of significant genes.
        genes = df_significant_genes.index.tolist()

        # Try to compute the enrichment score
        try:

            # Compute the enrichment score as the product of the
            # number of genes above the threshold and the number
            # of drivers divided by the product of the number of
            # genes above the threshold and the number of unique
            # drivers.
            e_score = \
                (len(set(genes) & set(genes_set)) * \
                    len(genes_all)) / \
                (len(set(genes)) * len(set(genes_set)))

            # Add the enrichment score to the list.
            e_scores.append({"genes_set" : genes_set_name,
                             "num_genes_in_set" : len(genes_set),
                             "num_genes_significant" : len(genes),
                             "e_score" : e_score})
        
        # If there is a division by zero
        except ZeroDivisionError:
            
            # Add a missing value to the list.
            e_scores.append({"genes_set" : genes_set_name,
                             "num_genes_in_set" : len(genes_set),
                             "num_genes_significant" : len(genes),
                             "e_score" : np.nan})
    
    #-----------------------------------------------------------------#
    
    # Create a data frame from the enrichment scores.
    df_e_scores = pd.DataFrame(e_scores)

    #-----------------------------------------------------------------#

    # Return the data frame.
    return df_e_scores


def perform_dea(obs_counts: pd.DataFrame,
                pred_means: pd.DataFrame,
                r_values: Optional[pd.DataFrame] = None,
                resolution: Optional[int] = None,
                alpha: float = 0.05,
                method: str = "fdr_bh",
                p_val: float = 0.05,
                q_val: float = 0.05,
                log2_fold_change: float = 2,
                genes_sets: Optional[dict[str, list[str]]] = None,
                pseudocount: int = 1) -> \
                    tuple[dict[str, pd.DataFrame],
                          pd.Series,
                          pd.DataFrame]:
    """Perform differential expression analysis (DEA) on multiple
    samples.

    Parameters
    ----------
    obs_counts : :class:`pandas.DataFrame`
        The observed gene counts in multiple sample.

        This is a data frame whose index contains the samples's names,
        and the columns contain either the genes' Ensembl IDs or
        names of fields containing additional information about the
        samples.
    
    pred_means : :class:`pandas.DataFrame`
        The predicted means of the distributions modelling the genes'
        counts in each sample.

        This is a data frame whose index contains the samples' names,
        and the columns contain either the genes' Ensembl IDs or
        names of fields containing additional information about the
        samples.
    
    r_values : :class:`pandas.DataFrame`, optional
        The predicted r-values of the negative binomial distributions
        modelling the genes' counts in each sample, if the genes'
        counts were modelled using negative binomial distributions.

        This is a data frame whose index contains the samples' names,
        and the columns contain either the genes' Ensembl IDs or
        names of fields containing additional information about the
        samples.

        If ``r_values`` is not provided, it is assumed that the genes'
        counts were modelled using Poisson distributions.
    
    resolution : :class:`int`, optional
        How accurate the calculation of the p-values should be.

        The ``resolution`` corresponds to the coarseness of the sum
        over the probability mass function of each distribution
        to compute the corresponding p-value.

        The higher the ``resolution``, the more accurate (and more
        computationally expensive) the calculation of the p-values
        will be.

        If not passed, the calculation will be exact.
    
    alpha : :class:`float`, ``0.05``
        The family-wise error rate for the calculation of the
        q-values (adjusted p-values).
    
    method : :class:`str`, ``"fdr_bh"``
        The method used to calculate the q-values (in other words, to
        adjust the p-values). The available methods are listed in the
        documentation for
        ``statsmodels.stats.multitest.multipletests``.
    
    p_val : :class:`float`, ``0.05``
        The p-value threshold to consider a gene as significant.
    
    q_val : :class:`float`, ``0.05``
        The q-value threshold to consider a gene as significant.
    
    log2_fold_change : :class:`float`, ``2``
        The log2-fold change threshold to consider a gene as
        significant. This value and its negative are used as the
        thresholds for the log2-fold change.
    
    genes_sets : :class:`dict`, optional
        A dictionary containing sets of genes of interest.

        The keys are the names of the gene sets, and the values are
        lists of genes.
    
    pseudocount : :class:`int`, ``1``
        A pseudocount to add to both the predicted means and observed
        counts to avoid artifacts.
    
    Returns
    -------
    dfs_stats : :class:`dict`
        A dictionary containing the data frames with the statistics
        for each sample.
    
    series_significant_genes : :class:`pandas.Series`
        A series containing the significant genes per sample.
    
    df_e_scores : :class:`pandas.DataFrame`
        A data frame containing the enrichment scores for each sample.

        If no gene sets were passed, the data frame will be empty.
    """

    # Initialize an empty dictionary to store the data frames
    # containing the statistics for each sample.
    dfs_stats = {}

    # Initialize an empty dictionary to store the significant genes
    # for each sample.
    significant_genes = {}

    # Initialize an empty list to store the enrichment scores for each
    # sample.
    e_scores = []

    #-----------------------------------------------------------------#

    # Get the sample's names.
    obs_counts_names = obs_counts.index.tolist()

    #-----------------------------------------------------------------#
    
    # For each sample
    for sample_name in obs_counts_names:

        # Set the options to perform the analysis.
        dea_options = \
            {"obs_counts" : obs_counts.loc[sample_name,:],
             "pred_means" : pred_means.loc[sample_name,:],
             "sample_name" : sample_name,
             "statistics" : \
                ["p_values", "q_values", "log2_fold_changes"],
             "resolution" : resolution,
             "alpha" : alpha,
             "method" : method,
             "pseudocount" : pseudocount}

        # If r-values were passed
        if r_values is not None:

            # Add the r-values for the current sample.
            dea_options["r_values"] = r_values.loc[sample_name,:]

        # Calculate the statistics for the current sample.
        df_stats, _ = get_statistics(**dea_options)

        #-------------------------------------------------------------#

        # Add a column containing the observed counts.
        df_stats["obs_counts"] = obs_counts.loc[sample_name,:]

        # Add a column containing the predicted means.
        df_stats["dgd_mean"] = pred_means.loc[sample_name,:]

        #-------------------------------------------------------------#

        # If the r-values were passed
        if r_values is not None:
            
            # Add a column containing the r-values
            df_stats["dgd_r"] = r_values.loc[sample_name,:]
        
        #-------------------------------------------------------------#
        
        # Add the data frame to the dictionary of data frames.
        dfs_stats[sample_name] = df_stats
        
        #-------------------------------------------------------------#

        # Get the significant genes.
        df_significant_genes = \
            get_significant_genes(df_stats,
                                  p_val = p_val,
                                  q_val = q_val,
                                  log2_fold_change = log2_fold_change)
        
        #-------------------------------------------------------------#

        # Add the significant genes to the final dictionary.
        significant_genes[sample_name] = \
            ".".join(df_significant_genes.index.tolist())
        
        #-------------------------------------------------------------#

        # If gene sets were passed
        if genes_sets is not None:

            # Get the enrichment scores.
            e_scores = \
                get_enrichment_scores(\
                    df_significant_genes = df_significant_genes,
                    genes_sets = genes_sets,
                    genes_all = df_stats.index.tolist())
            
            # Add a column containing the sample's name.
            e_scores["sample_name"] = sample_name

            # Append the enrichment scores to the list.
            e_scores.append(e_scores)

    #-----------------------------------------------------------------#

    # Create a series containing the significant genes per sample.
    series_significant_genes = pd.Series(significant_genes)
        
    #-----------------------------------------------------------------#

    # If gene sets were passed
    if genes_sets is not None:

        # Create a data frame with the enrichment scores.
        df_e_scores = pd.DataFrame(e_scores)

        # Set the 'sample_name' column as the index.
        df_e_scores.set_index("sample_name", inplace = True)

        # Append all the columns found in the original 'obs_counts'
        # data frame to the data frame containing the enrichment
        # scores.
        for col in [c for c in obs_counts.columns \
                    if not c.startswith("ENSG")]:

            # If the column is not already in the data frame
            if col not in df_e_scores.columns:

                # Add it to the data frame.
                df_e_scores[col] = obs_counts[col]

    # Otherwise
    else:

        # Create an empty data frame.
        df_e_scores = pd.DataFrame()
    
    #-----------------------------------------------------------------#

    # Return the data frames containing the statistics, the series
    # containing the significant genes, and the data frame containing
    # the enrichment scores.
    return dfs_stats, series_significant_genes, df_e_scores