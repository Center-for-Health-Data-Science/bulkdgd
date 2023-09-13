#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-

#    dea.py
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


# Description of the module
__doc__ = \
	"Utilities to perform differential expression analysis (DEA)."


# Standard library
import logging as log
# Third-party packages
import numpy as np
import pandas as pd
from scipy.stats import nbinom
from statsmodels.stats.multitest import multipletests
import torch


# Get the module's logger
logger = log.getLogger(__name__)


#------------------------- Private functions -------------------------#


def _log_prob_mass(k,
                   m,
                   r):
    """Compute the logarithm of the probability mass
    for a set of negative binomial distribution.

    Thr formula used to compute the logarithm of the
    probability mass is:

    .. math::

       logPDF_{NB(k,m,r)} &=
       log\\Gamma(k+r) - log\\Gamma(r) - log\\Gamma(k+1) \\\\ 
       &+ k \\cdot log(m \\cdot c + \\epsilon) +
       r \\cdot log(r \\cdot c)

    Where :math:`\\epsilon` is a small value to prevent
    underflow/overflow, and :math:`c` is equal to
    :math:`\\frac{1}{r+m+\\epsilon}`.

    The derivation of this formula from the non-logarithmic
    formulation of the probability mass function of the
    negative binomial distribution can be found below.

    Parameters
    ----------
    k : ``torch.Tensor``
        "Number of successes" seen before
        stopping the trials (each value in the
        tensor corresponds to the number of successes
        of a different negative binomial).

    m : ``torch.Tensor``
        Means of the negative binomials (each value in
        the tensor corresponds to the mean of a
        different negative binomial).

    r : ``torch.Tensor``
        "Number of failures" after which the trials
        end (each value in the tensor corresponds to
        the number of failures of a different negative
        binomial).

    Returns
    -------
    x : ``torch.Tensor``
        The log-probability mass of the negative binomials.
        This is a 1D tensor whose length corresponds to the
        number of negative binomials distributions considered,
        and each value in the tensor corresponds to the
        log-probability mass of a different negative binomial.

    Notes
    -----
    Here, we show how we derived the formula for the logarithm
    of the probability mass of the negative binomial
    distribution.

    We start from the non-logarithmic version of the
    probability mass for the negative binomial, which is:

    .. math::

       PDF_{NB(k,m,r)} = \
       \\binom{k+r-1}{k} (1-p)^{k} p^{r}

    However, since:

    * :math:`1-p` is equal to :math:`\\frac{m}{r+m}`
    * :math:`p` is equal to :math:`\\frac{r}{r+m}`
    * :math:`k+r-1` can be rewritten in terms of the
      gamma function as :math:`\\Gamma(k+r)`
    * :math:`k` can also be rewritten as
      :math:`\\Gamma(r) \\cdot k!`

    The formula becomes:

    .. math::

       PDF_{NB(k,m,r)} = \
       \\binom{\\Gamma(k+r)}{\\Gamma(r) \\cdot k!}
       \\left( \\frac{m}{r+m} \\right)^k
       \\left( \\frac{r}{r+m} \\right)^r

    However, :math:`k!` can be also be rewritten as
    :math:`\\Gamma(k+1)`, resulting in:

    .. math::

       PDF_{NB(k,m,r)} = \
       \\binom{\\Gamma(k+r)}{\\Gamma(r) \\cdot 
       \\Gamma(k+1)}
       \\left( \\frac{m}{r+m} \\right)^k
       \\left( \\frac{r}{r+m} \\right)^r

    Then, we get the natural logarithm:
    
    .. math::

       logPDF_{NB(k,m,r)} &= \
       log\\Gamma(k+r) - log\\Gamma(r) - log\\Gamma(k+1) \\\\
       &+ k \\cdot log \\left( \\frac{m}{r+m} \\right) +
       r \\cdot log \\left( \\frac{r}{r+m} \\right)
    
    Here, we are adding a small value :math:`\\epsilon` to
    prevent underflow/overflow:

    .. math::

       logPDF_{NB(k,m,r)} &= \
       log\\Gamma(k+r) - log\\Gamma(r) - log\\Gamma(k+1) \\\\
       &+ k \\cdot
       log \\left( m \\cdot \\frac{1}{r+m+\\epsilon} 
       + \\epsilon \\right) +
       r \\cdot
       log \\left( r \\cdot \\frac{1}{r+m+\\epsilon}
       \\right)

    Finally, we substitute :math:`\\frac{1}{r+m+\\epsilon}`
    with :math:`c` and we obtain:

    .. math::

       logPDF_{NB(k,m,r)} &= \
       log\\Gamma(k+r) - log\\Gamma(r) - log\\Gamma(k+1) \\\\
       &+ k \\cdot
       log \\left( m \\cdot c + \\epsilon \\right) +
       r \\cdot
       log \\left( r \\cdot c \\right)
    """

    # Convert the "number of successes" to a double-precision
    # floating point number
    k = k.double()
    
    # Set a small value used to prevent underflow and
    # overflow
    eps = 1.e-10
    
    # Set a constant used later in the equation defining
    # the log-probability mass
    c = 1.0 / (r + m + eps)
    
    # Get the log-probability mass of the negative binomials.
    #
    # The non-log version would be:
    #
    # NB(k,m,r) = \
    #   gamma(k+r) / (gamma(r) * k!) *
    #   (m/(r+m))^k *
    #   (r/(r+m))^r
    #
    # Since k! can be rewritten as gamma(k+1):
    #
    # NB(k,m,r) = \
    #   gamma(k+r) / (gamma(r) * gamma(k+1)) *
    #   (m/(r+m))^k *
    #   (r/(r+m))^r
    #
    # Getting the natural logarithm:
    #
    # log(NB(k,m,r)) = \
    #   lgamma(k+r) - lgamma(r) - lgamma(k+1) +
    #   k * log(m * 1/(r+m)) +
    #   r * log(r * 1/(r+m))
    #
    # Here, we are adding the small ``eps`` to
    # prevent underflow/overflow:
    #
    # log(NB(k,m,r)) = \
    #   lgamma(k+r) - lgamma(r) - lgamma(k+1) +
    #   k * log(m * 1/(r+m+eps) + eps) +
    #   r * log(r * 1/(r+m+eps))
    #
    # Substituting 1/(r+m+eps) with c:
    #
    # log(NB(k,m,r)) = \
    #   lgamma(k+r) - lgamma(r) - lgamma(k+1) +
    #   k * log(m * c + eps) +
    #   r * log(r * c)
    x = \
        torch.lgamma(k+r) - torch.lgamma(r) - \
        torch.lgamma(k+1) + k*torch.log(m*c+eps) + \
        r*torch.log(r*c)
    
    # Return the log-probability mass
    return x


def _rescale(means,
             scaling_factors):
    """Rescale the means of the negative binomial
    distributions.

    Parameters
    ----------
    means : ``torch.Tensor``
        A 1D tensor containing the means of the negative
        binomials o be rescaled.

        In the tensor, each value represents the 
        mean of a different negative binomial.

    scaling_factors : ``torch.Tensor``
        The scaling factors.

        This is a 1D tensor whose length is equal to the
        number of scaling factors to be used to rescale
        the means.

    Returns
    -------
    ``torch.Tensor``
        The rescaled means.

        This is a 1D tensor whose length is equal to the number
        of negative binomials whose means were rescaled.
    """
    
    # Return the rescaled values by multiplying the means
    # by the scaling factors
    return means * scaling_factors


def get_p_values(obs_counts,
                 pred_means,
                 r_values,
                 resolution = None):
    """Calculate the p-value associated to the predicted mean
    of each negative binomial by comparing it to the actual
    gene count for a single sample.

    Parameters
    ----------
    obs_counts : ``pandas.Series``
        The observed gene counts in a single sample.

        This is a series whose index contains
        either the genes' Ensembl IDs or names of fields 
        containing additional information about the sample.

    pred_means : ``pandas.Series``
        The (rescaled) predicted means of the negative binomials
        modeling the gene counts for a single sample.

        This is a series whose index contains
        either the genes' Ensembl IDs or names of fields 
        containing additional information about the sample.

    r_values : ``pandas.Series``
        A series containing one r-value for each negative binomial
        (= one r-value for each gene).

    resolution : ``int``, optional
        How accurate the calculation of the p-values should be.

        The ``resolution`` corresponds to the coarseness of the sum
        over the probability mass function of each negative binomial
        to compute the corresponding p-value.

        The higher the ``resolution``, the more accurate the
        calculation of the p-values.

        If ``None`` (the default), the calculation will be exact
        (but it will be more computationally expensive).

    Returns
    -------
    df_p_values : ``pandas.Series``
        A 1D array containing one p-value per gene.
    
    ks : ``numpy.ndarray``
        A 2D array containing the count values at 
        which the probability mass function was evaluated
        to compute the p-values.

        The array has as many
        rows as the number of genes and as many columns as
        the number of count values.
    
    pmfs : ``numpy.ndarray``
        A 2D array containing the value of the
        probability mass function for each count value
        at which it was evaluated. The array has as many
        rows as the number of genes and as many columns as
        the number of count values.
    """


    #------------------------ Check the genes ------------------------#


    # Get the names of the cells containing gene expression
    # data from the original series for the observed gene counts
    genes_obs = \
        [col for col in obs_counts.index \
         if col.startswith("ENSG")]

    # Get the names of the cells containing gene expression
    # data from the original series for the predicted means
    genes_pred = \
        [col for col in pred_means.index \
         if col.startswith("ENSG")]

    # Get the names of the cells containing r-values from the
    # series of r-values
    genes_r_values = r_values.index

    # Check that the lists contain the same genes
    if set(genes_obs) != set(genes_pred) != set(r_values):

        # Raise an error
        errstr = \
            "The set of genes in 'obs_counts', 'pred_means', " \
            "and 'r_values' must be the same."
        raise ValueError(errstr)


    #------------------ Preproces the samples' data ------------------#


    # Create a tensor with only those columns containing gene
    # expression data for the observed gene counts
    obs_counts = \
        torch.Tensor(pd.to_numeric(obs_counts.loc[genes_obs]).values)


    #---------------- Preprocess the decoder outputs -----------------#


    # Create a tensor with only those columns containing gene
    # expression data for the predicted means - the 'loc' should
    # return the selected columns in the correct order
    pred_means = \
        torch.Tensor(pd.to_numeric(pred_means.loc[genes_obs]).values)


    #-------------------- Preprocess the r-values --------------------#


    # Sort the r-values so be sure they are in the same order
    # that the genes in the observed counts/predicted genes
    r_values = \
        torch.Tensor(r_values.reindex(index = genes_obs).values)


    #----------------------- Get the p-values ------------------------#


    # Get the mean gene counts for the sample. The output is
    # a single value
    obs_counts_mean = torch.mean(obs_counts).unsqueeze(-1)

    # Get the rescaled predicted means of the negative binomials
    # (one for each gene). This is a 1D tensor with:
    #
    # - 1st dimension: the dimensionality of the output (= gene)
    #                  space
    pred_means = _rescale(pred_means,
                          obs_counts_mean)

    # Create an empty list to store the p-valued computed per gene
    # in the current sample, the value of the probability mass
    # function, and the 'k'
    results = []

    # For each gene's (rescaled) predicted mean count, observed
    # count, and r-value
    for pred_mean_gene_i, obs_count_gene_i, r_value_i \
        in zip(pred_means, obs_counts, r_values):


        #----------------------- Calculate 'p' -----------------------#


        # Calculate the probability of "success" from the r-value
        # (number of successes till the experiment is stopped) and
        # the mean of the negative binomial. This is a single value,
        # and is calculated from the mean 'm' as:
        #
        # m = r(1-p) / p
        # mp = r - rp
        # mp + rp = r
        # p(m+r) = r
        # p = r / (m + r)
        p_i = pred_mean_gene_i.item() / \
              (pred_mean_gene_i.item() + r_value_i.item())


        #-------------- Get the tail value for the sum ---------------#

        
        # Get the count value at which the value of the percent
        # point function (the inverse of the cumulative mass
        # function) is 0.99999. This corresponds to the value in
        # the probability mass function beyond which lies
        # 0.00001 of the mass. This is a single value.
        #
        # Since SciPy's negative binomial function is implemented
        # as function of the number of failures, their 'p' is
        # equivalent to our '1-p' and their 'n' is our 'r'
        tail = nbinom.ppf(q = 0.99999,
                          n = r_value_i.item(),
                          p = 1 - p_i).item()
        

        #---------------- Get the probability masses -----------------#


        # If no resolution was passed
        if resolution is None:
            
            # We are going to sum with steps of lenght 1.
            # This is a 1D tensor with length is equal to 'tail',
            # since we are taking steps of size 1 starting
            # from 0 and ending in 'tail'
            k = torch.arange(\
                    start = 0,
                    end = tail,
                    step = 1)
        
        # Otherwise
        else:
            
            # We are going to integrate with steps of length
            # 'resolution'. This is a 1D tensor whose length
            # is euqal to the number of 'resolution'-sized
            # steps between 0 and 'tail'
            k = torch.linspace(\
                    start = 0,
                    end = int(tail),
                    steps = int(resolution)).round().double()

        # Integrate to find the value of the probability mass
        # function for each count value in the 'k' tensor.
        # The output is a 1D tensor whose length is equal to
        # the length of 'k'
        pmf = \
            _log_prob_mass(\
                k = k,
                m = pred_mean_gene_i,
                r = r_value_i).to(torch.float64)


        #---------------------- Get the p-value ----------------------#


        # Find the value of the probability mass function for the
        # actual value of the count for gene 'i', 'obs_count_gene_i'.
        # This is a single value
        prob_obs_count_gene_i = \
            _log_prob_mass(\
                k = obs_count_gene_i,
                m = pred_mean_gene_i,
                r = r_value_i).to(torch.float64)

        # Find the probability that a point falls lower than the
        # observed count (= sum over all values of 'k' lower than
        # the value of the probability mass function at the actual
        # count value. Exponentiate it since for now we dealt with
        # log-probability masses, and we want the actual probability.
        # The output is a single value
        lower_probs = \
            pmf[pmf <= prob_obs_count_gene_i].exp().sum()

        # Get the total mass of the "discretized" probability mass
        # function we computed above
        norm_const = pmf.exp().sum()
        
        # Calculate the p-value as the ratio between the probability
        # mass associated to the event where a point falls lower
        # than the observed count and the total probability mass
        p_val = lower_probs / norm_const

        # Save the p-value found for the current gene, the value of
        # the probability mass function, and the k
        results.append((p_val.item(),
                       k.detach().numpy(),
                       pmf.detach().numpy()))

    # Create three lists containing all p-values, all PMFs, and
    # all 'k' values
    p_values, ks, pmfs = list(zip(*results))

    #------------------------ p-values series ------------------------#


    # Convert the p-values into a pandas Series
    series_p_values = pd.Series(np.array(p_values))

    # Set the index of the series equal to the genes' names
    series_p_values.index = genes_obs

    # Set the series' name
    series_p_values.name = "p_value"


    #--------------------- 'k' values data frame ---------------------#


    # Convert the 'k' values into a data frame
    df_ks = pd.DataFrame(np.stack(ks))

    # Set the index of the data frame equal to the genes' names
    df_ks.index = genes_obs


    #--------------------- PMF values data frame ---------------------#


    # Convert the PMF values into a data frame
    df_pmfs = pd.DataFrame(np.stack(pmfs))

    # Set the index of the data frame equal to the genes' names
    df_pmfs.index = genes_obs


    #-------------------- Return the data frames ---------------------#

    
    # Return the series/data frames
    return series_p_values, df_ks, df_pmfs


def get_q_values(p_values,
                 alpha = 0.05,
                 method = "fdr_bh"):
    """Get the q-values associated to a set of p-values. The q-values
    are the p-values adjusted for the false discovery rate.

    Parameters
    ----------
    p_values : ``pandas.Series``
        The p-values.

    alpha : ``float``, ``0.05``
        The family-wise error rate for the calculation of the
        q-values.

    method : ``str``, ``fdr_bh``
        The method used to adjust the p-values. The available
        methods are listed in the documentation for
        ``statsmodels.stats.multitest.multipletests``.

    Returns
    -------
    q_values : ``pandas.Series``
        A series containing the q-values (adjusted p-values).

        The index of the series is equal to the index of
        the p-values' series.

    rejected : ``pandas.Series``
        A series containing booleans defining whether a p-value
        in the input data frame was rejected (``True``) or
        not (``False``).

        The index of the series is equal to the index of
        the p-values' series.
    """

    # Get the genes' names from the p-values' index
    genes = p_values.index.tolist()

    # Adjust the p-values
    rejected, q_values, _, _ = multipletests(pvals = p_values.values,
                                             alpha = alpha,
                                             method = method)

    # Create a Series for the q-values
    series_q_values = pd.Series(q_values)

    # Set the index of the series
    series_q_values.index = genes

    # Set the series' name
    series_q_values.name = "q_value"

    # Create a Series for the boolean list
    series_rejected = pd.Series(rejected)

    # Set the index of the series
    series_rejected.index = genes

    # Set the series' name
    series_rejected.name = "is_p_value_rejected"

    # Return the q-values and the rejected p-values
    return series_q_values, series_rejected


def get_log2_fold_changes(obs_counts,
                          pred_means):
    """Get the log2-fold change of the gene expression.

    Parameters
    ----------
    obs_counts : ``pandas.Series``
        The observed gene counts in a single sample.

        This is a series whose index contains
        either the genes' Ensembl IDs or names of fields 
        containing additional information about the sample.

    pred_means : ``pandas.Series``
        The (rescaled) predicted means of the negative binomials
        modeling the gene counts for a single sample.

        This is a series whose index contains
        either the genes' Ensembl IDs or names of fields 
        containing additional information about the sample.
    
    Returns
    -------
    log2_fold_changes : ``torch.Tensor``
        The log2-fold change associated to each gene in the
        given sample.

        This is a series whose index correspond to the one of
        ``obs_counts`` and ``pred_means``.
    """


    #------------------------ Check the genes ------------------------#


    # Get the names of the cells containing gene expression
    # data from the original series for the observed gene counts
    genes_obs = \
        [col for col in obs_counts.index \
         if col.startswith("ENSG")]

    # Get the names of the cells containing gene expression
    # data from the original series for the predicted means
    genes_pred = \
        [col for col in pred_means.index \
         if col.startswith("ENSG")]

    # Check that the lists contain the same genes
    if set(genes_obs) != set(genes_pred):

        # Raise an error
        errstr = \
            "The set of genes in 'obs_counts' and 'pred_means' " \
            "must be the same."
        raise ValueError(errstr)


    #------------------ Preproces the samples' data ------------------#


    # Create a tensor with only those columns containing gene
    # expression data for the observed gene counts
    obs_counts = \
        torch.Tensor(obs_counts.loc[genes_obs].astype("int").values)


    #---------------- Preprocess the decoder outputs -----------------#


    # Create a tensor with only those columns containing gene
    # expression data for the predicted means - the 'loc' should
    # return the selected columns in the correct order
    pred_means = \
        torch.Tensor(pred_means.loc[genes_obs].astype("int").values)


    #--------------------- Get log2-fold changes ---------------------#
    

    # Get the log-fold change for each gene by dividing the
    # predicted mean count by the observed count. A small value
    # (1e-6) is added to ensure we do not divide by zero.
    log2_fold_changes = \
        torch.log2((pred_means + 1e-6) / (obs_counts + 1e-6))

    # Convert the tensor into a series
    series_log2_fold_changes = pd.Series(log2_fold_changes)

    # Set the index of the series
    series_log2_fold_changes.index = genes_obs

    # Set the series' name
    series_log2_fold_changes.name = "log2_fold_change"

    # Return the series
    return series_log2_fold_changes


def perform_dea(obs_counts,
                pred_means,
                sample_name = None,
                statistics = \
                    ["p_values", "q_values", "log2_fold_changes"],
                genes_names = None,
                r_values = None,
                resolution = 1,
                alpha = 0.05,
                method = "fdr_bh"):
    """Perform differential expression analysis (DEA).

    Parameters
    ----------
    obs_counts : ``pandas.Series``
        The observed gene counts in a single sample.

        This is a series whose index contains
        either the genes' Ensembl IDs or names of fields 
        containing additional information about the sample.

    pred_means : ``pandas.Series``
        The (rescaled) predicted means of the negative binomials
        modeling the gene counts for a single sample.

        This is a series whose index contains
        either the genes' Ensembl IDs or names of fields 
        containing additional information about the sample.

    sample_name : ``str``, optional
        The name of the sample under consideration. It is returned
        together with the results of the analysis to facilitate
        the identification of the sample when running the analysis
        in parallel for multiple samples (i.e., launching the
        function in parallel on multiple samples).

        There is no need to pass it if we are running the
        analysis for one sample at a time.

    statistics : ``list``,
                 {``["p_values", "q_values", "log2_fold_changes"]``}
        The statistics to be computed. By default, all of them
        will be computed (``"p_values"``, ``"q_values"``,
        ``"log2_fold_changes"``).

    genes_names : ``list``, optional
        The names of the genes on which DEA is performed. If provided,
        the genes will be the names of the rows of the output data
        frame. If not, the rows will be indexed starting from 0.

    r_values : ``torch.Tensor``
        A tensor containing one r-value for each negative binomial
        (= one r-value for each gene).

        This is a 1D tensor whose length is equal to the
        dimensionality of the gene space.

    resolution : ``int``, ``1``
        How accurate the calculation of the p-values should be.

        The ``resolution`` corresponds to the coarseness of the sum
        over the probability mass function of each negative binomial
        to compute the corresponding p-value.

        The lower the ``resolution``, the more accurate the
        calculation of the p-values.

        If ``1`` (the default), the calculation will be exact
        (but it will be more computationally expensive).

    alpha : ``float``, ``0.05``
        The family-wise error rate for the calculation of the
        q-values.

    method : ``str``, ``fdr_bh``
        The method used to calculate the q-values (in other
        words, to adjust the p-values). The available
        methods are listed in the documentation for
        ``statsmodels.stats.multitest.multipletests``.

    Returns
    -------
    ``pandas.DataFrame``
        A data frame whose rows represent the genes on which
        DEA was performed, and whose columns contain the statistics
        computed (p-values, q_values, log2-fold changes). If not
        all statistics were computed, the columns corresponding
        to the missing ones will be empty.
    """

    # Set a list of the available statistics
    AVAILABLE_STATISTICS = \
        ["p_values", "q_values", "log2_fold_changes"]
    
    # Initialize all the statistics to None
    p_values = None
    q_values = None
    log2_fold_changes = None

    # If no statistics were selected
    if not statistics:

        # Format a string for the available statistics
        available_stats_str = \
            ", ".join(f"'{s}'" for s in AVAILABLE_STATISTICS)

        # Raise an error
        errstr = \
            "The 'statistics' list should contain at least one " \
            "element. Available statistics are: " \
            f"{available_stats_str}."
        raise ValueError(errstr)


    #--------------------------- p-values ----------------------------#


    # If the user requested the calculation of p-values
    if "p_values" in statistics:

        # If no r-values were passed
        if r_values is None:

            # Raise an error
            errstr = \
                f"'r-values' are needed to compute the " \
                f"p-values."
            raise RuntimeError(errstr)

        # Calculate the p-values
        p_values, ks, pmfs = \
            get_p_values(obs_counts = obs_counts,
                         pred_means = pred_means,
                         r_values = r_values,
                         resolution = resolution)


    #--------------------------- q-values ----------------------------#


    # If the user requested the calculation of q-values
    if "q_values" in statistics:

        # If no p-values were calculated
        if p_values is None:

            # Raise an error
            errstr = \
                f"The calculation of p-values is needed to " \
                f"compute the q-values. This can be done " \
                f"by adding 'p_values' to 'stats'."
            raise RuntimeError(errstr)

        # Calculate the q-values
        q_values, rejected = \
            get_q_values(p_values = p_values)


    #----------------------- log2-fold changes -----------------------#


    # If the user requested the calculation of fold changes
    if "log2_fold_changes" in statistics:

        # Calculate the fold changes
        log2_fold_changes = \
            get_log2_fold_changes(obs_counts = obs_counts,
                                  pred_means = pred_means)

    # Get the results for the statistics that were computed
    stats_results = \
        [stat if stat is not None else pd.Series()
         for stat in (p_values, q_values, log2_fold_changes)]


    #----------------------- Output data frame -----------------------#


    # Create a data frame from the statistics computed
    df_stats = pd.concat(stats_results,
                         axis = 1)

    # Return the data frame
    return df_stats, sample_name