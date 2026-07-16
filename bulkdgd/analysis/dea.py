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
from typing import Iterator, Optional, Union

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


# Set the supported ways of computing the scaling factor of a sample.
# This mirrors 'GeneExpressionDataset.SCALING_FACTORS', and the two
# have to agree: the model is trained against one of these and the
# analysis has to undo the same one.
SCALING_FACTORS = ["mean", "median"]


########################## PRIVATE FUNCTIONS ##########################


def _get_scaling_factor(obs_counts: np.ndarray,
                        scaling_factor: str = "mean") -> float:
    """Get the factor a sample's predicted means are multiplied by to
    put them on the scale of the sample's own counts.

    The decoder does not emit counts. It emits a sample's profile, and
    the profile is multiplied by one number per sample to become the
    counts the sample actually has. That number is a property of the
    MODEL, not of this analysis: the decoder was fitted against it, and
    undoing it here with a different one leaves every predicted mean
    wrong by the ratio of the two - about a factor of three between the
    median and the mean - while every p-value still comes out looking
    like a p-value.

    This is why 'scaling_factor' has to be carried from the model's own
    configuration file to here, and why it is not simply assumed.

    Parameters
    ----------
    obs_counts : :class:`numpy.ndarray`
        The observed counts of the sample's genes.

    scaling_factor : :class:`str`, {``"mean"``, ``"median"``}, \
        ``"mean"``
        Which factor to compute. It must be the one the model was
        trained with.

    Returns
    -------
    factor : :class:`float`
        The factor.
    """

    # If the scaling factor is the mean.
    if scaling_factor == "mean":

        # Return the mean of the sample's counts.
        return np.mean(obs_counts)

    # If the scaling factor is the median.
    elif scaling_factor == "median":

        # Return the median of the sample's counts.
        return np.median(obs_counts)

    # Otherwise.
    else:

        # Raise an error.
        raise ValueError(
            f"Unsupported scaling factor '{scaling_factor}'. The "
            f"supported scaling factors are: "
            f"{', '.join(SCALING_FACTORS)}.")


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

            # Calculate the probability of "failure" of the negative
            # binomial from its mean 'm' and its r-value (the number of
            # successes at which the experiment is stopped).
            #
            # The mean, written in terms of the probability 'q' of a
            # FAILURE, is
            #
            #     m = r * q / (1 - q)
            #
            # so that
            #
            #     m (1 - q) = r q
            #     m         = q (m + r)
            #     q         = m / (m + r)
            #
            # which is what is calculated here. The derivation this
            # comment used to give was of the probability of a SUCCESS,
            # p = r / (m + r) - which is 1 - q, and not the line below.
            # The arithmetic was right; the reason given for it was not.
            p_i = pred_mean_gene_i / \
                  (pred_mean_gene_i + r_value_gene_i)

            #---------------------------------------------------------#

            # Set the percent point function to be calculated.
            ppf_dist = nbinom

            # Set the options to calculate the percent point function.
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


def _get_tails(pred_means: np.ndarray,
               r_values: Optional[np.ndarray] = None) -> np.ndarray:
    """For each gene, get the count value at which the value of the
    percent point function (the inverse of the cumulative mass
    function) is 0.99999.

    This is the vectorized counterpart of the per-gene percent point
    function evaluation performed in :func:`_yield_p_values`, and
    returns the same values.

    Parameters
    ----------
    pred_means : :class:`numpy.ndarray`
        A one-dimensional array containing the predicted scaled mean
        counts for the genes.

    r_values : :class:`numpy.ndarray`, optional
        A one-dimensional array containing the r-values for the genes.

    Returns
    -------
    tails : :class:`numpy.ndarray`
        A one-dimensional array containing the tail of each gene's
        distribution.
    """

    # If negative binomial distributions were used to model the genes'
    # counts
    if r_values is not None:

        # Calculate the probability of "failure" for all genes at once,
        # from the mean 'm' and the r-value: q = m / (m + r).
        p = pred_means / (pred_means + r_values)

        # Get the tail of each gene's distribution.
        #
        # SciPy's negative binomial counts the number of failures before
        # 'n' successes, and its 'p' is the probability of a SUCCESS.
        # Ours is the probability of a failure, so SciPy is given
        # 1 - p = r / (m + r), which returns the mean to 'm'.
        tails = nbinom.ppf(q = 0.99999,
                           n = r_values,
                           p = 1 - p)

    # If Poisson distributions were used to model the genes' counts
    else:

        # Get the tail of each gene's distribution.
        tails = poisson.ppf(q = 0.99999,
                            mu = pred_means)

    # A negative binomial with a vanishingly small r-value (as can
    # happen for genes the decoder predicts as essentially unexpressed)
    # drives 'p' to (numerically) exactly 1, which is a degenerate
    # parameterization SciPy's 'nbinom.ppf' resolves to NaN instead of
    # a finite value. Fall back to 0 in that case, as
    # '_yield_p_values' does.
    tails = np.where(np.isfinite(tails), tails, 0.0)

    # Return the tails.
    return tails


# Set the methods available to compute the p-values.
P_VALUES_METHODS = ["auto", "batched", "per-gene"]


def _resolve_p_values_method(method: str,
                             device: Union[str, torch.device],
                             resolution: Optional[int],
                             return_pmf_values: bool) -> str:
    """Get the method to be used to compute the p-values.

    Parameters
    ----------
    method : :class:`str`
        The method requested. It can be ``"auto"``, ``"batched"``, or
        ``"per-gene"``.

    device : :class:`str` or :class:`torch.device`
        The device on which the p-values will be computed.

    resolution : :class:`int`, optional
        The resolution at which the p-values will be computed.

    return_pmf_values : :class:`bool`
        Whether the points at which the log-probability mass function
        was evaluated need to be returned.

    Returns
    -------
    method : :class:`str`
        The method to be used. It is either ``"batched"`` or
        ``"per-gene"``.
    """

    # If the method is not one of the available ones
    if method not in P_VALUES_METHODS:

        # Raise an error.
        methods_str = \
            ", ".join(f"'{m}'" for m in P_VALUES_METHODS)
        errstr = \
            f"Unknown method '{method}' to compute the p-values. " \
            f"Available methods are: {methods_str}."
        raise ValueError(errstr)

    #-----------------------------------------------------------------#

    # Only the per-gene method yields the points at which the
    # log-probability mass function was evaluated.
    if return_pmf_values:

        # If the batched method was explicitly requested
        if method == "batched":

            # Raise an error, rather than silently returning something
            # other than what was asked for.
            errstr = \
                "The 'batched' method cannot return the points at " \
                "which the log-probability mass function was " \
                "evaluated. Either use the 'per-gene' method, or do " \
                "not request the points."
            raise ValueError(errstr)

        # Go gene by gene.
        return "per-gene"

    #-----------------------------------------------------------------#

    # If a method was explicitly requested
    if method != "auto":

        # Use it - the user knows what their machine looks like better
        # than we do.
        return method

    #-----------------------------------------------------------------#

    # From here on, the method is chosen automatically.

    # On a GPU, computing the p-values for all the genes at once is
    # always the better choice - it is what lets the GPU be used at
    # all.
    if torch.device(device).type != "cpu":

        # Compute the p-values for all the genes at once.
        return "batched"

    #-----------------------------------------------------------------#

    # On a CPU, it depends on how the log-probability mass function is
    # evaluated.
    #
    # At a fixed resolution, every gene is evaluated at the same number
    # of points, so the batch is a dense rectangle with no padding in
    # it, and evaluating it in one go is faster than looping over the
    # genes in Python.
    #
    # In the exact calculation, instead, each gene is evaluated at as
    # many points as its own tail requires. The batch is then ragged,
    # and has to be padded - and, on a CPU, there are no spare cores to
    # absorb the cost of the padding. So, go gene by gene.
    if resolution is not None:

        # Compute the p-values for all the genes at once.
        return "batched"

    # Go gene by gene.
    return "per-gene"


def _compute_p_values(obs_counts: np.ndarray,
                      pred_means: np.ndarray,
                      r_values: Optional[np.ndarray] = None,
                      resolution: Optional[int] = None,
                      device: Union[str, torch.device] = "cpu",
                      max_elements: int = 2**26) -> np.ndarray:
    """Calculate the p-value for all the genes in a sample at once.

    This is a vectorized re-formulation of :func:`_yield_p_values`. It
    evaluates the log-probability mass function of all the genes'
    distributions in a batch instead of gene by gene, which makes the
    calculation suitable for running on a GPU.

    The p-value of a gene depends only on that gene's distribution, so
    the genes are independent of each other, and the calculation
    parallelizes exactly.

    The genes' distributions have different tails, so the points at
    which the log-probability mass function needs to be evaluated form
    a "ragged" set. Here, the genes are sorted by the length of their
    set of points and split into chunks. Inside a chunk, the sets of
    points are padded to the length of the longest one, and the
    log-probability mass at the padded points is set to negative
    infinity so that it contributes zero probability mass. Sorting the
    genes beforehand keeps the amount of padding (and, therefore, the
    amount of wasted computation) small.

    Parameters
    ----------
    obs_counts : :class:`numpy.ndarray`
        A one-dimensional array containing the observed counts for the
        genes.

    pred_means : :class:`numpy.ndarray`
        A one-dimensional array containing the predicted scaled mean
        counts for the genes.

    r_values : :class:`numpy.ndarray`, optional
        A one-dimensional array containing the r-values for the genes.

    resolution : :class:`int`, optional
        The resolution at which to perform the p-value calculation.

    device : :class:`str` or :class:`torch.device`, ``"cpu"``
        The device on which to perform the calculation.

    max_elements : :class:`int`, ``2**26``
        The maximum number of points at which the log-probability mass
        function is evaluated in one chunk. This caps the memory used
        on the device.

    Returns
    -------
    p_values : :class:`numpy.ndarray`
        A one-dimensional array containing the p-value of each gene.
    """

    # Get the device on which to perform the calculation.
    device = torch.device(device)

    # Get the number of genes.
    n_genes = obs_counts.shape[0]

    # Get the tail of each gene's distribution.
    tails = _get_tails(pred_means = pred_means,
                       r_values = r_values)

    #-----------------------------------------------------------------#

    # If no resolution was passed
    if resolution is None:

        # The log-probability mass function is evaluated at steps of
        # width 1, from 0 up to (but excluding) the tail, so each gene
        # needs as many points as the ceiling of its tail.
        n_points = np.ceil(tails).astype(np.int64)

    # Otherwise
    else:

        # The log-probability mass function is evaluated at
        # 'resolution' points for every gene.
        n_points = np.full(shape = n_genes,
                           fill_value = int(resolution),
                           dtype = np.int64)

    #-----------------------------------------------------------------#

    # Move the genes' data to the device, in double precision, so that
    # the calculation matches the NumPy one.
    obs_counts_t = torch.as_tensor(obs_counts,
                                   dtype = torch.float64,
                                   device = device)
    pred_means_t = torch.as_tensor(pred_means,
                                   dtype = torch.float64,
                                   device = device)
    tails_t = torch.as_tensor(tails,
                              dtype = torch.float64,
                              device = device)
    n_points_t = torch.as_tensor(n_points,
                                 dtype = torch.int64,
                                 device = device)

    # If the genes' counts were modelled using negative binomial
    # distributions
    if r_values is not None:

        # Move the r-values to the device, too.
        r_values_t = torch.as_tensor(r_values,
                                     dtype = torch.float64,
                                     device = device)

    #-----------------------------------------------------------------#

    # Sort the genes by the number of points at which the
    # log-probability mass function needs to be evaluated, so that
    # genes needing a similar number of points end up in the same
    # chunk, and little padding is needed.
    order = np.argsort(n_points, kind = "stable")

    # Create an empty tensor to store the p-values.
    p_values_t = torch.empty(n_genes,
                             dtype = torch.float64,
                             device = device)

    #-----------------------------------------------------------------#

    # Get the number of points needed by each gene, sorted.
    n_points_sorted = n_points[order]

    # Set the index of the first gene in the current chunk.
    start = 0

    # Until all genes have been processed
    while start < n_genes:

        # Find how many genes fit in the current chunk. The chunk is
        # padded to the number of points needed by its longest-tailed
        # gene, so the chunk's size is capped so that the padded chunk
        # holds at most 'max_elements' points.
        #
        # The genes are sorted by the number of points they need, so
        # the longest-tailed gene of a chunk is its last one, and the
        # size of a chunk of 's' genes starting at 'start' is
        # 's * n_points_sorted[start+s-1]'. This grows with 's', so the
        # largest 's' whose chunk fits can be found by binary search.
        lo = 1
        hi = n_genes - start

        # While the search has not converged
        while lo < hi:

            # Try the larger half.
            mid = (lo + hi + 1) // 2

            # If a chunk of 'mid' genes fits
            if mid * int(n_points_sorted[start+mid-1]) <= max_elements:

                # It is a lower bound on the chunk's size.
                lo = mid

            # Otherwise
            else:

                # It is an upper bound on the chunk's size.
                hi = mid - 1

        # A chunk always holds at least one gene, even if that gene
        # alone needs more than 'max_elements' points.
        chunk_size = max(1, lo)

        # Get the indices of the genes in the current chunk.
        idx = order[start : start + chunk_size]

        # Get the number of points needed by the longest-tailed gene
        # of the chunk (the genes are sorted, so it is the last one).
        n_points_chunk = int(n_points[idx[-1]])

        # Get the indices of the genes in the chunk, on the device.
        idx_t = torch.as_tensor(idx,
                                dtype = torch.int64,
                                device = device)

        # Get the data of the genes in the chunk.
        obs_counts_chunk = obs_counts_t[idx_t].unsqueeze(1)
        pred_means_chunk = pred_means_t[idx_t].unsqueeze(1)
        tails_chunk = tails_t[idx_t].unsqueeze(1)
        n_points_gene = n_points_t[idx_t].unsqueeze(1)

        #-------------------------------------------------------------#

        # If no resolution was passed
        if resolution is None:

            # The log-probability mass function is evaluated at steps
            # of width 1, starting from 0.
            k = torch.arange(n_points_chunk,
                             dtype = torch.float64,
                             device = device).unsqueeze(0)

            # A gene's points beyond its own tail are padding.
            mask = \
                torch.arange(n_points_chunk,
                             dtype = torch.int64,
                             device = device).unsqueeze(0) \
                < n_points_gene

        # Otherwise
        else:

            # The log-probability mass function is evaluated at
            # 'resolution' evenly spaced points between 0 and the
            # gene's tail, rounded.
            #
            # The points are built the way 'numpy.linspace' builds
            # them - as 'arange(resolution) * step', with the last
            # point pinned to the gene's tail - so that they are
            # bit-for-bit the ones the per-gene calculation uses.
            # Computing them in any other way (say, as
            # 'linspace(0, 1) * tail') can be off by one unit in the
            # last place, which is enough to make a point that sits
            # exactly halfway between two integers round the other way.
            tails_trunc = torch.trunc(tails_chunk)

            # Get the points at which to evaluate the function.
            steps = \
                torch.arange(int(resolution),
                             dtype = torch.float64,
                             device = device).unsqueeze(0)

            # If more than one point is requested
            if int(resolution) > 1:

                # Space the points evenly between 0 and the tail.
                k = steps * (tails_trunc / (int(resolution) - 1))

                # Pin the last point to the tail.
                k[:, -1] = tails_trunc[:, 0]

            # Otherwise
            else:

                # There is a single point, at 0.
                k = steps * tails_trunc

            # Round the points.
            k = torch.round(k)

            # Every gene is evaluated at the same number of points, so
            # there is no padding.
            mask = torch.ones_like(k, dtype = torch.bool)

        #-------------------------------------------------------------#

        # If the genes' counts were modelled using negative binomial
        # distributions
        if r_values is not None:

            # Get the r-values of the genes in the chunk.
            r_values_chunk = r_values_t[idx_t].unsqueeze(1)

            # Get the log-probability mass at each point.
            pmf = _util.log_prob_mass_nb_torch(k = k,
                                               m = pred_means_chunk,
                                               r = r_values_chunk)

            # Get the log-probability mass at the observed count.
            prob_obs = \
                _util.log_prob_mass_nb_torch(k = obs_counts_chunk,
                                             m = pred_means_chunk,
                                             r = r_values_chunk)

        # If the genes' counts were modelled using Poisson
        # distributions
        else:

            # Get the log-probability mass at each point.
            pmf = _util.log_prob_mass_poisson_torch(k = k,
                                                    m = pred_means_chunk)

            # Get the log-probability mass at the observed count.
            prob_obs = \
                _util.log_prob_mass_poisson_torch(k = obs_counts_chunk,
                                                  m = pred_means_chunk)

        #-------------------------------------------------------------#

        # Set the log-probability mass at the padded points to negative
        # infinity, so that they carry no probability mass.
        pmf = torch.where(mask,
                          pmf,
                          torch.tensor(-float("inf"),
                                       dtype = torch.float64,
                                       device = device))

        # Get the probability mass at each point.
        prob_mass = torch.exp(pmf)

        # Get the probability that a point falls lower than the
        # observed count. The padded points carry no probability mass,
        # so they contribute nothing to the sum.
        lower_probs = \
            torch.where(pmf <= prob_obs,
                        prob_mass,
                        torch.zeros_like(prob_mass)).sum(dim = 1)

        # Get the total mass of the "discretized" probability mass
        # function computed above.
        norm_const = prob_mass.sum(dim = 1)

        # Calculate the p-values of the genes in the chunk. Genes whose
        # tail is 0 have no points, and therefore a total mass of 0 -
        # they get a NaN p-value, as they do in '_yield_p_values'.
        p_values_t[idx_t] = lower_probs / norm_const

        #-------------------------------------------------------------#

        # Move on to the next chunk.
        start += len(idx)

    #-----------------------------------------------------------------#

    # Return the p-values.
    return p_values_t.cpu().numpy()


########################## PUBLIC FUNCTIONS ###########################


def get_p_values(obs_counts: pd.Series,
                 pred_means: pd.Series,
                 r_values: Optional[pd.Series] = None,
                 resolution: Optional[int] = None,
                 return_pmf_values: bool = False,
                 pseudocount: int = 1,
                 device: Union[str, torch.device] = "cpu",
                 p_values_method: str = "auto",
                 max_elements: Optional[int] = None,
                 scaling_factor: str = "mean") -> \
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

        Setting it to ``True`` forces the p-values to be computed gene
        by gene on the CPU, ignoring ``device``.

    pseudocount : :class:`int`, ``1``
        A pseudocount to add to both the predicted means and observed
        counts to avoid artifacts.

    device : :class:`str` or :class:`torch.device`, ``"cpu"``
        The device on which to compute the p-values.

        The genes are independent of each other, so the calculation of
        the p-values parallelizes exactly, and running it on a GPU
        (for instance, by passing ``"cuda"``) speeds it up
        considerably.

        It is ignored if ``return_pmf_values`` is ``True``.

    p_values_method : :class:`str`, ``"auto"``
        How to compute the p-values. The methods give the same
        p-values, but differ in how they use the machine.

        ``"batched"`` computes the p-values for all the genes at once.
        This is what allows them to be computed on a GPU. On a CPU,
        :mod:`torch` spreads the computation over the cores by itself,
        so it should be used in a single process.

        ``"per-gene"`` computes the p-values one gene at a time. It is
        meant to be parallelized over the samples - by
        :func:`perform_dea`'s caller, or by ``bulkdgd_dea``'s ``-n``
        option - which is how to use a machine with many cores.

        The two kinds of parallelism do not compose: several processes
        each running a ``"batched"`` computation would fight over the
        CPU's cores, and the result would be slower than either kind of
        parallelism on its own.

        ``"auto"`` uses ``"batched"`` on a GPU, and ``"per-gene"`` on a
        CPU for the exact calculation (where the batch would have to be
        padded), and ``"batched"`` on a CPU otherwise.

    max_elements : :class:`int`, optional
        The maximum number of points at which the log-probability mass
        function is evaluated in one batch, which caps the memory used
        by the ``"batched"`` method.

        If not passed, it defaults to ``2**26`` on a GPU and ``2**22``
        on a CPU, where the batch sits in RAM that several processes
        may be sharing.

    scaling_factor : :class:`str`, \
        {``"mean"``, ``"median"``}, ``"mean"``
        How the model computes the scaling factor of a sample - the
        number its predicted means are multiplied by to reach the scale
        of the sample's own counts.

        It must be the one the model was **trained** with, which is in
        the model's own configuration file as ``"scaling_factor"``. The
        decoder is fitted against it, and undoing it here with the
        other one leaves every predicted mean wrong by the ratio of the
        two - about three, between the median and the mean - while
        every p-value still looks like a p-value.

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

    # Get the sample's scaling factor - the mean or the median of its
    # counts, whichever the model was trained against.
    #
    # The output is a single value.
    obs_counts_scale = \
        _get_scaling_factor(obs_counts = obs_counts,
                            scaling_factor = scaling_factor)

    #-----------------------------------------------------------------#

    # Rescale the predicted means by it.
    #
    # The output is a 1D tensor containing the rescaled means.
    pred_means = pred_means * obs_counts_scale

    #-----------------------------------------------------------------#

    # Add a pseudocount of 1 to the predicted means.
    pred_means = pred_means + pseudocount

    # Add a pseudocount of 1 to the observed counts.
    obs_counts = obs_counts + pseudocount

    #-----------------------------------------------------------------#

    # Get the method to be used to compute the p-values.
    p_values_method = \
        _resolve_p_values_method(method = p_values_method,
                                 device = device,
                                 resolution = resolution,
                                 return_pmf_values = return_pmf_values)

    #-----------------------------------------------------------------#

    # If the p-values should be computed for all the genes at once
    if p_values_method == "batched":

        # If no cap on the memory used was passed
        if max_elements is None:

            # Set the default cap for the device. A CPU has to keep the
            # batch in RAM, which several processes may be sharing, so
            # it gets a smaller one.
            max_elements = \
                2**26 if torch.device(device).type != "cpu" else 2**22

        # Compute the p-values of all the genes in the sample at once.
        # This is equivalent to the per-gene calculation below.
        p_values = _compute_p_values(obs_counts = obs_counts,
                                     pred_means = pred_means,
                                     r_values = r_values,
                                     resolution = resolution,
                                     device = device,
                                     max_elements = max_elements)

        # Convert the p-values into a series.
        series_p_values = pd.Series(p_values)

        # Set the genes as the index of the series.
        series_p_values.index = genes_obs

        # Name the series.
        series_p_values.name = "p_value"

        # Return the p-values, and two empty data frames, since the
        # points at which the log-probability mass function was
        # evaluated and the values of the function at those points were
        # not requested.
        return series_p_values, pd.DataFrame(), pd.DataFrame()

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
                          pseudocount: int = 1,
                          scaling_factor: str = "mean") -> pd.Series:
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

    scaling_factor : :class:`str`, \
        {``"mean"``, ``"median"``}, ``"mean"``
        How the model computes the scaling factor of a sample - the
        number its predicted means are multiplied by to reach the scale
        of the sample's own counts.

        It must be the one the model was **trained** with, which is in
        the model's own configuration file as ``"scaling_factor"``. The
        decoder is fitted against it, and undoing it here with the
        other one leaves every predicted mean wrong by the ratio of the
        two - about three, between the median and the mean - while
        every fold change still looks like a fold change.

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

    # Get the sample's scaling factor - the mean or the median of its
    # counts, whichever the model was trained against.
    #
    # The output is a single value.
    obs_counts_scale = \
        _get_scaling_factor(obs_counts = obs_counts,
                            scaling_factor = scaling_factor)

    #-----------------------------------------------------------------#

    # Rescale the predicted means by it.
    #
    # The output is a 1D tensor containing the rescaled means.
    pred_means = pred_means * obs_counts_scale

    #-----------------------------------------------------------------#

    # Get the log-fold change for each gene by dividing the observed
    # count by the predicted mean count. A small value is added
    # to ensure we do not divide by zero and avoid artifacts.
    #
    # The observed count over the predicted one, and not the other way
    # round: the sample over the healthy counterpart the model decoded
    # for it. So a POSITIVE log2 fold change means the gene is HIGHER in
    # the sample than the model expected of a healthy one, which is the
    # convention DESeq2 uses for its case over its control, and the
    # convention anybody reading the number will assume.
    #
    # It used to be the reciprocal - the predicted over the observed -
    # so that a positive value meant the gene was LOWER in the sample.
    # Nothing that CALLS a gene significant was affected, because
    # everything thresholds on the absolute value; but every statement
    # about a gene being up or down was backwards, and the fold changes
    # compared against DESeq2's correlated at -0.74 where they should
    # have been at +0.74. The pseudocount is symmetric, so the fix is
    # exactly a change of sign, and the tables already computed were
    # corrected by negating the column rather than by being recomputed.
    log2_fold_changes = \
        np.log2((obs_counts + pseudocount) / \
                (pred_means + pseudocount))

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
                   pseudocount: int = 1,
                   device: Union[str, torch.device] = "cpu",
                   p_values_method: str = "auto",
                   max_elements: Optional[int] = None,
                   scaling_factor: str = "mean") -> \
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

    device : :class:`str` or :class:`torch.device`, ``"cpu"``
        The device on which to compute the p-values.

        The genes are independent of each other, so the calculation of
        the p-values parallelizes exactly, and running it on a GPU
        (for instance, by passing ``"cuda"``) speeds it up
        considerably.

    p_values_method : :class:`str`, ``"auto"``
        How to compute the p-values. The methods give the same
        p-values, but differ in how they use the machine.

        ``"batched"`` computes the p-values for all the genes at once.
        This is what allows them to be computed on a GPU. On a CPU,
        :mod:`torch` spreads the computation over the cores by itself,
        so it should be used in a single process.

        ``"per-gene"`` computes the p-values one gene at a time. It is
        meant to be parallelized over the samples - by
        :func:`perform_dea`'s caller, or by ``bulkdgd_dea``'s ``-n``
        option - which is how to use a machine with many cores.

        The two kinds of parallelism do not compose: several processes
        each running a ``"batched"`` computation would fight over the
        CPU's cores, and the result would be slower than either kind of
        parallelism on its own.

        ``"auto"`` uses ``"batched"`` on a GPU, and ``"per-gene"`` on a
        CPU for the exact calculation (where the batch would have to be
        padded), and ``"batched"`` on a CPU otherwise.

    max_elements : :class:`int`, optional
        The maximum number of points at which the log-probability mass
        function is evaluated in one batch, which caps the memory used
        by the ``"batched"`` method.

        If not passed, it defaults to ``2**26`` on a GPU and ``2**22``
        on a CPU, where the batch sits in RAM that several processes
        may be sharing.

    scaling_factor : :class:`str`, \
        {``"mean"``, ``"median"``}, ``"mean"``
        How the model computes the scaling factor of a sample - the
        number its predicted means are multiplied by to reach the scale
        of the sample's own counts.

        It must be the one the model was **trained** with, which is in
        the model's own configuration file as ``"scaling_factor"``. The
        decoder is fitted against it, and undoing it here with the
        other one leaves every predicted mean wrong by the ratio of the
        two - about three, between the median and the mean - while
        every p-value still looks like a p-value.

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
                         pseudocount = pseudocount,
                         device = device,
                         p_values_method = p_values_method,
                         max_elements = max_elements,
                         scaling_factor = scaling_factor)

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
                             return_pmf_values = False,
                             device = device,
                             p_values_method = p_values_method,
                             max_elements = max_elements,
                             scaling_factor = scaling_factor)

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
                                  pseudocount = pseudocount,
                                  scaling_factor = scaling_factor)

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
                pseudocount: int = 1,
                device: Union[str, torch.device] = "cpu",
                p_values_method: str = "auto",
                max_elements: Optional[int] = None,
                scaling_factor: str = "mean") -> \
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

    device : :class:`str` or :class:`torch.device`, ``"cpu"``
        The device on which to compute the p-values.

        The genes are independent of each other, so the calculation of
        the p-values parallelizes exactly, and running it on a GPU
        (for instance, by passing ``"cuda"``) speeds it up
        considerably.

    p_values_method : :class:`str`, ``"auto"``
        How to compute the p-values. The methods give the same
        p-values, but differ in how they use the machine.

        ``"batched"`` computes the p-values for all the genes at once.
        This is what allows them to be computed on a GPU. On a CPU,
        :mod:`torch` spreads the computation over the cores by itself,
        so it should be used in a single process.

        ``"per-gene"`` computes the p-values one gene at a time. It is
        meant to be parallelized over the samples, which is how to use
        a machine with many cores.

        The two kinds of parallelism do not compose: several processes
        each running a ``"batched"`` computation would fight over the
        CPU's cores, and the result would be slower than either kind of
        parallelism on its own.

        ``"auto"`` uses ``"batched"`` on a GPU, and ``"per-gene"`` on a
        CPU for the exact calculation (where the batch would have to be
        padded), and ``"batched"`` on a CPU otherwise.

    max_elements : :class:`int`, optional
        The maximum number of points at which the log-probability mass
        function is evaluated in one batch, which caps the memory used
        by the ``"batched"`` method.

        If not passed, it defaults to ``2**26`` on a GPU and ``2**22``
        on a CPU, where the batch sits in RAM that several processes
        may be sharing.

    scaling_factor : :class:`str`, \
        {``"mean"``, ``"median"``}, ``"mean"``
        How the model computes the scaling factor of a sample - the
        number its predicted means are multiplied by to reach the scale
        of the sample's own counts.

        It must be the one the model was **trained** with, which is in
        the model's own configuration file as ``"scaling_factor"``. The
        decoder is fitted against it, and undoing it here with the
        other one leaves every predicted mean wrong by the ratio of the
        two - about three, between the median and the mean - while
        every p-value still looks like a p-value.

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
             "pseudocount" : pseudocount,
             "device" : device,
             "p_values_method" : p_values_method,
             "max_elements" : max_elements,
             "scaling_factor" : scaling_factor}

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