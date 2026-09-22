#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-

#    _util.py
#
#    Private utilities for the analyses.
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
__doc__ = "Private utilities for the analyses."


#######################################################################


# Import from the standard library.
import logging as log
from typing import Optional

# Import from third-party packages.
import numpy as np
import pandas as pd
from scipy.special import gammaln
import torch


#######################################################################


# Get the module's logger.
logger = log.getLogger(__name__)


#######################################################################


def get_aligned_genes(obs_counts: pd.Series,
                      pred_means: pd.Series,
                      r_values: Optional[pd.Series] = None) -> \
                        list[str]:
    """Get the genes of a sample, checking that the observed counts,
    the predicted means, and the r-values list the same genes in the
    same order.

    Parameters
    ----------
    obs_counts : :class:`pandas.Series`
        Observed gene counts, indexed by Ensembl ID or metadata field.

    pred_means : :class:`pandas.Series`
        Predicted means of the genes' count distributions.

    r_values : :class:`pandas.Series`, optional
        Predicted r-values of the negative binomial distributions.

    Returns
    -------
    genes : :class:`list`
        The genes' Ensembl IDs, in the order the inputs list them.
    """

    # Set the inputs to be checked.
    inputs = {"obs_counts" : obs_counts,
              "pred_means" : pred_means,
              "r_values" : r_values}

    # Get the genes listed in each input passed, in order.
    genes = \
        {name : [ix for ix in series.index if ix.startswith("ENSG")]
         for name, series in inputs.items() if series is not None}

    # Get the inputs whose genes differ from the observed counts'.
    names_diff = \
        [name for name in genes if genes[name] != genes["obs_counts"]]

    # If any input differs
    if names_diff:

        # Raise an error.
        names_str = " and ".join(f"'{name}'" for name in names_diff)
        errstr = \
            f"The genes in {names_str} differ from those in " \
            "'obs_counts' or are in a different order. The genes " \
            "must be the same, in the same order, in all inputs. " \
            "It is assumed that the genes are specified using " \
            "their Ensembl IDs."
        raise ValueError(errstr)

    # Return the genes.
    return genes["obs_counts"]


def log_prob_mass_nb(k: np.ndarray,
                     m: np.ndarray,
                     r: np.ndarray) -> np.ndarray:
    """Compute the natural logarithm of the probability mass for a set
    of negative binomial distributions.

    Parameters
    ----------
    k : :class:`numpy.ndarray`
        A one-dimensional array containing the "number of successes"
        seen before stopping the trials, one per negative binomial.

    m : :class:`numpy.ndarray`
        A one-dimensional array containing the means of the negative
        binomials.

    r : :class:`numpy.ndarray`
        A one-dimensional array containing the "number of failures"
        after which the trials end, one per negative binomial.

    Returns
    -------
    x : :class:`numpy.ndarray`
        A one-dimensional array containing the log-probability mass of
        each negative binomial.

    Notes
    -----
    The log-probability mass is:

    .. math::

       logPDF_{NB(k,m,r)} &=
       log\\Gamma(k+r) - log\\Gamma(r) - log\\Gamma(k+1) \\\\
       &+ k \\cdot log(m \\cdot c + \\epsilon) +
       r \\cdot log(r \\cdot c)

    Where :math:`\\epsilon` is a small value to prevent underflow/
    overflow, and :math:`c` is equal to
    :math:`\\frac{1}{r+m+\\epsilon}`.

    It derives from the probability mass:

    .. math::

       PDF_{NB(k,m,r)} = \
       \\binom{k+r-1}{k} (1-p)^{k} p^{r}

    Where:

    * :math:`1-p` is equal to :math:`\\frac{m}{r+m}`
    * :math:`p` is equal to :math:`\\frac{r}{r+m}`
    * :math:`\\binom{k+r-1}{k}` is equal to
      :math:`\\frac{\\Gamma(k+r)}{\\Gamma(r) \\cdot \\Gamma(k+1)}`

    Therefore:

    .. math::

       PDF_{NB(k,m,r)} = \
       \\frac{\\Gamma(k+r)}{\\Gamma(r) \\cdot \\Gamma(k+1)}
       \\left( \\frac{m}{r+m} \\right)^k
       \\left( \\frac{r}{r+m} \\right)^r

    Taking the natural logarithm of both sides, adding
    :math:`\\epsilon`, and substituting :math:`c` yields the formula
    above.
    """

    # Convert the "number of successes" to a double-precision
    # floating point number.
    k = k.astype(np.float64)

    # Set a small value used to prevent underflow and overflow.
    eps = 1.e-10

    # Set a constant used later in the equation defining the log-
    # probability mass.
    c = 1.0 / (r + m + eps)

    # Get the log-probability mass of the negative binomials.
    x = \
        gammaln(k+r) - gammaln(r) - \
        gammaln(k+1) + k*np.log(m*c+eps) + \
        r*np.log(r*c)

    # Return the log-probability mass for the negative binomials.
    return x


def log_prob_mass_poisson(k: np.ndarray,
                          m: np.ndarray) -> np.ndarray:
    """Compute the natural logarithm of the probability mass for a
    set of Poisson distributions.

    Parameters
    ----------
    k : :class:`numpy.ndarray`
        A one-dimensional array containing the "number of successes"
        seen before stopping the trials, one per Poisson distribution.

    m : :class:`numpy.ndarray`
        A one-dimensional array containing the means of the Poisson
        distributions.

    Returns
    -------
    x : :class:`numpy.ndarray`
        A one-dimensional array containing the log-probability mass of
        each Poisson distribution.

    Notes
    -----
    The log-probability mass is:

    .. math::

       logPDF_{Poisson(k,m)} &=
       k * log(m + \\epsilon) - m - log\\Gamma(k+1)

    Where :math:`\\epsilon` is a small value to prevent underflow/
    overflow.

    It derives from the probability mass, where :math:`k!` is
    rewritten as :math:`\\Gamma(k+1)`:

    .. math::

       PDF_{Poisson(k,m)} = \
       \\frac{m^{k}e^{-m}}{k!} = \
       \\frac{m^{k}e^{-m}}{\\Gamma(k+1)}

    Taking the natural logarithm of both sides and adding
    :math:`\\epsilon` yields the formula above.
    """

    # Convert the "number of successes" to a double-precision
    # floating point number.
    k = k.astype(np.float64)

    # Set a small value used to prevent underflow and overflow.
    eps = 1.e-10

    # Get the log-probability mass of the Poisson distributions.
    x = k * np.log(m + eps) - m - gammaln(k + 1)

    # Return the log-probability mass for the Poisson
    # distributions.
    return x


def log_prob_mass_nb_torch(k: torch.Tensor,
                           m: torch.Tensor,
                           r: torch.Tensor) -> torch.Tensor:
    """Compute the natural logarithm of the probability mass for a set
    of negative binomial distributions, using :mod:`torch` so that the
    computation can run on a GPU.

    Parameters
    ----------
    k : :class:`torch.Tensor`
        The "number of successes" seen before stopping the trials.

    m : :class:`torch.Tensor`
        The means of the negative binomials.

    r : :class:`torch.Tensor`
        The r-values of the negative binomials.

    Returns
    -------
    x : :class:`torch.Tensor`
        The log-probability mass of the negative binomials, evaluated
        at the given points.
    """

    # Set a small value used to prevent underflow and overflow.
    eps = 1.e-10

    # Set a constant used later in the equation defining the log-
    # probability mass.
    c = 1.0 / (r + m + eps)

    # Get the log-probability mass of the negative binomials.
    x = \
        torch.lgamma(k+r) - torch.lgamma(r) - \
        torch.lgamma(k+1) + k*torch.log(m*c+eps) + \
        r*torch.log(r*c)

    # Return the log-probability mass for the negative binomials.
    return x


def log_prob_mass_poisson_torch(k: torch.Tensor,
                                m: torch.Tensor) -> torch.Tensor:
    """Compute the natural logarithm of the probability mass for a set
    of Poisson distributions, using :mod:`torch` so that the
    computation can run on a GPU.

    Parameters
    ----------
    k : :class:`torch.Tensor`
        The "number of successes" seen before stopping the trials.

    m : :class:`torch.Tensor`
        The means of the Poisson distributions.

    Returns
    -------
    x : :class:`torch.Tensor`
        The log-probability mass of the Poisson distributions,
        evaluated at the given points.
    """

    # Set a small value used to prevent underflow and overflow.
    eps = 1.e-10

    # Get the log-probability mass of the Poisson distributions.
    x = k * torch.log(m + eps) - m - torch.lgamma(k + 1)

    # Return the log-probability mass for the Poisson distributions.
    return x
