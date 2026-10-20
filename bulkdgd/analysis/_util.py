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
import torch

# Import from 'bulkdgd'.
from bulkdgd.core import outputmodules


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
            "'obs_counts' or are in a different order. All inputs " \
            "must list the same Ensembl IDs, in the same order."
        raise ValueError(errstr)

    # Return the genes.
    return genes["obs_counts"]


def log_prob_mass_nb(k: np.ndarray,
                     m: np.ndarray,
                     r: np.ndarray,
                     eps: Optional[float] = None) -> np.ndarray:
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

    eps : :class:`float`, optional
        The small value of the log-probability mass (see
        :meth:`core.outputmodules.OutputModuleNB.log_prob_mass`).

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
       &+ k \\cdot log(\\frac{m}{r+m}) +
       r \\cdot log(\\frac{r}{r+m})

    Where :math:`m` is offset by ``MEAN_OFFSET`` to keep its logarithm
    finite.

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

    Taking the natural logarithm of both sides yields the formula
    above. Above ``NB_R_SWITCH`` (``NB_R_SWITCH_EPS`` if ``eps`` is
    passed), :func:`core.outputmodules.log_prob_mass_nb_large_r` is
    used.
    """

    # Return the log-probability mass.
    return outputmodules.OutputModuleNB.log_prob_mass(
        k = torch.as_tensor(k,
                            dtype = torch.float64),
        m = torch.as_tensor(m,
                            dtype = torch.float64),
        r = torch.as_tensor(r,
                            dtype = torch.float64),
        eps = eps).numpy()


def log_prob_mass_poisson(k: np.ndarray,
                          m: np.ndarray,
                          eps: Optional[float] = None) -> np.ndarray:
    """Compute the natural logarithm of the probability mass for a
    set of Poisson distributions.

    Parameters
    ----------
    k : :class:`numpy.ndarray`
        A one-dimensional array containing the observed counts, one
        per Poisson distribution.

    m : :class:`numpy.ndarray`
        A one-dimensional array containing the means of the Poisson
        distributions.

    eps : :class:`float`, optional
        The small value of the log-probability mass (see
        :meth:`core.outputmodules.OutputModulePoisson.log_prob_mass`).

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

    # Return the log-probability mass.
    return outputmodules.OutputModulePoisson.log_prob_mass(
        k = torch.as_tensor(k,
                            dtype = torch.float64),
        m = torch.as_tensor(m,
                            dtype = torch.float64),
        eps = eps).numpy()
