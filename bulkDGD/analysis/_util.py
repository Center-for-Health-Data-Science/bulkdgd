#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-

#    _util.py
#
#    Private utilities for the analyses.
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
__doc__ = "Private utilities for the analyses."


#######################################################################


# Import from the standard library.
import logging as log
# Import from third-party packages.
import torch


#######################################################################


# Get the module's logger.
logger = log.getLogger(__name__)


#######################################################################


def log_prob_mass_nb(k,
                     m,
                     r):
    """Compute the natural logarithm of the probability mass for a set
    of negative binomial distributions.

    Thr formula used to compute the logarithm of the probability mass
    is:

    .. math::

       logPDF_{NB(k,m,r)} &=
       log\\Gamma(k+r) - log\\Gamma(r) - log\\Gamma(k+1) \\\\ 
       &+ k \\cdot log(m \\cdot c + \\epsilon) +
       r \\cdot log(r \\cdot c)

    Where :math:`\\epsilon` is a small value to prevent underflow/
    overflow, and :math:`c` is equal to
    :math:`\\frac{1}{r+m+\\epsilon}`.

    The derivation of this formula from the non-logarithmic formulation
    of the probability mass function of the negative binomial
    distribution can be found below.

    Parameters
    ----------
    k : :class:`torch.Tensor`
        A one-dimensional tensor containing he "number of successes"
        seen before stopping the trials.

        Each value in the tensor corresponds to the number of
        successes in a different negative binomial.

    m : :class:`torch.Tensor`
        A one-dimensional tensor containing the means of the negative
        binomials.

        Each value in the tensor corresponds to the mean of a different
        negative binomial.

    r : :class:`torch.Tensor`
        A one-dimensional tensor containing the "number of failures"
        after which the trials end.

        Each value in the tensor corresponds to the number of failures
        in a different negative binomial.

    Returns
    -------
    x : :class:`torch.Tensor`
        A one-dimensional tensor containing the lhe log-probability
        mass of each negative binomials.

        Each value in the tensor corresponds to the log-probability
        mass of a different negative binomial.

    Notes
    -----
    Here, we show how we derived the formula for the logarithm of the
    probability mass of the negative binomial distribution.

    We start from the non-logarithmic version of the probability mass
    for the negative binomial, which is:

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

    Then, we get the natural logarithm of both sides:
    
    .. math::

       logPDF_{NB(k,m,r)} &= \
       log\\Gamma(k+r) - log\\Gamma(r) - log\\Gamma(k+1) \\\\
       &+ k \\cdot log \\left( \\frac{m}{r+m} \\right) +
       r \\cdot log \\left( \\frac{r}{r+m} \\right)
    
    Here, we are adding a small value :math:`\\epsilon` to prevent
    underflow/overflow:

    .. math::

       logPDF_{NB(k,m,r)} &= \
       log\\Gamma(k+r) - log\\Gamma(r) - log\\Gamma(k+1) \\\\
       &+ k \\cdot
       log \\left( m \\cdot \\frac{1}{r+m+\\epsilon} 
       + \\epsilon \\right) +
       r \\cdot
       log \\left( r \\cdot \\frac{1}{r+m+\\epsilon}
       \\right)

    Finally, we substitute :math:`\\frac{1}{r+m+\\epsilon}` with
    :math:`c` and we obtain:

    .. math::

       logPDF_{NB(k,m,r)} &= \
       log\\Gamma(k+r) - log\\Gamma(r) - log\\Gamma(k+1) \\\\
       &+ k \\cdot
       log \\left( m \\cdot c + \\epsilon \\right) +
       r \\cdot
       log \\left( r \\cdot c \\right)
    """

    # Convert the "number of successes" to a double-precision floating
    # point number.
    k = k.double()
    
    # Set a small value used to prevent underflow and overflow.
    eps = 1.e-10
    
    # Set a constant used later in the equation defining the log-
    # probability mass.
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
    # Since k! can be rewritten as 'gamma(k+1)':
    #
    # NB(k,m,r) = \
    #   gamma(k+r) / (gamma(r) * gamma(k+1)) *
    #   (m/(r+m))^k *
    #   (r/(r+m))^r
    #
    # Getting the natural logarithm of both sides results in:
    #
    # log(NB(k,m,r)) = \
    #   lgamma(k+r) - lgamma(r) - lgamma(k+1) +
    #   k * log(m * 1/(r+m)) +
    #   r * log(r * 1/(r+m))
    #
    # Here, we are adding the small 'eps' to prevent underflow/
    # overflow:
    #
    # log(NB(k,m,r)) = \
    #   lgamma(k+r) - lgamma(r) - lgamma(k+1) +
    #   k * log(m * 1/(r+m+eps) + eps) +
    #   r * log(r * 1/(r+m+eps))
    #
    # Substituting '1/(r+m+eps)' with 'c' yields:
    #
    # log(NB(k,m,r)) = \
    #   lgamma(k+r) - lgamma(r) - lgamma(k+1) +
    #   k * log(m * c + eps) +
    #   r * log(r * c)
    x = \
        torch.lgamma(k+r) - torch.lgamma(r) - \
        torch.lgamma(k+1) + k*torch.log(m*c+eps) + \
        r*torch.log(r*c)

    # Return the log-probability mass for the negative binomials.
    return x


def log_prob_mass_poisson(k,
                          m):
    """Compute the natural logarithm of the probability mass for a
    set of Poisson distributions.

    The formula used to compute the logarithm of the probability
    mass is:

    .. math::

       logPDF_{Poisson(k,m)} &=
       k * log(m + \\epsilon) - m - log\\Gamma(k+1)

    Where :math:`\\epsilon` is a small value to prevent underflow/
    overflow.

    The derivation of this formula from the non-logarithmic
    formulation of the probability mass function of the Poisson
    distribution can be found below.

    Parameters
    ----------
    k : ``torch.Tensor``
        A one-dimensional tensor containing he "number of
        successes" seen before stopping the trials.

        Each value in the tensor corresponds to the number of
        successes in a different Poisson distribution.

    m : ``torch.Tensor``
        A one-dimensional tensor containing the means of the
        Poisson distributions.

        Each value in the tensor corresponds to the mean of a
        different Poisson distribution.

    Returns
    -------
    x : :class:`torch.Tensor`
        A one-dimensional tensor containing the lhe log-probability
        mass of each Poisson distribution.

        Each value in the tensor corresponds to the log-probability
        mass of a different Poisson distribution.

    Notes
    -----
    Here, we show how we derived the formula for the logarithm of
    the probability mass of the Poisson distribution.

    We start from the non-logarithmic version of the probability
    mass for the Poisson distribution, which is:

    .. math::

       PDF_{Poisson(k,m)} = \
       \\frac{m^{k}e^{-m}}{k!}

    However, since:

    * :math:`k!` can be rewritten in terms of the
      gamma function as :math:`\\Gamma(k+1)`

    The formula becomes:

    .. math::

       PDF_{Poisson(k,m)} = \
       \\frac{m^{k}e^{-m}}{\\Gamma(k+1)}

    Then, we get the natural logarithm of both sides:
    
    .. math::

       logPDF_{Poisson(k,m)} &= \
       k * log(m) - m - log\\Gamma(k+1)
    
    Finally, we add a small value :math:`\\epsilon` to prevent
    underflow/overflow:

    .. math::

       logPDF_{Poisson(k,m)} &= \
       k * log(m + \\epsilon) - m - log\\Gamma(k+1)
    """

    # Convert the "number of successes" to a double-precision
    # floating point number.
    k = k.double()
    
    # Set a small value used to prevent underflow and overflow.
    eps = 1.e-10
    
    # Get the log-probability mass of the Poisson distributions.
    x =  k * torch.log(m + eps) - m - torch.lgamma(k + 1)
    
    # Return the log-probability mass for the Poisson
    # distributions.
    return x
