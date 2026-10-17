#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-

#    outputmodules.py
#
#    This module contains the classes defining the output layer of the
#    :class:`core.decoder.Decoder`.
#
#    The code was originally developed by Viktoria Schuster,
#    Inigo Prada Luengo, and Anders Krogh.
#
#    Valentina Sora modified and complemented it for the purposes
#    of this package.
#
#    Copyright (C) 2026 Valentina Sora
#                       <sora.valentina1@gmail.com>
#                       Viktoria Schuster
#                       <viktoria.schuster@sund.ku.dk>
#                       Inigo Prada Luengo
#                       <inlu@diku.dk>
#                       Anders Krogh
#                       <akrogh@di.ku.dk>
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
    "This module contains the classes defining the output layer of " \
    "the :class:`core.decoder.Decoder`."


#######################################################################


# Import from the standard library.
import logging as log
import math
from typing import Optional

# Import from third-party libraries.
import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F


#######################################################################


# Get the module's logger.
logger = log.getLogger(__name__)


#######################################################################


# Set the r-value above which the negative binomial's log-probability
# mass comes from an expansion instead of the closed form.
NB_R_SWITCH = 1.0e6

# Set the r-value above which the expansion replaces the closed form
# with an eps.
NB_R_SWITCH_EPS = 1.0e15

# Set the range of the r-values, keeping the gradients finite.
NB_R_MIN = 1.0e-100
NB_R_MAX = 1.0e150

# Set the offset added to the means, keeping their logarithm finite.
MEAN_OFFSET = 1.0e-30

# Set the value of m / (r + m) below which, for r-values above the
# switch, the tail probabilities come from the Poisson limit.
NB_POISSON_Q = 1.0e-8


#######################################################################


def log_prob_mass_nb_large_r(k: torch.Tensor,
                             m: torch.Tensor,
                             r: torch.Tensor) -> torch.Tensor:
    """Compute the natural logarithm of the probability mass for a set
    of negative binomial distributions with large r-values.

    Parameters
    ----------
    k : :class:`torch.Tensor`
        The "number of successes" seen before stopping the trials.

    m : :class:`torch.Tensor`
        The means of the negative binomials.

    r : :class:`torch.Tensor`
        The r-values of the negative binomials, between
        ``NB_R_SWITCH`` and ``NB_R_MAX``.

    Returns
    -------
    x : :class:`torch.Tensor`
        The log-probability mass of the negative binomials.

    Notes
    -----
    Stirling's series for :math:`\\log \\Gamma(k+r) - \\log \\Gamma(r)`
    gives

    .. math::

       \\log P(k \\mid m, r) &=
       k \\left[ \\log(1 + k/r) - \\log(1 + m/r) \\right]
       + (r - 1/2) \\log(1 + k/r) - k \\\\
       &+ \\frac{1}{12(k+r)} - \\frac{1}{12r}
       - \\log \\Gamma(k+1) + k \\log m - r \\log(1 + m/r)
    """

    # Get log(1 + k/r) and log(1 + m/r).
    log1p_k = torch.log1p(k / r)
    log1p_m = torch.log1p(m / r)

    # Return the log-probability mass.
    return \
        k * (log1p_k - log1p_m) + (r - 0.5) * log1p_k - k + \
        1.0 / (12.0 * (k + r)) - 1.0 / (12.0 * r) - \
        torch.lgamma(k + 1) + k * torch.log(m) - r * log1p_m


#######################################################################


class OutputModuleBase(nn.Module):

    """
    Base class for the decoder's output modules.
    """


    ######################## PUBLIC ATTRIBUTES ########################


    # Supported activation functions.
    ACTIVATION_FUNCTIONS = ["sigmoid", "softplus"]


    ######################### INITIALIZATION ##########################


    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 activation: str = "softplus",
                 eps: Optional[float] = None) -> None:
        """Initialize an instance of the class.

        Parameters
        ----------
        input_dim : :class:`int`
            The dimensionality of the input.

        output_dim : :class:`int`
            The dimensionality of the output.

        activation : :class:`str`, {``"sigmoid"``, ``"softplus"``}, \
            ``"softplus"``
            The name of the activation function to be used.

            Available options are:

            * ``"sigmoid"``: the sigmoid activation function.
            * ``"softplus"``: the softplus activation function.

        eps : :class:`float`, optional
            The small value the log-probability mass adds inside its
            logarithms. If not passed, the exact form is used.
        """

        # Initialize the instance.
        super().__init__()

        # Set the small value of the log-probability mass.
        self._eps = eps

        # Set the dimensionality of the input.
        self._input_dim = input_dim

        # Set the dimensionality of the output.
        self._output_dim = output_dim

        # Get the name of the activation that will be used.
        self._activation = \
            self._get_activation(activation = activation)


    def _get_activation(self,
                        activation: str) -> str:
        """Get the name of the activation function after checking
        that it is supported.

        Parameters
        ----------
        activation : :class:`str`
            The name of the activation function to be used.

        Returns
        -------
        activation : :class:`str`
            The name of the activation function to be used.
        """

        # If the provided activation function is not supported
        if activation not in self.ACTIVATION_FUNCTIONS:

            # Raise an exception.
            errstr = \
                f"Unknown 'activation' ({activation}) for " \
                f"{self.__class__.__name__}. Supported activation " \
                f"functions are: " \
                f"{', '.join(self.ACTIVATION_FUNCTIONS)}."
            raise ValueError(errstr)

        # Return the name of the activation function.
        return activation


    ########################### PROPERTIES ############################


    @property
    def input_dim(self):
        """The dimensionality of the input.
        """

        return self._input_dim


    @input_dim.setter
    def input_dim(self,
                  value):
        """Raise an exception if the user tries to modify the value
        of ``input_dim`` after initialization.

        Parameters
        ----------
        value : :class:`int`
            The new value.
        """

        # Raise an error.
        errstr = \
            "The value of 'input_dim' is set at initialization and " \
            "cannot be changed."
        raise ValueError(errstr)


    @property
    def output_dim(self):
        """The dimensionality of the output.
        """

        return self._output_dim


    @output_dim.setter
    def output_dim(self,
                   value):
        """Raise an exception if the user tries to modify the value
        of ``output_dim`` after initialization.

        Parameters
        ----------
        value : :class:`int`
            The new value.
        """

        # Raise an error.
        errstr = \
            "The value of 'output_dim' is set at initialization and " \
            "cannot be changed."
        raise ValueError(errstr)


    @property
    def activation(self):
        """The activation function used.
        """

        return self._activation


    @property
    def eps(self):
        """The small value of the log-probability mass, if any.
        """

        return self._eps


    @activation.setter
    def activation(self,
                   value):
        """Raise an exception if the user tries to modify the value
        of ``activation`` after initialization.

        Parameters
        ----------
        value : :class:`str`
            The new value.
        """

        # Raise an error.
        errstr = \
            "The value of 'activation' is set at initialization and " \
            "cannot be changed."
        raise ValueError(errstr)


    ######################### STATIC METHODS ##########################


    @staticmethod
    def rescale(means: torch.Tensor,
                scaling_factors: torch.Tensor) -> torch.Tensor:
        """Rescale the means of the distributions.

        Parameters
        ----------
        means : :class:`torch.Tensor`
            A 1D tensor containing the means of the distributions.

            In the tensor, each value represents the mean of a
            different distribution.

        scaling_factors : :class:`torch.Tensor`
            The scaling factors.

            This is a 1D tensor whose length is equal to the number of
            scaling factors to be used to rescale the means.

        Returns
        -------
        rescaled_means : :class:`torch.Tensor`
            The rescaled means.

            This is a 1D tensor whose length is equal to the number
            of distributions whose means were rescaled.
        """

        # Return the rescaled values by multiplying the means by the
        # scaling factors.
        return means * scaling_factors


    def diagnostics(self) -> dict:
        """Get the module's internal state, for the training loop to
        log once an epoch.

        Returns
        -------
        diagnostics : :class:`dict`
            The module's internal state (empty by default).
        """

        # By default, a module reports nothing.
        return {}


    def dispersion_regularization(self,
                                  pred_means,
                                  pred_log_r_values,
                                  reduction = "sum"):
        """Get the penalty an output module adds to the training loss
        to regularize its predicted dispersions.

        Parameters
        ----------
        pred_means : :class:`torch.Tensor`
            The predicted scaled means, before the sample's scaling
            factor is applied.

        pred_log_r_values : :class:`torch.Tensor`
            The predicted log-r-values.

        reduction : :class:`str`, {``"sum"``, ``"mean"``}, ``"sum"``
            How to reduce the penalty (as the reconstruction loss is
            reduced).

        Returns
        -------
        penalty : :class:`torch.Tensor` or :class:`float`
            The penalty (``0.0`` by default).
        """

        # By default, a module adds no penalty.
        return 0.0


class OutputModulePoisson(OutputModuleBase):

    """
    Class for the decoder's output modules modelling Poisson
    distributions.
    """


    ######################### INITIALIZATION ##########################


    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 activation: str = "softplus",
                 eps: Optional[float] = None) -> None:
        """Initialize an instance of the class.

        Parameters
        ----------
        input_dim : :class:`int`
            The dimensionality of the input.

        output_dim : :class:`int`
            The dimensionality of the output.

        activation : :class:`str`, {``"sigmoid"``, ``"softplus"``}, \
            ``"softplus"``
            The name of the activation function to be used.

            Available options are:

            * ``"sigmoid"``: the sigmoid activation function.
            * ``"softplus"``: the softplus activation function.

        eps : :class:`float`, optional
            The small value the log-probability mass adds inside its
            logarithms. If not passed, the exact form is used.
        """

        # Initialize the instance.
        super().__init__(input_dim = input_dim,
                         output_dim = output_dim,
                         activation = activation,
                         eps = eps)

        # Set the layer that will contain the means of the Poisson
        # distributions.
        self._layer_means = \
            nn.Linear(in_features = input_dim,
                      out_features = output_dim)


    ######################### STATIC METHODS ##########################


    @staticmethod
    def log_prob_mass(k: torch.Tensor,
                      m: torch.Tensor,
                      eps: Optional[float] = None) -> torch.Tensor:
        """Compute the natural logarithm of the probability mass for a
        set of Poisson distributions.

        The formula used to compute the logarithm of the probability
        mass is:

        .. math::

           logPDF_{Poisson(k,m)} &=
           k * log(m + \\epsilon) - m - log\\Gamma(k+1)

        Where :math:`\\epsilon` is a small value to prevent underflow/
        overflow.

        Parameters
        ----------
        k : :class:`torch.Tensor`
            A one-dimensional tensor containing the "number of
            successes" seen before stopping the trials.

            Each value in the tensor corresponds to the number of
            successes in a different Poisson distribution.

        m : :class:`torch.Tensor`
            A one-dimensional tensor containing the means of the
            Poisson distributions.

            Each value in the tensor corresponds to the mean of a
            different Poisson distribution.

        eps : :class:`float`, optional
            The value of :math:`\\epsilon`. If not passed, it is
            ``MEAN_OFFSET``.

        Returns
        -------
        x : :class:`torch.Tensor`
            A one-dimensional tensor containing the log-probability
            mass of each Poisson distribution.

            Each value in the tensor corresponds to the log-probability
            mass of a different Poisson distribution.

        Notes
        -----
        The log-probability mass is

        .. math::

           \\log P(k \\mid m) =
           k \\log(m + \\epsilon) - m - \\log \\Gamma(k+1)

        with a small :math:`\\epsilon` preventing underflow.
        """

        # Convert the counts and the means to double precision.
        k = k.double()
        m = m.double()

        # Get the log-probability mass of the Poisson distributions.
        x = k * torch.log(m + (MEAN_OFFSET if eps is None else eps)) - \
            m - torch.lgamma(k + 1)

        # Return the log-probability mass for the Poisson
        # distributions.
        return x


    ######################### PUBLIC METHODS ##########################


    def forward(self,
                x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : :class:`torch.Tensor`
            The input tensor.

        Returns
        -------
        m : :class:`torch.Tensor`
            A tensor containing the means of the Poisson distributions.
        """

        # Pass the input through the layer.
        _m = self._layer_means(x)

        #-------------------------------------------------------------#

        # If the activation function is a sigmoid
        if self.activation == "sigmoid":

            # Get the predicted means of the Poisson distributions.
            m = torch.sigmoid(_m)

        # If the activation function is a softplus
        elif self.activation == "softplus":

            # Get the predicted means of the Poisson distributions.
            m = F.softplus(_m)

        #-------------------------------------------------------------#

        # Return the means of the Poisson distributions.
        return m


    def log_prob(self,
                 obs_counts: torch.Tensor,
                 pred_means: torch.Tensor,
                 scaling_factors: torch.Tensor) -> torch.Tensor:
        """Get the log-probability mass of the Poisson distributions.

        Parameters
        ----------
        obs_counts : :class:`torch.Tensor`
            The observed gene counts.

            The first dimension of this tensor must have a length
            equal to the number of samples whose counts are
            reported.

        pred_means : :class:`torch.Tensor`
            The predicted means of the Poisson distributions.

            This is a tensor whose shape must match that of
            ``obs_counts``.

        scaling_factors : :class:`torch.Tensor`
            The scaling factors.

            This is a 1D tensor whose length must match that
            of the first dimension of ``obs_counts`` and
            ``pred_means``.

        Returns
        -------
        log_prob_mass : :class:`torch.Tensor`
            The log-probability mass of the Poisson distributions.

            This is a 2D tensor where:

            * The first dimension has a length equal to the length
              of the first dimension of ``obs_counts`` and
              ``pred_means``.

            * The second dimension has a length equal to the length
              of the second dimension of ``obs_counts`` and
              ``pred_means``.
        """

        # Get the rescaled means of the Poisson distributions.
        m = self.__class__.rescale(means = pred_means,
                                   scaling_factors = scaling_factors)

        # Return the log-probability mass for the Poisson
        # distributions.
        return self.__class__.log_prob_mass(k = obs_counts,
                                            m = m,
                                            eps = self.eps)


    def loss(self,
             obs_counts: torch.Tensor,
             pred_means: torch.Tensor,
             scaling_factors: torch.Tensor,
             contamination: float = 0.0,
             contamination_r: float = 0.05) -> torch.Tensor:
        """Compute the loss, the negative log-probability mass of the
        Poisson distributions (or, if ``contamination`` is above zero,
        of a mixture with a negative binomial outlier component).

        Parameters
        ----------
        obs_counts : :class:`torch.Tensor`
            The observed gene counts.

        pred_means : :class:`torch.Tensor`
            The predicted means of the Poisson distributions.

            This is a tensor whose shape must match that of
            ``obs_counts``.

        scaling_factors : :class:`torch.Tensor`
            The scaling factors.

            This is a 1D tensor whose length must match that of the
            first dimension of ``obs_counts`` and ``pred_means``.

        contamination : :class:`float`, ``0.0``
            The weight of the outlier component.

        contamination_r : :class:`float`, ``0.05``
            The r-value of the outlier component.

        Returns
        -------
        loss : :class:`torch.Tensor`
            The loss for each count.

            This is a 2D tensor where:

            * The first dimension has a length equal to the length
              of the first dimension of ``obs_counts`` and
              ``pred_means``.

            * The second dimension has a length equal to the length
              of the second dimension of ``obs_counts`` and
              ``pred_means``.
        """

        # Get the loss for each of the Poisson distributions.
        nll = - self.log_prob(obs_counts = obs_counts,
                              pred_means = pred_means,
                              scaling_factors = scaling_factors)

        # If there is no contamination, return the loss.
        if not contamination:
            return nll

        #-------------------------------------------------------------#

        # Get the log r-values of the outlier component.
        log_r_outlier = \
            torch.full_like(pred_means,
                            math.log(contamination_r))

        # Get the loss under the outlier component, a negative
        # binomial with the same means.
        nll_outlier = \
            - OutputModuleNB.log_prob_mass(
                k = obs_counts,
                m = self.__class__.rescale(
                        means = pred_means,
                        scaling_factors = scaling_factors),
                r = torch.exp(log_r_outlier.double()),
                eps = self.eps)

        #-------------------------------------------------------------#

        # Get the weighted log-probabilities of the two components.
        log_weights = \
            torch.stack(
                [math.log1p(-contamination) - nll,
                 math.log(contamination) - nll_outlier])

        # Return -log[(1-eps) * Poisson + eps * NB(r_outlier)].
        return - torch.logsumexp(log_weights, dim = 0)


    def sample(self,
               n: int,
               pred_means: torch.Tensor,
               scaling_factors: torch.Tensor) -> torch.Tensor:
        """Get samples from the Poisson distributions.

        Parameters
        ----------
        n : :class:`int`
            The number of samples to get.

        pred_means : :class:`torch.Tensor`
            The predicted means of the Poisson distributions.

        scaling_factors : :class:`torch.Tensor`
            A tensor containing the scaling factors.

            This is a 1D tensor whose length must match that
            of the first dimension of ``pred_means``.

        Returns
        -------
        samples : :class:`torch.Tensor`
            The samples drawn from the Poisson distributions.

            The shape of this tensor depends on the shape of ``n``
            and ``pred_means``, but the first dimension always has
            a length equal to the number of samples drawn from the
            Poisson distribution.
        """

        # Disable the gradient calculation.
        with torch.no_grad():

            # Get the rescaled means of the Poisson distributions.
            m = self.__class__.rescale(\
                    means = pred_means,
                    scaling_factors = scaling_factors)

            # Sample from the Poisson distributions.
            poisson = dist.Poisson(rate = m)

            # Get 'n' samples from the distributions.
            return poisson.sample([n]).squeeze()


class OutputModuleNB(OutputModuleBase):

    """
    Base class for the decoder's output modules modelling negative
    binomial distributions.
    """


    ######################### INITIALIZATION ##########################


    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 activation: str = "softplus",
                 eps: Optional[float] = None) -> None:
        """Initialize an instance of the class.

        Parameters
        ----------
        input_dim : :class:`int`
            The dimensionality of the input.

        output_dim : :class:`int`
            The dimensionality of the output.

        activation : :class:`str`, {``"sigmoid"``, ``"softplus"``}, \
            ``"softplus"``
            The name of the activation function to be used.

            Available options are:

            * ``"sigmoid"``: the sigmoid activation function.
            * ``"softplus"``: the softplus activation function.

        eps : :class:`float`, optional
            The small value the log-probability mass adds inside its
            logarithms. If not passed, the exact form is used.
        """

        # Initialize the instance.
        super().__init__(input_dim = input_dim,
                         output_dim = output_dim,
                         activation = activation,
                         eps = eps)


    ######################### STATIC METHODS ##########################


    @staticmethod
    def log_prob_mass(k: torch.Tensor,
                      m: torch.Tensor,
                      r: torch.Tensor,
                      eps: Optional[float] = None) -> torch.Tensor:
        """Compute the natural logarithm of the probability mass for a
        set of negative binomial distributions.

        The formula used to compute the logarithm of the probability
        mass is:

        .. math::

           logPDF_{NB(k,m,r)} &=
           log\\Gamma(k+r) - log\\Gamma(r) - log\\Gamma(k+1) \\\\
           &+ k \\cdot log(\\frac{m}{r+m}) +
           r \\cdot log(\\frac{r}{r+m})

        Where :math:`m` is offset by a small value to prevent underflow.
        Above ``NB_R_SWITCH``, :func:`log_prob_mass_nb_large_r` is used.

        Parameters
        ----------
        k : :class:`torch.Tensor`
            A one-dimensional tensor containing the "number of
            successes" seen before stopping the trials.

            Each value in the tensor corresponds to the number of
            successes in a different negative binomial.

        m : :class:`torch.Tensor`
            A one-dimensional tensor containing the means of the
            negative binomial distributions.

            Each value in the tensor corresponds to the mean of a
            different negative binomial.

        r : :class:`torch.Tensor`
            A one-dimensional tensor containing the "number of
            failures" after which the trials end.

            Each value in the tensor corresponds to the number of
            failures in a different negative binomial.

        eps : :class:`float`, optional
            If passed, the closed form is
            :math:`k \\cdot log(m \\cdot c + \\epsilon) +
            r \\cdot log(r \\cdot c)`, with :math:`c` equal to
            :math:`\\frac{1}{r+m+\\epsilon}`, up to
            ``NB_R_SWITCH_EPS``.

        Returns
        -------
        x : :class:`torch.Tensor`
            A one-dimensional tensor containing the log-probability
            mass of each negative binomial distribution.

            Each value in the tensor corresponds to the log-probability
            mass of a different negative binomial.
        """

        # Compute the log-probability mass in double precision.
        k = k.double()
        m = m.double()
        r = r.double()

        # Offset the means.
        m_off = m + MEAN_OFFSET

        # Keep the r-values in their range.
        r = torch.clamp(r,
                        min = NB_R_MIN,
                        max = NB_R_MAX)

        # Get the r-value above which the expansion is used.
        switch = NB_R_SWITCH if eps is None else NB_R_SWITCH_EPS

        #-------------------------------------------------------------#

        # Find the r-values above the switch.
        large = r > switch

        # Get whether any r-value exceeds it.
        any_large = bool(large.any())

        # Cap the r-values of the closed form at the switch, if needed.
        r_low = torch.clamp(r,
                            max = switch) \
            if any_large else r

        # If no eps was passed
        if eps is None:

            # Get the log-probability mass from the closed form.
            x = \
                torch.lgamma(k+r_low) - torch.lgamma(r_low) - \
                torch.lgamma(k+1) + k*torch.log(m_off) - \
                k*torch.log(r_low+m_off) - \
                r_low*torch.log1p(m_off/r_low)

        # Otherwise
        else:

            # Set the constant 'c' of the log-probability mass.
            c = 1.0 / (r_low + m + eps)

            # Get the log-probability mass from the closed form with
            # the eps.
            x = \
                torch.lgamma(k+r_low) - torch.lgamma(r_low) - \
                torch.lgamma(k+1) + k*torch.log(m*c+eps) + \
                r_low*torch.log(r_low*c)

        # If no r-value exceeds the switch, return the mass.
        if not any_large:
            return x

        #-------------------------------------------------------------#

        # Get the log-probability mass from the expansion.
        x_large = \
            log_prob_mass_nb_large_r(
                k = k,
                m = m_off,
                r = torch.clamp(r,
                                min = switch))

        # Return the log-probability mass for the negative binomial
        # distributions, from the expansion above the switch.
        return torch.where(large,
                           x_large,
                           x)


class OutputModuleNBFeatureDispersion(OutputModuleNB):

    """
    Class implementing an output layer representing the means of the
    negative binomial distributions modeling the outputs (i.e., the
    means of the gene expression counts). One negative binomial
    distribution with trainable mean is used for each gene.
    """


    ######################### INITIALIZATION ##########################


    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 r_init: int,
                 activation: str = "softplus",
                 eps: Optional[float] = None) -> None:
        """Initialize an instance of the class.

        Parameters
        ----------
        input_dim : :class:`int`
            The dimensionality of the input.

        output_dim : :class:`int`
            The dimensionality of the output.

        r_init : :class:`int`
            The initial 'r' value.

        activation : :class:`str`, {``"sigmoid"``, ``"softplus"``}, \
            ``"softplus"``
            The name of the activation function to be used.

            Available options are:

            * ``"sigmoid"``: the sigmoid activation function.
            * ``"softplus"``: the softplus activation function.

        eps : :class:`float`, optional
            The small value the log-probability mass adds inside its
            logarithms. If not passed, the exact form is used.
        """

        # Initialize the instance.
        super().__init__(input_dim = input_dim,
                         output_dim = output_dim,
                         activation = activation,
                         eps = eps)

        # Initialize the value of the log of r.
        self._log_r = \
            self._get_log_r(r_init = r_init,
                            output_dim = output_dim)

        # Set the layer that will contain the means of the negative
        # binomials.
        self._layer_means = \
            nn.Linear(in_features = input_dim,
                      out_features = output_dim)


    def _get_log_r(self,
                   r_init: int,
                   output_dim: int) -> torch.Tensor:
        """Get a tensor with dimensions (1, ``output_dim``) filled with
        the natural logarithm of the initial value of 'r' ("number of
        failures" after which the "trials" stop).

        Parameters
        ----------
        r_init : :class:`int`
            The initial value for 'r', representing the "number
            of failures" after which the "trials" stop.

        output_dim : :class:`int`
            The dimensionality of the output.

        Returns
        -------
        log_r : :class:`torch.Tensor`
            A tensor containing the ``r_init`` value as many times
            as the number of dimensions of the space the negative
            binomials live in.
        """

        # Return the natural logarithm of the initial value of
        # 'r'.
        return nn.Parameter(torch.full(fill_value = math.log(r_init),
                                       size = (1, output_dim)),
                            requires_grad = True)


    ########################### PROPERTIES ############################


    @property
    def log_r(self) -> torch.Tensor:
        """The natural logarithm of the 'r' values associated with
        the negative binomial distributions.
        """

        return self._log_r


    @log_r.setter
    def log_r(self,
              value: torch.Tensor) -> None:
        """Raise an exception if the user tries to modify the value
        of ``log_r`` after initialization.

        Parameters
        ----------
        value : :class:`torch.Tensor`
            The new value.
        """

        # Raise an error.
        errstr = \
            "The value of 'log_r' is set at initialization and " \
            "cannot be changed."
        raise ValueError(errstr)


    ######################### PUBLIC METHODS ##########################


    def forward(self,
                x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : :class:`torch.Tensor`
            The input tensor.

        Returns
        -------
        m : :class:`torch.Tensor`
            A tensor containing the means of the negative binomial
            distributions.
        """

        # Pass the input through the layer.
        _m = self._layer_means(x)

        #-------------------------------------------------------------#

        # If the activation function is a sigmoid
        if self.activation == "sigmoid":

            # Get the predicted scaled means of the negative binomial
            # distributions.
            m = torch.sigmoid(_m)

        # If the activation function is a softplus
        elif self.activation == "softplus":

            # Get the predicted scaled means of the negative binomial
            # distributions.
            m = F.softplus(_m)

        #-------------------------------------------------------------#

        # Return the means of the negative binomial distributions.
        return m


    def log_prob(self,
                 obs_counts: torch.Tensor,
                 pred_means: torch.Tensor,
                 scaling_factors: torch.Tensor) -> torch.Tensor:
        """Get the log-probability mass of the negative binomial
        distributions.

        Parameters
        ----------
        obs_counts : :class:`torch.Tensor`
            The observed gene counts.

            The first dimension of this tensor must have a length
            equal to the number of samples whose counts are
            reported.

        pred_means : :class:`torch.Tensor`
            The predicted scaled means of the negative binomial
            distributions.

            This is a tensor whose shape must match that of
            ``obs_counts``.

        scaling_factors : :class:`torch.Tensor`
            The scaling factors.

            This is a 1D tensor whose length must match that
            of the first dimension of ``obs_counts`` and
            ``pred_means``.

        Returns
        -------
        log_prob_mass : :class:`torch.Tensor`
            The log-probability mass of the negative binomial
            distributions.

            This is a 2D tensor where:

            * The first dimension has a length equal to the length
              of the first dimension of ``obs_counts`` and
              ``pred_means``.

            * The second dimension has a length equal to the length
              of the second dimension of ``obs_counts`` and
              ``pred_means``.
        """

        # Get the rescaled means of the negative binomial
        # distributions.
        m = self.__class__.rescale(means = pred_means,
                                   scaling_factors = scaling_factors)

        # Get the 'r' values of the negative binomial distributions.
        # Exponentiate in double precision, within the range.
        r = torch.exp(torch.clamp(self.log_r.double(),
                                  min = math.log(NB_R_MIN),
                                  max = math.log(NB_R_MAX)))

        # Return the log-probability mass for the negative binomial
        # distributions.
        return self.__class__.log_prob_mass(k = obs_counts,
                                            m = m,
                                            r = r,
                                            eps = self.eps)


    def loss(self,
             obs_counts: torch.Tensor,
             pred_means: torch.Tensor,
             scaling_factors: torch.Tensor,
             contamination: float = 0.0,
             contamination_r: float = 0.05) -> torch.Tensor:
        """Compute the loss, the negative log-probability mass of the
        negative binomial distributions (or, if ``contamination`` is
        above zero, of a mixture with an outlier component).

        Parameters
        ----------
        obs_counts : :class:`torch.Tensor`
            The observed gene counts.

        pred_means : :class:`torch.Tensor`
            The predicted scaled means of the negative binomial
            distributions.

            This is a tensor whose shape must match that of
            ``obs_counts``.

        scaling_factors : :class:`torch.Tensor`
            The scaling factors.

            This is a 1D tensor whose length must match that of the
            first dimension of ``obs_counts`` and ``pred_means``.

        contamination : :class:`float`, ``0.0``
            The weight of the outlier component.

        contamination_r : :class:`float`, ``0.05``
            The r-value of the outlier component.

        Returns
        -------
        loss : :class:`torch.Tensor`
            The loss for each count.

            This is a 2D tensor where:

            * The first dimension has a length equal to the length
              of the first dimension of ``obs_counts`` and
              ``pred_means``.

            * The second dimension has a length equal to the length
              of the second dimension of ``obs_counts`` and
              ``pred_means``.
        """

        # Get the loss for each of the negative binomial distributions.
        nll = - self.log_prob(obs_counts = obs_counts,
                              pred_means = pred_means,
                              scaling_factors = scaling_factors)

        # If there is no contamination, return the loss.
        if not contamination:
            return nll

        #-------------------------------------------------------------#

        # Get the log r-values of the outlier component.
        log_r_outlier = \
            torch.full_like(self.log_r,
                            math.log(contamination_r))

        # Get the loss under the outlier component.
        nll_outlier = \
            - self.__class__.log_prob_mass(
                k = obs_counts,
                m = self.__class__.rescale(
                        means = pred_means,
                        scaling_factors = scaling_factors),
                r = torch.exp(log_r_outlier.double()),
                eps = self.eps)

        #-------------------------------------------------------------#

        # Get the weighted log-probabilities of the two components.
        log_weights = \
            torch.stack(
                [math.log1p(-contamination) - nll,
                 math.log(contamination) - nll_outlier])

        # Return -log[(1-eps) * NB(r) + eps * NB(r_outlier)].
        return - torch.logsumexp(log_weights, dim = 0)


    def sample(self,
               n: int,
               pred_means: torch.Tensor,
               scaling_factors: torch.Tensor) -> torch.Tensor:
        """Get samples from the negative binomial distributions.

        Parameters
        ----------
        n : :class:`int`
            The number of samples to get.

        pred_means : :class:`torch.Tensor`
            The predicted scaled means of the negative binomial
            distributions.

        scaling_factors : :class:`torch.Tensor`
            A tensor containing the scaling factors.

            This is a 1D tensor whose length must match that
            of the first dimension of ``pred_means``.

        Returns
        -------
        samples : :class:`torch.Tensor`
            The samples drawn from the negative binomial distributions.

            The shape of this tensor depends on the shape of ``n``
            and ``pred_means``, but the first dimension always has
            a length equal to the number of samples drawn from the
            negative binomial distribution.
        """

        # Disable the gradient calculation.
        with torch.no_grad():

            # Get the rescaled means of the negative binomial
            # distributions.
            m = self.__class__.rescale(\
                    means = pred_means,
                    scaling_factors = scaling_factors)

            # Get the r-values of the negative binomial distributions.
            # Exponentiate in double precision, within the range.
            r = torch.exp(torch.clamp(self.log_r.double(),
                                      min = math.log(NB_R_MIN),
                                      max = math.log(NB_R_MAX)))

            # Get the probabilities from the means using the formula:
            # m = p * r / (1-p), so p = m / (m+r)
            probs = m / (m + r)

            # Sample from the negative binomial distributions with the
            # calculated probabilities.
            nb = dist.NegativeBinomial(total_count = r,
                                       probs = probs)

            # Get 'n' samples from the distributions.
            return nb.sample([n]).squeeze()


class OutputModuleNBFullDispersion(OutputModuleNB):

    """
    Class implementing an output layer representing the means of the
    negative binomial distributions modeling the outputs (i.e., the
    means of the gene expression counts). One negative binomial
    distribution with trainable parameters is used for each gene.
    """


    ######################### INITIALIZATION ##########################


    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 activation: str = "softplus",
                 eps: Optional[float] = None) -> None:
        """Initialize an instance of the class.

        Parameters
        ----------
        input_dim : :class:`int`
            The dimensionality of the input.

        output_dim : :class:`int`
            The dimensionality of the output.

        activation : :class:`str`, {``"sigmoid"``, ``"softplus"``}, \
            ``"softplus"``
            The name of the activation function to be used.

            Available options are:

            * ``"sigmoid"``: the sigmoid activation function.
            * ``"softplus"``: the softplus activation function.

        eps : :class:`float`, optional
            The small value the log-probability mass adds inside its
            logarithms. If not passed, the exact form is used.
        """

        # Initialize the instance.
        super().__init__(input_dim = input_dim,
                         output_dim = output_dim,
                         activation = activation,
                         eps = eps)

        # Set the layer that will contain the means of the negative
        # binomial distributions.
        self._layer_means = \
            nn.Linear(in_features = input_dim,
                      out_features = output_dim)

        # Set the layer that will contain the predicted logarithm of
        # the 'r' values of the negative binomial distributions.
        self._layer_r_values = \
            nn.Linear(in_features = input_dim,
                      out_features = output_dim)


    ######################### PUBLIC METHODS ##########################


    def forward(self,
                x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        x : :class:`torch.Tensor`
            The input tensor.

        Returns
        -------
        m : :class:`torch.Tensor`
            A tensor containing the means of the negative binomial
            distributions.

        log_r : :class:`torch.Tensor`
            A tensor containing the logarithm of the 'r' values of
            the negative binomial distributions.
        """

        # Pass the input through the first output layer.
        _m = self._layer_means(x)

        #-------------------------------------------------------------#

        # If the activation function is a sigmoid
        if self.activation == "sigmoid":

            # Get the predicted scaled means of the negative binomial
            # distributions.
            m = torch.sigmoid(_m)

        # If the activation function is a softplus
        elif self.activation == "softplus":

            # Get the predicted scaled means of the negative binomial
            # distributions.
            m = F.softplus(_m)

        #-------------------------------------------------------------#

        # Pass the input through the second output layer.
        log_r = self._layer_r_values(x)

        #-------------------------------------------------------------#

        # Return the means and the logarithm of the 'r' values of the
        # negative binomial distributions.
        return m, log_r


    def log_prob(self,
                 obs_counts: torch.Tensor,
                 pred_means: torch.Tensor,
                 pred_log_r_values: torch.Tensor,
                 scaling_factors: torch.Tensor) -> torch.Tensor:
        """Get the log-probability mass of the negative binomial
        distributions.

        Parameters
        ----------
        obs_counts : :class:`torch.Tensor`
            The observed gene counts.

            The first dimension of this tensor must have a length
            equal to the number of samples whose counts are
            reported.

        pred_means : :class:`torch.Tensor`
            The predicted scaled means of the negative binomial
            distributions.

            This is a tensor whose shape must match that of
            ``obs_counts``.

        pred_log_r_values : :class:`torch.Tensor`
            The predicted logarithm of the r-values of the negative
            binomial distributions.

            This is a tensor whose shape must match that of
            ``obs_counts`` and ``pred_means``.

        scaling_factors : :class:`torch.Tensor`
            The scaling factors.

            This is a 1D tensor whose length must match that of the
            first dimension of ``obs_counts``, ``pred_means``, and
            ``pred_log_r_values``.

        Returns
        -------
        log_prob_mass : :class:`torch.Tensor`
            The log-probability mass.

            This is a 2D tensor where:

            * The first dimension has a length equal to the length
              of the first dimension of ``obs_counts``,``pred_means``,
              and ``pred_log_r_values``.

            * The second dimension has a length equal to the length
              of the second dimension of ``obs_counts``,``pred_means``,
              and ``pred_log_r_values``.
        """

        # Get the rescaled means of the negative binomial
        # distributions.
        m = self.__class__.rescale(means = pred_means,
                                   scaling_factors = scaling_factors)

        # Get the r-values of the negative binomial distributions.
        # Exponentiate in double precision, within the range.
        r = torch.exp(torch.clamp(pred_log_r_values.double(),
                                  min = math.log(NB_R_MIN),
                                  max = math.log(NB_R_MAX)))

        # Return the log-probability mass for the negative binomial
        # distributions.
        return self.__class__.log_prob_mass(k = obs_counts,
                                            m = m,
                                            r = r,
                                            eps = self.eps)


    def loss(self,
             obs_counts: torch.Tensor,
             pred_means: torch.Tensor,
             pred_log_r_values: torch.Tensor,
             scaling_factors: torch.Tensor,
             contamination: float = 0.0,
             contamination_r: float = 0.05) -> torch.Tensor:
        """Compute the loss, the negative log-probability mass of the
        negative binomial distributions (or, if ``contamination`` is
        above zero, of a mixture with an outlier component).

        Parameters
        ----------
        obs_counts : :class:`torch.Tensor`
            The observed gene counts.

        pred_means : :class:`torch.Tensor`
            The predicted scaled means of the negative binomial
            distributions.

            This is a tensor whose shape must match that of
            ``obs_counts``.

        pred_log_r_values : :class:`torch.Tensor`
            The predicted logarithm of the r-values of the negative
            binomial distributions.

            This is a tensor whose shape must match that of
            ``obs_counts`` and ``pred_means``.

        scaling_factors : :class:`torch.Tensor`
            The scaling factors.

            This is a 1D tensor whose length must match that of the
            first dimension of ``obs_counts``, ``pred_means``, and
            ``pred_log_r_values``.

        contamination : :class:`float`, ``0.0``
            The weight of the outlier component.

        contamination_r : :class:`float`, ``0.05``
            The r-value of the outlier component.

        Returns
        -------
        loss : :class:`torch.Tensor`
            The loss for each count.

            This is a 2D tensor where:

            * The first dimension has a length equal to the length
              of the first dimension of ``obs_counts``,
              ``pred_means``, and ``pred_log_r_values``.

            * The second dimension has a length equal to the length
              of the second dimension of ``obs_counts``,
              ``pred_means``, and ``pred_log_r_values``.
        """

        # Get the loss for each of the negative binomial distributions.
        nll = - self.log_prob(obs_counts = obs_counts,
                              pred_means = pred_means,
                              pred_log_r_values = pred_log_r_values,
                              scaling_factors = scaling_factors)

        # If there is no contamination, return the loss.
        if not contamination:
            return nll

        #-------------------------------------------------------------#

        # Get the log r-values of the outlier component.
        log_r_outlier = \
            torch.full_like(pred_log_r_values,
                            math.log(contamination_r))

        # Get the loss under the outlier component.
        nll_outlier = \
            - self.log_prob(obs_counts = obs_counts,
                            pred_means = pred_means,
                            pred_log_r_values = log_r_outlier,
                            scaling_factors = scaling_factors)

        #-------------------------------------------------------------#

        # Get the weighted log-probabilities of the two components.
        log_weights = \
            torch.stack(
                [math.log1p(-contamination) - nll,
                 math.log(contamination) - nll_outlier])

        # Return -log[(1-eps) * NB(r) + eps * NB(r_outlier)].
        return - torch.logsumexp(log_weights, dim = 0)


    def sample(self,
               n: int,
               pred_means: torch.Tensor,
               pred_log_r_values: torch.Tensor,
               scaling_factors: torch.Tensor) -> torch.Tensor:
        """Get samples from the negative binomial distributions.

        Parameters
        ----------
        n : :class:`int`
            The number of samples to get.

        pred_means : :class:`torch.Tensor`
            The predicted scaled means of the negative binomial
            distributions.

        pred_log_r_values : :class:`torch.Tensor`
            The predicted logarithm of the r-values of the negative
            binomial distributions.

            This is a 2D tensor whose shape must match that of
            ``pred_means``.

        scaling_factors : :class:`torch.Tensor`
            A tensor containing the scaling factors.

            This is a 1D tensor whose length must match that
            of the first dimension of ``pred_means`` and
            ``pred_log_r_values``.

        Returns
        -------
        samples : :class:`torch.Tensor`
            The samples drawn from the negative binomial distributions.

            The shape of this tensor depends on the shape of ``n``
            and ``pred_means``/``pred_log_r_values``, but the first
            dimension always has a length equal to the number of
            samples drawn from the negative binomial distribution.
        """

        # Disable the gradient calculation.
        with torch.no_grad():

            # Get the rescaled means of the negative binomial
            # distributions.
            m = self.__class__.rescale(\
                    means = pred_means,
                    scaling_factors = scaling_factors)

            # Get the r-values of the negative binomial distributions.
            # Exponentiate in double precision, within the range.
            r = torch.exp(torch.clamp(pred_log_r_values.double(),
                                      min = math.log(NB_R_MIN),
                                      max = math.log(NB_R_MAX)))

            # Get the probabilities from the means using the formula:
            # m = p * r / (1-p), so p = m / (m+r)
            probs = m / (m + r)

            # Sample from the negative binomial distributions with the
            # calculated probabilities.
            nb = dist.NegativeBinomial(total_count = r,
                                       probs = probs)

            # Get 'n' samples from the distributions.
            return nb.sample([n]).squeeze()


#######################################################################


# Set the available output modules.
OUTPUT_MODULES = {

    # Output module for Poisson distributions.
    "poisson" : OutputModulePoisson,

    # Output module for negative binomial distributions with feature
    # dispersion.
    "nb_feature_dispersion" : OutputModuleNBFeatureDispersion,

    # Output module for negative binomial distributions with full
    # dispersion.
    "nb_full_dispersion" : OutputModuleNBFullDispersion,

    }
