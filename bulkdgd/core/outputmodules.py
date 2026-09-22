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

# Import from third-party libraries.
import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F


#######################################################################


# Get the module's logger.
logger = log.getLogger(__name__)


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
                 activation: str = "softplus") -> None:
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
        """

        # Initialize the instance.
        super().__init__()

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
                 activation: str = "softplus") -> None:
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
        """

        # Initialize the instance.
        super().__init__(input_dim = input_dim,
                         output_dim = output_dim,
                         activation = activation)

        # Set the layer that will contain the means of the Poisson
        # distributions.
        self._layer_means = \
            nn.Linear(in_features = input_dim,
                      out_features = output_dim)


    ######################### STATIC METHODS ##########################


    @staticmethod
    def log_prob_mass(k: torch.Tensor,
                      m: torch.Tensor) -> torch.Tensor:
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

        # Convert the "number of successes" to a double-precision
        # floating point number.
        k = k.double()

        # Set a small value used to prevent underflow and overflow.
        eps = 1.e-10

        # Get the log-probability mass of the Poisson distributions.
        x = k * torch.log(m + eps) - m - torch.lgamma(k + 1)

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
                                            m = m)


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
                r = torch.exp(log_r_outlier.double()))

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
                 activation: str = "softplus") -> None:
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
        """

        # Initialize the instance.
        super().__init__(input_dim = input_dim,
                         output_dim = output_dim,
                         activation = activation)


    ######################### STATIC METHODS ##########################


    @staticmethod
    def log_prob_mass(k: torch.Tensor,
                      m: torch.Tensor,
                      r: torch.Tensor) -> torch.Tensor:
        """Compute the natural logarithm of the probability mass for a
        set of negative binomial distributions.

        The formula used to compute the logarithm of the probability
        mass is:

        .. math::

           logPDF_{NB(k,m,r)} &=
           log\\Gamma(k+r) - log\\Gamma(r) - log\\Gamma(k+1) \\\\
           &+ k \\cdot log(m \\cdot c + \\epsilon) +
           r \\cdot log(r \\cdot c)

        Where :math:`\\epsilon` is a small value to prevent underflow/
        overflow, and :math:`c` is equal to
        :math:`\\frac{1}{r+m+\\epsilon}`.

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

        Returns
        -------
        x : :class:`torch.Tensor`
            A one-dimensional tensor containing the log-probability
            mass of each negative binomial distribution.

            Each value in the tensor corresponds to the log-probability
            mass of a different negative binomial.

        Notes
        -----
        The log-probability mass is

        .. math::

           \\log P(k \\mid m, r) &=
           \\log \\Gamma(k+r) - \\log \\Gamma(r) - \\log \\Gamma(k+1)
           \\\\
           &+ k \\log \\frac{m}{r+m} + r \\log \\frac{r}{r+m}

        with a small :math:`\\epsilon` preventing underflow.
        """

        # Compute the log-probability mass in double precision.
        k = k.double()
        m = m.double()
        r = r.double()

        # Set a small value used to prevent underflow and overflow.
        eps = 1.e-10

        #-------------------------------------------------------------#

        # Set the constant 'c' of the log-probability mass.
        c = 1.0 / (r + m + eps)

        # Get the log-probability mass of the negative binomial
        # distributions (see the Notes for the derivation).
        x = \
            torch.lgamma(k+r) - torch.lgamma(r) - \
            torch.lgamma(k+1) + k*torch.log(m*c+eps) + \
            r*torch.log(r*c)

        # Return the log-probability mass for the negative binomial
        # distributions.
        return x


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
                 activation: str = "softplus") -> None:
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
        """

        # Initialize the instance.
        super().__init__(input_dim = input_dim,
                         output_dim = output_dim,
                         activation = activation)

        # Initialize the value of the log of r. Real-valued positive
        # parameters are usually used as their log equivalent.
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
        # Exponentiate in double precision.
        r = torch.exp(self.log_r.double())

        # Return the log-probability mass for the negative binomial
        # distributions.
        return self.__class__.log_prob_mass(k = obs_counts,
                                            m = m,
                                            r = r)


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
                r = torch.exp(log_r_outlier.double()))

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
            # Exponentiate in double precision.
            r = torch.exp(self.log_r.double())

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
                 activation: str = "softplus") -> None:
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
        """

        # Initialize the instance.
        super().__init__(input_dim = input_dim,
                         output_dim = output_dim,
                         activation = activation)

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
        # Exponentiate in double precision.
        r = torch.exp(pred_log_r_values.double())

        # Return the log-probability mass for the negative binomial
        # distributions.
        return self.__class__.log_prob_mass(k = obs_counts,
                                            m = m,
                                            r = r)


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

        # Return -log[(1-eps) * NB(r) + eps * NB(r_outlier)], using
        # 'logsumexp' to survive underflow in the first component.
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
            # Exponentiate in double precision.
            r = torch.exp(pred_log_r_values.double())

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
