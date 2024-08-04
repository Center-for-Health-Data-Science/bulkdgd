#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-

#    outputmodules.py
#
#    This module contains the classes defining the decoder's
#    output layer.
#
#    The code was originally developed by Viktoria Schuster,
#    Inigo Prada Luengo, and Anders Krogh.
#    
#    Valentina Sora modified and complemented it for the purposes
#    of this package.
#
#    Copyright (C) 2024 Valentina Sora 
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
    "This module contains the classes defining the layer " \
    "representing the negative binomial distributions used to " \
    "model the expression of the genes " \
    "(:class:`core.decoder.NBLayer`). "


#######################################################################


# Import from the standard library.
import logging as log
import math
# Import from third-party packages.
import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F


#######################################################################


# Get the module's logger.
logger = log.getLogger(__name__)


########################### PUBLIC CLASSES ############################


class OutputModuleNB(nn.Module):


    ######################## PUBLIC ATTRIBUTES ########################


    # Supported activation functions.
    ACTIVATION_FUNCTIONS = ["sigmoid", "softplus"]


    ######################### INITIALIZATION ##########################


    def __init__(self,
                 input_dim,
                 output_dim,
                 activation = "softplus"):
        """Initialize an instance of the class.

        Parameters
        ----------
        input_dim : ``int``
            The dimensionality of the input.

        output_dim : ``int``
            The dimensionality of the output.

        activation : ``str``, {``"sigmoid"``, ``"softplus"``}, \
            ``"softplus"``
            The name of the activation function to be used.
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
                        activation):
        """Get the name of the activation function after checking
        that it is supported.

        Parameters
        ----------
        activation : ``str``
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
        """
        
        errstr = \
            "The value of 'input_dim' is set at initialization " \
            "and cannot be changed. If you want to change the " \
            "dimensionality of the input, initialize a new instance " \
            f"of '{self.__class__.__name__}'."
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
        """
        
        errstr = \
            "The value of 'output_dim' is set at initialization " \
            "and cannot be changed. If you want to change the " \
            "dimensionality of the output, initialize a new " \
            f"instance of '{self.__class__.__name__}'."
        raise ValueError(errstr)


    @property
    def activation(self):
        """The activation function used in the layer.
        """

        return self._activation


    @activation.setter
    def activation(self,
                   value):
        """Raise an exception if the user tries to modify the value
        of ``activation`` after initialization.
        """
        
        errstr = \
            "The value of 'activation' is set at initialization and " \
            "cannot be changed. If you want to change the " \
            "activation function used in the layer, initialize a " \
            f"new instance of '{self.__class__.__name__}'."
        raise ValueError(errstr)


    ######################### STATIC METHODS ##########################


    @staticmethod
    def rescale(means,
                scaling_factors):
        """Rescale the means of the negative binomial distributions.

        Parameters
        ----------
        means : ``torch.Tensor``
            A 1D tensor containing the means of the negative binomials
            to be rescaled.

            In the tensor, each value represents the mean of a
            different negative binomial distribution.

        scaling_factors : ``torch.Tensor``
            The scaling factors.

            This is a 1D tensor whose length is equal to the number of
            scaling factors to be used to rescale the means.

        Returns
        -------
        ``torch.Tensor``
            The rescaled means.

            This is a 1D tensor whose length is equal to the number
            of negative binomials whose means were rescaled.
        """
        
        # Return the rescaled values by multiplying the means by the
        # scaling factors.
        return means * scaling_factors


    @staticmethod
    def log_prob_mass(k,
                      m,
                      r):
        """Compute the natural logarithm of the probability mass for a
        set of negative binomial distributions.

        Thr formula used to compute the logarithm of the probability
        mass is:

        .. math::

           logPDF_{NB(k,m,r)} &=
           log\\Gamma(k+r) - log\\Gamma(r) - log\\Gamma(k+1) \\\\ 
           &+ k \\cdot log(m \\cdot c + \\epsilon) +
           r \\cdot log(r \\cdot c)

        Where :math:`\\epsilon` is a small value to prevent underflow/
        overflow, and :math:`c` is equal to
        :math:`\\frac{1}{r+m+\\epsilon}`.

        The derivation of this formula from the non-logarithmic
        formulation of the probability mass function of the negative
        binomial distribution can be found below.

        Parameters
        ----------
        k : ``torch.Tensor``
            A one-dimensional tensor containing he "number of
            successes" seen before stopping the trials.

            Each value in the tensor corresponds to the number of
            successes in a different negative binomial.

        m : ``torch.Tensor``
            A one-dimensional tensor containing the means of the
            negative binomials.

            Each value in the tensor corresponds to the mean of a
            different negative binomial.

        r : ``torch.Tensor``
            A one-dimensional tensor containing the "number of
            failures" after which the trials end.

            Each value in the tensor corresponds to the number of
            failures in a different negative binomial.

        Returns
        -------
        x : ``torch.Tensor``
            A one-dimensional tensor containing the lhe log-probability
            mass of each negative binomials.

            Each value in the tensor corresponds to the log-probability
            mass of a different negative binomial.

        Notes
        -----
        Here, we show how we derived the formula for the logarithm of
        the probability mass of the negative binomial distribution.

        We start from the non-logarithmic version of the probability
        mass for the negative binomial, which is:

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

        # Convert the "number of successes" to a double-precision
        # floating point number.
        k = k.double()
        
        # Set a small value used to prevent underflow and overflow.
        eps = 1.e-10
        
        # Set a constant used later in the equation defining the
        # log-probability mass.
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
        
        # Return the log-probability mass for the negative binomials.
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
                 input_dim,
                 output_dim,
                 r_init,
                 activation = "softplus"):
        """Initialize an instance of the class.

        Parameters
        ----------
        input_dim : ``int``
            The dimensionality of the input.

        output_dim : ``int``
            The dimensionality of the output.

        r_init : ``float``
            The initial 'r' value.

        activation : ``str``, {``"sigmoid"``, ``"softplus"``}, \
            ``"softplus"``
            The name of the activation function to be used.
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
                   r_init,
                   output_dim):
        """Get a tensor with dimensions (1, ``dim``) filled with the
        natural logarithm of the initial value of 'r' ("number of
        failures" after which the "trials" stop).

        Parameters
        ----------
        r_init : ``int``
            The initial value for 'r', representing the "number
            of failures" after which the "trials" stop.

        output_dim : ``int``
            The dimensionality of the output.

        Returns
        -------
        log_r : ``torch.Tensor``
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
    def log_r(self):
        """The natural logarithm of the 'r' values associated with
        the negative binomial distributions (the "number of failures"
        after which the "trials" end).
        """

        return self._log_r


    @log_r.setter
    def log_r(self,
              value):
        """Raise an exception if the user tries to modify the value
        of ``log_r`` after initialization.
        """
        
        errstr = \
            "The value of 'log_r' is set at initialization and " \
            "depends on the input 'r_init' value. Therefore, it " \
            "cannot be changed. If you want to change the 'r_init' " \
            "value, initialize a new instance of " \
            f"'{self.__class__.__name__}'."
        raise ValueError(errstr)


    ######################### PUBLIC METHODS ##########################


    def forward(self,
                x):
        """Forward pass.

        Parameters
        ----------
        x : ``torch.Tensor``
            The input tensor.

        Returns
        -------
        m : ``torch.Tensor``
            A tensor containing the means of the negative binomials.
        """

        # Pass the input through the layer.
        x = self._layer_means(x)

        #-------------------------------------------------------------#

        # If the activation function is a sigmoid
        if self.activation == "sigmoid":
            
            # Get the predicted means of the negative binomials.
            m = torch.sigmoid(x)
        
        # If the activation function is a softplus
        elif self.activation == "softplus":

            # Get the predicted means of the negative binomials.
            m = F.softplus(x)

        #-------------------------------------------------------------#

        # Return the means of the negative binomials.
        return m
    

    def log_prob(self,
                 obs_counts,
                 pred_means,
                 scaling_factors):
        """Get the log-probability mass of the negative binomials.

        Parameters
        ----------
        obs_counts : ``torch.Tensor``
            The observed gene counts.

            The first dimension of this tensor must have a length
            equal to the number of samples whose counts are
            reported.

        pred_means : ``torch.Tensor``
            The predicted means of the negative binomials.

            This is a tensor whose shape must match that of
            ``obs_counts``.

        scaling_factors : ``torch.Tensor``
            The scaling factors.

            This is a 1D tensor whose length must match that
            of the first dimension of ``obs_counts`` and
            ``pred_means``.
        
        Returns
        -------
        log_prob_mass : ``torch.Tensor``
            The log-probability mass.

            This is a 2D tensor where:

            * The first dimension has a length equal to the length
              of the first dimension of ``obs_counts`` and
              ``pred_means``.

            * The second dimension has a length equal to the length
              of the second dimension of ``obs_counts`` and
              ``pred_means``.
        """

        # Get the rescaled means of the negative binomials.
        m = self.__class__.rescale(means = pred_means,
                                   scaling_factors = scaling_factors)

        # Get the 'r' values of the negative binomials.
        r = torch.exp(self.log_r)
        
        # Return the log-probability mass for the negative binomials.
        return self.__class__.log_prob_mass(k = obs_counts,
                                            m = m,
                                            r = r)

    def loss(self,
             obs_counts,
             pred_means,
             scaling_factors):
        """Compute the loss given observed the means ``obs_counts``
        and predicted means ``pred_means``, the latter rescaled by
        ``scaling_factors``.

        The loss corresponds to the negative log-probability mass.

        Parameters
        ----------
        obs_counts : ``torch.Tensor``
            The observed gene counts.

        pred_means : ``torch.Tensor``
            The predicted means of the negative binomials.

            This is a tensor whose shape must match that of
            ``obs_counts``.

        scaling_factors : ``torch.Tensor``
            The scaling factors.

            This is a 1D tensor whose length must match that of the
            first dimension of ``obs_counts`` and ``pred_means``.

        Returns
        -------
        loss : ``torch.Tensor``
            The loss associated with the input ``x``.

            This is a 2D tensor where:

            * The first dimension has a length equal to the length
              of the first dimension of ``obs_counts`` and
              ``pred_means``.

            * The second dimension has a length equal to the length
              of the second dimension of ``obs_counts`` and
              ``pred_means``.
        """  
            
        # Return a tensor with as many values as the dimensions of the
        # input 'x' (the loss for each of the negative binomials
        # associated with 'x')
        return - self.log_prob(obs_counts = obs_counts,
                               pred_means = pred_means,
                               scaling_factors = scaling_factors)


    def sample(self,
               n,
               pred_means,
               scaling_factors):
        """Get samples from the negative binomials.

        Parameters
        ----------
        n : ``int``
            The number of samples to get.

        pred_means : ``torch.Tensor``
            The predicted means of the negative binomials.

        scaling_factors : ``torch.Tensor``
            A tensor containing the scaling factors.

            This is a 1D tensor whose length must match that
            of the first dimension of ``pred_means``.
        
        Returns
        -------
        samples : ``torch.Tensor``
            The samples drawn from the negative binomial distributions.
            
            The shape of this tensor depends on the shape of ``n``
            and ``pred_means``, but the first dimension always has
            a length equal to the number of samples drawn from the
            negative binomial distribution.
        """
        
        # Disable the gradient calculation.
        with torch.no_grad():
            
            # Get the rescaled means.
            m = self.__class__.rescale(\
                    means = pred_means,
                    scaling_factors = scaling_factors)

            # Get the r-values.
            r = torch.exp(self.log_r)
            
            # Get the probabilities from the means using the formula:
            # m = p * r / (1-p), so p = m / (m+r)
            probs = m / (m + r)

            # Sample from the negative binomials with the calculated
            # probabilities.
            nb = dist.NegativeBinomial(total_count = r,
                                       probs = probs)
            
            # Get 'n' samples from the distributions.
            return nb.sample([n]).squeeze()


class OutputModuleNBFullDispersion(OutputModuleNB):
    
    """
    Class implementing an output layer representing the means of the
    negative binomial distributions modeling the outputs (i.e., the
    means of the gene expression counts). One negative binomial
    distribution with trainable parameters  is used for each gene.
    """


    ######################### INITIALIZATION ##########################


    def __init__(self,
                 input_dim,
                 output_dim,
                 activation = "softplus"):
        """Initialize an instance of the class.

        Parameters
        ----------
        input_dim : ``int``
            The dimensionality of the input.

        output_dim : ``int``
            The dimensionality of the output.

        activation : ``str``, {``"sigmoid"``, ``"softplus"``}, \
            ``"softplus"``
            The name of the activation function to be used.
        """
        
        # Initialize the instance.
        super().__init__(input_dim = input_dim,
                         output_dim = output_dim,
                         activation = activation)

        # Set the layer that will contain the means of the negative
        # binomials.
        self._layer_means = \
            nn.Linear(in_features = input_dim,
                      out_features = output_dim)

        # Set the layer that will contain the predicted logaritm of
        # the 'r' values of the negative binomials.
        self._layer_r_values = \
            nn.Linear(in_features = input_dim,
                      out_features = output_dim)


    ######################### PUBLIC METHODS ##########################


    def forward(self,
                x):
        """Forward pass.

        Parameters
        ----------
        x : ``torch.Tensor``
            The input tensor.

        Returns
        -------
        m : ``torch.Tensor``
            A tensor containing the means of the negative binomials.

        log_r : ``torch.Tensor``
            A tensor containing the logarithm of the 'r' values of
            the negative binomials.
        """
            
        # Pass the input through the layer.
        _m = self._layer_means(x)

        #-------------------------------------------------------------#

        # If the activation function is a sigmoid
        if self.activation == "sigmoid":
            
            # Get the predicted means of the negative binomials.
            m = torch.sigmoid(_m)
        
        # If the activation function is a softplus
        elif self.activation == "softplus":

            # Get the predicted means of the negative binomials.
            m = F.softplus(_m)

        #-------------------------------------------------------------#

        # Get the logarithm of the 'r' values of the negative
        # binomials.
        _log_r = self._layer_r_values(x)

        #-------------------------------------------------------------#

        # If the activation function is a sigmoid
        if self.activation == "sigmoid":
            
            # Get the predicted means of the negative binomials.
            log_r = torch.sigmoid(_log_r)
        
        # If the activation function is a softplus
        elif self.activation == "softplus":

            # Get the predicted means of the negative binomials.
            log_r = F.softplus(_log_r)

        #-------------------------------------------------------------#

        # Return the means and the logarithm of the 'r' values of the
        # negative binomials.
        return m, log_r


    def log_prob(self,
                 obs_counts,
                 pred_means,
                 pred_log_r_values,
                 scaling_factors):
        """Get the log-probability mass of the negative binomials.

        Parameters
        ----------
        obs_counts : ``torch.Tensor``
            The observed gene counts.

            The first dimension of this tensor must have a length
            equal to the number of samples whose counts are
            reported.

        pred_means : ``torch.Tensor``
            The predicted means of the negative binomials.

            This is a tensor whose shape must match that of
            ``obs_counts``.

        pred_log_r_values : ``torch.Tensor``
            The predicted logarithm of the 'r' values of the negative
            binomials.

            This is a tensor whose shape must match that of
            ``obs_counts`` and. ``pred_means``.   

        scaling_factors : ``torch.Tensor``
            The scaling factors.

            This is a 1D tensor whose length must match that of the
            first dimension of ``obs_counts``, ``pred_means``, and
            ``pred_log_r_values``.
        
        Returns
        -------
        log_prob_mass : ``torch.Tensor``
            The log-probability mass.

            This is a 2D tensor where:

            * The first dimension has a length equal to the length
              of the first dimension of ``obs_counts``,``pred_means``,
              and ``pred_log_r_values``.

            * The second dimension has a length equal to the length
              of the second dimension of ``obs_counts``,``pred_means``,
              and ``pred_log_r_values``.
        """

        # Get the rescaled means of the negative binomials.
        m = self.__class__.rescale(means = pred_means,
                                   scaling_factors = scaling_factors)

        # Get the 'r' values of the negative binomials.
        r = torch.exp(pred_log_r_values)

        # Return the log-probability mass for the negative binomials.
        return self.__class__.log_prob_mass(k = obs_counts,
                                            m = m,
                                            r = r)

    def loss(self,
             obs_counts,
             pred_means,
             pred_log_r_values,
             scaling_factors):
        """Compute the loss given observed the means ``obs_counts``,
        the predicted means ``pred_means`` (rescaled by
        ``scaling_factors``), and the predicted logarithm of the
        r-values (``pred_log_r_values``).

        The loss corresponds to the negative log-probability mass.

        Parameters
        ----------
        obs_counts : ``torch.Tensor``
            The observed gene counts.

        pred_means : ``torch.Tensor``
            The predicted means of the negative binomials.

            This is a tensor whose shape must match that of
            ``obs_counts``.

        pred_log_r_values : ``torch.Tensor``
            The predicted logarithm of the 'r' values of the negative
            binomials.

            This is a tensor whose shape must match that of
            ``obs_counts`` and. ``pred_means``.   

        scaling_factors : ``torch.Tensor``
            The scaling factors.

            This is a 1D tensor whose length must match that of the
            first dimension of ``obs_counts``, ``pred_means``, and
            ``pred_log_r_values``.

        Returns
        -------
        loss : ``torch.Tensor``
            The loss associated with the input ``x``.

            This is a 2D tensor where:

            * The first dimension has a length equal to the length
              of the first dimension of ``obs_counts``,``pred_means``,
              and ``pred_log_r_values``.

            * The second dimension has a length equal to the length
              of the second dimension of ``obs_counts``,``pred_means``,
              and ``pred_log_r_values``.
        """  
            
        # Return a tensor with as many values as the dimensions of the
        # input 'x' (the loss for each of the negative binomials
        # associated with 'x')
        return - self.log_prob(obs_counts = obs_counts,
                               pred_means = pred_means,
                               pred_log_r_values = pred_log_r_values,
                               scaling_factors = scaling_factors)


    def sample(self,
               n,
               pred_means,
               pred_log_r_values,
               scaling_factors):
        """Get samples from the negative binomials.

        Parameters
        ----------
        n : ``int``
            The number of samples to get.

        pred_means : ``torch.Tensor``
            The predicted means of the negative binomials.

        pred_log_r_values : ``torch.Tensor``
            The predicted logarithm of the r-values of the negative
            binomials.

            This is a 2D tensor whose shape must match that of
            ``pred_means``.

        scaling_factors : ``torch.Tensor``
            A tensor containing the scaling factors.

            This is a 1D tensor whose length must match that
            of the first dimension of ``pred_means`` and
            ``pred_log_r_values``.
        
        Returns
        -------
        samples : ``torch.Tensor``
            The samples drawn from the negative binomial distributions.
            
            The shape of this tensor depends on the shape of ``n``
            and ``pred_means``/``pred_log_r_values``, but the first
            dimension always has a length equal to the number of
            samples drawn from the negative binomial distribution.
        """
        
        # Disable the gradient calculation.
        with torch.no_grad():
            
            # Get the rescaled means.
            m = self.__class__.rescale(\
                    means = pred_means,
                    scaling_factors = scaling_factors)

            # Get the r-values.
            r = torch.exp(pred_log_r_values)
            
            # Get the probabilities from the means using the formula:
            # m = p * r / (1-p), so p = m / (m+r)
            probs = m / (m + r)

            # Sample from the negative binomials with the calculated
            # probabilities.
            nb = dist.NegativeBinomial(total_count = r,
                                       probs = probs)
            
            # Get 'n' samples from the distributions.
            return nb.sample([n]).squeeze()
