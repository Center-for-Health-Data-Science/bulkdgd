#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-

#    decoder.py
#
#    Module containing the classes defining the decoder,
#    the representation layer (input of the decoder), the
#    negative binomial layer (output of the decoder),
#    and helper functions.
#
#    The code was originally developed by Viktoria Schuster,
#    Inigo Prada Luengo, Yuhu Liang, and Anders Krogh.
#    
#    Valentina Sora rearranged it for the purposes of this package.
#    Therefore, only functions/methods needed for the purposes
#    of this package were retained in the code.
#
#    Copyright (C) 2023 Valentina Sora 
#                       <sora.valentina1@gmail.com>
#                       Viktoria Schuster
#                       <viktoria.schuster@sund.ku.dk>
#                       Inigo Prada Luengo
#                       <inlu@diku.dk>
#                       Yuhu Liang
#                       <yuhu.liang@di.ku.dk>
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


# Description of the module
__doc__ = \
    "Module containing the classes defining the decoder, " \
    "the representation layer (input of the decoder), the " \
    "negative binomial layer (output of the decoder), " \
    "and helper functions."


# Standard library
import logging as log
import math
# Third-party packages
import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F


# Get the module's logger
logger = log.getLogger(__name__)


#------------------------- Helper functions --------------------------#


def reshape_scaling_factor(x,
                           out_dim):
    """Reshape the scaling factor.

    Parameters
    ----------
    x : ``torch.Tensor``
        The input tensor to be reshaped.

    out_dim : ``int``
        The dimensionality of the output tensor.

    Returns
    -------
    x : ``torch.Tensor``
        The reshaped tensor.
    """
    
    # Get the dimensionality of the input tensor
    start_dim = len(x.shape)
    
    # For each extra dimension that the output tensor needs to
    # have with respect to the input tensor
    for i in range(out_dim - start_dim):

        # Add a singleton dimension
        x = x.unsqueeze(1)
    
    # Return the reshaped tensor
    return x


#------------------------ Representation layer -----------------------#


class RepresentationLayer(nn.Module):
    
    """
    Class implementing a representation layer accumulating
    ``PyTorch`` gradients.
    """
    
    def __init__(self,
                 values = None,
                 dist = "normal",
                 dist_params = None):
        """The representations are vectors in an N-dimensional real
        space.


        If no ``values`` for the representations are passed, the
        representations will be initialized as a two-dimensional
        tensor of shape (``dist_params["n_samples"]``,
        ``dist_params["dim"]``) from ``dist``.

        Parameters
        ----------
        values : ``torch.Tensor``, optional
            The tensor used to initialize the representations.

            If it is not passed, the representations will be
            initialized by sampling the distribution specified
            with `dist`.

        dist : ``str``, {``"normal"``}, default: ``"normal"``
            The name of the distribution used to sample the
            representations, if no ``values`` are passed.

            By default, the distribution is a ``"normal"``
            distribution.

        dist_params : ``dict``, optional
            A dictionary containing the parameters to sample the
            representations from the distribution, if
            ``values`` is not passed.

            For any distribution the following keys and associated
            parameters must be provided:

            * ``"dim"`` : the dimensionality of the representations
              to sample from the distribution.
            * ``"n_samples"`` : the number of samples to draw from
              the distribution.


            If ``dist`` is ``"normal"``, the dictionary must contain
            these additional key/value pairs:

            * ``"mean"`` : the mean of the normal distribution used
              to generate the representations.
            * ``"stddev"`` : the standard deviation of the normal
              distribution used to generate the representations.
        """

        # Available distributions to sample the representations
        # from
        AVAILABLE_DISTS = {"normal"}
        
        # Initialize the class
        super().__init__()
        
        # Initialize the gradients with respect to the
        # representations to None
        self.dz = None

        # If a tensor of values was passed
        if values is not None:

            # Get the representations from the tensor
            self._n_samples, self._dim, self._mean, \
                self._stddev, self._z = \
                    self._get_rep_from_values(\
                        values = values)      
        
        # Otherwise
        else:

            # If the representations are to be sampled from a
            # normal distribution
            if dist == "normal":

                # Sample the representations from a normal
                # distribution
                self._n_samples, self._dim, self._mean, \
                    self._stddev, self._z = \
                        self._get_rep_from_normal(\
                            params = dist_params)

            # Otherwise
            else:

                # Raise an error since only the normal distribution
                # is implemented so far
                available_dists_str = \
                    ", ".join(f'{d}' for d in AVAILABLE_DISTS)
                errstr = \
                    f"An invalid distribution '{dist}' was passed. " \
                    f"So far, the only distributions for which " \
                    f"the sampling of the representations has been " \
                    f"implemented are: {available_dists_str}."
                raise ValueError(errstr)

    
    #-------------------- Initialization methods ---------------------#


    def _get_rep_from_values(self,
                             values):
        """Get the representations from a given tensor of values.

        Parameters
        ----------
        values : ``torch.Tensor``
            The tensor used to initialize the representations.

        Returns
        -------
        ``tuple``
            A tuple containing:

            * An ``int`` representing the number of samples
              found in the input tensor (and, therefore,
              the number of representations).
            * An ``int`` representing the dimensionality of
              the representations.
            * A `None` representing the mean of the distribution
              from which the representations were sampled (this
              is populated by something other than ``None`` only
              when the representations are generated from a
              distribution, and not when ``values`` is passed).
            * A `None` representing the standard deviation of the
              distribution from which the representations were
              sampled (this is populated by something other than
              ``None`` only when the representations are generated
              from a distribution, and not when ``values`` is passed).
            * A ``torch.Tensor`` containing the representations.
        """

        # Inform the user that the representations will be
        # initialized from the values passed
        infostr = \
            f"The representations will be initialized from " \
            f"'values'."
        logger.info(infostr)

        # Get the number of samples from the first
        # dimension of the tensor
        n_samples = values.shape[0]
        
        # Get the dimensionality of the representations
        # from the last dimension of the tensor
        dim = values.shape[-1]

        # Initialize the mean and the standard
        # deviation to None, since no normal
        # distribution is needed to initialize
        # the representations
        mean, stddev = None, None

        # Initialize a tensor with the representations
        z = nn.Parameter(torch.zeros_like(values), 
                         requires_grad = True)

        # Fill the tensor with the given values
        with torch.no_grad():
            z.copy_(values)

        # Return the representations and the parameters used
        # to generate them
        return n_samples, dim, mean, stddev, z


    def _get_rep_from_normal(self,
                             params):
        """Get representations by sampling from a normal
        distribution.

        Parameters
        ----------
        params : ``dict``
            A dictionary containing the parameters to sample the
            representations from a normal distribution.

            The dictionary must contains the following keys,
            associated to the corresponding parameters:

            * ``"dim"`` : the dimensionality of the representations
              to sample from the normal distribution.
            * ``"n_samples"`` : the number of samples to draw from
              the normal distribution.
            * ``"mean"`` : the mean of the normal distribution used
              to generate the representations.
            * ``"stddev"`` : the standard deviation of the normal
              distribution used to generate the representations.

        Returns
        -------
        ``tuple``
            A tuple containing:

            * An ``int`` representing the number of samples
              found in the input tensor (and, therefore,
              the number of representations).
            * An ``int`` representing the dimensionality of
              the representations.
            * A number representing the mean of the normal
              distribution from which the representations were
              sampled.
            * A number representing the standard deviation of the
              normal distribution from which the representations
              were sampled.
            * A ``torch.Tensor`` containing the representations.
        """

        # Get the desired number of samples
        n_samples = params["n_samples"]

        # Get the dimensionality of the desired representations
        dim = params["dim"]

        # Get the mean of the normal distribution from which
        # the representations should be sampled from
        mean = params["mean"]

        # Get the standard deviation of the normal distribution
        # from which the representations should be sampled from
        stddev = params["stddev"]

        # Get the representations
        z = \
            nn.Parameter(\
                torch.normal(mean,
                             stddev,
                             size = (n_samples, dim),
                             requires_grad = True))
        
        # Return the representations and the parameters used
        # to generate them
        return n_samples, dim, mean, stddev, z


    #--------------------------- Properties --------------------------#


    @property
    def n_samples(self):
        """The number of samples for which a representation must
        be found.
        """

        return self._n_samples


    @n_samples.setter
    def n_samples(self,
                  value):
        """Raise an exception if the user tries to modify
        the value of ``n_samples`` after initialization.
        """
        
        errstr = \
            "The value of 'n_samples' cannot be changed " \
            "after initialization."
        raise ValueError(errstr)


    @property
    def dim(self):
        """The dimensionality of the representations.
        """

        return self._dim


    @dim.setter
    def dim(self,
            value):
        """Raise an exception if the user tries to modify
        the value of ``dim`` after initialization.
        """
        
        errstr = \
            "The value of 'dim' cannot be changed " \
            "after initialization."
        raise ValueError(errstr)


    @property
    def mean(self):
        """The mean of the distribution the representations
        were sampled from, if the user passed no ``values``
        for the representations when initializing the
        ``RepresentationLayer``.
        """

        return self._mean


    @mean.setter
    def mean(self,
             value):
        """Raise an exception if the user tries to modify
        the value of ``mean`` after initialization.
        """
        
        errstr = \
            "The value of 'mean' cannot be changed " \
            "after initialization."
        raise ValueError(errstr)


    @property
    def stddev(self):
        """The standard deviation of the distribution the
        representations were sampled from, if the user
        passed no ``values`` for the representations
        when initializing the ``RepresentationLayer``.
        """

        return self._stddev


    @stddev.setter
    def stddev(self,
               value):
        """Raise an exception if the user tries to modify
        the value of ``stddev`` after initialization.
        """
        
        errstr = \
            "The value of 'stddev' cannot be changed " \
            "after initialization."
        raise ValueError(errstr)


    @property
    def z(self):
        """The representations.
        """

        return self._z


    @z.setter
    def z(self,
          value):
        """Raise an exception if the user tries to modify
        the value of ``z`` after initialization.
        """
        
        errstr = \
            "The value of 'z' cannot be changed " \
            "after initialization."
        raise ValueError(errstr)


    #------------------------ Public methods -------------------------#


    def forward(self,
                ixs = None):
        """Forward pass. It returns the representations. You can
        select a subset of representations to be returned using
        their numerical indexes (``ixs``).

        Parameters
        ----------
        ixs : ``list``, optional
            The indexes of the samples whose representations should
            be returned. If not passed, all representations
            will be returned.

        Returns
        -------
        ``torch.Tensor``
            A tensor containing the representations for the samples
            of interest.
        """

        # If no indexes were provided
        if ixs is None:
            
            # Return all representations
            return self.z
        
        # Otherwise
        else:

            # Return the representations of the
            # samples corresponding to the
            # given indexes
            return self.z[ixs]


    def zero_grad(self):
        """Set the gradients associated with the representations
        to zero.
        """

        # If the gradients are not None
        if self.z.grad is not None:

            # Create a tensor which detaches the output
            # from the computational graph (no gradients
            # will be backpropagated along this variable)
            self.z.grad.detach_()

            # Set the gradients to zero
            self.z.grad.zero_()


    def rescale(self):
        """Rescale the representations by subtracting the mean
        of all representations from each of them and dividing
        them by the standard deviation of all representations.

        Given :math:`N` samples, we can indicate with :math:`z^{n}`
        the representation of sample :math:`x^{n}`. The rescaled
        representation :math:`z^{n}_{rescaled}` will be, therefore:
        
        .. math::

           z^{n}_{rescaled} = \\frac{z^{n} - \\bar{z}}{s}

        Where :math:`\\bar{z}` is the mean of the representations
        and :math:`s` is the standard deviation.
        """
        
        # Flatten the tensor with the representations
        z_flat = torch.flatten(self.z.cpu().detach())
        
        # Get the mean and the standard deviation
        sd, m = torch.std_mean(z_flat)
        
        # Disable the gradient calculation
        with torch.no_grad():

            # Subtract the mean from the representations
            self.z -= m

            # Divide the representations by the standard
            # deviation
            self.z /= sd


#---------------------- Negative binomial layer ----------------------#


class NBLayer(nn.Module):
    
    """
    Class implementing an output layer representing the means
    of the negative binomial distributions modeling the outputs
    (e.g., the means of the gene expression counts). One negative
    binomial distribution with trainable parameters is used for
    each gene.

    The negative binomial models the number of "successes" in a
    sequence of independent and identically distributed Bernoulli
    trials before a specified number of "failures" occur.
    
    A Bernoulli trial is a trial where there are only two
    possible mutually exclusive outcomes. 
    """

    # Supported scaling types for the means of the negative binomial
    # distrbutions
    SCALING_TYPES = \
        ["library", "total_count", "mean", "median"]


    def __init__(self,
                 dim,
                 r_init,
                 scaling_type = "library",
                 reduction = None):
        """Initialize an instance of the class.

        Parameters
        ----------
        dim : ``int``
            The dimensionality of the negative binomial
            distributions.

        r_init : ``int``
            The initial value for ``r``, representing the "number
            of failures" after which the "trials" stop.

        scaling_type : ``str``, {``"library"``, ``"total_count"``, \
                       ``"mean"``, ``"median"``}, default: \
                       ``"library"``
            The type of scaling applied to the means of the
            negative binomial distributions.

            The type of scaling determines the activation
            function used in the ``NBLayer``.

            If the scaling type is ``"library"`` or ``"total_count"``,
            the acivation function will be ``"sigmoid"``.

            If the scaling type is ``"mean"`` or ``"median"``,
            the activation function will be ``"softplus"``.

        reduction : ``str``, optional, {``"sum"``}
            Whether to reduce the output by summing over its
            components when computing the output's loss
            (``"sum"``) or to compute a per-component loss
            (if not passed).
        """
        
        # Initialize the class
        super().__init__()

        # Set the dimensionality of the NBLayer
        self._dim = dim

        # Set the type of scaling
        self._scaling_type = \
            self._get_scaling_type(scaling_type = scaling_type)

        # Initialize the value of the log of r. Real-valued positive
        # parameters are usually used as their log equivalent.
        self._log_r = \
            self._get_log_r(r_init = r_init,
                            dim = self.dim)
        
        # Get the activation function to be used
        self._activation = \
            self._get_activation(scaling_type = self.scaling_type)
        
        # Set the reduction
        self._reduction = reduction


    #-------------------- Initialization methods ---------------------#


    def _get_scaling_type(self,
                          scaling_type):
        """Return the scaling type after checking whether it is
        supported.

        Parameters
        ----------
        scaling_type : ``str``, {``"library"``, ``"total_count"``, \
                       ``"mean"``, ``"median"``}, default: \
                       ``"library"``
            The type of scaling applied to the means of the
            negative binomial distributions.

            The type of scaling determines the activation
            function used in the ``NBLayer``.

            If the scaling type is ``"library"`` or ``"total_count"``,
            the acivation function will be ``"sigmoid"``.

            If the scaling type is ``"mean"`` or ``"median"``,
            the activation function will be ``"softplus"``.

        Returns
        -------
        scaling_type : ``str``, {``"library"``, ``"total_count"``, \
                       ``"mean"``, ``"median"``}, default: \
                       ``"library"``
            The type of scaling applied to the means of the
            negative binomial distributions.

            The type of scaling determines the activation
            function used in the ``NBLayer``.

            If the scaling type is ``"library"`` or ``"total_count"``,
            the acivation function will be ``"sigmoid"``.

            If the scaling type is ``"mean"`` or ``"median"``,
            the activation function will be ``"softplus"``.
        """

        # If the scaling type is not supported
        if not scaling_type in self.SCALING_TYPES:

            # Raise an error
            supported_stypes = \
                ", ".join([f"'{st}'" for st in self.SCALING_TYPES])
            errstr = \
                f"Invalid 'scaling_type' provided " \
                f"('{scaling_type}'). Supported scaling types " \
                f"are: {supported_stypes}."
            raise ValueError(errstr)

        # Return the scaling type
        return scaling_type
    

    def _get_log_r(self,
                   r_init,
                   dim):
        """Get a tensor with dimensions (1, ``dim``) filled with the
        natural logarithm of the initial value of ``r`` ("number of
        failures" after which the "trials" stop).

        Parameters
        ----------
        dim : ``int``
            The dimensionality of the negative binomial
            distributions.

        r_init : ``int``
            The initial value for ``r``, representing the "number
            of failures" after which the "trials" stop.

        Returns
        -------
        ``torch.Tensor``
            Tensor containing the ``r_init`` value as many times
            as the number of dimensions of the space the
            negative binomials live in.
        """

        return \
            nn.Parameter(torch.full(fill_value = math.log(r_init),
                                    size = (1, dim)),
                         requires_grad = True)


    def _get_activation(self,
                        scaling_type):
        """Get the activation function based on the scaling
        requested.

        Parameters
        ----------
        scaling_type : ``str``, {``"library"``, ``"total_count"``, \
                       ``"mean"``, ``"median"``}, default: \
                       ``"library"``
            The type of scaling applied to the means of the
            negative binomial distributions.

            The type of scaling determines the activation
            function used in the ``NBLayer``.

            If the scaling type is ``"library"`` or ``"total_count"``,
            the acivation function will be ``"sigmoid"``.

            If the scaling type is ``"mean"`` or ``"median"``,
            the activation function will be ``"softplus"``.

        Returns
        -------
        activation : ``str``, {``sigmoid``, ``softplus``}
            The name of the activation function to be used
            (depending on the provided scaling type). If the
            scaling type is ``"library"`` or ``"total_count"``
            the acivation function will be ``"sigmoid"``.
            If the scaling type is ``"mean"`` or ``"median"``
            the activation function will be ``"softplus"``.
        """

        # If scaling based on the library or total count
        # was requested
        if scaling_type in ["library", "total_count"]:
            
            # The activation function will be a sigmoid
            activation = "sigmoid"
        
        # If scaling based on the mean or median was requested
        elif scaling_type in ["mean", "median"]:

            # The activation function will be a softplus
            activation = "softplus"
        
        # Otherwise
        else:

            # Raise an exception
            errstr = \
                f"Unknown 'scaling_type' ({scaling_type}). " \
                f"Supported values are: " \
                f"{', '.join(self.SCALING_TYPES)}."
            raise ValueError(errstr)

        # Return the activation
        return activation


    #-------------------------- Properties ---------------------------#


    @property
    def dim(self):
        """The dimensionality of the ``NBLayer``.
        """

        return self._dim


    @dim.setter
    def dim(self,
            value):
        """Raise an exception if the user tries to modify
        the value of ``dim`` after initialization.
        """
        
        errstr = \
            "The value of 'dim' cannot be changed " \
            "after initialization."
        raise ValueError(errstr)


    @property
    def scaling_type(self):
        """The type of scaling used.
        """

        return self._scaling_type


    @scaling_type.setter
    def scaling_type(self,
                     value):
        """Raise an exception if the user tries to modify
        the value of ``scaling_type`` after initialization.
        """
        
        errstr = \
            "The value of 'scaling_type' cannot be changed " \
            "after initialization."
        raise ValueError(errstr)


    @property
    def log_r(self):
        """The natural logarithm of the ``r`` values associated
        with the negative binomial distributions (the "number
        of failures" after which the "trials" end).
        """

        return self._log_r


    @log_r.setter
    def log_r(self,
              value):
        """Raise an exception if the user tries to modify
        the value of ``log_r`` after initialization.
        """
        
        errstr = \
            "The value of 'log_r' cannot be changed " \
            "after initialization."
        raise ValueError(errstr)


    @property
    def activation(self):
        """The activation function used in the ``NBLayer``,
        which depends on the scaling type.
        """

        return self._activation


    @activation.setter
    def activation(self,
                   value):
        """Raise an exception if the user tries to modify
        the value of ``activation`` after initialization.
        """
        
        errstr = \
            "The value of 'activation' cannot be changed " \
            "after initialization."
        raise ValueError(errstr)


    @property
    def reduction(self):
        """Whether to reduce the output by summing over its
        components when computing the output's loss or to
        compute a per-component loss.
        """

        return self._reduction


    @reduction.setter
    def reduction(self,
                  value):
        """Raise an exception if the user tries to modify
        the value of ``reduction`` after initialization.
        """
        
        errstr = \
            "The value of 'reduction' cannot be changed " \
            "after initialization."
        raise ValueError(errstr)


    #------------------------ Public methods -------------------------#


    def forward(self,
                x):
        """Forward pass. Pass the input tensor through the
        activation function and return the result.

        Parameters
        ----------
        x : ``torch.Tensor``
            The input tensor (containing the mean gene
            expression counts).

        Returns
        -------
        ``torch.Tensor``
            A tensor containing the result of passing the input
            tensor through the activation function.
        """
        
        # If the activation function is a sigmoid
        if self.activation == "sigmoid":
            
            # Pass the input through the sigmoid
            # function
            return torch.sigmoid(x)
        
        # If the activation function is a softplus
        elif self.activation == "softplus":

            # Pass the input through the softplus
            # function
            return F.softplus(x)


    def log_nb_density(self,
                       k,
                       m,
                       r):
        """Compute the logarithm of the probability density
        for the negative binomial distributions.

        Thr formula used to compute the logarithm of the
        probability density is:

        .. math::

           logPDF_{NB(k,m,r)} =
           log\\Gamma(k+r) - log\\Gamma(r) - log\\Gamma(k+1) +
           k \\cdot log(m \\cdot c + \\epsilon) +
           r \\cdot log(r \\cdot c)

        Where :math:`\\epsilon` is a small value to prevent
        underflow/overflow, and :math:`c` is equal to
        :math:`\\frac{1}{r+m+\\epsilon}`.

        The derivation of this formula from the non-logarithmic
        formulation of the probability density function of the
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
            binomials).

        Returns
        -------
        x : ``torch.Tensor``
            The log-density of the negative binomials (each
            value in the tensor corresponds to the
            log-density of a different negative binomial).

        Notes
        -----
        Here, we show how we derived the formula for the logarithm
        of the probability density of the negative binomial
        distribution.

        We start from the non-logarithmic version of the
        probability density for the negative binomial, which is:

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

           logPDF_{NB(k,m,r)} = \
           log\\Gamma(k+r) - log\\Gamma(r) - log\\Gamma(k+1) +
           k \\cdot log \\left( \\frac{m}{r+m} \\right) +
           r \\cdot log \\left( \\frac{r}{r+m} \\right)
        
        Here, we are adding a small value :math:`\\epsilon` to
        prevent underflow/overflow:

        .. math::

           logPDF_{NB(k,m,r)} = \
           log\\Gamma(k+r) - log\\Gamma(r) - log\\Gamma(k+1) +
           k \\cdot
           log \\left( m \\cdot \\frac{1}{r+m+\\epsilon} 
           + \\epsilon \\right) +
           r \\cdot
           log \\left( r \\cdot \\frac{1}{r+m+\\epsilon}
           \\right)

        Finally, we substitute :math:`\\frac{1}{r+m+\\epsilon}`
        with :math:`c` and we obtain:

        .. math::

           logPDF_{NB(k,m,r)} = \
           log\\Gamma(k+r) - log\\Gamma(r) - log\\Gamma(k+1) +
           k \\cdot
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
        # the log-density
        c = 1.0 / (r + m + eps)
        
        # Get the log-density of the negative binomials.
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
        
        # Return the log-density
        return x


    def rescale(self,
                scaling_factor,
                mean):
        """Rescale the mean of the means of the negative binomial
        distributions by a per-batch scaling factor.

        Parameters
        ----------
        scaling_factor : ``torch.Tensor``
            The scaling factor.

            This tensor has dimensions (n_batches, 1),
            meaning there is a scaling factor for each batch
            of input samples (for instance, the mean gene
            expression in that batch).

        mean : ``torch.Tensor``
            A one-dimensional tensor containing the predicted
            means of the negative binomials (the output of the
            decoder).

            In the tensor, each value represents the 
            predicted mean of the expression of a gene.

        Returns
        -------
        ``torch.Tensor``
            The rescaled values.
        """
        
        # Return the rescaled values by multiplying the
        # scaling factor by the means
        return scaling_factor * mean


    def loss(self,
             x,
             scaling_factor,
             mean):
        """Compute the loss for a given input ``x``.

        The loss is calculated as the negative log-likelihood
        of ``x``. In other words, minimizing the loss for
        ``x`` corresponds to maximizing the log-likelihood
        of ``x``.

        Parameters
        ----------
        x : ``torch.Tensor``
            Th input tensor.

        scaling_factor : ``torch.Tensor``
            The scaling factor.

            This tensor has dimensions (n_batches, 1),
            meaning there is a scaling factor for each batch
            of input samples (for instance, the mean gene
            expression in that batch).

        mean : ``torch.Tensor``
            A one-dimensional tensor containing the predicted
            means of the negative binomials (the output of the
            decoder).

            In the tensor, each value represents the 
            predicted mean of the expression of a gene.

        Returns
        -------
        ``torch.Tensor``
            * If the ``reduction`` property is ``None``, the
              tensor will contain as many values as the
              dimensions of the input ``x`` (the loss for each
              of the negative binomials associated with ``x``).

            * If the ``reduction`` property is ``sum``, the tensor
              will contain a single value (the total loss for
              the given input ``x``).
        """

        # If no reduction method was defined
        if self.reduction is None:
            
            # Return a tensor with as many values as the
            # dimensions of the input ``x`` (the loss for each
            # of the negative binomials associated with ``x``)
            return - self.log_nb_density(\
                            x,
                            self.rescale(scaling_factor, mean),
                            torch.exp(self.log_r))
        
        # If a reduction of the output needs to be performed by
        # summing up its components
        elif self.reduction == "sum":
            
            # Return a tensor with only one value (the total loss
            # associated with ``x``)
            return - self.log_nb_density(\
                            x,
                            self.rescale(scaling_factor, mean),
                            torch.exp(self.log_r)).sum()


    def log_prob(self,
                 x,
                 scaling_factor,
                 mean):
        """Get the log-likelihood of an input ``x``.

        Parameters
        ----------
        x : ``torch.Tensor``
            Input tensor.

        scaling_factor : ``torch.Tensor``
            The scaling factor.

            This tensor has dimensions (n_batches, 1),
            meaning there is a scaling factor for each batch
            of input samples (for instance, the mean gene
            expression in that batch).

        mean : ``torch.Tensor``
            A one-dimensional tensor containing the predicted
            means of the negative binomials (the output of the
            decoder).

            In the tensor, each value represents the 
            predicted mean of the expression of a gene.
        
        Returns
        -------
        ``torch.Tensor``
            Log-likelihood of the input.
        """
        
        return self.log_nb_density(\
                    x,
                    self.rescale(scaling_factor, mean),
                    torch.exp(self.log_r))


    def sample(self,
               n,
               scaling_factor,
               mean):
        """Get ``n`` samples from the negative binomials.

        Parameters
        ----------
        n : ``int``
            Number of samples to get.

        scaling_factor : ``torch.Tensor``
            The scaling factor.

            This tensor has dimensions (n_batches, 1),
            meaning there is a scaling factor for each batch
            of input samples (for instance, the mean gene
            expression in that batch).

        mean : ``torch.Tensor``
            A one-dimensional tensor containing the predicted
            means of the negative binomials (the output of the
            decoder).

            In the tensor, each value represents the 
            predicted mean of the expression of a gene.
        
        Returns
        -------
        ???
            ???
        """
        
        # Disable gradient calculation
        with torch.no_grad():
            
            # Rescale the means
            m = self.rescale(scaling_factor, mean)
            
            # Get the probabilities from the means
            # m = p * r / (1-p), so p = m / (m+r)
            probs = m / (m + torch.exp(self.log_r))

            # Sample from the negative binomials with the
            # calculated probabilities
            nb = \
                dist.NegativeBinomial(\
                    torch.exp(self.log_r),
                    probs = probs)
            
            # Get 'n' samples from the distributions
            return nb.sample([n]).squeeze()


#------------------------------ Decoder ------------------------------#


class Decoder(nn.Module):
    
    """
    Class implementing the decoder.
    """

    def __init__(self,
                 n_neurons_latent,
                 n_neurons_hidden1,
                 n_neurons_hidden2,
                 n_neurons_out,
                 r_init = 2,
                 scaling_type = "library"):
        """Initialize an instance of the neural network representing
        the decoder.

        Parameters
        ----------
        n_neurons_latent : ``int``
            The mumber of neurons in the layer facing the latent
            space (input layer).

        n_neurons_hidden1 : ``int``
            The number of neurons in the first hidden layer.

        n_neurons_hidden2 : ``int``
            The number of neurons in the second hidden layer.

        n_neurons_out : ``int``
            The number of neurons in the output layer.

        r_init : ``int``
            The initial value for ``r``, representing the "number
            of failures" after which the "trials" stop in the
            ``NBLayer``.

        scaling_type : ``str``, {``"library"``, ``"total_count"``, \
                       ``"mean"``, ``"median"``}, default: \
                       ``"library"``
            The type of scaling applied to the means of the
            negative binomial distributions.

            The type of scaling determines the activation
            function used in the ``NBLayer``.

            If the scaling type is ``"library"`` or ``"total_count"``,
            the acivation function will be ``"sigmoid"``.

            If the scaling type is ``"mean"`` or ``"median"``,
            the activation function will be ``"softplus"``.
        """

        # Initialize the class
        super().__init__()

        # Create a list of modules
        self.main = nn.ModuleList()

        # Add layers to the decoder
        self.main.extend(\
            [nn.Linear(n_neurons_latent, n_neurons_hidden1),
             nn.ReLU(True),
             nn.Linear(n_neurons_hidden1, n_neurons_hidden2),
             nn.ReLU(True),
             nn.Linear(n_neurons_hidden2, n_neurons_out)])

        # Create the output layer
        self.nb = \
            NBLayer(dim = n_neurons_out,
                    r_init = r_init,
                    scaling_type = scaling_type)


    def forward(self,
                z):
        """Perform the forward pass through the neural network.

        Parameters
        ----------
        z : ``torch.Tensor``
            A tensor holding the representations to pass
            through the decoder.

        Returns
        -------
        ``torch.Tensor``
            A tensor holding the outputs of the decoder
            for the given representations.
        """

        # For each layer of the neural network
        for i in range(len(self.main)):
            
            # Pass through the layer and find the
            # intermediate (or final) representation
            z = self.main[i](z)

        # Return the final representations
        return self.nb(z)