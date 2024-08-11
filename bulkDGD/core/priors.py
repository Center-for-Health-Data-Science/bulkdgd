#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-

#    priors.py
#
#    This module contains the classes implementing some of the prior
#    distributions used in the DGD model. The 'softball' distribution
#    (:class:`core.priors.SoftballPrior`) is used as a prior over the
#    means of the components of the Gaussian mixture, while a
#    Gaussian distribution (:class:`core.priors.GaussianPrior`) is
#    used as a prior over the log-variance of the Gaussian mixture.
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
    "This module contains the classes implementing some of the " \
    "prior distributions used in the DGD model. The 'softball' " \
    "distribution (:class:`core.priors.SoftballPrior`) is used " \
    "as a prior over the means of the components of the " \
    "Gaussian mixture, while a Gaussian distribution " \
    "(:class:`core.priors.GaussianPrior`) is used as a prior " \
    "over the log-variance of the Gaussian mixture."


#######################################################################


# Import from the standard library.
import logging as log
import math
# Import from third-party packages.
import torch


#######################################################################


# Get the module's logger.
logger = log.getLogger(__name__)


#######################################################################


class SoftballPrior:
    
    """
    Class implementing a "softball" prior distribution.

    It is an almost uniform prior for an m-dimensional ball, with
    the logistic function making a soft (differentiable) boundary.
    """


    ######################### INITIALIZATION ##########################

    
    def __init__(self,
                 dim,
                 radius,
                 sharpness):
        """Initialize an instance of the softball prior distribution.

        Parameters
        ----------
        dim : :class:`int`
            The dimensionality of the distribution.

        radius : :class:`float`
            The radius of the soft ball.

        sharpness : :class:`int`
            The sharpness of the soft boundary of the ball.
        """
        
        # Set the dimensionality of the prior.
        self._dim = dim

        # Set the radius of the ball.
        self._radius = radius

        # Set the sharpness of the boundary.
        self._sharpness = sharpness


    ########################### PROPERTIES ############################


    @property
    def dim(self):
        """The dimensionality of the softball distribution.
        """

        return self._dim


    @dim.setter
    def dim(self,
            value):
        """Raise an exception if the user tries to modify the value of
        ``dim`` after initialization.
        """
        
        errstr = \
            "The value of 'dim' is set at initialization and cannot " \
            "be changed. If you want to change the dimensionality " \
            "of the distribution, initialize a new instance of " \
            f"'{self.__class__.__name__}'."
        raise ValueError(errstr)


    @property
    def radius(self):
        """The radius of the soft ball.
        """

        return self._radius


    @radius.setter
    def radius(self,
               value):
        """Raise an exception if the user tries to modify the value of
        ``radius`` after initialization.
        """
        
        errstr = \
            "The value of 'radius' is set at initialization and " \
            "cannot be changed. If you want to change the radius of " \
            "the soft ball, initialize a new instance of " \
            f"'{self.__class__.__name__}'."
        raise ValueError(errstr)


    @property
    def sharpness(self):
        """The sharpness of the soft boundary of the ball.
        """

        return self._sharpness


    @sharpness.setter
    def sharpness(self,
                  value):
        """Raise an exception if the user tries to modify the value of
        ``sharpness`` after initialization.
        """
        
        errstr = \
            "The value of 'sharpness' is set at initialization and " \
            "cannot be changed. If you want to change the sharpness " \
            "of the soft boundary of the ball, initialize a new " \
            f"instance of '{self.__class__.__name__}'."
        raise ValueError(errstr)


    ######################### PUBLIC METHODS ##########################
    

    def sample(self,
               n_samples):
        """Get samples from the softball distribution.
        
        Parameters
        ----------
        n_samples : :class:`int`
            The number of samples to be drawn.

        Returns
        -------
        samples : :class:`torch.Tensor`
            The samples drawn from the softball distribution.
        """

        # Disable gradient calculation.
        with torch.no_grad():
            
            # Get a tensor filled with random numbers sampled from a 
            # normal distribution with a mean of 0 and a standard
            # deviation of 1 - 'sample' is a tensor with dimensions
            # [n_samples, dim].
            samples = torch.randn((n_samples, self.dim))
            
            # Get the norm of the tensor calculated on the last
            # dimension of the tensor. Retain 'dim' in the output
            # tensor. Divide the first element of the norm by the
            # second element of the norm.
            #
            # In brief, get 'n' random directions.
            samples.div_(samples.norm(dim = -1,
                                      keepdim = True))
            
            # Get 'n' random lengths.
            local_len = \
                self.radius * \
                torch.pow(torch.rand((n_samples, 1)), 1.0 / self.dim)
            
            # ???
            samples.mul_(local_len.expand(-1, self.dim))
        
        # Return the new samples.
        return samples
    

    def log_prob(self,
                 x):
        """Return the log of the probability density function evaluated
        at ``x``.

        Parameters
        ----------
        x : :class:`torch.Tensor`
            The input tensor.

        Returns
        -------
        log_prob : :class:`torch.Tensor`
            The log of the probability density function evaluated at
            ``x``.
        """

        # Compute the norm.
        norm = \
            math.lgamma(1 + self.dim * 0.5) - \
            self.dim * (math.log(self.radius) + \
            0.5 * math.log(math.pi))
        
        # Return the log of the probability density function evaluated
        # at 'x'.
        return (norm - \
                torch.log(1 + \
                          torch.exp(\
                            self.sharpness * (x.norm(dim = -1) / \
                            self.radius-1))))


class GaussianPrior:
    
    """
    Class implementing a Gaussian prior distribution.
    """


    ######################### INITIALIZATION ##########################


    def __init__(self,
                 dim,
                 mean,
                 stddev):
        """Initialize an instance of the Gaussian distribution.

        Parameters
        ----------
        dim : :class:`int`
            The dimensionality of the distribution.
        
        mean : :class:`float`
            The mean of the Gaussian distribution.

        stddev : :class:`float`
            The standard deviation of the Gaussian distribution.
        """

        # Set the dimensionality of the distribution.
        self._dim = dim
        
        # Set the mean of the distribution.
        self._mean = mean

        # Set the standard deviation of the distribution.
        self._stddev = stddev

        # Set a normal distribution with the given mean and standard
        # deviation.
        self._dist = \
            torch.distributions.normal.Normal(self.mean,
                                              self.stddev)


    ########################### PROPERTIES ############################


    @property
    def dim(self):
        """The dimensionality of the Gaussian distribution.
        """

        return self._dim


    @dim.setter
    def dim(self,
            value):
        """Raise an exception if the user tries to modify the value of
        ``dim`` after initialization.
        """
        
        errstr = \
            "The value of 'dim' is set at initialization and cannot " \
            "be changed. If you want to change the dimensionality " \
            "of the distribution, initialize a new instance of " \
            f"'{self.__class__.__name__}'."
        raise ValueError(errstr)


    @property
    def mean(self):
        """The mean of the Gaussian distribution.
        """

        return self._mean


    @mean.setter
    def mean(self,
             value):
        """Raise an exception if the user tries to modify the value of
        ``mean`` after initialization.
        """
        
        errstr = \
            "The value of 'mean' is set at initialization and " \
            "cannot be changed. If you want to change the mean of " \
            "the distribution, initialize a new instance of " \
            f"'{self.__class__.__name__}'."
        raise ValueError(errstr)


    @property
    def stddev(self):
        """The standard deviation of the Gaussian distribution.
        """

        return self._stddev


    @stddev.setter
    def stddev(self,
               value):
        """Raise an exception if the user tries to modify the value of
        ``stddev`` after initialization.
        """
        
        errstr = \
            "The value of 'stddev' is set at initialization and " \
            "cannot be changed. If you want to change the standard " \
            "deviation of the distribution, initialize a new " \
            f"instance of '{self.__class__.__name__}'."
        raise ValueError(errstr)


    ######################### PUBLIC METHODS ##########################


    def sample(self,
               n_samples):
        """Get samples from the Gaussian distribution.

        Parameters
        ----------
        n_samples : :class:`int`
            The number of samples to be drawn.

        Returns
        -------
        samples : :class:`torch.Tensor`
            The samples drawn from the Gaussian distribution.
        """
        
        # Get the samples from the distribution and return them.
        return self._dist.sample((n_samples, self.dim))

    
    def log_prob(self,
                 x):
        """Return the log of the probability density function
        evaluated at ``x``.

        Parameters
        ----------
        x : :class:`torch.Tensor`
            The input tensor.

        Returns
        -------
        log_prob : :class:`torch.Tensor`
            The log of the probability density function evaluated
            at ``x``.
        """
        
        # Return the log probability of the density function
        # evaluated at the input(s).
        return self._dist.log_prob(x)
