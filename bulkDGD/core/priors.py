#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-

#    priors.py
#
#    Module containing the classes defining the priors.
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
    "Module containing the classes defining the priors."


# Standard library
import math
# Third-party packages
import torch


class softball():
    
    """
    Class implementing a "softball" prior.

    It is an almost uniform prior for an m-dimensional ball, with
    the logistic function making a soft (differentiable) boundary.
    """
    
    def __init__(self,
                 dim,
                 radius,
                 sharpness):
        """Initialize an instance of the "softball" distribution.

        Parameters
        ----------
        dim : ``int``
            Dimensionality of the prior.

        radius : ``float``
            Radius of the ball.

        sharpness : ``int``
            Sharpness of the "soft" boundary.
        """
        
        # Set the dimensionality of the prior
        self._dim = dim

        # Set the radius of the ball
        self._radius = radius

        # Set the sharpness of the boundary
        self._sharpness = sharpness


    #-------------------------- Properties ---------------------------#


    @property
    def dim(self):
        """The dimensionality of the softball prior.
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
    def radius(self):
        """The radius of the ball.
        """

        return self._radius


    @radius.setter
    def radius(self,
               value):
        """Raise an exception if the user tries to modify
        the value of ``radius`` after initialization.
        """
        
        errstr = \
            "The value of 'radius' cannot be changed " \
            "after initialization."
        raise ValueError(errstr)


    @property
    def sharpness(self):
        """The sharpness of the "soft" boundary of the ball.
        """

        return self._sharpness


    @sharpness.setter
    def sharpness(self,
                  value):
        """Raise an exception if the user tries to modify
        the value of ``sharpness`` after initialization.
        """
        
        errstr = \
            "The value of 'sharpness' cannot be changed " \
            "after initialization."
        raise ValueError(errstr)


    #------------------------ Public methods -------------------------#
    

    def sample(self,
               n):
        """Get ``n`` random samples from the softball distribution.
        The sampling is uniform from the ``dim``-dimensional ball,
        and approximate.
        
        Parameters
        ----------
        n : ``int``
            Number of random samples.

        Returns
        -------
        ``torch.Tensor``
            Sampled points from the softball distribution.
        """

        # Disable gradient calculation
        with torch.no_grad():
            
            # Get a tensor filled with random numbers sampled from a 
            # normal distribution with mean of 0 and a standard
            # deviation of 1 - 'sample' is a tensor with dimensions
            # [n_samples, dim]
            sample = torch.randn((n, self.dim))
            
            # Get the norm of the tensor calculated on the last
            # dimension of the tensor. Retain 'dim' in the output
            # tensor. Divide the first element of the norm by
            # the second element of the norm.
            # In brief, get 'n' random directions.
            sample.div_(sample.norm(dim = -1,
                                    keepdim = True))
            
            # Get 'n' random lengths
            local_len = \
                self.radius * \
                torch.pow(torch.rand((n, 1)), 1.0 / self.dim)
            
            # ???
            sample.mul_(local_len.expand(-1, self.dim))
        
        # Return the new sample
        return sample
    

    def log_prob(self,
                 z):
        """Return the log probabilities of the elements of a tensor
        (the last dimension is assumed to be ``z`` vectors).

        Parameters
        ----------
        z : ``torch.Tensor``
            The tensor to find the log probabilities of.

        Returns
        -------
        ``torch.Tensor``
            The log probabilities of the input tensor.
        """

        # Compute the norm
        norm = \
            math.lgamma(1 + self.dim*0.5) - \
            self.dim * (math.log(self.radius) + \
            0.5 * math.log(math.pi))
        
        # Return the log probabilities
        return (norm - \
                torch.log(1 + \
                          torch.exp(\
                            self.sharpness * (z.norm(dim = -1) / \
                            self.radius-1))))


class gaussian():
    
    """
    Class implementing a Gaussian prior used to initialize
    the Gaussian mixture model's means.
    """

    def __init__(self,
                 dim,
                 mean,
                 stddev):
        """Initialize an instance of the Gaussian distribution.

        Parameters
        ----------
        dim : ``int``
            Dimensionality of the space.
        
        mean : ``float``
            Mean of the Gaussian distribution.

        stddev : ``float``
            Standard deviation of the Gaussian distribution.
        """

        # Set the dimensionality
        self._dim = dim
        
        # Set the mean
        self._mean = mean

        # Set the standard deviation
        self._stddev = stddev

        # Set a normal distribution with the given mean
        # and standard deviation  
        self._dist = \
            torch.distributions.normal.Normal(self.mean,
                                              self.stddev)


    #-------------------------- Properties ---------------------------#


    @property
    def dim(self):
        """The dimensionality of the gaussian prior.
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
        """The mean of the Gaussian distribution.
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
        """The standard deviation of the Gaussian distribution.
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
    def dist(self):
        """The distribution associated with the prior.
        """

        return self._dist


    @dist.setter
    def dist(self,
             value):
        """Raise an exception if the user tries to modify
        the value of ``stddev`` after initialization.
        """
        
        errstr = \
            "The value of 'dist' is automatically determined " \
            "from 'mean' and 'stddev' during initialization " \
            "and it cannot be changed."
        raise ValueError(errstr)


    #------------------------ Public methods -------------------------#


    def sample(self,
               n_samples):
        """Sample from the Gaussian distribution.

        Parameters
        ----------
        n_samples : ``int``
            Number of samples.

        Returns
        -------
        ``torch.Tensor``
            Sampled points from the Gaussian distribution.
        """
        
        return self.dist.sample((n_samples, self.dim))

    
    def log_prob(self,
                 x):
        """Return the log probabilities of a tensor.

        Parameters
        ----------
        x : ``torch.Tensor``
            The input tensor.

        Returns
        -------
        ``torch.Tensor``
            The log probabilities of the input tensor.
        """
        
        return self.dist.log_prob(x)