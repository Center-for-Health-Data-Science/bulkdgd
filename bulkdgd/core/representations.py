#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-

#    representations.py
#
#    This module contains the class implementing the representation
#    layer of the :class:`core.models.BulkDGD`
#    (:class:`core.representations.RepresentationLayer`).
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
    "The representation layer of :class:`core.models.BulkDGD` " \
    "(:class:`core.representations.RepresentationLayer`)."


#######################################################################


# Import from the standard library.
from typing import Dict, Optional, Union

# Import from third-party libraries.
import torch
import torch.nn as nn


#######################################################################


class RepresentationLayer(nn.Module):

    """
    A class implementing a representation layer, storing the
    optimizable representations of samples in the latent space.
    """


    ######################## PUBLIC ATTRIBUTE #########################


    # Set the available distributions to sample the representations
    # from.
    AVAILABLE_DISTS = \
       ["normal",
        "uniform",
        "laplace",
        "student_t",
        "cauchy",
        "uniform_ball",
        "zeros"]


    ######################### INITIALIZATION ##########################


    def __init__(self,
                 values: Optional[torch.Tensor] = None,
                 dist: str = "normal",
                 dist_options: Optional[Dict[str, Union[int, float]]] \
                    = None,
                 device: Optional[Union[str, torch.device]] = "cpu") \
                    -> None:
        """Initialize a representation layer.

        Parameters
        ----------
        values : :class:`torch.Tensor`, optional
            A tensor used to initialize the representations in
            the layer.

            This is a 2D tensor where:

            * The first dimension has a length equal to the number
              of representations in the tensor.

            * The second dimension has a length equal to the
              dimensionality of the representations.

            If the tensor is not passed, the representations will be
            initialized by sampling the distribution specified
            by ``dist``.

        dist : :class:`str`, {``"normal"``, ``"uniform"``, \
            ``"laplace"``, ``"student_t"``, ``"cauchy"``, \
                ``"uniform_ball"``, ``"zeros"``}, ``"normal"``
            The name of the distribution used to sample the
            representations, if no ``values`` are passed.

            Available options are:

            * ``"normal"`` : sample the representations from a normal
              distribution.

            * ``"uniform"`` : sample the representations from a
              uniform distribution.

            * ``"laplace"`` : sample the representations from a
              Laplace distribution.

            * ``"student_t"`` : sample the representations from a
              Student's t-distribution.

            * ``"cauchy"`` : sample the representations from a
              Cauchy distribution.

            * ``"uniform_ball"`` : sample the representations from a
              uniform distribution in a ball.

            * ``"zeros"`` : initialize the representations to zero.

        dist_options : :class:`dict`, optional
            A dictionary containing the parameters to sample the
            representations from the distribution, if no ``values``
            are passed.

            For any distribution the following keys and associated
            parameters must be provided:

            * ``"n_samples"`` : the number of samples to draw from
              the distribution.

            * ``"dim"`` : the dimensionality of the representations
              to sample from the distribution.

            Distribution-specific parameters:

            - For ``"normal"``:

              * ``"mean"`` : the mean (default: 0.0)
              * ``"stddev"`` : the standard deviation (default: 1.0)

            - For ``"uniform"``:

              * ``"low"`` : lower bound (default: -1.0)
              * ``"high"`` : upper bound (default: 1.0)

            - For ``"laplace"``:

              * ``"loc"`` : location parameter (default: 0.0)
              * ``"scale"`` : scale parameter (default: 1.0)

            - For ``"student_t"``:

              * ``"df"`` : degrees of freedom (default: 3.0)
              * ``"scale"`` : scale parameter (default: 1.0)

            - For ``"cauchy"``:

              * ``"scale"`` : scale parameter (default: 1.0)

            - For ``"uniform_ball"``:

              * ``"radius"`` : radius of the ball (default: 1.0)

            - For ``"zeros"``: No additional parameters

        device : :class:`str` or :class:`torch.device`, ``"cpu"``
            The device on which the representations should be
            initialized.

            If not specified, the representations will be initialized
            on the CPU.
        """

        # Initialize an instance of the 'nn.Module' class.
        super().__init__()

        #-------------------------------------------------------------#

        # Initialize the gradients with respect to the representations
        # to None.
        self.dz = None

        #-------------------------------------------------------------#

        # Set the device the representations are created on.
        self._device = torch.device(device) \
            if device is not None else torch.device("cpu")

        #-------------------------------------------------------------#

        # If a tensor of values was passed
        if values is not None:

            # Get the number, dimensionality, and values of the
            # representations from the tensor.
            self._n_rep, self._dim, self._z, self._options = \
                self._get_rep_from_values(values = values)

        # Otherwise
        else:

            # If the representations are to be initialized to zero
            if dist == "zeros":

                # Initialize the representations to zero.
                self._n_rep, self._dim, self._z, self._options = \
                    self._get_rep_from_zeros(options = dist_options)

            # If the representations are to be sampled from a normal
            # distribution
            elif dist == "normal":

                # Sample the representations from a normal
                # distribution.
                self._n_rep, self._dim, self._z, self._options = \
                    self._get_rep_from_normal(options = dist_options)

            # If the representations are to be sampled from a uniform
            # distribution
            elif dist == "uniform":

                # Sample the representations from a uniform
                # distribution.
                self._n_rep, self._dim, self._z, self._options = \
                    self._get_rep_from_uniform(options = dist_options)

            # If the representations are to be sampled from a uniform
            # distribution in a ball
            elif dist == "uniform_ball":

                # Sample the representations uniformly from a ball.
                self._n_rep, self._dim, self._z, self._options = \
                    self._get_rep_from_uniform_ball(\
                        options = dist_options)

            # If the representations are to be sampled from a Laplace
            # distribution
            elif dist == "laplace":

                # Sample the representations from a Laplace
                # distribution.
                self._n_rep, self._dim, self._z, self._options = \
                    self._get_rep_from_laplace(options = dist_options)

            # If the representations are to be sampled from a Student's
            # t-distribution
            elif dist == "student_t":

                # Sample the representations from a Student's
                # t-distribution.
                self._n_rep, self._dim, self._z, self._options = \
                    self._get_rep_from_student_t(\
                        options = dist_options)

            # If the representations are to be sampled from a Cauchy
            # distribution
            elif dist == "cauchy":

                # Sample the representations from a Cauchy
                # distribution.
                self._n_rep, self._dim, self._z, self._options = \
                    self._get_rep_from_cauchy(options = dist_options)

            # Otherwise
            else:

                # Get a string listing the available distributions.
                available_dists_str = \
                    ", ".join(f'{d}' for d in self.AVAILABLE_DISTS)

                # Raise an error.
                errstr = \
                    f"Unsupported distribution '{dist}'. Supported " \
                    f"distributions are: {available_dists_str}."
                raise ValueError(errstr)

        #-------------------------------------------------------------#

        # Move the representations to the device, as a parameter.
        self._z = nn.Parameter(self._z.to(self._device),
                               requires_grad = True)


    def _get_rep_from_values(self,
                             values: torch.Tensor) -> \
                                tuple[int, int,
                                      torch.nn.Parameter,
                                      None]:
        """Get the representations from a given tensor of values.

        Parameters
        ----------
        values : :class:`torch.Tensor`
            The tensor used to initialize the representations.

        Returns
        -------
        n_rep : :class:`int`
            The number of representations found in the input tensor.

        dim : :class:`int`
            The dimensionality of the representations.

        rep : :class:`torch.Tensor`
            The values of the representations.

            This is a 2D tensor where:

            * The first dimension has a length equal to the number of
              representations.

            * The second dimension has a length equal to the
              dimensionality of the representations.

        options : :obj:`None`
            No options, since the representations were not sampled.
        """

        # Get the number of representations from the first dimension of
        # the tensor.
        n_rep = values.shape[0]

        # Get the dimensionality of the representations from the last
        # dimension of the tensor.
        dim = values.shape[-1]

        #-------------------------------------------------------------#

        # Initialize a tensor with the representations.
        z = nn.Parameter(torch.zeros_like(values),
                         requires_grad = True)

        # Disable gradient calculation.
        with torch.no_grad():

            # Fill the tensor with the given values.
            z.copy_(values)

        #-------------------------------------------------------------#

        # Return the number of representations, their dimensionality,
        # their values, and no options.
        return n_rep, \
               dim, \
               z, \
               None


    def _get_rep_from_zeros(self,
                            options: dict[str, object]) -> \
                                tuple[int, int,
                                      torch.nn.Parameter,
                                      dict[str, object]]:
        """Get the representations initialized as zeros.

        Parameters
        ----------
        options : :class:`dict`
            A dictionary containing the parameters for zero
            initialization.

            The dictionary must contain the following keys,
            associated with the corresponding parameters:

            * ``"n_samples"`` : the number of representations to
              generate.

            * ``"dim"`` : the dimensionality of the representations
              to generate.

        Returns
        -------
        n_rep : :class:`int`
            The number of representations.

        dim : :class:`int`
            The dimensionality of the representations.

        rep : :class:`torch.Tensor`
            The values of the representations.

        options : :class:`dict`
            A dictionary containing the options used to initialize
            the representations.
        """

        # Get the number of representations to generate.
        n_rep = options["n_samples"]

        # Get the dimensionality of the desired representations.
        dim = options["dim"]

        #-------------------------------------------------------------#

        # Create a tensor of zeros.
        samples = torch.zeros(n_rep,
                              dim,
                              device = self.device)

        #-------------------------------------------------------------#

        # Get the values of the representations.
        z = nn.Parameter(samples, requires_grad = True)

        #-------------------------------------------------------------#

        # Return the number of representations, their dimensionality,
        # their values, and the options used to generate them.
        return n_rep, \
               dim, \
               z, \
               {"dist_name" : "zeros"}


    def _get_rep_from_normal(self,
                             options: dict[str, object]) -> \
                                tuple[int, int,
                                      torch.nn.Parameter,
                                      dict[str, object]]:
        """Get the representations by sampling from a normal
        distribution.

        Parameters
        ----------
        options : :class:`dict`
            A dictionary containing the parameters to sample the
            representations from a normal distribution.

            The dictionary must contain the following keys,
            associated with the corresponding parameters:

            * ``"n_samples"`` : the number of samples to draw from
              the normal distribution.

            * ``"dim"`` : the dimensionality of the representations
              to sample from the normal distribution.

            Optional parameters:

            * ``"mean"`` : the mean of the normal distribution used
              to generate the representations (default: 0.0).

            * ``"stddev"`` : the standard deviation of the normal
              distribution used to generate the representations
              (default: 1.0).

        Returns
        -------
        n_rep : :class:`int`
            The number of representations.

        dim : :class:`int`
            The dimensionality of the representations.

        rep : :class:`torch.Tensor`
            The values of the representations.

            This is a 2D tensor where:

            * The first dimension has a length equal to the number of
              representations.

            * The second dimension has a length equal to the
              dimensionality of the representations.

        options : :class:`dict`
            A dictionary containing the options used to initialize
            the representations.
        """

        # Get the desired number of representations to be drawn.
        n_rep = options["n_samples"]

        # Get the dimensionality of the desired representations.
        dim = options["dim"]

        # Get the mean of the normal distribution from which the
        # representations should be sampled.
        mean = options.get("mean", 0.0)

        # Get the standard deviation of the normal distribution
        # from which the representations should be sampled.
        stddev = options.get("stddev", 1.0)

        #-------------------------------------------------------------#

        # Sample the values of the representations.
        z = \
            nn.Parameter(\
                torch.normal(mean,
                             stddev,
                             size = (n_rep, dim),
                             requires_grad = True))

        #-------------------------------------------------------------#

        # Return the number of representations, their dimensionality,
        # their values, and the options used to generate them.
        return n_rep, \
               dim, \
               z, \
               {"dist_name" : "normal",
                "mean" : mean,
                "stddev" : stddev}


    def _get_rep_from_uniform(self,
                              options: dict[str, object]) -> \
                                tuple[int, int,
                                      torch.nn.Parameter,
                                      dict[str, object]]:
        """Get the representations by sampling from a uniform
        distribution.

        Parameters
        ----------
        options : :class:`dict`
            A dictionary containing the parameters to sample the
            representations from a uniform distribution.

            The dictionary must contain the following keys,
            associated with the corresponding parameters:

            * ``"n_samples"`` : the number of samples to draw from
              the distribution.

            * ``"dim"`` : the dimensionality of the representations
              to sample from the distribution.

            Optional parameters:

            * ``"low"`` : the lower bound of the uniform distribution
              (default: -1.0).

            * ``"high"`` : the upper bound of the uniform distribution
              (default: 1.0).

        Returns
        -------
        n_rep : :class:`int`
            The number of representations.

        dim : :class:`int`
            The dimensionality of the representations.

        rep : :class:`torch.Tensor`
            The values of the representations.

        options : :class:`dict`
            A dictionary containing the options used to initialize
            the representations.
        """

        # Get the desired number of representations to be drawn.
        n_rep = options["n_samples"]

        # Get the dimensionality of the desired representations.
        dim = options["dim"]

        # Get the lower bound of the uniform distribution.
        low = options.get("low", -1.0)

        # Get the upper bound of the uniform distribution.
        high = options.get("high", 1.0)

        #-------------------------------------------------------------#

        # Sample the values of the representations.
        z = nn.Parameter(\
                torch.empty(n_rep,
                            dim,
                            device = self.device).uniform_(low, high),
                requires_grad = True)

        #-------------------------------------------------------------#

        # Return the number of representations, their dimensionality,
        # their values, and the options used to generate them.
        return n_rep, \
               dim, \
               z, \
               {"dist_name" : "uniform",
                "low" : low,
                "high" : high}


    def _get_rep_from_uniform_ball(self,
                                   options: dict[str, object]) -> \
                                        tuple[int, int,
                                              torch.nn.Parameter,
                                               dict[str, object]]:
        """Get the representations by sampling uniformly from a ball.

        Parameters
        ----------
        options : :class:`dict`
            A dictionary containing the parameters to sample the
            representations uniformly from a ball.

            The dictionary must contain the following keys,
            associated with the corresponding parameters:

            * ``"n_samples"`` : the number of samples to draw from
              the distribution.

            * ``"dim"`` : the dimensionality of the representations
              to sample from the distribution.

            Optional parameters:

            * ``"radius"`` : the radius of the ball (default: 1.0).

        Returns
        -------
        n_rep : :class:`int`
            The number of representations.

        dim : :class:`int`
            The dimensionality of the representations.

        rep : :class:`torch.Tensor`
            The values of the representations.

        options : :class:`dict`
            A dictionary containing the options used to initialize
            the representations.
        """

        # Get the desired number of representations to be drawn.
        n_rep = options["n_samples"]

        # Get the dimensionality of the desired representations.
        dim = options["dim"]

        # Get the radius of the ball from which the representations
        # will be drawn.
        radius = options.get("radius", 1.0)

        #-------------------------------------------------------------#

        # Get random directions by normalizing samples from a normal
        # distribution to unit vectors.
        normal_samples = torch.randn(n_rep,
                                     dim,
                                     device = self.device)
        normal_samples_norm = torch.norm(normal_samples,
                                         dim = 1,
                                         keepdim = True)
        unit_directions = normal_samples / normal_samples_norm

        # Get random radii as u^(1/dim), with u uniform in (0, 1).
        u = torch.rand(n_rep,
                       1,
                       device = self.device)
        random_radii = radius * u.pow(1.0 / dim)

        # Generate the values of the representations by scaling the
        # unit directions by the random radii.
        samples = unit_directions * random_radii

        #-------------------------------------------------------------#

        # Create a tensor with the values of the representations.
        z = nn.Parameter(samples,
                         requires_grad = True)

        #-------------------------------------------------------------#

        # Return the number of representations, their dimensionality,
        # their values, and the options used to generate them.
        return n_rep, \
               dim, \
               z, \
               {"dist_name" : "uniform_ball",
                "radius" : radius}


    def _get_rep_from_laplace(self,
                              options: dict[str, object]) -> \
                                    tuple[int, int,
                                          torch.nn.Parameter,
                                          dict[str, object]]:
        """Get the representations by sampling from a Laplace
        distribution.

        Parameters
        ----------
        options : :class:`dict`
            A dictionary containing the parameters to sample the
            representations from a Laplace distribution.

            The dictionary must contain the following keys,
            associated with the corresponding parameters:

            * ``"n_samples"`` : the number of samples to draw from
              the distribution.

            * ``"dim"`` : the dimensionality of the representations
              to sample from the distribution.

            Optional parameters:

            * ``"loc"`` : the location (default: 0.0).

            * ``"scale"`` : the scale (default: 1.0).

        Returns
        -------
        n_rep : :class:`int`
            The number of representations.

        dim : :class:`int`
            The dimensionality of the representations.

        rep : :class:`torch.Tensor`
            The values of the representations.

        options : :class:`dict`
            A dictionary containing the options used to initialize
            the representations.
        """

        # Get the number of representations to generate.
        n_rep = options["n_samples"]

        # Get the dimensionality of the desired representations.
        dim = options["dim"]

        # Get the location of the Laplace distribution from which the
        # representations will be sampled.
        loc = options.get("loc", 0.0)

        # Get the scale of the Laplace distribution from which the
        # representations will be sampled.
        scale = options.get("scale", 1.0)

        #-------------------------------------------------------------#

        # Sample from a uniform distribution.
        uniform = torch.empty(n_rep,
                              dim,
                              device = self.device).uniform_(0, 1)

        # Get the sign of each centred uniform sample.
        sign = torch.sign(uniform - 0.5)

        # Convert the uniform samples to Laplace samples using the
        # inverse CDF.
        samples = \
            loc - scale * sign * \
                torch.log(1 - 2 * torch.abs(uniform - 0.5))

        #-------------------------------------------------------------#

        # Create a tensor with the values of the representations.
        z = nn.Parameter(samples,
                         requires_grad = True)

        #-------------------------------------------------------------#

        # Return the number of representations, their dimensionality,
        # their values, and the options used to generate them.
        return n_rep, \
               dim, \
               z, \
               {"dist_name" : "laplace",
                "loc" : loc,
                "scale" : scale}


    def _get_rep_from_student_t(self,
                                options: dict[str, object]) -> \
                                    tuple[int, int,
                                          torch.nn.Parameter,
                                          dict[str, object]]:
        """Get the representations by sampling from a Student's t
        distribution.

        Parameters
        ----------
        options : :class:`dict`
            A dictionary containing the parameters to sample the
            representations from a Student's t distribution.

            The dictionary must contain the following keys,
            associated with the corresponding parameters:

            * ``"n_samples"`` : the number of samples to draw from
              the distribution.

            * ``"dim"`` : the dimensionality of the representations
              to sample from the distribution.

            Optional parameters:

            * ``"df"`` : the degrees of freedom (default: 3.0).

            * ``"scale"`` : the scale (default: 1.0).

        Returns
        -------
        n_rep : :class:`int`
            The number of representations.

        dim : :class:`int`
            The dimensionality of the representations.

        rep : :class:`torch.Tensor`
            The values of the representations.

        options : :class:`dict`
            A dictionary containing the options used to initialize
            the representations.
        """

        # Get the number of representations to generate.
        n_rep = options["n_samples"]

        # Get the dimensionality of the desired representations.
        dim = options["dim"]

        # Get the degrees of freedom of the Student's t-distribution
        # from which the representations will be sampled.
        df = options.get("df", 3.0)

        # Get the scale of the Student's t-distribution from which the
        # representations will be sampled.
        scale = options.get("scale", 1.0)

        #-------------------------------------------------------------#

        # Get the Student's t-distribution.
        t_dist = torch.distributions.StudentT(df = df)

        # Sample from the Student's t-distribution.
        samples = t_dist.sample((n_rep, dim)).to(self.device)

        # Scale the samples.
        samples = samples * scale

        #-------------------------------------------------------------#

        # Create a tensor with the values of the representations.
        z = nn.Parameter(samples,
                         requires_grad = True)

        #-------------------------------------------------------------#

        # Return the number of representations, their dimensionality,
        # their values, and the options used to generate them.
        return n_rep, \
               dim, \
               z, \
               {"dist_name" : "student_t",
                "df" : df,
                "scale" : scale}


    def _get_rep_from_cauchy(self,
                             options: dict[str, object]) -> \
                                tuple[int, int,
                                      torch.nn.Parameter,
                                      dict[str, object]]:
        """Get the representations by sampling from a Cauchy
        distribution.

        Parameters
        ----------
        options : :class:`dict`
            A dictionary containing the parameters to sample the
            representations from a Cauchy distribution.

            The dictionary must contain the following keys,
            associated with the corresponding parameters:

            * ``"n_samples"`` : the number of samples to draw from
              the distribution.

            * ``"dim"`` : the dimensionality of the representations
              to sample from the distribution.

            Optional parameters:

            * ``"scale"`` : the scale (default: 1.0).

        Returns
        -------
        n_rep : :class:`int`
            The number of representations.

        dim : :class:`int`
            The dimensionality of the representations.

        rep : :class:`torch.Tensor`
            The values of the representations.

        options : :class:`dict`
            A dictionary containing the options used to initialize
            the representations.
        """

        # Get the number of representations to generate.
        n_rep = options["n_samples"]

        # Get the dimensionality of the desired representations.
        dim = options["dim"]

        # Get the scale of the Cauchy distribution from which the
        # representations will be sampled.
        scale = options.get("scale", 1.0)

        #-------------------------------------------------------------#

        # Get the Cauchy distribution.
        cauchy_dist = torch.distributions.Cauchy(loc = 0.0,
                                                 scale = scale)

        # Sample from the Cauchy distribution.
        samples = cauchy_dist.sample((n_rep, dim)).to(self.device)

        #-------------------------------------------------------------#

        # Create a tensor with the values of the representations.
        z = nn.Parameter(samples,
                         requires_grad = True)

        #-------------------------------------------------------------#

        # Return the number of representations, their dimensionality,
        # their values, and the options used to generate them.
        return n_rep, \
               dim, \
               z, \
               {"dist_name": "cauchy",
                "scale": scale}


    ########################### PROPERTIES ############################


    @property
    def n_rep(self):
        """The number of representations in the layer.
        """

        return self._n_rep


    @n_rep.setter
    def n_rep(self,
              value):
        """Raise an exception if the user tries to modify the value
        of ``n_rep`` after initialization.

        Parameters
        ----------
        value : :class:`int`
            The new value.
        """

        # Raise an error.
        errstr = \
            "The value of 'n_rep' is set at initialization and " \
            "cannot be changed. Initialize a new instance of " \
            f"'{self.__class__.__name__}' instead."
        raise ValueError(errstr)


    @property
    def dim(self):
        """The dimensionality of the representations.
        """

        return self._dim


    @dim.setter
    def dim(self,
            value):
        """Raise an exception if the user tries to modify the value of
        ``dim`` after initialization.

        Parameters
        ----------
        value : :class:`int`
            The new value.
        """

        # Raise an error.
        errstr = \
            "The value of 'dim' is set at initialization and cannot " \
            "be changed. Initialize a new instance of " \
            f"'{self.__class__.__name__}' instead."
        raise ValueError(errstr)


    @property
    def options(self):
        """The dictionary of options used to generate the
        representations, if no values were passed when initializing
        the layer.
        """

        return self._options


    @options.setter
    def options(self,
                value):
        """Raise an exception if the user tries to modify the value of
        ``options`` after initialization.

        Parameters
        ----------
        value : :class:`dict`
            The new value.
        """

        # Raise an error.
        errstr = \
            "The value of 'options' is set at initialization and " \
            "cannot be changed. Initialize a new instance of " \
            f"'{self.__class__.__name__}' instead."
        raise ValueError(errstr)


    @property
    def z(self):
        """The values of the representations.
        """

        return self._z


    @z.setter
    def z(self,
          value):
        """Raise an exception if the user tries to modify the value of
        ``z`` after initialization.

        Parameters
        ----------
        value : :class:`torch.Tensor`
            The new value.
        """

        # Raise an error.
        errstr = \
            "The value of 'z' is set at initialization and cannot " \
            "be changed. Initialize a new instance of " \
            f"'{self.__class__.__name__}' instead."
        raise ValueError(errstr)


    @property
    def device(self):
        """The device where the representations are.
        """

        return self._device


    @device.setter
    def device(self,
               value):
        """Move the representation layer to the selected device.

        Parameters
        ----------
        value : :class:`str` or :class:`torch.device`
            The device.
        """

        # Move the representation layer to the specified device.
        self.to(device = torch.device(value))

        # Update the device the representation layer is on.
        self._device = torch.device(value)


    ######################### PUBLIC METHODS ##########################


    @classmethod
    def load(cls,
             file: str,
             device: Optional[str | torch.device] = None) \
                -> 'RepresentationLayer':
        """Load representations from a file.

        Parameters
        ----------
        file : :class:`str`
            The file from which to load the representations.

        device : :class:`str` or :class:`torch.device`, optional
            The device where to load the representations.

        Returns
        -------
        rep_layer : \
            :class:`bulkdgd.core.representations.RepresentationLayer`
            The representation layer.
        """

        # Load the state dictionary from the specified file.
        state_dict = torch.load(file,
                                map_location = "cpu")

        # Build the representation layer on the specified device.
        rep_layer = cls(values = state_dict["z"],
                        device = device)

        # Return the representation layer.
        return rep_layer


    def forward(self,
                ixs: Optional[list[int] | torch.Tensor] = None,
                index_map: Optional[dict[int, int]] = None,
                batch_size: Optional[int] = None) -> torch.Tensor:
        """Get the values of the representations (forward pass).

        Parameters
        ----------
        ixs : :class:`list` or :class:`torch.Tensor`, optional
            The indexes of the samples whose representations should
            be returned. If not passed, all representations will be
            returned.

        index_map : :class:`dict`, optional
            A mapping from dataset indexes to representation indexes.
            Unmapped indexes raise an error.

        batch_size : :class:`int`, optional
            The size of the batches the representations are gathered
            in.

        Returns
        -------
        reps : :class:`torch.Tensor`
            A tensor containing the values of the representations for
            the samples of interest.

            This is a 2D tensor where:

            * The first dimension has a length equal to the number
              of representations.

            * The second dimension has a length equal to the
              dimensionality of the representations.
        """

        # If no indexes were provided
        if ixs is None:

            # Return the values for all representations.
            return self.z

        #-------------------------------------------------------------#

        # If an index mapping was passed
        if index_map is not None:

            # Get the indexes the mapping does not cover.
            ixs_unmapped = \
                [int(idx) for idx in ixs if int(idx) not in index_map]

            # If any index is not covered
            if ixs_unmapped:

                # Raise an error.
                errstr = \
                    f"'index_map' does not map {len(ixs_unmapped)} " \
                    "of the indexes passed (the first are: " \
                    f"{', '.join(str(i) for i in ixs_unmapped[:3])})."
                raise KeyError(errstr)

            # If the indexes are provided as a tensor
            if isinstance(ixs, torch.Tensor):

                # Map the indexes using the provided map.
                mapped_ixs = \
                    torch.tensor(\
                        [index_map[idx.item()] for idx in ixs],
                        device = ixs.device)

            # If the indexes are provided as a list
            else:

                # Map the indexes using the provided map.
                mapped_ixs = [index_map[idx] for idx in ixs]

        # Otherwise
        else:

            # If the indexes are provided as a tensor
            if isinstance(ixs, torch.Tensor):

                # Get whether any index is out of range.
                out_of_range = \
                    bool(((ixs < 0) | (ixs >= len(self.z))).any())

            # Otherwise
            else:

                # Get whether any index is out of range.
                out_of_range = \
                    any(not 0 <= idx < len(self.z) for idx in ixs)

            # If any index is out of range
            if out_of_range:

                # Raise an error.
                errstr = \
                    "Some indexes are outside the range of the " \
                    f"{len(self.z)} representations in the layer."
                raise IndexError(errstr)

            # Use the indexes as they are.
            mapped_ixs = ixs

        #-------------------------------------------------------------#

        # If a batch size was passed and the number of
        # representations to return is greater than it
        if batch_size is not None and len(mapped_ixs) > batch_size:

            # Initialize a list to hold the result chunks.
            result_chunks = []

            # For each batch of indexes
            for i in range(0,
                           len(mapped_ixs),
                           batch_size):

                # Get the current batch of indexes.
                batch_ixs = mapped_ixs[i:i+batch_size]

                # Add the representations for the current batch to the
                # result list.
                result_chunks.append(self.z[batch_ixs])

            # Concatenate the result chunks along the first dimension
            # to return a single tensor.
            return torch.cat(result_chunks,
                             dim = 0)

        #-------------------------------------------------------------#

        # Return the representations for the specified indexes.
        return self.z[mapped_ixs]


    def rescale(self) -> None:
        """Rescale the representations to zero mean and unit standard
        deviation, computed jointly across all representations.
        """

        # Flatten the tensor containing the representations' values.
        z_flat = torch.flatten(self.z.cpu().detach())

        # Get the mean and the standard deviation of the
        # representations.
        sd, m = torch.std_mean(z_flat)

        # Disable the calculation of the gradients.
        with torch.no_grad():

            # Subtract the mean from the representations.
            self.z -= m

            # Divide the representations by the standard deviation.
            self.z /= sd


    def save(self,
             file: str) -> None:
        """Save the representations to a .pth file.

        Parameters
        ----------
        file : :class:`str`
            The .pth file where to save the representations.
        """

        # Build the state dictionary containing the parameters of the
        # representations.
        state_dict = {"z" : self.z.detach().cpu(),
                      "n_rep" : self._n_rep,
                      "dim" : self._dim,
                      "options" : self._options}

        # Save the state dictionary to the specified file.
        torch.save(state_dict, file)
