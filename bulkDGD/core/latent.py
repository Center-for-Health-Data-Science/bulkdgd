#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-

#    latent.py
#
#    The code was originally developed by Viktoria Schuster,
#    Inigo Prada Luengo, and Anders Krogh.
#    
#    Valentina Sora modified and complemented it for the purposes
#    of this package.
#
#    Copyright (C) 2023 Valentina Sora 
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


# Description of the module
__doc__ = \
    "This module contains the classes implementing the components " \
    "of the DGD model's latent space, namely the Gaussian " \
    "mixture model (:class:`core.latent.GaussianMixtureModel`) " \
    "and the representation layer " \
    "(:class:`core.latent.RepresentationLayer`), which feeds " \
    "the (:class:`core.decoder.Decoder`)."


# Standard library
import logging as log
import math
# Third-party packages
import torch
import torch.distributions
import torch.nn as nn
# bulkDGD
from . import priors


# Get the module's logger
logger = log.getLogger(__name__)


class GaussianMixtureModel(nn.Module):
    
    """
    A class implementing a mixture of multivariate Gaussians
    (Gaussian mixture model or GMM).
    """

    # Supported priors over the means of the components
    # of the Gaussian mixture model
    MEANS_PRIORS = ["softball"]

    # Supported priors over the weights of the components
    # of the Gaussian mixture model
    WEIGHTS_PRIORS = ["dirichlet"]

    # Supported priors over the negative log-variance of the
    # components of the Gaussian mixture model
    LOG_VAR_PRIORS = ["gaussian"]


    def __init__(self,
                 dim,
                 n_comp,
                 means_prior_name,
                 weights_prior_name,
                 log_var_prior_name,
                 means_prior_options,
                 weights_prior_options,
                 log_var_prior_options,
                 cm_type = "diagonal"):
        """Initialize an instance of the GMM.

        Parameters
        ----------
        dim : ``int``
            The dimensionality of the Gaussian mixture model.

        n_comp : ``int``
            The number of components in the Gaussian mixture.

        means_prior_name : ``str``, {``"softball"``}
            The name of the prior over the means of the components
            of the Gaussian mixture.

        weights_prior_name : ``str``, {``"dirichlet"``}
            The name of the prior over the weights of the components
            of the Gaussian mixture.

        log_var_prior_name : ``str``, {``"gaussian"``}
            The name of the prior over the negative log-variance of the
            components of the Gaussian mixture.

        means_prior_options : ``dict``
            A dictionary containing the options needed to set
            up the prior over the means of the components of the
            Gaussian mixture model.

            It varies according to the selected prior.

            If ``means_prior_name`` is ``"softball"``, the
            options that must be provided are:

            * ``"radius"``, namely the radius of the multi-
              dimensional soft ball.

            * ``"sharpness"``, namely the sharpness of the
              soft boundary of the ball.

        weights_prior_options : ``dict``
            A dictionary containing the options needed to set
            up the prior over the weights of the components of
            the Gaussian mixture model.

            It varies according to the selected prior.

            If ``weights_prior_name`` is ``"dirichlet"``, the
            options that must be provided are:

            * ``"alpha"``, namely the alpha of the Dirichlet
              distribution.

        log_var_prior_options : ``dict``
            A dictionary containing the options needed to set
            up the prior over the negative log-variance of the
            Gaussian mixture model.

            It varies according to the selected prior.

            If ``log_var_prior_name`` is ``"gaussian"``, the
            options that must be provided are:

            * ``"mean"``, namely the mean of the Gaussian
              distribution.

            * ``"stddev"``, namely the standard deviation of the
              Gaussian distribution.

        cm_type : ``str``, {``"fixed"``, ``"isotropic"``, \
        ``"diagonal"``}, ``"diagonal"``
            The shape of the covariance matrix.
        """

        # Initialize the class
        super().__init__()
        
        # Set the dimensionality of the space
        self._dim = dim

        # Set the number of components in the mixture
        self._n_comp = n_comp

        # Set the type of the covariance matrix
        self._cm_type = cm_type

        #--------------------------- Means ---------------------------#

        # Set the prior over the means of the components
        self._means_prior = \
            self._get_means_prior(\
                means_prior_name = means_prior_name,
                means_prior_options = means_prior_options)

        # Set the means
        self._means = self._get_means()

        #-------------------------- Weights --------------------------#

        # Set the prior over the weights of the components
        self._weights_prior = \
            self._get_weights_prior(\
                weights_prior_name = weights_prior_name,
                weights_prior_options = weights_prior_options)

        # Set the weights
        self._weights = self._get_weights()

        #----------------------- Log-variance ------------------------#

        # Set the prior over the log-variance of the components
        self._log_var_prior = \
            self._get_log_var_prior(\
                log_var_prior_name = log_var_prior_name,
                log_var_prior_options = log_var_prior_options)

        # Get the log-variance
        self._log_var = self._get_log_var()


    #-------------------- Initialization methods ---------------------#


    def _get_means_prior(self,
                         means_prior_name,
                         means_prior_options):
        """Get the prior over the means of the components of the
        Gaussian mixture model.

        Parameters
        ----------
        means_prior_name : ``str``
            The name of the prior.

        means_prior_options : ``dict``
            The options to set up the prior.

        Returns
        -------
        means_prior_dict : ``dict``
            A dictionary containing the name of the prior and
            the options and distribution associated with it.
        """

        #---------------------- Softball prior -----------------------#

        # If the prior is the softball distribution
        if means_prior_name == "softball":

            # Get the distribution
            dist = \
                priors.SoftballPrior(dim = self.dim,
                                     **means_prior_options)

            # Return the dictionary with the name of the prior
            # and associated options and distribution
            return {"name" : means_prior_name,
                    "options" : \
                        {"dim" : self.dim,
                         **means_prior_options},
                    "dist" : dist}

        #--------------------- Unsupported prior ---------------------#

        # Otherwise
        else:

            # Raise an error
            errstr = \
                f"Unrecognized prior '{means_prior_name}' passed " \
                "to 'means_prior_name'. Supported priors are: " \
                f"{', '.join(self.MEANS_PRIOR)}."


    def _get_means(self):
        """Return the prior on the means of the Gaussians
        and the the means themselves.

        Returns
        -------
        means : ``torch.Tensor``
            The means of the Gaussian mixture components
            sampled from the prior.

            This is a 2D tensor where:
            
            * The first dimension has a length equal to the
              number of components in the Gaussian mixture.

            * The second dimension has a length equal to the
              dimensionality of the Gaussian mixture model.
        """

        # Get the distribution representing the prior
        dist_prior = self.means_prior["dist"]

        # Means with shape: n_comp, dim
        means = \
            nn.Parameter(dist_prior.sample(n_samples = self.n_comp),
                         requires_grad = True)

        # Return the means sampled from the prior
        return means


    def _get_weights_prior(self,
                           weights_prior_name,
                           weights_prior_options):
        """Get the prior over the weights of the components of the
        Gaussian mixture model.

        Parameters
        ----------
        weights_prior_name : ``str``
            The name of the prior.

        weights_prior_options : ``dict``
            The options to set up the prior.

        Returns
        -------
        weights_prior_dict : ``dict``
            A dictionary containing the name of the prior and
            the options associated with it.
        """

        #---------------------- Dirichlet prior ----------------------#

        # If the prior is the Dirichlet distribution
        if weights_prior_name == "dirichlet":

            # Get the alpha
            alpha = weights_prior_options["alpha"]

            # Calculate the Dirichlet constant
            constant = math.lgamma(self.n_comp * alpha) - \
                                   self.n_comp * math.lgamma(alpha)

            # Return the dictionary with the name of the prior
            # and associated options
            return {"name" : weights_prior_name,
                    "options" : \
                        {"alpha" : alpha,
                         "constant" : constant}}

        #--------------------- Unsupported prior ---------------------#

        # Otherwise
        else:

            # Raise an error
            errstr = \
                f"Unrecognized prior '{weights_prior_name}' passed " \
                "to 'weights_prior_name'. Supported priors are: " \
                f"{', '.join(self.WEIGHTS_PRIORS)}."


    def _get_weights(self):
        """Return the weights of the components in the Gaussian
        mixture model.

        Returns
        -------
        ``torch.Tensor``
            The weights of the components in the Gaussian mixture.
            
            The is a 1D tensor having a length equal to the number
            of components in the Gaussian mixture model.
        """

        return nn.Parameter(torch.ones(self.n_comp),
                            requires_grad = True)


    def _get_log_var_prior(self,
                               log_var_prior_name,
                               log_var_prior_options):
        """Get the prior over the log-variance of the components
        of the Gaussian mixture model.

        Parameters
        ----------
        log_var_prior_name : ``str``
            The name of the prior.

        log_var_prior_options : ``dict``
            The options to set up the prior.

        Returns
        -------
        log_var_prior_dict : ``dict``
            A dictionary containing the name of the prior and
            the options and distribution associated with it.
        """

        #---------------------- Gaussian prior -----------------------#

        # If the prior is the Gaussian distribution
        if log_var_prior_name == "gaussian":

            # Get the mean of the Gaussian distribution
            dist_mean = log_var_prior_options["mean"]

            # Get the standard deviation of the Gaussian
            # distribution
            dist_stddev = log_var_prior_options["stddev"]

            #---------------- Fixed covariance matrix ----------------#

            # If the covariance matrix is fixed
            if self.cm_type == "fixed":

                # The log-variance factor will be half of the
                # dimensionality of the space
                dist_factor = self.dim * 0.5

                # The dimensionality of the log-variance factor
                # will be 1
                dist_dim = 1

                # Gradients will not be required
                requires_grad = False

            #-------------- Isotropic covariance matrix --------------#

            # If the covariance matrix is isotropic
            elif self.cm_type == "isotropic":

                # The log-variance factor will be half of the
                # dimensionality of the space
                dist_factor = self.dim * 0.5

                # The dimensionality of the log-variance factor
                # will be 1
                dist_dim = 1

                # Gradients will be required
                requires_grad = True

            #-------------- Diagonal covariance matrix ---------------#
        
            # If the covariance matrix is diagonal
            elif self.cm_type == "diagonal":
                
                # The log-variance factor will be 1/2
                dist_factor = 0.5

                # The dimensionality of the log-variance
                # will be the dimensionality of the space
                dist_dim = self.dim

                # Gradients will be required
                requires_grad = True

            # Get the distribution
            dist = \
                priors.GaussianPrior(dim = dist_dim,
                                     mean = -2 * math.log(dist_mean),
                                     stddev = dist_stddev)

            # Return the dictionary with the name of the prior
            # and associated options and distribution
            return {"name" : log_var_prior_name,
                    "options" : \
                        {"factor" : dist_factor,
                         "dim" : dist_dim,
                         "requires_grad" : requires_grad,
                         "mean" : dist_mean,
                         "stddev" : dist_stddev},
                     "dist" : dist}

        #--------------------- Unsupported prior ---------------------#

        # Otherwise
        else:

            # Raise an error
            errstr = \
                f"Unrecognized prior '{log_var_prior_name}' " \
                "passed to 'log_var_prior_name'. Supported " \
                f"priors are: {', '.join(self.LOG_VAR_PRIORS)}."


    def _get_log_var(self):
        """Get the log-variance of the components of the
        Gaussian mixture model.

        Returns
        -------
        log_var : ``torch.Tensor``
            The log-variance.

            It is a 2D tensor where:

            * The first dimension has a length equal to the number
              of components in the Gaussian mixture.

            * The second dimension has a length equal to the
              dimensionality of the Gaussian mixture model.
        """

        # Get the dimension of the log-variance
        log_var_dim = \
            self.log_var_prior["options"]["dim"]
        
        # Get whether the log-variance requires gradient calculation
        requires_grad = \
            self.log_var_prior["options"]["requires_grad"]

        # Get the log-variance
        log_var = nn.Parameter(\
                    torch.empty(self.n_comp,
                                log_var_dim),
                    requires_grad = requires_grad)

        # Get the name of the prior
        log_var_prior_mame = self.log_var_prior["name"]

        #---------------------- Gaussian prior -----------------------#

        # If the prior is a Gaussian distribution
        if log_var_prior_mame == "gaussian":

            # Get the mean of the Gaussian distribution
            dist_mean = self.log_var_prior["options"]["mean"]

            # Disable gradient calculation
            with torch.no_grad():

                # Populate the log-variance
                log_var.fill_(-2 * math.log(dist_mean))

        #--------------------- Unsupported prior ---------------------#

        # Otherwise
        else:

            # Raise an error
            errstr = \
                f"Unrecognized prior '{log_var_prior_name}' " \
                "passed to 'log_var_prior_name'. Supported " \
                f"priors are: {', '.join(self.log_var_PRIORS)}."

        # Return the log-variance
        return log_var


    #-------------------------- Properties ---------------------------#


    @property
    def dim(self):
        """The dimensionality of the Gaussian mixture model.
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
    def n_comp(self):
        """The number of components in the Gaussian mixture.
        """

        return self._n_comp


    @n_comp.setter
    def n_comp(self,
               value):
        """Raise an exception if the user tries to modify
        the value of ``n_comp`` after initialization.
        """
        
        errstr = \
            "The value of 'n_comp' cannot be changed " \
            "after initialization."
        raise ValueError(errstr)


    @property
    def cm_type(self):
        """The type of the covariance matrix.
        """
        
        return self._cm_type


    @cm_type.setter
    def cm_type(self,
                value):
        """Raise an exception if the user tries to modify
        the value of ``cm_type`` after initialization.
        """
        
        errstr = \
            "The value of 'cm_type' cannot be changed " \
            "after initialization."
        raise ValueError(errstr)


    @property
    def means_prior(self):
        """A dictionary containing the name of the prior over
        the means of the components of the Gaussian mixture model
        and the options used to set it up.
        """

        return self._means_prior


    @means_prior.setter
    def means_prior(self,
                    value):
        """Raise an exception if the user tries to modify
        the value of ``means_prior`` after initialization.
        """
        
        errstr = \
            "The value of 'means_prior' cannot be changed " \
            "after initialization."
        raise ValueError(errstr)


    @property
    def means(self):
        """The means of the components of the Gaussian
        mixture model.
        """

        return self._means


    @means.setter
    def means(self,
              value):
        """Raise an exception if the user tries to modify
        the value of ``means`` after initialization.
        """
        
        errstr = \
            "The value of 'means' cannot be changed " \
            "after initialization."
        raise ValueError(errstr)


    @property
    def weights_prior(self):
        """A dictionary containing the name of the prior
        over the weights of the components of the Gaussian
        mixture model and the options used to set it up.
        """

        return self._weights_prior


    @weights_prior.setter
    def weights_prior(self,
                      value):
        """Raise an exception if the user tries to modify
        the value of ``weights_prior`` after initialization.
        """
        
        errstr = \
            "The value of 'weights_prior' cannot be changed " \
            "after initialization."
        raise ValueError(errstr)  
    

    @property
    def weights(self):
        """The weights of the components of the Gaussian
        mixture model.
        """

        return self._weights


    @weights.setter
    def weights(self,
                value):
        """Raise an exception if the user tries to modify
        the value of ``weights`` after initialization.
        """
        
        errstr = \
            "The value of 'weights' cannot be changed " \
            "after initialization."
        raise ValueError(errstr)


    @property
    def log_var_prior(self):
        """A dictionary containing the name of the prior over
        the log-variance of the components of the Gaussian mixture
        model and the options used to set it up.
        """

        return self._log_var_prior


    @log_var_prior.setter
    def log_var_prior(self,
                      value):
        """Raise an exception if the user tries to modify
        the value of ``log_var_prior`` after initialization.
        """
        
        errstr = \
            "The value of 'log_var_prior' cannot be changed " \
            "after initialization."
        raise ValueError(errstr)


    @property
    def log_var(self):
        """The log-variance of the components of the
        Gaussian mixture model.
        """

        return self._log_var


    @log_var.setter
    def log_var(self,
                value):
        """Raise an exception if the user tries to modify
        the value of ``log_var`` after initialization.
        """
        
        errstr = \
            "The value of 'log_var' cannot be changed " \
            "after initialization."
        raise ValueError(errstr)


    #------------------------ Private methods ------------------------#


    def _get_log_prob_comp(self,
                           x):
        """Get the per-data-point, per-component log-probability.

        Parameters
        ----------
        x : ``torch.Tensor``
            The input data points. This is a 2D tensor where:

            * The first dimension has a length equal to the number of
              data points.

            * The second dimension has a length equal to the
              dimensionality of the data points.

        Returns
        -------
        log_prob_comp : ``torch.Tensor``
            The per-sample, per-component log-probbaility.

            This is a 2D tensor where:

            * The first dimension has a length equal to the number
              of data points in the input tensor.

            * The second dimension has a length equal to the
              dimensionality of the data points.
        """

        # The formula to get the probability density is:
        #
        # p(x) = (1 / (2*pi)^(dim/2)) 
        #      * e^(-0.5 * ((x - means)^2 / var)))
        #
        # If we want the logarithm of the probability density,
        # it becomes:
        #
        # log(p(x)) = log(1) + 
        #            - log((2*pi)^(dim/2) * sqrt(var)) +
        #            - 0.5 * ((x - means)^2 / var))
        #
        # Since the logarithm of 1 is zero and the logarithm of
        # a product is equal to the sum of the logarithms of the
        # terms, we can rewrite the formula as:
        #
        # log(p(x)) = - log((2*pi)^(dim/2)) +
        #             - log(sqrt(var)) +
        #             - 0.5 * ((x - means)^2 / var))
        #
        # Since sqrt(var) can be rewritten as var^(1/2) and
        # an exponent inside a logarithm can be brough out of
        # the logarithm, we can rewrite as:
        #
        # log(p(x)) = - 0.5 * dim * log(2*pi) +
        #             - 0.5 * log(var) + 
        #             - 0.5 * ((x - means)^2 / var))
        #
        # We can rewrite the third term by incorporating the
        # multiplication by 0.5 into the denominator:
        #
        # log(p(x)) = - 0.5 * dim * log(2*pi) +
        #             - 0.5 * log(var) + 
        #             - ((x - means)^2 / 2 * var))
        #
        # We can now compute each term separately and then
        # add them together.
        
        # First, we get the first term, which contains 'pi'.
        pi_term = - 0.5 * self.dim * math.log(2 * math.pi)

        # Then, we get the second term, where 0.5 is replaced
        # by dim * 0.5 if the covariance matrix is fixed or
        # isotropic.
        cm_dependent_term = \
            - (self.log_var_prior["options"]["factor"] * \
               self.log_var.sum(-1))

        # Then, we get the third term
        mean_term = \
            - (x.unsqueeze(-2) - self.means).square().div(\
                2 * torch.exp(self.log_var)).sum(-1)

        # We can now get the log-probability density
        log_prob = pi_term + cm_dependent_term + mean_term

        # Add the log of the softmax of the weights of the
        # components. The output tensor has dimensionality:
        # (n_samples, n_comp)
        log_prob = \
            log_prob + torch.log_softmax(self.weights,
                                         dim = 0)

        # Return the tensor
        return log_prob


    #------------------------ Public methods -------------------------#


    def get_mixture_probs(self):
        """Convert the weights into mixture probabilities using
        the softmax function.

        Returns
        -------
        mixture_probs : ``torch.Tensor``
            The mixture probabilities.

            This is a 1D tensor whose size equals the number of
            components in the Gaussian mixture model.
        """
        
        return torch.softmax(self.weights,
                             dim = -1)


    def get_prior_log_prob(self):
        """Calculate the log probability of the prior on the means,
        log-variance, and mixture coefficients.

        Returns
        -------
        p : ``float``
            The probability of the prior.
        """

        # Initialize the probability to 0.0
        p = 0.0

        #---------------------- Weights' prior -----------------------#

        # Get the name of the prior over the weights of the
        # components
        weights_prior_name = self.weights_prior["name"]

        # If the prior over the weights is the Dirichlet prior
        if weights_prior_name == "dirichlet":

            # Get the alpha
            alpha = self.weights_prior["options"]["alpha"]

            # Get the Dirichlet constant
            p = self.weights_prior["options"]["constant"]

            # If the alpha is different from 1
            if alpha != 1:

                # Add the log probability to the mixture coefficients
                p = p + \
                    (alpha - 1.0) * \
                    (self.get_mixture_probs().log().sum())

        # Otherwise
        else:

            # Raise an error
            errstr = \
                f"Unsupported prior '{weights_prior_name}' for " \
                "the weights of the components of the Gaussian " \
                "mixture model. Supported priors are: " \
                f"{', '.join(WEIGHTS_PRIORS)}."

        #----------------------- Means' prior ------------------------#

        # Get the name of the prior over the means of the
        # components
        means_prior_name = self.means_prior["name"]

        # If the prior over the means is the softball prior
        if means_prior_name == "softball":

            # Get the prior distribution
            dist_means_prior = self.means_prior["dist"]
            
            # Add the log probability of the means
            p = p + dist_means_prior.log_prob(self.means).sum()

        # Otherwise
        else:

            # Raise an error
            errstr = \
                f"Unsupported prior '{means_prior_name}' for " \
                "the means of the components of the Gaussian " \
                "mixture model. Supported priors are: " \
                f"{', '.join(MEANS_PRIOR)}."

        #------------------- Log-variance's prior --------------------#

        # Get the name of the prior over the log-variance
        log_var_prior_name = self.log_var_prior["name"]

        # If the prior over the log-variance is the gaussian prior
        if log_var_prior_name == "gaussian":

            # Get the prior distribution
            dist_log_var_prior = self.log_var_prior["dist"]

            # Add the log-variance probability
            p =  p + \
                dist_log_var_prior.log_prob(self.log_var).sum()

        # Otherwise
        else:

            # Raise an error
            errstr = \
                f"Unsupported prior '{log_var_prior_name}' for " \
                "the negative log-variance of the Gaussian " \
                "mixture model. Supported priors are: " \
                f"{', '.join(log_var_PRIORS)}."

        # Return the probability
        return p

          
    def forward(self,
                x):
        """Forward pass - compute the absolute log-probability density
        for a set of data points.

        Parameters
        ----------
        x : ``torch.Tensor``
            The input data points. This is a 2D tensor where:

            * The first dimension has a length equal to the number
              of data points.

            * The second dimension has a length equal to the
              dimensionality of the data points.
            
        Returns
        -------
        y : ``torch.Tensor``
            The result of the forward pass.

            This is a 1D tensor whose size is equal to the number
            of input data points.

            Each element of the tensor is the absolute log-probability
            density of a data point.
        """

        # Get the per-sample, per-component log-probability
        y = self._get_log_prob_comp(x = x)

        # Get the log of summed exponentials of each row
        # of the tensor in the last dimension (the number
        # of components in the Gaussian mixture). The output
        # tensor has dimensionality:
        # (n_samples)
        y = torch.logsumexp(y,
                            dim = -1)
        
        # Add the log-probability of the prior
        y = y + self.get_prior_log_prob()
        
        # Return the negative log-probability density
        return -y


    def sample_probs(self,
                     x):
        """Get the probability density per sample per component.

        Parameters
        ----------
        x : ``torch.Tensor``
            The input data points. This is a 2D tensor where:

            * The first dimension has a length equal to the number
              of data points.

            * The second dimension has a length equal to the
              dimensionality of the data points.

        Returns
        -------
        probs : ``torch.Tensor``
            The probability densities.

            This is a 2D tensor where:

            * The first dimension has a length equal to the number
              of input data points.

            * The second dimension has a length equal to the number
              of components in the Gaussian mixture.

            Each element of the tensor stores a per-data point,
            per-component probability density.
        """

        # Get the per-sample, per-component log-probability
        y = self._get_log_prob_comp(x = x)

        # Return the probability
        return torch.exp(y)


    def log_prob(self,
                 x):
        """Get the log-probability density of the samples
        ``x`` being drawn from the GMM.

        Parameters
        ----------
        x : ``torch.Tensor``
            The input data points. This is a 2D tensor where:

            * The first dimension has a length equal to the number
              of data points.

            * The second dimension has a length equal to the
              dimensionality of the data points.

        Returns
        -------
        log_prob : ``torch.Tensor``
            A 1D tensor storing the log-probability density
            of each input data points to be drawn from the GMM.

            The tensor has a size equal to the number of data points
            passed.
        """
        
        return - self.forward(x)


    #------------------------ New data points ------------------------#


    def sample_new_points(self,
                          n_points,
                          n_samples_per_comp = 1,
                          sampling_method = "mean"):
        """Draw samples for new data points from each component
        of the Gaussian mixture model

        Parameters
        ----------
        n_points : ``int``
            Number of data points for which samples should be drawm.

        n_samples_per_comp : ``int``, ``1``
            Number of samples to draw per data point per component
            of the Gaussian mixture.

        sampling_method : ``str``, {``"mean"``}, \
                          ``"mean"``
            How to draw the samples for the given data points:

            * ``"mean"`` means taking the mean of each component as
              the value of each ``n_samples_per_comp`` sample taken
              for each data point.

        Returns
        -------
        new_points : ``torch.Tensor``
            The samples drawn.

            This is a 2D tensor where:

            * The first dimension has a length equal to
              ``n_points * n_reps_per_mix_comp * n_comp``.

            * The second dimension has a length equal to the
              dimensionality of the Gaussian mixture model.
        """

        # Supported sampling methods
        SAMPLING_METHODS = ["mean"]

        # Get the total number of samples to be taken from the
        # model
        n_samples = n_points * n_samples_per_comp

        #------------------ Sampling from the means ------------------#

        # If the user selected the option to take the mean
        # of each component as initial representation for
        # the data points
        if sampling_method == "mean":

            # Disable gradient calculation
            with torch.no_grad():

                # Get the representations
                out = \
                    torch.repeat_interleave(\
                        self.means.clone().cpu().detach().unsqueeze(0),
                        n_samples,
                        dim = 0)

        #--------------------- Invalid sampling ----------------------#

        # Otherwise
        else:

            # Warn the user that they have passed an invalid option,
            # and raise an exception
            errstr = \
                f"Please specify how to correctly initialize new " \
                f"representations. The supported methods are: " \
                f"{', '.join(SAMPLING_METHODS)}."
            raise ValueError(errstr)

        #-------------------- Reshape the output ---------------------#
        
        # Return the representations for the new points as a 2D
        # tensor with:
        #
        # - 1st dimension: the number of data points times number of
        #                  samples drawn per component per data point
        #                  times the number of components in the
        #                  mixture ->
        #                  'n_points' *
        #                  'n_comp' *
        #                  'n_samples_per_comp'
        #
        # - 2nd dimension: the dimensionality of the Gaussian mixture
        #                  model ->
        #                  'dim'
        return out.view(n_samples * self.n_comp,
                        self.dim)


#------------------------ Representation layer -----------------------#


class RepresentationLayer(nn.Module):
    
    """
    Class implementing a representation layer accumulating
    ``PyTorch`` gradients.
    """

    # Available distributions to sample the representations
    # from
    AVAILABLE_DISTS = ["normal"]
    
    def __init__(self,
                 values = None,
                 dist = "normal",
                 dist_options = None):
        """Initialize a representation layer.

        Parameters
        ----------
        values : ``torch.Tensor``, optional
            A tensor used to initialize the representations in
            the representation layer.

            This is a 2D tensor where:

            * The first dimension has a length equal to the number
              of representations in the tensor.

            * The second dimension has a length equal to the
              dimensionality of the representations.

            If the tensor is not passed, the representations will be
            initialized by sampling the distribution specified
            by ``dist``.

        dist : ``str``, {``"normal"``}, default: ``"normal"``
            The name of the distribution used to sample the
            representations, if no ``values`` are passed.

            By default, the distribution is a ``"normal"``
            distribution.

        dist_options : ``dict``, optional
            A dictionary containing the parameters to sample the
            representations from the distribution, if no
            ``values`` are passed.

            For any distribution the following keys and associated
            parameters must be provided:

            * ``"n_samples"`` : the number of samples to draw from
              the distribution.

            * ``"dim"`` : the dimensionality of the representations
              to sample from the distribution.

            If ``dist`` is ``"normal"``, the dictionary must contain
            these additional key/value pairs:

            * ``"mean"`` : the mean of the normal distribution used
              to generate the representations.

            * ``"stddev"`` : the standard deviation of the normal
              distribution used to generate the representations.
        """
        
        # Initialize the class
        super().__init__()
        
        # Initialize the gradients with respect to the
        # representations to None
        self.dz = None

        # If a tensor of values was passed
        if values is not None:

            # Set the options used to initialize the
            # representations to an empty dictionary, since
            # they have not been samples from a distribution
            self._options = {}

            # Get the representations from the tensor
            self._n_samples, self._dim, self._z = \
                self._get_rep_from_values(\
                    values = values)      
        
        # Otherwise
        else:

            # If the representations are to be sampled from a
            # normal distribution
            if dist == "normal":

                # Sample the representations from a normal
                # distribution
                self._n_samples, self._dim, self._z, self._options = \
                    self._get_rep_from_normal(\
                        options = dist_options)

            # Otherwise
            else:

                # Raise an error since only the normal distribution
                # is implemented so far
                available_dists_str = \
                    ", ".join(f'{d}' for d in self.AVAILABLE_DISTS)
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
        n_samples : ``int``
            The number of samples found in the input tensor (and,
            therefore, the number of representations).

        dim : ``int``
            The dimensionality of the representations.

        rep : ``torch.Tensor``
            The representations.

            This is a 2D tensor where:

            * The first dimension has a length equal to the number of
              representations.

            * The second dimension has a length equal to the
              dimensionality of the representations.
        """

        # Get the number of samples from the first
        # dimension of the tensor
        n_samples = values.shape[0]
        
        # Get the dimensionality of the representations
        # from the last dimension of the tensor
        dim = values.shape[-1]

        # Initialize a tensor with the representations
        z = nn.Parameter(torch.zeros_like(values), 
                         requires_grad = True)

        # Fill the tensor with the given values
        with torch.no_grad():
            z.copy_(values)

        # Return the representations and the parameters used
        # to generate them
        return n_samples, \
               dim, \
               z


    def _get_rep_from_normal(self,
                             options):
        """Get representations by sampling from a normal
        distribution.

        Parameters
        ----------
        options : ``dict``
            A dictionary containing the parameters to sample the
            representations from a normal distribution.

            The dictionary must contains the following keys,
            associated to the corresponding parameters:

            * ``"n_samples"`` : the number of samples to draw from
              the normal distribution.

            * ``"dim"`` : the dimensionality of the representations
              to sample from the normal distribution.

            * ``"mean"`` : the mean of the normal distribution used
              to generate the representations.

            * ``"stddev"`` : the standard deviation of the normal
              distribution used to generate the representations.

        Returns
        -------
        n_samples : ``int``
            The number of samples found in the input tensor (and,
            therefore, the number of representations).

        dim : ``int``
            The dimensionality of the representations.

        rep : ``torch.Tensor``
            The representations.

            This is a 2D tensor where:

            * The first dimension has a length equal to the number of
              representations.

            * The second dimension has a length equal to the
              dimensionality of the representations.

        options : ``dict``
            A dictionary containing the options used to initialize
            the representations.
        """

        # Get the desired number of samples
        n_samples = options["n_samples"]

        # Get the dimensionality of the desired representations
        dim = options["dim"]

        # Get the mean of the normal distribution from which
        # the representations should be sampled from
        mean = options["mean"]

        # Get the standard deviation of the normal distribution
        # from which the representations should be sampled from
        stddev = options["stddev"]

        # Get the representations
        z = \
            nn.Parameter(\
                torch.normal(mean,
                             stddev,
                             size = (n_samples, dim),
                             requires_grad = True))
        
        # Return the representations and the parameters used
        # to generate them
        return n_samples, \
               dim, \
               z, \
               {"dist_name" : "normal",
                "mean" : mean,
                "stddev" : stddev}


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
    def options(self):
        """Dictionary ot options used to generate the
        representations, if no values were passed at
        initialization time.
        """

        return self._options


    @options.setter
    def options(self,
                value):
        """Raise an exception if the user tries to modify
        the value of ``options`` after initialization.
        """
        
        errstr = \
            "The value of 'options' cannot be changed " \
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

            This is a 2D tensor where:

            * The first dimension has a length equal to the number
              of representations.

            * The second dimension has a length equal to the
              dimensionality of the representations.
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

        Returns
        -------
        ``torch.Tensor``
            The rescaled representations.

            This is a 2D tensor where:

            * The first dimension has a length equal to the number
              of representations.

            * The second dimension has a length equal to the
              dimensionality of the representations.
        """
        
        # Flatten the tensor with the representations
        z_flat = torch.flatten(self.z.cpu().detach())
        
        # Get the mean and the standard deviation of the
        # representations
        sd, m = torch.std_mean(z_flat)
        
        # Disable the gradient calculation
        with torch.no_grad():

            # Subtract the mean from the representations
            self.z -= m

            # Divide the representations by the standard
            # deviation
            self.z /= sd