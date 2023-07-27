#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-

#    GMM.py
#
#    Module containing the class defining the Gaussian mixture model.
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
    "Module containing the class defining the Gaussian " \
    "mixture model."


# Standard library
import logging as log
import math
# Third-party packages
import torch
import torch.distributions as dist
import torch.nn as nn
# BulkDGD
from . import priors


# Get the module's logger
logger = log.getLogger(__name__)


class GaussianMixtureModel(nn.Module):
    
    """
    A class implementing a mixture of multivariate Gaussians
    (Gaussian mixture model or GMM).
    """

    def __init__(self,
                 dim,
                 n_comp,
                 mean_prior = None,
                 cm_type = "isotropic",
                 alpha = 1, 
                 softball_params = (0.0, 1.0),
                 logbeta_params = (0.5, 0.5)):
        """Initialize an instance of the GMM.

        Parameters
        ----------
        dim : ``int``
            The dimensionality of the Gaussian mixture model.

        n_comp : ``int``
            The number of components in the mixture.

        mean_prior : any object representing a prior, optional
            Instance of a class representing a prior. If
            not passed, the ``bulkDGD.core.priors.softball``
            prior will be used.

        cm_type : ``str``, {``"fixed"``, ``"isotropic"``, \
        ``"diagonal"``}, ``"isotropic"``
            Shape of the covariance matrix.

        alpha : ``int``, ``1``
            Alpha of the Dirichlet distribution determining
            the uniformity of the weights of the components
            in the mixture.
        
        softball_params : ``tuple``, ``(0.0, 1.0)``
            Parameters (radius, sharpness) of the softball
            prior, if no prior was passed.

        logbeta_params : ``tuple``, ``(0.5, 0.5)``
            Tuple containing the parameters to initialize the
            Gaussian prior on the log-variances of the components:

            * Mean: ``2 * log(logbeta_params[0])``

            * Standard deviation: ``logbeta_params[1]``
        
        Notes
        -----
        The difference between giving a prior beforehand and
        giving only init values is that with a given prior
        the logbetas will be sampled from it. Otherwise,
        the alpha determines the Dirichlet prior on the
        mixture coefficients. The mixture coefficients
        are initialized uniformly, while other parameters
        are sampled from the prior.
        """

        # Initialize the class
        super().__init__()
        
        # Set the dimensionality of the space
        self._dim = dim

        # Set the number of components in the mixture
        self._n_comp = n_comp

        # Set the parameters (radius, sharpness) of the
        # softball prior, if no prior was passed
        self._softball_params = softball_params

        # Set the parameters to initialize the Gaussian prior
        # on the log-variances of the components
        self._logbeta_params = logbeta_params

        # Set the Dirichlet alpha determining the uniformity of
        # the weights of the components in the mixture
        self._alpha = alpha

        # Set the Dirichlet constant
        self._dirichlet_constant = \
            self._get_dirichlet_constant(n_comp = self.n_comp,
                                         alpha = self.alpha)

        # Set the weights, which are initialized uniformly so that
        # the components start out as equiprobable
        self._weight = \
            self._get_weight(n_comp = self.n_comp)
        
        # Initialize the prior on the means and the means themselves
        self._mean_prior, self._mean = \
            self._get_means(mean_prior = mean_prior,
                            softball_params = self.softball_params,
                            dim = self.dim,
                            n_comp = self.n_comp)

        # Initialize the covariance matrix and related parameters
        self._logbeta, self._logbeta_dim, \
            self._logbeta_factor, self._logbeta_prior = \
                self._get_logbeta(\
                    cm_type = cm_type,
                    logbeta_params = self.logbeta_params,
                    n_comp = self.n_comp,
                    dim = self.dim)


    #-------------------- Initialization methods ---------------------#


    def _get_dirichlet_constant(self,
                                n_comp,
                                alpha):
        """Return the Dirichlet constant.

        Parameters
        ----------
        n_comp : ``int``
            The number of components in the mixture.

        alpha : ``int``
            Alpha of the Dirichlet distribution determining
            the uniformity of the weights of the components
            in the mixture.

        Returns
        -------
        ``float``
            The Dirichlet constant.
        """

        return math.lgamma(n_comp * alpha) - \
               n_comp * math.lgamma(alpha)


    def _get_weight(self,
                    n_comp):
        """Return the weights of the components in the mixture.

        Parameters
        ----------
        n_comp : ``int``
            The number of components in the Gaussian mixture.

        Returns
        -------
        ``torch.Tensor``
            A one-dimensional tensor containing the weights of
            the components in the mixture. The tensor has a
            length equal to the number of components in the
            Gaussian mixture.
        """

        return nn.Parameter(torch.ones(n_comp),
                            requires_grad = True)


    def _get_means(self,
                   mean_prior,
                   softball_params,
                   dim,
                   n_comp):
        """Return the prior on the means of the Gaussians
        and the the means themselves.

        Parameters
        ----------
        mean_prior : any object representing a prior
            Instance of a class representing a prior. If
            not passed, the ``DGDPerturbations.core.priors.softball``
            prior will be used.

        softball_params : ``tuple``
            Parameters (radius, sharpness) of the softball
            prior, if no prior was passed.

        dim : ``int``
            The dimensionality of the Gaussian mixture model.

        n_comp : ``int``
            The number of components in the mixture.

        Returns
        -------
        ``tuple``
            A tuple containing:

            * The prior used over the means of the mixture
            components.

            * The tensor containing the means of the mixture
            components samples from the prior. This tensor
            is two-dimensional, with the first dimension having
            a length equal to the number of components in the
            Gaussian mixture and the second dimension having
            a length equal to the dimensionality of the Gaussian
            mixture model.
        """

        # If the user has passed a prior
        if mean_prior is not None:

            # Check the prior
            self._check_mean_prior(mean_prior = mean_prior)

            # Inform the user that the prior passed will be
            # used
            infostr = \
                f"The {mean_prior.__class__.__name__} prior " \
                f"will be used as prior over the means of " \
                f"the mixture components."
            logger.info(infostr)

        # Otherwise
        else:

            # Get the radius and sharpness of the softball
            # prior that will be used
            radius, sharpness = softball_params

            # Get the mean prior as a softball prior
            mean_prior = \
                priors.softball(dim = dim,
                                radius = radius,
                                sharpness = sharpness)

            # Inform the user that the softball prior will be
            # used
            infostr = \
                f"The softball prior with radius {radius} and " \
                f"sharpness {sharpness} will be used as prior " \
                f"over the means of the mixture components."
            logger.info(infostr)

        # Means with shape: n_comp, dim
        mean = \
            nn.Parameter(mean_prior.sample(n = n_comp),
                         requires_grad = True)

        # Return the prior on the means and the means
        return mean_prior, mean


    def _get_logbeta(self,
                     cm_type,
                     logbeta_params,
                     n_comp,
                     dim):
        """Return the parameters to learn the covariance matrix
        as a negative log variance to ensure it is positive
        definite.

        Parameters
        ----------
        cm_type : ``str``, {``"fixed"``, ``"isotropic"``, \
                  ``"diagonal"``"}
            Shape of the covariance matrix.

        logbeta_params : ``tuple``
            Tuple containing the parameters to initialize the
            Gaussian prior on the log-variances of the components:
            
            * Mean: ``2 * log(logbeta_params[0])``
            * Standard deviation: ``logbeta_params[1]``

        n_comp : ``int``
            The number of components in the mixture.

        dim : ``int``
            The dimensionality of the Gaussian mixture model.

        Returns
        -------
        ``tuple``
            A tuple containing:

            * The log-variance. It is a two-dimensional tensor
            where the first dimension has a length equal to the
            number of components in the Gaussian mixture and the
            second dimension has a length equal to the dimensionality
            of the Gaussian mixture model.

            * The dimensionality of the Gaussian mixture model.

            * The log-variance factor.

            * The prior over the log-variance.
        """
        
        # If the covariance matrix is fixed
        if cm_type == "fixed":

            # The log-variance factor will be half
            # of the dimensionality of the space
            logbeta_factor = dim * 0.5

            # The dimensionality of the log-variance
            # factor will be 1
            logbeta_dim = 1

            # No gradient is needed for training
            logbeta = \
                nn.Parameter(torch.empty(n_comp,
                                         logbeta_dim),
                             requires_grad = False)

        # If the covariance matrix is isotropic
        elif cm_type == "isotropic":

            # The log-variance factor will be half
            # of the dimensionality of the space
            logbeta_factor = dim * 0.5

            # The dimensionality of the log-variance
            # factor will be 1
            logbeta_dim = 1

            # Gradients are required
            logbeta = \
                nn.Parameter(torch.empty(n_comp,
                                         logbeta_dim),
                             requires_grad = True)
        
        # If the covariance matrix is diagonal
        elif cm_type == "diagonal":
            
            # The log-variance factor will be 1/2
            logbeta_factor = 0.5

            # The dimensionality of the log-variance
            # will be the dimensionality of the space
            logbeta_dim = dim

            # Gradients are required
            logbeta = \
                nn.Parameter(torch.empty(n_comp,
                                         logbeta_dim),
                             requires_grad = True)

        # If an invalid covariance matrix type was passed
        else:

            # Raise an error
            errstr = \
                "'cm_type' must be 'isotropic' (default), " \
                "'diagonal', or 'fixed'."
            raise ValueError(errstr)

        # Disable gradient calculation
        with torch.no_grad():
            logbeta.fill_(2 * math.log(logbeta_params[0]))

        # Get the prior over the log-variance
        logbeta_prior = \
            priors.gaussian(dim = logbeta_dim,
                            mean = 2 * math.log(logbeta_params[0]),
                            stddev = logbeta_params[1])

        # Return the log-variance, its dimensionality, the
        # log-variance factor, and the prior over the
        # log-variance
        return logbeta, logbeta_dim, logbeta_factor, logbeta_prior


    def _check_mean_prior(self,
                          mean_prior):
        """Check whether the user-provided prior for the means of
        the mixture components is valid.

        Parameters
        ----------
        mean_prior : any object representing a prior
            Instance of a class representing a prior. If
            not passed, the ``DGDPerturbations.core.priors.softball``
            prior will be used.
        """

        # If the prior does not have a 'sample' method
        if getattr(mean_prior, "sample", None) is None:

            # Warn the user and raise an error
            errstr = \
                f"The {mean_prior.__class__.__name__} prior must " \
                f"implement a 'sample' method to be used as a " \
                f"prior over the means of the mixture components."
            raise AttributeError(errstr)

        # If the prior does not have a 'log_prob' method
        if getattr(mean_prior, "log_prob", None) is None:

            # Warn the user and raise an error
            errstr = \
                f"The {mean_prior.__class__.__name__} prior must " \
                f"implement a 'log_prob' method to be used as a " \
                f"prior over the means of the mixture components."
            raise AttributeError(errstr)


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
    def softball_params(self):
        """The parameters (radius, sharpness) of the softball
        prior that will be used, if no prior is passed.
        """

        return self._softball_params


    @softball_params.setter
    def softball_params(self,
                        value):
        """Raise an exception if the user tries to modify
        the value of ``softball_params`` after initialization.
        """
        
        errstr = \
            "The value of 'softball_params' cannot be changed " \
            "after initialization."
        raise ValueError(errstr)


    @property
    def logbeta_params(self):
        """The parameters used to inizialize the Gaussian
        prior on the log-variances of the components:

        * Mean: ``2 * log(logbeta_params[0])``
        * Standard deviation: ``logbeta_params[1]``
        """

        return self._logbeta_params


    @logbeta_params.setter
    def logbeta_params(self,
                       value):
        """Raise an exception if the user tries to modify
        the value of ``logbeta_params`` after initialization.
        """
        
        errstr = \
            "The value of 'logbeta_params' cannot be changed " \
            "after initialization."
        raise ValueError(errstr)


    @property
    def alpha(self):
        """The Dirichlet alpha determining the uniformity of the
        weights of the components in the Gaussian mixture.
        """

        return self._alpha


    @alpha.setter
    def alpha(self,
              value):
        """Raise an exception if the user tries to modify
        the value of ``alpha`` after initialization.
        """
        
        errstr = \
            "The value of 'alpha' cannot be changed " \
            "after initialization."
        raise ValueError(errstr)


    @property
    def dirichlet_constant(self):
        """The Dirichlet constant.
        """

        return self._dirichlet_constant


    @dirichlet_constant.setter
    def dirichlet_constant(self,
                           value):
        """Raise an exception if the user tries to modify
        the value of ``dirichlet_constant`` after initialization.
        """
        
        errstr = \
            "The value of 'dirichlet_constant' cannot be changed " \
            "after initialization."
        raise ValueError(errstr)


    @property
    def weight(self):
        """The weights of the components in the Gaussian mixture.
        """

        return self._weight


    @weight.setter
    def weight(self,
               value):
        """Raise an exception if the user tries to modify
        the value of ``weight`` after initialization.
        """
        
        errstr = \
            "The value of 'weight' cannot be changed " \
            "after initialization."
        raise ValueError(errstr)


    @property
    def mean_prior(self):
        """The prior on the means of the Gaussian components
        in the mixture.
        """

        return self._mean_prior


    @mean_prior.setter
    def mean_prior(self,
                   value):
        """Raise an exception if the user tries to modify
        the value of ``mean_prior`` after initialization.
        """
        
        errstr = \
            "The value of 'mean_prior' cannot be changed " \
            "after initialization."
        raise ValueError(errstr)


    @property
    def mean(self):
        """The means of the Gaussian components in the mixture.
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
    def logbeta(self):
        """The log-variance of the Gaussian components in the
        mixture.
        """

        return self._logbeta


    @logbeta.setter
    def logbeta(self,
                value):
        """Raise an exception if the user tries to modify
        the value of ``logbeta`` after initialization.
        """
        
        errstr = \
            "The value of 'logbeta' cannot be changed " \
            "after initialization."
        raise ValueError(errstr)


    @property
    def logbeta_dim(self):
        """The dimensionality of the log-variance.
        """

        return self._logbeta_dim


    @logbeta_dim.setter
    def logbeta_dim(self,
                    value):
        """Raise an exception if the user tries to modify
        the value of ``logbeta_dim`` after initialization.
        """
        
        errstr = \
            "The value of 'logbeta_dim' cannot be changed " \
            "after initialization."
        raise ValueError(errstr)


    @property
    def logbeta_factor(self):
        """The log-variance factor.
        """

        return self._logbeta_factor


    @logbeta_factor.setter
    def logbeta_factor(self,
                       value):
        """Raise an exception if the user tries to modify
        the value of ``logbeta_factor`` after initialization.
        """
        
        errstr = \
            "The value of 'logbeta_factor' cannot be changed " \
            "after initialization."
        raise ValueError(errstr)


    @property
    def logbeta_prior(self):
        """The prior over the log-variance.
        """

        return self._logbeta_prior


    @logbeta_prior.setter
    def logbeta_prior(self,
                      value):
        """Raise an exception if the user tries to modify
        the value of ``logbeta_prior`` after initialization.
        """
        
        errstr = \
            "The value of 'logbeta_prior' cannot be changed " \
            "after initialization."
        raise ValueError(errstr)


    #------------------------ Private methods ------------------------#


    def _get_log_prob_comp(self,
                           x):
        """Get the per-sample, per-component log-probability.

        Parameters
        ----------
        x : ``torch.Tensor``
            The input data points. This is a 2D tensor where:

            * The first dimension has a length equal to the number of
              data points.

            * The second dimension has a length equal to the
              dimensionality of the data points.
        """

        # "Unsqueeze" the input tensor by adding a dimension
        # just before the last one. If the input tensor is
        # a two-dimensional tensor with the number of samples
        # as the first dimension and the dimensionality of the
        # samples' representations as the second dimension,
        # the "unsqueezed" tensor will be three-dimensional
        # with dimensionality:
        # (n_samples, 1, dim)
        y = x.unsqueeze(-2)

        # Subtract the means of the components from the
        # tensor and square the result.
        # This new tensor has dimensionality:
        # (n_samples, n_comp, dim)
        y = (y - self.mean).square()

        # Multiply the tensor by 0.5 * e^log_var (e^log_var = log_var).
        y = y.mul(0.5 * torch.exp(self.logbeta))

        # Sum the tensor over its last dimension and take the
        # negative. Since we are summing over the dimensionality
        # of the representations, the new tensor has dimensionality:
        # (n_samples, n_comp)
        y = - y.sum(-1)

        # The log-variance has dimensionality:
        # (n_comp, dim)
        #
        # Sum over the log variance's last dimension, meaning that
        # the resulting tensor will be one-dimensional with size
        # equal to the number of components in the Gaussian mixture
        # (for which all dimensions have been summmed together).
        # The new tensor has dimensionality:
        # (n_comp)
        log_var_sum = self.logbeta.sum(-1)
        
        # Add the summed log-variance multiplied by the log-variance
        # factor to the previous tensor.
        # The output tensor has dimensionality:
        # (n_samples, n_comp)
        y = y + (self.logbeta_factor * log_var_sum)

        # Subtract 0.5 * dim * log(2 * pi).
        # The output tensor has dimensionality:
        # (n_samples, n_comp)
        y = y - (0.5 * self.dim * math.log(2 * math.pi))

        # Add the log of the softmax of the weights of the
        # components. The output tensor has dimensionality:
        # (n_samples, n_comp)
        y = y + torch.log_softmax(self.weight,
                                  dim = 0)

        # Return the tensor
        return y


    #------------------------ Public methods -------------------------#


    def get_mixture_probs(self):
        """Convert the weights into mixture probabilities using
        the softmax function.

        Returns
        -------
        ``torch.Tensor``
            The mixture probabilities. This is a 1D tensor whose
            size equals the number of components in the Gaussian
            mixture.
        """
        
        return torch.softmax(self.weight,
                             dim = -1)


    def get_distribution(self):
        """Create a distribution from the GMM for sampling.

        Returns
        -------
        ``torch.distributions.MixtureSameFamily``
            A distribution created from the GMM.
        """

        # Disable gradient calculation
        with torch.no_grad():

            mix = \
                dist.Categorical(probs = torch.softmax(self.weight,
                                                       dim = -1))
            
            # Get the components - reinterpret some of the batch
            # dimensions as event dimensions
            comp = \
                dist.Independent(\
                    base_distribution = \
                        dist.Normal(self.mean,
                                    torch.exp(-0.5 * self.logbeta)),
                    reinterpreted_batch_ndims = 1)
            
            # Return the distribution
            return dist.MixtureSameFamily(mix, comp)


    def get_prior_log_prob(self):
        """Calculate the log probability of the prior on the means,
        log-variance, and mixture coefficients.

        Returns
        -------
        p : ``float``
            The probability of the prior.
        """

        # Get the Dirichlet constant
        p = self.dirichlet_constant

        # If the alpha is different from 1
        if self.alpha != 1:

            # Add the log probability to the mixture coefficients
            p = p + \
                (self.alpha - 1.0) * \
                (self.get_mixture_probs().log().sum())
        
        # Add the log probability of the means
        p = p + self.mean_prior.log_prob(self.mean).sum()

        # If the logbeta prior is not None
        if self.logbeta_prior is not None:

            # Add the logbeta probability
            p =  p + self.logbeta_prior.log_prob(self.logbeta).sum()
        
        # Return the probability
        return p

          
    def forward(self,
                x):
        """Forward pass - compute the negative log-probability density
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
        ``torch.Tensor``
            This is a 1D tensor whose size is equal to the number
            of input data points.

            Each element of the tensor is the negative log-probability
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
        
        # The forward pass consists in computing the
        # negative log-probability of the input samples.
        # The output tensor has dimensionality:
        # (n_samples)
        neg_log_prob = -y

        # Return the negative log-probability
        return neg_log_prob


    def sample_probs(self,
                     x):
        """Return the probability density per sample per component.

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
        ``torch.Tensor``
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
        """Get the log-probability density of the samples ``x``
        being drawn from the GMM.

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
        ``torch.Tensor``
            A 1D tensor storing the log-probability density of each
            input data points to be drawn from the GMM. The tensor
            has a size equal to the number of data points passed.
        """
        
        return - self.forward(x)


    def sample(self,
               n_samples):
        """Create samples from the GMM distribution.

        Parameters
        ----------
        n : ``int``
            The number of samples to be drawn from the GMM distribution.

        Returns
        -------
        ``torch.Tensor``
            The points sampled from the GMM distribution.

            This is a 2D tensor where:
            
            * The first dimension has a length equal to the number
              of samples drawn.

            * The second dimension has a length equal to the
              dimensionality of the samples drawn (= the
              dimensionality of the Gaussian mixture model).
        """
        
        # Disable gradient calculation
        with torch.no_grad():

            # Get the GMM distribution
            gmm_dist = self.get_distribution()

            # Sample from the distribution
            return gmm_dist.sample(torch.tensor([n_samples]))


    def component_sample(self,
                         n_samples):
        """Sample from each component of the GMM. 

        Parameters
        ----------
        n_samples : ``int``
            The number of samples to be drawn for each component
            of the Gaussian mixture.

        Returns
        -------
        ``torch.Tensor``
            The sampled points from the components of the Gaussian
            mixture. This is a 3D tensor where:
            
            * The first dimension has a length equal to the number
              of samples drawn from each component.

            * The second dimension has a length equal to the number
              of components in the Gaussian mixture.

            * The third dimension has a length equal to the
              dimensionality of the Gaussian mixture model.
        """

        # Disable gradient calculation
        with torch.no_grad():

            # Get the components - reinterpret some of the batch
            # dimensions as event dimensions
            comp = \
                dist.Independent(\
                    base_distribution = \
                        dist.Normal(self.mean,
                                    torch.exp(-0.5 * self.logbeta)),
                    reinterpreted_batch_ndims = 1)

            # Return the sample from each component
            return comp.sample(torch.tensor([n_samples]))


    #------------------------ New data points ------------------------#


    def sample_new_points(self,
                          n_points,
                          n_samples_per_comp = 1,
                          sampling_method = "random"):
        """Draw samples for new data points from each component
        of the Gaussian mixture model

        Parameters
        ----------
        n_points : ``int``
            Number of data points for which samples should be drawm.

        n_samples_per_comp : ``int``, ``1``
            Number of samples to draw per data point per component
            of the Gaussian mixture.

        sampling_method : ``str``, {``"mean"``, ``"random"``}, \
                          ``"random"``
            How to draw the samples for the given data points:

            * ``"random"`` means sampling ``n_samples_per_comp``
               vectors from each mixture component at random.

            * ``"mean"`` means taking the mean of each component as
              the value of each ``n_samples_per_comp`` sample taken
              for each data point.

            Setting ``sampling_method`` to ``random`` is equivalent
            to calling the ``component_sample()`` method with
            ``n_samples`` set to
            ``n_points * n_samples_per_comp``.

        Returns
        -------
        ``torch.Tensor``
            The samples drawn. This is a 2D tensor where:

            * The first dimension has a length equal to
              ``n_points * n_reps_per_mix_comp * n_comp``.

            * The second dimension has a length equal to the
              dimensionality of the Gaussian mixture model.
        """

        # Supported sampling methods
        SAMPLING_METHODS = ["random", "mean"]

        # Get the total number of samples to be taken from the
        # model
        n_samples = n_points * n_samples_per_comp

        #---------------------- Random sampling ----------------------#

        # If the user selected the option to sample
        # 'n_new' vectors from each component
        if sampling_method == "random":

            # Sample from each component
            out = self.component_sample(n_samples = n_samples)

        #------------------ Sampling from the means ------------------#

        # If the user selected the option to take the mean
        # of each component as initial representation for
        # the data points
        elif sampling_method == "mean":

            # Disable gradient calculation
            with torch.no_grad():

                # Get the representations
                out = \
                    torch.repeat_interleave(\
                        self.mean.clone().cpu().detach().unsqueeze(0),
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
        were sampled from, if the user did not pass any
        ``values`` for the representations when initializing the
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
        representations were sampled from, if the user did
        not pass any ``values`` for the representations
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