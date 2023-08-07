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
                 mean_prior,
                 cm_type = "isotropic",
                 alpha = 1,
                 log_var_params = (0.5, 0.5)):
        """Initialize an instance of the GMM.

        Parameters
        ----------
        dim : ``int``
            The dimensionality of the Gaussian mixture model.

        n_comp : ``int``
            The number of components in the mixture.

        mean_prior : ``object``
            Instance of a class representing a prior.

        cm_type : ``str``, {``"fixed"``, ``"isotropic"``, \
        ``"diagonal"``}, ``"isotropic"``
            Shape of the covariance matrix.

        alpha : ``int``, ``1``
            Alpha of the Dirichlet distribution determining
            the uniformity of the weights of the components
            in the mixture.

        log_var_params : ``tuple``, ``(0.5, 0.5)``
            Tuple containing the parameters to initialize the
            Gaussian prior on the log-variances of the components:

            * Mean: ``2 * log(log_var_params[0])``

            * Standard deviation: ``log_var_params[1]``
        """

        # The difference between giving a prior beforehand and
        # giving only init values is that with a given prior
        # the log_vars will be sampled from it. Otherwise,
        # the alpha determines the Dirichlet prior on the
        # mixture coefficients. The mixture coefficients
        # are initialized uniformly, while other parameters
        # are sampled from the prior.

        # Initialize the class
        super().__init__()
        
        # Set the dimensionality of the space
        self._dim = dim

        # Set the number of components in the mixture
        self._n_comp = n_comp

        # Set the parameters to initialize the Gaussian prior
        # on the log-variances of the components
        self._log_var_params = log_var_params

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
                            dim = self.dim,
                            n_comp = self.n_comp)

        # Initialize the covariance matrix and related parameters
        self._log_var, self._log_var_dim, \
            self._log_var_factor, self._log_var_prior = \
                self._get_log_var(\
                    cm_type = cm_type,
                    log_var_params = self.log_var_params,
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
            The number of components in the Gaussian mixture.

        alpha : ``int``
            The alpha of the Dirichlet distribution determining
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
            The weights of the components in the Gaussian mixture.
            
            The is a 1D tensor having a length equal to the number
            of components in the Gaussian mixture.
        """

        return nn.Parameter(torch.ones(n_comp),
                            requires_grad = True)


    def _get_means(self,
                   mean_prior,
                   dim,
                   n_comp):
        """Return the prior on the means of the Gaussians
        and the the means themselves.

        Parameters
        ----------
        mean_prior : ``object``
            Instance of a class representing a prior.

        dim : ``int``
            The dimensionality of the Gaussian mixture model.

        n_comp : ``int``
            The number of components in the Gaussian mixture.

        Returns
        -------
        mean_prior : ``object``
            The prior used over the means of the mixture
            components.
        
        means : ``torch.Tensor``
            The means of the mixture components sampled from the
            prior.

            This is a 2D tensor where:
            
            * The first dimension has a length equal to the
              number of components in the Gaussian mixture.

            * The second dimension has a length equal to the
              dimensionality of the Gaussian mixture model.
        """

        # Check the prior
        self._check_mean_prior(mean_prior = mean_prior)

        # Inform the user that the prior passed will be
        # used
        infostr = \
            f"The {mean_prior.__class__.__name__} prior " \
            f"will be used as prior over the means of " \
            f"the mixture components."
        logger.info(infostr)


        # Means with shape: n_comp, dim
        mean = \
            nn.Parameter(mean_prior.sample(n = n_comp),
                         requires_grad = True)

        # Return the prior on the means and the means
        return mean_prior, mean


    def _get_log_var(self,
                     cm_type,
                     log_var_params,
                     n_comp,
                     dim):
        """Return the parameters to learn the covariance matrix
        as a negative log variance to ensure it is positive
        definite.

        Parameters
        ----------
        cm_type : ``str``
            The shape of the covariance matrix.

        log_var_params : ``tuple``
            A tuple containing the parameters to initialize the
            Gaussian prior on the log-variances of the components:
            
            * Mean: ``2 * log(log_var_params[0])``
            * Standard deviation: ``log_var_params[1]``

        n_comp : ``int``
            The number of components in the mixture.

        log_var_dim : ``int``
            The dimensionality of the log-variance.

        Returns
        -------
        log_var : ``torch.Tensor``
            The log-variance.

            It is a 2D tensor where:

            * The first dimension has a length equal to the number
              of components in the Gaussian mixture.

            * The second dimension has a length equal to the
              dimensionality of the Gaussian mixture model.
        
        dim : ``int``
            The dimensionality of the Gaussian mixture model.
        
        log_var_factor : ``float``
            The log-variance factor.
        
        log_var_prior: ``bulkDGD.core.priors.GaussianPrior``
            The prior over the log-variance.
        """
        
        # If the covariance matrix is fixed
        if cm_type == "fixed":

            # The log-variance factor will be half
            # of the dimensionality of the space
            log_var_factor = dim * 0.5

            # The dimensionality of the log-variance
            # factor will be 1
            log_var_dim = 1

            # No gradient is needed for training
            log_var = \
                nn.Parameter(torch.empty(n_comp,
                                         log_var_dim),
                             requires_grad = False)

        # If the covariance matrix is isotropic
        elif cm_type == "isotropic":

            # The log-variance factor will be half
            # of the dimensionality of the space
            log_var_factor = dim * 0.5

            # The dimensionality of the log-variance
            # factor will be 1
            log_var_dim = 1

            # Gradients are required
            log_var = \
                nn.Parameter(torch.empty(n_comp,
                                         log_var_dim),
                             requires_grad = True)
        
        # If the covariance matrix is diagonal
        elif cm_type == "diagonal":
            
            # The log-variance factor will be 1/2
            log_var_factor = 0.5

            # The dimensionality of the log-variance
            # will be the dimensionality of the space
            log_var_dim = dim

            # Gradients are required
            log_var = \
                nn.Parameter(torch.empty(n_comp,
                                         log_var_dim),
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
            log_var.fill_(2 * math.log(log_var_params[0]))

        # Get the prior over the log-variance
        log_var_prior = \
            priors.GaussianPrior(\
                dim = log_var_dim,
                mean = 2 * math.log(log_var_params[0]),
                stddev = log_var_params[1])

        # Return the log-variance, its dimensionality, the
        # log-variance factor, and the prior over the
        # log-variance
        return log_var, log_var_dim, log_var_factor, log_var_prior


    def _check_mean_prior(self,
                          mean_prior):
        """Check whether the user-provided prior for the means of
        the mixture components is valid.

        Parameters
        ----------
        mean_prior : ``object``
            An instance of a class representing a prior.
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
    def log_var_params(self):
        """The parameters used to inizialize the Gaussian
        prior on the log-variances of the components:

        * Mean: ``2 * log(log_var_params[0])``
        * Standard deviation: ``log_var_params[1]``
        """

        return self._log_var_params


    @log_var_params.setter
    def log_var_params(self,
                       value):
        """Raise an exception if the user tries to modify
        the value of ``log_var_params`` after initialization.
        """
        
        errstr = \
            "The value of 'log_var_params' cannot be changed " \
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
    def log_var(self):
        """The log-variance of the Gaussian components in the
        mixture.
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


    @property
    def log_var_dim(self):
        """The dimensionality of the log-variance.
        """

        return self._log_var_dim


    @log_var_dim.setter
    def log_var_dim(self,
                    value):
        """Raise an exception if the user tries to modify
        the value of ``log_var_dim`` after initialization.
        """
        
        errstr = \
            "The value of 'log_var_dim' cannot be changed " \
            "after initialization."
        raise ValueError(errstr)


    @property
    def log_var_factor(self):
        """The log-variance factor.
        """

        return self._log_var_factor


    @log_var_factor.setter
    def log_var_factor(self,
                       value):
        """Raise an exception if the user tries to modify
        the value of ``log_var_factor`` after initialization.
        """
        
        errstr = \
            "The value of 'log_var_factor' cannot be changed " \
            "after initialization."
        raise ValueError(errstr)


    @property
    def log_var_prior(self):
        """The prior over the log-variance.
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
        y = y.mul(0.5 * torch.exp(self.log_var))

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
        log_var_sum = self.log_var.sum(-1)
        
        # Add the summed log-variance multiplied by the log-variance
        # factor to the previous tensor.
        # The output tensor has dimensionality:
        # (n_samples, n_comp)
        y = y + (self.log_var_factor * log_var_sum)

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
        mixture_probs : ``torch.Tensor``
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
        dist : ``torch.distributions.MixtureSameFamily``
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
                                    torch.exp(-0.5 * self.log_var)),
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

        # If the log_var prior is not None
        if self.log_var_prior is not None:

            # Add the log_var probability
            p =  p + self.log_var_prior.log_prob(self.log_var).sum()
        
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
        y : ``torch.Tensor``
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
        log_prob : ``torch.Tensor``
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
        samples : ``torch.Tensor``
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
        component_samples : ``torch.Tensor``
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
                                    torch.exp(-0.5 * self.log_var)),
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
        new_points : ``torch.Tensor``
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
        n_samples : ``int``
            The number of samples found in the input tensor (and,
            therefore, the number of representations).

        dim : ``int``
            The dimensionality of the representations.

        mean : ``None``
            The mean of the distribution from which the representations
            were sampled (this is populated by something other than
            ``None`` only when the representations are generated from a
            distribution, and not when ``values`` is passed).

        stddev : ``None``
            The standard deviation of the distribution from which
            the representations were sampled (this is populated by
            something other than ``None`` only when the representations
            are generated from a distribution, and not when ``values``
            is passed).

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
        n_samples : ``int``
            The number of samples found in the input tensor (and,
            therefore, the number of representations).

        dim : ``int``
            The dimensionality of the representations.

        mean : ``float``
            The mean of the distribution from which the representations
            were sampled.

        stddev : ``float``
            The standard deviation of the distribution from which
            the representations were sampled.

        rep : ``torch.Tensor``
            The representations.

            This is a 2D tensor where:

            * The first dimension has a length equal to the number of
              representations.

            * The second dimension has a length equal to the
              dimensionality of the representations.
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
        
        # Get the mean and the standard deviation
        sd, m = torch.std_mean(z_flat)
        
        # Disable the gradient calculation
        with torch.no_grad():

            # Subtract the mean from the representations
            self.z -= m

            # Divide the representations by the standard
            # deviation
            self.z /= sd