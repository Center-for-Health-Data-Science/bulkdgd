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
                 n_mix_comp,
                 mean_prior = None,
                 cm_type = "isotropic",
                 alpha = 1, 
                 softball_params = (0.0, 1.0),
                 logbeta_params = (0.5, 0.5)):
        """Initialize an instance of the GMM.

        Parameters
        ----------
        dim : ``int``
            The dimensionality of the latent space.

        n_mix_comp : ``int``
            The number of components in the mixture.

        mean_prior : any object representing a prior
            Instance of a class representing a prior. If
            not passed, the ``DGDPerturbations.core.priors.softball``
            prior will be used.

        cm_type : ``str``, {``"fixed"``, ``"isotropic"``, ``"diagonal"``}, ``"isotropic"``
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
            - mean: ``2 * log(logbeta_params[0])``
            - stddev: ``logbeta_params[1]``
        
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
        self._n_mix_comp = n_mix_comp

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
            self._get_dirichlet_constant(n_mix_comp = self.n_mix_comp,
                                         alpha = self.alpha)

        # Set the weights, which are initialized uniformly so that
        # the components start out as equiprobable
        self._weight = \
            self._get_weight(n_mix_comp = self.n_mix_comp)
        
        # Initialize the prior on the means and the means themselves
        self._mean_prior, self._mean = \
            self._get_means(mean_prior = mean_prior,
                            softball_params = self.softball_params,
                            dim = self.dim,
                            n_mix_comp = self.n_mix_comp)

        # Initialize the covariance matrix and related parameters
        self._logbeta, self._logbeta_dim, \
            self._logbeta_factor, self._logbeta_prior = \
                self._get_logbeta(\
                    cm_type = cm_type,
                    logbeta_params = self.logbeta_params,
                    n_mix_comp = self.n_mix_comp,
                    dim = self.dim)


    #-------------------- Initialization methods ---------------------#


    def _get_dirichlet_constant(self,
                                n_mix_comp,
                                alpha):
        """Return the Dirichlet constant.

        Parameters
        ----------
        n_mix_comp : ``int``
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

        return math.lgamma(n_mix_comp * alpha) - \
               n_mix_comp * math.lgamma(alpha)


    def _get_weight(self,
                    n_mix_comp):
        """Return the weights of the components in the mixture.

        Parameters
        ----------
        n_mix_comp : ``int``
            The number of components in the mixture.

        Returns
        -------
        ``torch.Tensor``
            The weights of the components in the mixture.
        """

        return nn.Parameter(torch.ones(n_mix_comp),
                            requires_grad = True)


    def _get_means(self,
                   mean_prior,
                   softball_params,
                   dim,
                   n_mix_comp):
        """Return the prior on the means of the Gaussians
        and the the means themselves.

        Parameters
        ----------
        mean_prior : any object representing a prior
            Instance of a class representing a prior. If
            not passed, the ``DGDPerturbations.core.priors.softball``
            prior will be used.

        softball_params : ``tuple``, ``(0.0, 1.0)``
            Parameters (radius, sharpness) of the softball
            prior, if no prior was passed.

        dim : ``int``
            The dimensionality of the latent space.

        n_mix_comp : ``int``
            The number of components in the mixture.

        Returns
        -------
        ``tuple``
            Tuple containing the prior to be used over the
            means of the mixture components and the tensor
            containing the means of the mixture components
            sampled from the prior.
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

        # Means with shape: n_mix_comp, dim
        mean = \
            nn.Parameter(mean_prior.sample(n = n_mix_comp),
                         requires_grad = True)

        # Return the prior on the means and the means
        return mean_prior, mean


    def _get_logbeta(self,
                     cm_type,
                     logbeta_params,
                     n_mix_comp,
                     dim):
        """Return the parameters to learn the covariance matrix
        as a negative log variance to ensure it is positive
        definite.

        Parameters
        ----------
        cm_type : ``str``, {``"fixed"``, ``"isotropic"``, ``"diagonal"``"}
            Shape of the covariance matrix.

        logbeta_params : ``tuple``
            Tuple containing the parameters to initialize the
            Gaussian prior on the log-variances of the components:
            - mean: 2 * log(logbeta_params[0])
            - stddev: logbeta_params[1]

        n_mix_comp : ``int``
            The number of components in the mixture.

        dim : ``int``
            The dimensionality of the latent space.

        Returns
        -------
        ``tuple``
            Tuple containing the log-variance, its dimensionality,
            the log-variance factor, and the prior over the
            log-variance.
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
                nn.Parameter(torch.empty(n_mix_comp,
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
                nn.Parameter(torch.empty(n_mix_comp,
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
                nn.Parameter(torch.empty(n_mix_comp,
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
    def n_mix_comp(self):
        """The number of components in the mixture.
        """

        return self._n_mix_comp


    @n_mix_comp.setter
    def n_mix_comp(self,
                   value):
        """Raise an exception if the user tries to modify
        the value of ``n_mix_comp`` after initialization.
        """
        
        errstr = \
            "The value of 'n_mix_comp' cannot be changed " \
            "after initialization."
        raise ValueError(errstr)


    @property
    def softball_params(self):
        """The parameters (radius, sharpness) of the
        prior that will be used if no prior is passed.
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
        Mean: ``2 * log(logbeta_params[0])``
        Standard deviation: ``logbeta_params[1]``
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
        """The Dirichlet alpha determining the uniformity
        of the weights of the components in the mixture.
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
        """The weights of the components in the mixture.
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
        """The prior on the means of the Gaussians.
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
        """The means of the Gaussians in the mixture.
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
        """The log-variance.
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


    #------------------------ Public methods -------------------------#


    def get_prior_log_prob(self):
        """Calculate the log probability of the prior on the means,
        logbeta, and mixture coefficients.

        Returns
        -------
        p : ``float``
            The probability of the prior.
        """

        # Get the Dirichlet constant
        p = self.dirichlet_constant

        # If the alpha is different from 1
        if self.alpha != 1:

            # Add the log probability on the mixture coefficients
            p = p + \
                (self.alpha - 1.0) * \
                (self.get_mixture_probs().log().sum())
        
        # Add the log probability of the means
        p = p + self.mean_prior.log_prob(self.mean).sum()

        # If the logbeta prior is not None
        # VALE: verify that the None scenario can actually happen,
        # otherwise just remove the if clause
        if self.logbeta_prior is not None:

            # Add the logbeta probability
            p =  p + self.logbeta_prior.log_prob(self.logbeta).sum()
        
        # Return the probability
        return p

          
    def forward(self,
                x):
        """Forward pass - compute the negative log-density
        for a sample.

        VALE: this follows what Viki has done in the new code,
        since in the old code used in the DGD bulk paper the
        absolute value of y was returned, but the function was
        never called anyway.

        Parameters
        ----------
        x : ``torch.Tensor``
            The input samples.
            
        Returns
        -------
        ???
            ???
        """

        # x is unsqueezed to (n_sample, 1, dim) so that the
        # broadcasting of the mean works. Sum terms for
        # each component over the last dimension. 
        y = - (x.unsqueeze(-2) - self.mean).square().mul(\
                0.5 * torch.exp(self.logbeta)).sum(-1)

        y = y + self.logbeta_factor * self.logbeta.sum(-1)

        y = y - 0.5 * self.dim * math.log(2 * math.pi)

        # For each component, multiply by the mixture probabilities
        y = y + torch.log_softmax(self.weight,
                                  dim = 0)
        
        y = torch.logsumexp(y,
                            dim = -1)
        
        y = y + self.get_prior_log_prob()

        # Return the negative log probability density
        return (-y)


    def log_prob(self,
                 x):
        """Get the log density of the probability
        of ``x`` being drawn from the GMM.

        Parameters
        ----------
        x : ``torch.Tensor``
            The input samples.

        Returns
        -------
        ???
            ???
        """
        
        return - self.forward(x)


    def get_mixture_probs(self):
        """Convert the weights into mixture probabilities using
        the softmax function.

        Returns
        -------
        ``torch.Tensor``
            The mixture probabilities.
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
            
            return dist.MixtureSameFamily(mix, comp)


    def sample(self,
               n_samples):
        """Create samples from the GMM distribution.

        Parameters
        ----------
        n : ``int``
            Number of samples.

        Returns
        -------
        ``torch.Tensor``
            Sampled points from the GMM distribution.
        """
        
        # Disable gradient calculation
        with torch.no_grad():

            # Get the GMM distribution
            gmm_dist = self.get_distribution()

            # Sample from the distribution
            return gmm_dist.sample(torch.tensor([n_samples]))


    def component_sample(self,
                         n_samples):
        """Returns a sample from each component of the GMM. 
        The shape of the tensor returned is
        (``n_samples``, ``n_mix_comp``, ``dim``).

        Parameters
        ----------
        n_samples : ``int``
            Number of samples.

        Returns
        -------
        ``torch.Tensor``
            The sampled points from the GMM components.
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


    def sample_probs(self,
                     x):
        """Return the probability densities per sample per
        component.

        Parameters
        ----------
        x : ``torch.Tensor``
            The input samples.

        Returns
        -------
        ``torch.Tensor``
            The probability densities per sample per component.
        """

        # x is unsqueezed to (n_sample, 1, dim) so that the
        # broadcasting of the mean works. Sum terms for
        # each component over the last dimension. 
        y = - (x.unsqueeze(-2) - self.mean).square().mul(\
                0.5 * torch.exp(self.logbeta)).sum(-1)

        y = y + self.logbeta_factor * self.logbeta.sum(-1)

        y = y - 0.5 * self.dim * math.log(2 * math.pi)

        # For each component, multiply by the mixture probabilities
        y = y + torch.log_softmax(self.weight,
                                  dim = 0)

        # Return the probability densities per sample per
        # component
        return torch.exp(y)


    #------------------------ New data points ------------------------#


    def sample_new_points(self,
                          n_points,
                          n_reps_per_mix_comp = 1,
                          sampling_method = "random"):
        """Generate samples for new data points.

        ``n_rep`` representations will be generated for each data
        point.

        Parameters
        ----------
        n_points : ``int``
            Number of new data points for which representations
            have to be found.

        n_reps_per_mix_comp : ``int``, ``1``
            Number of new representations per component per sample.

        sampling_method : ``str``, {``"mean"``, ``"random"``}, ``"random"``
            How to sample the representations for the given data points:

            * ``"random"`` means sampling ``n_rep`` vectors from each
              mixture component.

            * ``"mean"`` means taking the mean of each component as
              value of each ``n_rep`` vector sampled for each data
              point.

        Returns
        -------
        ``torch.Tensor``
            The representations for the new data points in a 2D
            tensor with:

            * 1st dimension: ``n_points`` * ``n_reps_per_mix_comp``
              * number of components in the GMM.

            * 2nd dimension: dimensionality of the GMM.
        """

        # Supported sampling methods
        SAMPLING_METHODS = ["random", "mean"]

        # Get the number of new data points to find a representation
        # for
        self.new_samples = n_reps_per_mix_comp

        # Get the total number of samples to be taken from the
        # model
        n_samples = n_points * n_reps_per_mix_comp

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
        #                  representations found per component per
        #                  data point times number of components
        #                  in the mixture ->
        #                  'n_points' *
        #                  'n_mix_comp' *
        #                  'n_reps_per_mix_comp'
        #
        # - 2nd dimension: dimensionality of the Gaussian mixture
        #                  model ->
        #                  'dim'
        return out.view(n_samples * self.n_mix_comp,
                        self.dim)


    def reshape_targets(self,
                        y,
                        y_type = "true"):
        """Reshape the output to calculate the losses, since we have
        multiple representations for the same data point.

        Parameters
        ----------
        y : ``torch.Tensor``
            The representations for a new data point.

        y_type : ``str``, {``"true"``, ``"predicted"``, ``"reverse"``}, ``"true"``
            Type of input ``y``. If ``"true"``, ``y`` represents the
            true targets. If ``"predicted"``, ``y`` represents the
            model's predictions. If ``"reverse"``, ``y`` represents
            the 4-dimensional representation or the loss.

        Returns
        -------
        ``torch.Tensor``
            The reshaped input tensor.
        """

        # Supported y_type options
        Y_TYPE_OPTIONS = ["true", "predicted", "reverse"]
        
        # If we are dealing with the true targets
        if y_type == "true":

            # If y has more than 2 dimensions
            if len(y.shape) > 2:

                # Raise an exception
                errstr = \
                    f"If 'y_type' is 'true', 'y' must have " \
                    f"at most 2 dimensions. Instead, 'y' " \
                    f"has {len(y.shape)} dimensions."
                raise ValueError(errstr)

            return y.unsqueeze(1).unsqueeze(1).expand(\
                    -1, self.new_samples, self.n_mix_comp, -1)
        
        # If we are dealing with the model's predictions
        elif y_type == "predicted":

            # If y has more than 2 dimensions
            if len(y.shape) > 2:
                
                # Raise an exception
                errstr = \
                    f"If 'y_type' is 'predicted', 'y' must have " \
                    f"at most 2 dimensions. Instead, 'y' " \
                    f"has {len(y.shape)} dimensions."
                raise ValueError(errstr)

            n_points = \
                int(torch.numel(y) / \
                (self.new_samples * self.n_mix_comp * y.shape[-1]))
            
            return y.view(n_points,
                          self.new_samples,
                          self.n_mix_comp,
                          y.shape[-1])
        
        # If we are dealing with losses
        elif "reverse":
            
            # If they have less than 4 dimensions (i.e, shape 
            # equal to (n_points,self.new_samples,self.n_mix_comp))
            if len(y.shape) < 4:
                
                return y.view(y.shape[0] * \
                              self.new_samples * \
                              self.n_mix_comp)
            
            # Otherwise
            else:
                
                return y.view(y.shape[0] * \
                              self.new_samples * \
                              self.n_mix_comp,
                              y.shape[-1])
        
        # Otherwise
        else:

            # Raise an exception
            supported_ytypes = \
                ", ".join([f"'{yt}'" for yt in Y_TYPE_OPTIONS])
            errstr = \
                f"Unrecognized 'y_type' ('{y_type}''). Supported " \
                f"values are: {supported_ytypes}."
            raise ValueError(errstr)

 
    def choose_best_representations(self,
                                    x,
                                    losses):
        """For each new data point, chose the representation
        that minimizes the loss.

        Parameters
        ----------
        x : ``torch.Tensor``
            The newly learned representations.

        losses : ``torch.Tensor``
            The losses associated to the representations.
            ``x`` and ``losses`` need to have the same shape in
            the first dimension. Make sure that the losses are
            only summed over the output dimension.

        Returns
        -------
        ``torch.Tensor``
            The best representations for the new data points.
        """

        n_points = \
            int(torch.numel(losses) / \
            (self.new_samples * self.n_mix_comp))
        
        # Get the best sample
        best_sample = \
            torch.argmin(\
                losses.view(-1,
                            self.new_samples * self.n_mix_comp),
                            dim = 1).squeeze(-1)

        # Get the best representation for each sample
        best_rep = \
            x.view(n_points,
                   self.new_samples * self.n_mix_comp,
                   self.dim)[range(n_points), best_sample]

        # Return the best representation
        return best_rep