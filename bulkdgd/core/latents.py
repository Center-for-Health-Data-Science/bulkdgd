#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-

#    latent.py
#
#    This module contains the classes implementing the components of
#    the latent space of the :class:`core.model.BulkDGD`, namely
#    the Gaussian mixture model
#    (:class:`core.latent.GaussianMixtureModel`) and the representation
#    layer (:class:`core.latent.RepresentationLayer`).
#
#    The code was originally developed by Viktoria Schuster,
#    Inigo Prada Luengo, and Anders Krogh.
#
#    Valentina Sora modified and complemented it for the purposes
#    of this package.
#
#    Copyright (C) 2026 Valentina Sora
#                       <sora.valentina1@gmail.com>
#                       Adrián Avelino Sousa-Poza
#                       <asp@di.ku.dk>
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
    "This module contains the classes implementing the components " \
    "of the latent space of the :class:`core.model.BulkDGD`, " \
    "namely the Gaussian mixture model " \
    "(:class:`core.latent.GaussianMixtureModel`) and the " \
    "representation layer (:class:`core.latent.RepresentationLayer`)."


#######################################################################


# Import from the standard library.
import logging as log
import math
from typing import Dict, Optional, Union

# Import from third-party libraries.
import torch
import torch.nn as nn
import tgmm

# Import from 'bulkdgd'.
from . import priors


#######################################################################


# Get the module's logger.
logger = log.getLogger(__name__)


#######################################################################


class GaussianMixtureModelLegacy(nn.Module):
    
    """
    A class implementing the legacy mixture of multivariate Gaussian
    distributions (Gaussian mixture model or GMM).
    """

    # Set the supported priors over the means of the components of
    # the Gaussian mixture model.
    MEANS_PRIORS = ["softball"]

    # Set the supported priors over the weights of the components
    # of the Gaussian mixture model.
    WEIGHTS_PRIORS = ["dirichlet"]

    # Set the supported priors over the negative log-variance of the
    # components of the Gaussian mixture model.
    LOG_VAR_PRIORS = ["gaussian"]

    # Set the supported types of covariance matrix.
    COVARIANCE_TYPES = ["fixed", "isotropic", "diagonal"] 


    def __init__(self,
                 dim: int,
                 n_components: int,
                 means_prior_name: str,
                 weights_prior_name: str,
                 log_var_prior_name: str,
                 means_prior_options: dict[str, object],
                 weights_prior_options: dict[str, object],
                 log_var_prior_options: dict[str, object],
                 covariance_type: str = "diagonal") -> None:
        """Initialize an instance of the GMM.

        Parameters
        ----------
        dim : :class:`int`
            The dimensionality of the Gaussian mixture model.

        n_components : :class:`int`
            The number of components in the Gaussian mixture.

        means_prior_name : :class:`str`, {``"softball"``}
            The name of the prior over the means of the components
            of the Gaussian mixture.

        weights_prior_name : :class:`str`, {``"dirichlet"``}
            The name of the prior over the weights of the components
            of the Gaussian mixture.

        log_var_prior_name : :class:`str`, {``"gaussian"``}
            The name of the prior over the negative log-variance of the
            components of the Gaussian mixture.

        means_prior_options : :class:`dict`
            A dictionary containing the options needed to set up the
            prior over the means of the components of the Gaussian
            mixture model.

            It varies according to the selected prior.

            If ``means_prior_name`` is ``"softball"``, the
            options that must be provided are:

            * ``"radius"``, namely the radius of the multi-
              dimensional soft ball.

            * ``"sharpness"``, namely the sharpness of the
              soft boundary of the ball.

        weights_prior_options : :class:`dict`
            A dictionary containing the options needed to set up the
            prior over the weights of the components of the Gaussian
            mixture model.

            It varies according to the selected prior.

            If ``weights_prior_name`` is ``"dirichlet"``, the options
            that must be provided are:

            * ``"alpha"``, namely the alpha of the Dirichlet
              distribution.

        log_var_prior_options : :class:`dict`
            A dictionary containing the options needed to set up the
            prior over the negative log-variance of the Gaussian
            mixture model.

            It varies according to the selected prior.

            If ``log_var_prior_name`` is ``"gaussian"``, the
            options that must be provided are:

            * ``"mean"``, namely the mean of the Gaussian
              distribution.

            * ``"stddev"``, namely the standard deviation of the
              Gaussian distribution.

        covariance_type : :class:`str`, {``"fixed"``, ``"isotropic"``, \
            ``"diagonal"``}, ``"diagonal"``
            The type of covariance matrix used.
        """

        # Initialize an instance of 'nn.Module'.
        super().__init__()
        
        # Set the dimensionality of the Gaussian mixture model.
        self._dim = dim

        # Set the number of components in the mixture.
        self._n_components = n_components

        # Set the type of the covariance matrix.
        self._covariance_type = covariance_type

        #-------------------------------------------------------------#

        # Set the prior over the means of the components.
        self._means_prior = \
            self._get_means_prior(\
                means_prior_name = means_prior_name,
                means_prior_options = means_prior_options)

        # Set the means.
        self._means = self._get_means()

        # Set a string with the options used for the prior over the
        # means.
        means_prior_opts_str = \
            ", ".join([f"{opt} = '{val}'" \
                       if isinstance(val, str) \
                       else f"{opt} = {val}" \
                       for opt, val in means_prior_options.items()])

        # Inform the user that the prior over the means was set.
        logstr = \
            "The prior over the means of the components of the " \
            "Gaussian mixture model was set. Prior " \
            f"'{means_prior_name}' ({means_prior_opts_str})."
        log.info(logstr)

        #-------------------------------------------------------------#

        # Set the prior over the weights of the components.
        self._weights_prior = \
            self._get_weights_prior(\
                weights_prior_name = weights_prior_name,
                weights_prior_options = weights_prior_options)

        # Set the weights.
        self._weights = self._get_weights()

        # Set a string with the options used for the prior over the
        # weights.
        weights_prior_opts_str = \
            ", ".join([f"{opt} = '{val}'" \
                       if isinstance(val, str) \
                       else f"{opt} = {val}" \
                       for opt, val in weights_prior_options.items()])

        # Inform the user that the prior over the weights was set.
        logstr = \
            "The prior over the weights of the components of the " \
            "Gaussian mixture model was set. Prior " \
            f"'{weights_prior_name}' ({weights_prior_opts_str})."
        log.info(logstr)

        #-------------------------------------------------------------#

        # Set the prior over the log-variance of the components.
        self._log_var_prior = \
            self._get_log_var_prior(\
                log_var_prior_name = log_var_prior_name,
                log_var_prior_options = log_var_prior_options)

        # Get the log-variance.
        self._log_var = self._get_log_var()

        # Set a string with the options used for the prior over the
        # log-variance.
        log_var_prior_opts_str = \
            ", ".join([f"{opt} = '{val}'" \
                       if isinstance(val, str) \
                       else f"{opt} = {val}" \
                       for opt, val in log_var_prior_options.items()])

        # Inform the user that the prior over the log-variance was set.
        logstr = \
            "The prior over the log-variance of the components of " \
            "the Gaussian mixture model was set. Prior " \
            f"'{log_var_prior_name}' ({log_var_prior_opts_str})."
        log.info(logstr)


    ######################### INITIALIZATION ##########################


    def _get_means_prior(self,
                         means_prior_name: str,
                         means_prior_options: dict[str, object]) -> \
                            dict[str, object]:
        """Get the prior over the means of the components of the
        Gaussian mixture model.

        Parameters
        ----------
        means_prior_name : :class:`str`
            The name of the prior.

        means_prior_options : :class:`dict`
            The options to set up the prior.

        Returns
        -------
        means_prior_dict : :class:`dict`
            A dictionary containing the name of the prior and the
            options and distribution associated with it.
        """

        # If the prior is the softball distribution
        if means_prior_name == "softball":

            # Get the distribution.
            dist = \
                priors.SoftballPrior(dim = self.dim,
                                     **means_prior_options)

            # Return the dictionary with the name of the prior and
            # associated options and distribution.
            return {"name" : means_prior_name,
                    "options" : \
                        {"dim" : self.dim,
                         **means_prior_options},
                    "dist" : dist}

        #-------------------------------------------------------------#

        # Otherwise
        else:

            # Raise an error.
            errstr = \
                f"Unrecognized prior '{means_prior_name}' passed " \
                "to 'means_prior_name'. Supported priors are: " \
                f"{', '.join(self.MEANS_PRIORS)}."
            raise ValueError(errstr)


    def _get_means(self) -> nn.Parameter:
        """Return the prior on the means of the Gaussian distributions
        and the means themselves.

        Returns
        -------
        means : :class:`torch.nn.Parameter`
            The means of the Gaussian mixture components sampled from
            the prior.

            This is a 2D tensor where:
            
            * The first dimension has a length equal to the number of
              components in the Gaussian mixture.

            * The second dimension has a length equal to the
              dimensionality of the Gaussian mixture model.
        """

        # Get the distribution representing the prior.
        dist_prior = self.means_prior["dist"]

        # Get the means of the mixture. This is a two dimensional
        # entity with dimensionality 'n_components', 'dim'.
        means = \
            nn.Parameter(dist_prior.sample(
                            n_samples = self.n_components),
                         requires_grad = True)

        # Return the means sampled from the prior.
        return means


    def _get_weights_prior(
            self,
            weights_prior_name: str,
            weights_prior_options: dict[str, object]) -> \
                dict[str, object]:
        """Get the prior over the weights of the components of the
        Gaussian mixture model.

        Parameters
        ----------
        weights_prior_name : :class:`str`
            The name of the prior.

        weights_prior_options : :class:`dict`
            The options to set up the prior.

        Returns
        -------
        weights_prior_dict : :class:`dict`
            A dictionary containing the name of the prior and the
            options associated with it.
        """

        # If the prior is the Dirichlet distribution
        if weights_prior_name == "dirichlet":

            # Get the alpha.
            alpha = weights_prior_options.get("alpha")

            # If the 'alpha' was not provided
            if alpha is None:

                # Raise an error.
                errstr = \
                    "If 'weights_prior_name' is 'dirichlet', " \
                    "'weights_prior_options' must contain " \
                    "the alpha of the Dirichlet distribution " \
                    "('alpha')."
                raise KeyError(errstr)

            # Calculate the Dirichlet constant.
            constant = \
                math.lgamma(self.n_components * alpha) - \
                            self.n_components * math.lgamma(alpha)

            # Return the dictionary with the name of the prior and
            # associated options.
            return {"name" : weights_prior_name,
                    "options" : \
                        {"alpha" : alpha,
                         "constant" : constant}}

        #-------------------------------------------------------------#

        # Otherwise
        else:

            # Raise an error.
            errstr = \
                f"Unrecognized prior '{weights_prior_name}' passed " \
                "to 'weights_prior_name'. Supported priors are: " \
                f"{', '.join(self.WEIGHTS_PRIORS)}."
            raise ValueError(errstr)


    def _get_weights(self) -> torch.nn.Parameter:
        """Return the weights of the components in the Gaussian
        mixture model.

        Returns
        -------
        weights : :class:`torch.nn.Parameter`
            The weights of the components in the Gaussian mixture.
            
            The is a 1D tensor having a length equal to the number
            of components in the Gaussian mixture model.
        """

        # Return the weights of the components.
        return nn.Parameter(torch.ones(self.n_components),
                            requires_grad = True)


    def _get_log_var_prior(
            self,
            log_var_prior_name: str,
            log_var_prior_options: dict[str, object]) -> object:
        """Get the prior over the log-variance of the components of
        the Gaussian mixture model.

        Parameters
        ----------
        log_var_prior_name : :class:`str`
            The name of the prior.

        log_var_prior_options : :class:`dict`
            The options to set up the prior.

        Returns
        -------
        log_var_prior_dict : :class:`dict`
            A dictionary containing the name of the prior and the
            options and distribution associated with it.
        """

        # If the prior is the Gaussian distribution
        if log_var_prior_name == "gaussian":

            # Get the mean of the Gaussian distribution.
            dist_mean = log_var_prior_options.get("mean")

            # Get the standard deviation of the Gaussian distribution.
            dist_stddev = log_var_prior_options.get("stddev")

            # If the mean of the Gaussian distribution was not provided
            if dist_mean is None:

                # Raise an error.
                errstr = \
                    "If 'log_var_prior_name' is 'gaussian', " \
                    "'log_var_prior_options' must contain the " \
                    "mean of the Gaussian distribution ('mean')."
                raise KeyError(errstr)

            # If the standard deviation of the Gaussian distribution
            # was not provided
            if dist_stddev is None:

                # Raise an error.
                errstr = \
                    "If 'log_var_prior_name' is 'gaussian', " \
                    "'log_var_prior_options' must contain the " \
                    "standard deviation of the Gaussian " \
                    "distribution ('stddev')."
                raise KeyError(errstr)

            #---------------------------------------------------------#

            # If the covariance matrix is fixed
            if self.covariance_type == "fixed":

                # The log-variance factor will be half of the
                # dimensionality of the space.
                dist_factor = self.dim * 0.5

                # The dimensionality of the log-variance factor will
                # be 1.
                dist_dim = 1

                # Gradients will not be required.
                requires_grad = False

            #---------------------------------------------------------#

            # If the covariance matrix is isotropic
            elif self.covariance_type == "isotropic":

                # The log-variance factor will be half of the
                # dimensionality of the space.
                dist_factor = self.dim * 0.5

                # The dimensionality of the log-variance factor will
                # be 1.
                dist_dim = 1

                # Gradients will be required.
                requires_grad = True

            #---------------------------------------------------------#
        
            # If the covariance matrix is diagonal
            elif self.covariance_type == "diagonal":
                
                # The log-variance factor will be 1/2.
                dist_factor = 0.5

                # The dimensionality of the log-variance will be the
                # dimensionality of the space.
                dist_dim = self.dim

                # Gradients will be required.
                requires_grad = True

            #---------------------------------------------------------#

            # Get the distribution.
            dist = \
                priors.GaussianPrior(dim = dist_dim,
                                     mean = -2 * math.log(dist_mean),
                                     stddev = dist_stddev)

            #---------------------------------------------------------#

            # Return the dictionary with the name of the prior and
            # associated options and distribution.
            return {"name" : log_var_prior_name,
                    "options" : \
                        {"factor" : dist_factor,
                         "dim" : dist_dim,
                         "requires_grad" : requires_grad,
                         "mean" : dist_mean,
                         "stddev" : dist_stddev},
                     "dist" : dist}

        #-------------------------------------------------------------#

        # Otherwise
        else:

            # Raise an error
            errstr = \
                f"Unrecognized prior '{log_var_prior_name}' " \
                "passed to 'log_var_prior_name'. Supported " \
                f"priors are: {', '.join(self.LOG_VAR_PRIORS)}."
            raise ValueError(errstr)


    def _get_log_var(self) -> torch.nn.Parameter:
        """Get the log-variance of the components of the Gaussian
        mixture model.

        Returns
        -------
        log_var : :class:`torch.Tensor`
            The negative log-variance of the components.

            It is a 2D tensor where:

            * The first dimension has a length equal to the number
              of components in the Gaussian mixture.

            * The second dimension has a length equal to the
              dimensionality of the Gaussian mixture model.
        """

        # Get the dimension of the log-variance of the components.
        log_var_dim = \
            self.log_var_prior["options"]["dim"]
        
        # Get whether the log-variance requires gradient calculation.
        requires_grad = \
            self.log_var_prior["options"]["requires_grad"]

        # Get the log-variance.
        log_var = nn.Parameter(torch.empty(self.n_components,
                                           log_var_dim),
                               requires_grad = requires_grad)

        # Get the name of the prior.
        log_var_prior_name = self.log_var_prior["name"]

        #-------------------------------------------------------------#

        # If the prior is a Gaussian distribution
        if log_var_prior_name == "gaussian":

            # Get the mean of the Gaussian distribution.
            dist_mean = self.log_var_prior["options"]["mean"]

            # Disable gradient calculation.
            with torch.no_grad():

                # Populate the log-variance.
                log_var.fill_(2 * math.log(dist_mean))

        #-------------------------------------------------------------#

        # Otherwise
        else:

            # Raise an error.
            errstr = \
                f"Unrecognized prior '{log_var_prior_name}' " \
                "passed to 'log_var_prior_name'. Supported " \
                f"priors are: {', '.join(self.LOG_VAR_PRIORS)}."
            raise ValueError(errstr)

        # Return the log-variance.
        return log_var


    ########################### PROPERTIES ############################


    @property
    def dim(self):
        """The dimensionality of the Gaussian mixture model.
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
            "of the Gaussian mixture model, initialize a new " \
            f"instance of '{self.__class__.__name__}'."
        raise ValueError(errstr)
    

    #-----------------------------------------------------------------#


    @property
    def n_components(self):
        """The number of components in the Gaussian mixture.
        """

        return self._n_components


    @n_components.setter
    def n_components(self,
               value):
        """Raise an exception if the user tries to modify the value of
        ``n_components`` after initialization.
        """
        
        errstr = \
            "The value of 'n_components' is set at initialization " \
            "and cannot be changed. If you want to change the " \
            "number of components of the Gaussian mixture model, " \
            "initialize a new instance " \
            f"of '{self.__class__.__name__}'."
        raise ValueError(errstr)


    #-----------------------------------------------------------------#


    @property
    def covariance_type(self):
        """The type of the covariance matrix.
        """
        
        return self._covariance_type


    @covariance_type.setter
    def covariance_type(self,
                value):
        """Raise an exception if the user tries to modify the value of
        ``covariance_type`` after initialization.
        """
        
        errstr = \
            "The value of 'covariance_type' is set at initialization and " \
            "cannot be changed. If you want to change the type of " \
            "covariance matrix used for the Gaussian mixture model, " \
            "initialize a new instance of " \
            f"'{self.__class__.__name__}'."
        raise ValueError(errstr)


    #-----------------------------------------------------------------#


    @property
    def means_prior(self):
        """A dictionary containing the name of the prior over the
        means of the components of the Gaussian mixture model and the
        options used to set it up.
        """

        return self._means_prior


    @means_prior.setter
    def means_prior(self,
                    value):
        """Raise an exception if the user tries to modify the value of
        ``means_prior`` after initialization.
        """
        
        errstr = \
            "The value of 'means_prior' is set at initialization " \
            "and cannot be changed. If you want to change the prior " \
            "over the means of the components of the Gaussian " \
            "model, initialize a new instance of " \
            f"'{self.__class__.__name__}' and change the " \
            "'means_prior_name' and/or the 'means_prior_options'."
        raise ValueError(errstr)


    #-----------------------------------------------------------------#


    @property
    def means(self):
        """The means of the components of the Gaussian mixture model.
        """

        return self._means


    @means.setter
    def means(self,
              value):
        """Raise an exception if the user tries to modify the value
        of ``means`` after initialization.
        """
        
        errstr = \
            "The value of 'means' is set at initialization and " \
            "cannot be changed. The means of the components of the " \
            "Gaussian mixture model are initialized according to " \
            "the specified prior (defined by 'means_prior_name' " \
            "and 'means_prior_options')."
        raise ValueError(errstr)


    #-----------------------------------------------------------------#


    @property
    def weights_prior(self):
        """A dictionary containing the name of the prior over the
        weights of the components of the Gaussian mixture model and
        the options used to set it up.
        """

        return self._weights_prior


    @weights_prior.setter
    def weights_prior(self,
                      value):
        """Raise an exception if the user tries to modify the value of
        ``weights_prior`` after initialization.
        """
        
        errstr = \
            "The value of 'weights_prior' is set at initialization " \
            "and cannot be changed. If you want to change the prior " \
            "over the weights of the components of the Gaussian " \
            "mixture model, initialize a new instance of " \
            f"'{self.__class__.__name__}' and change the " \
            "'weights_prior_name' and/or the 'weights_prior_options'."
        raise ValueError(errstr)


    #-----------------------------------------------------------------#
    

    @property
    def weights(self):
        """The weights of the components of the Gaussian mixture model.
        """

        return self._weights


    @weights.setter
    def weights(self,
                value):
        """Raise an exception if the user tries to modify the value of
        ``weights`` after initialization.
        """
        
        errstr = \
            "The value of 'weights' is set at initialization and " \
            "cannot be changed. The weights of the components of " \
            "the Gaussian mixture model are initialized according " \
            "to the specified prior (defined by " \
            "'weights_prior_name' and 'weights_prior_options')."
        raise ValueError(errstr)


    #-----------------------------------------------------------------#


    @property
    def log_var_prior(self):
        """A dictionary containing the name of the prior over the
        log-variance of the components of the Gaussian mixture model
        and the options used to set it up.
        """

        return self._log_var_prior


    @log_var_prior.setter
    def log_var_prior(self,
                      value):
        """Raise an exception if the user tries to modify the value of
        ``log_var_prior`` after initialization.
        """
        
        errstr = \
            "The value of 'log_var_prior' is set at initialization " \
            "and cannot be changed. If you want to change the prior " \
            "over the log-variance of the components of the " \
            "Gaussian mixture model, initialize a new instance of " \
            f"'{self.__class__.__name__}' and change the " \
            "'log_var_prior_name' and/or the 'log_var_prior_options'."
        raise ValueError(errstr)


    #-----------------------------------------------------------------#


    @property
    def log_var(self):
        """The log-variance of the components of the Gaussian mixture
        model.
        """

        return self._log_var


    @log_var.setter
    def log_var(self,
                value):
        """Raise an exception if the user tries to modify the value of
        ``log_var`` after initialization.
        """
        
        errstr = \
            "The value of 'log_var' is set at initialization and " \
            "cannot be changed. The log-variance of the components " \
            "of the Gaussian mixture model is initialized according " \
            "to the specified prior (defined by " \
            "'log_var_prior_name' and 'log_var_prior_options')."
        raise ValueError(errstr)


    ######################### PRIVATE METHODS #########################


    def _get_log_prob_comp(self,
                           x: torch.Tensor) -> torch.Tensor:
        """Get the per-data-point, per-component log-probability.

        Parameters
        ----------
        x : :class:`torch.Tensor`
            The input data points. This is a 2D tensor where:

            * The first dimension has a length equal to the number of
              data points.

            * The second dimension has a length equal to the
              dimensionality of the data points.

        Returns
        -------
        log_prob_comp : :class:`torch.Tensor`
            The per-sample, per-component negative log-probability.

            This is a 2D tensor where:

            * The first dimension has a length equal to the number
              of data points in the input tensor.

            * The second dimension has a length equal to the
              dimensionality of the data points.
        """

        # Get the log-variance factor.
        log_var_factor = self.log_var_prior["options"]["factor"]

        #-------------------------------------------------------------#

        # Compute the covariance matrix of the components of the
        # Gaussian mixture model. The covariance matrix is
        # the exponential of the log-variance.
        covariance = torch.exp(self.log_var)

        #-------------------------------------------------------------#

        # Get the 'pi' term.
        pi_term = - 0.5 * self.dim * math.log(2 * math.pi)

        #-------------------------------------------------------------#

        # Get the means term.
        y = \
            -(x.unsqueeze(-2) - self.means).square().div(\
                2 * covariance).sum(-1)
        
        # Add the log-variance term.
        y = y - (log_var_factor * self.log_var.sum(-1))

        # Add the 'pi' term.
        y = y + pi_term

        # Add the log of the softmax of the weights of the
        # components.
        y = y + torch.log_softmax(self.weights,
                                  dim = 0)

        #-------------------------------------------------------------#

        # Return the tensor
        return y


    ######################### PUBLIC METHODS ##########################


    def set_means(self,
                  means: torch.Tensor) -> None:
        """Set the means of the components of the Gaussian mixture model.

        Parameters
        ----------
        means : :class:`torch.Tensor`
            The means of the components of the Gaussian mixture model.

            This is a 2D tensor where:

            * The first dimension has a length equal to the number of
              components in the Gaussian mixture.

            * The second dimension has a length equal to the
              dimensionality of the Gaussian mixture model.
        """

        # Set the means.
        self._means = means


    def set_weights(self,
                    weights: torch.Tensor) -> None:
        """Set the weights of the components of the Gaussian mixture
        model.

        Parameters
        ----------
        weights : :class:`torch.Tensor`
            The weights of the components of the Gaussian mixture model.

            This is a 1D tensor whose size equals the number of
            components in the Gaussian mixture model.
        """

        # Set the weights.
        self._weights = weights
    

    def set_log_var(self,
                    log_var: torch.Tensor) -> None:
        """Set the log-variance of the components of the Gaussian
        mixture model.

        Parameters
        ----------
        log_var : :class:`torch.Tensor`
            The log-variance of the components of the Gaussian mixture
            model.

            This is a 2D tensor where:

            * The first dimension has a length equal to the number of
              components in the Gaussian mixture.

            * The second dimension has a length equal to the
              dimensionality of the Gaussian mixture model.
        """

        # Set the log-variance.
        self._log_var = log_var


    def get_mixture_probs(self) -> torch.Tensor:
        """Convert the weights into mixture probabilities using the
        softmax function.

        Returns
        -------
        mixture_probs : :class:`torch.Tensor`
            The mixture probabilities.

            This is a 1D tensor whose size equals the number of
            components in the Gaussian mixture model.
        """
        
        # Return the mixture probabilities.
        return torch.softmax(self.weights,
                             dim = -1)


    def get_prior_log_prob(self) -> float:
        """Calculate the log-probability of the prior over the means,
        log-variance, and mixture coefficients.

        Returns
        -------
        p : :class:`float`
            The log-probability of the prior.
        """

        # Initialize the probability to 0.0.
        p = 0.0

        #-------------------------------------------------------------#

        # Get the name of the prior over the weights of the
        # components.
        weights_prior_name = self.weights_prior["name"]

        # If the prior over the weights is the Dirichlet prior
        if weights_prior_name == "dirichlet":

            # Get the alpha.
            alpha = self.weights_prior["options"]["alpha"]

            # Get the Dirichlet constant.
            p = self.weights_prior["options"]["constant"]

            # If the alpha is different from 1
            if alpha != 1:

                # Add the log-probability to the mixture coefficients.
                p = p + \
                    (alpha - 1.0) * \
                    (self.get_mixture_probs().log().sum())

        # Otherwise
        else:

            # Raise an error.
            errstr = \
                f"Unsupported prior '{weights_prior_name}' for " \
                "the weights of the components of the Gaussian " \
                "mixture model. Supported priors are: " \
                f"{', '.join(self.WEIGHTS_PRIORS)}."
            raise ValueError(errstr)

        #-------------------------------------------------------------#

        # Get the name of the prior over the means of the
        # components.
        means_prior_name = self.means_prior["name"]

        # If the prior over the means is the softball prior
        if means_prior_name == "softball":

            # Get the prior distribution.
            dist_means_prior = self.means_prior["dist"]
            
            # Add the log probability of the means.
            p = p + dist_means_prior.log_prob(self.means).sum()

        # Otherwise
        else:

            # Raise an error.
            errstr = \
                f"Unsupported prior '{means_prior_name}' for " \
                "the means of the components of the Gaussian " \
                "mixture model. Supported priors are: " \
                f"{', '.join(self.MEANS_PRIORS)}."
            raise ValueError(errstr)

        #-------------------------------------------------------------#

        # Get the name of the prior over the log-variance of the
        # components.
        log_var_prior_name = self.log_var_prior["name"]

        # If the prior over the log-variance is the gaussian prior
        if log_var_prior_name == "gaussian":

            # Get the prior distribution.
            dist_log_var_prior = self.log_var_prior["dist"]

            # Add the log-probability of the log-variance.
            p =  p + \
                dist_log_var_prior.log_prob(-self.log_var).sum()

        # Otherwise
        else:

            # Raise an error.
            errstr = \
                f"Unsupported prior '{log_var_prior_name}' for " \
                "the log-variance of the Gaussian mixture model. " \
                "Supported priors are: " \
                f"{', '.join(self.LOG_VAR_PRIORS)}."
            raise ValueError(errstr)

        # Return the probability.
        return p

          
    def forward(self,
                x: torch.Tensor) -> torch.Tensor:
        """Forward pass - compute the absolute log-probability density
        for a set of data points.

        Parameters
        ----------
        x : :class:`torch.Tensor`
            The input data points. This is a 2D tensor where:

            * The first dimension has a length equal to the number of
              data points.

            * The second dimension has a length equal to the
              dimensionality of the data points.
            
        Returns
        -------
        y : :class:`torch.Tensor`
            The result of the forward pass.

            This is a 1D tensor whose size is equal to the number of
            input data points.

            Each element of the tensor is the absolute log-probability
            density of a data point.
        """

        # Get the per-sample, per-component log-probability.
        y = self._get_log_prob_comp(x = x)

        # Get the log of summed exponentials of each row of the tensor
        # in the last dimension (= the number of components in the
        # mixture).
        #
        # The output is a 1D tensor whose length is equal to the number
        # of samples.
        y = torch.logsumexp(y,
                            dim = -1)
        
        # Add the log-probability of the priors.
        # We divide by the batch size so that upon .sum(), the prior
        # is added exactly once for the batch instead of multiplied.
        y = y + self.get_prior_log_prob() / x.shape[0]
        
        # Return the negative log-probability density.
        return -y


    def sample_probs(self,
                     x: torch.Tensor) -> torch.Tensor:
        """Get the probability density per sample per component.

        Parameters
        ----------
        x : :class:`torch.Tensor`
            The input data points. This is a 2D tensor where:

            * The first dimension has a length equal to the number
              of data points.

            * The second dimension has a length equal to the
              dimensionality of the data points.

        Returns
        -------
        probs : :class:`torch.Tensor`
            The probability densities.

            This is a 2D tensor where:

            * The first dimension has a length equal to the number
              of input data points.

            * The second dimension has a length equal to the number
              of components in the Gaussian mixture.

            Each element of the tensor stores a per-data point,
            per-component probability density.
        """

        # Get the per-sample, per-component log-probability.
        y = self._get_log_prob_comp(x = x)

        # Return the probability density.
        return torch.exp(y)


    def log_prob(self,
                 x: torch.Tensor) -> torch.Tensor:
        """Get the log-probability density of a set of samples drawn
        from the Gaussian mixture model.

        Parameters
        ----------
        x : :class:`torch.Tensor`
            The input data points. This is a 2D tensor where:

            * The first dimension has a length equal to the number
              of data points.

            * The second dimension has a length equal to the
              dimensionality of the data points.

        Returns
        -------
        log_prob : :class:`torch.Tensor`
            A 1D tensor storing the log-probability density of each
            input data point to be drawn from the Gaussian mixture
            model.

            The tensor has a size equal to the number of data points
            passed.
        """
        
        return - self.forward(x)


    def sample_new_points(self,
                          n_points: int,
                          n_samples_per_comp: int = 1,
                          sampling_method: str = "mean") -> \
                            torch.Tensor:
        """Draw samples for new data points from each component
        of the Gaussian mixture model.

        Parameters
        ----------
        n_points : :class:`int`
            The number of data points for which samples should be
            drawn.

        n_samples_per_comp : :class:`int`, ``1``
            The number of samples to draw per data point per component
            of the Gaussian mixture.

        sampling_method : :class:`str`, {``"mean"``}, ``"mean"``
            How to draw the samples for the given data points.

            Available options are:

            * ``"mean"`` means taking the mean of each component as
              the value of each ``n_samples_per_comp`` sample taken
              for each data point.

        Returns
        -------
        new_points : :class:`torch.Tensor`
            The samples drawn.

            This is a 2D tensor where:

            * The first dimension has a length equal to
              ``n_points * n_reps_per_mix_comp * n_components``.

            * The second dimension has a length equal to the
              dimensionality of the Gaussian mixture model.
        """

        # Set the supported sampling methods.
        SAMPLING_METHODS = ["mean"]

        # Get the total number of samples to be taken from the Gaussian
        # mixture model.
        n_samples = n_points * n_samples_per_comp

        #-------------------------------------------------------------#

        # If the user selected the option to take the mean of each
        # component as initial representation for each data point.
        if sampling_method == "mean":

            # Disable gradient calculation.
            with torch.no_grad():

                # Get the representations.
                out = \
                    torch.repeat_interleave(\
                        self.means.clone().cpu().detach().unsqueeze(0),
                        n_samples,
                        dim = 0)

        #-------------------------------------------------------------#

        # Otherwise
        else:

            # Raise an error.
            errstr = \
                "Please specify how to correctly initialize new " \
                "representations. The supported methods are: " \
                f"{', '.join(SAMPLING_METHODS)}."
            raise ValueError(errstr)

        #-------------------------------------------------------------#
        
        # Return the representations for the new points as a 2D tensor
        # with:
        #
        # - 1st dimension: the number of data points times number of
        #                  samples drawn per component per data point
        #                  times the number of components in the
        #                  mixture ->
        #                  'n_points' *
        #                  'n_components' *
        #                  'n_samples_per_comp'
        #
        # - 2nd dimension: the dimensionality of the Gaussian mixture
        #                  model ->
        #                  'dim'
        return out.view(n_samples * self.n_components,
                        self.dim)


    def save(self,
             file: str) -> None:
        """Save the legacy GMM parameters to a .pth file.

        Parameters
        ----------
        file : :class:`str`
            The .pth file where to save the model parameters.
        """

        # Save the module state dictionary to the specified file.
        torch.save(self.state_dict(),
                   file)


class GaussianMixtureModelTGMM(tgmm.GaussianMixture):

    """Compatibility wrapper for :class:`tgmm.GaussianMixture`.
    """

    # Set the supported types of covariance matrix.
    COVARIANCE_TYPES = \
        ["full", "spherical", "diag", "tied_full", "tied_spherical",
         "tied_diag"] 

    # Set the supported initializaton methods for the means of the
    # components of the Gaussian mixture model.
    INIT_MEANS_METHODS = \
        ["kmeans", "kpp", "random", "points", "maxdist"]

    # Set the supported initialization methods for the weights of the
    # components of the Gaussian mixture model.
    INIT_WEIGHTS_METHODS = \
        ["uniform", "random", "kmeans"]
    
    # Set the supported initialization methods for the covariance of
    # the components of the Gaussian mixture model.
    INIT_COVARIANCES_METHODS = \
        ["empirical", "eye", "random", "global"]


    ########################### PROPERTIES ############################


    @property
    def dim(self):
        """Alias for TGMM ``n_features``."""

        return self.n_features

    @dim.setter
    def dim(self,
            value):
        """Alias setter for TGMM ``n_features``."""

        self.n_features = value


    #-----------------------------------------------------------------#


    @property
    def means(self):
        """Alias for TGMM ``means_``."""

        return self.means_

    @means.setter
    def means(self,
              value):
        """Alias setter for TGMM ``means_``."""

        self.means_ = value


    #-----------------------------------------------------------------#


    @property
    def weights(self):
        """Alias for TGMM ``weights_``."""

        return self.weights_

    @weights.setter
    def weights(self,
                value):
        """Alias setter for TGMM ``weights_``."""

        self.weights_ = value
    

    ######################## PRIVATE METHODS ##########################


    def _get_log_prob_comp(self,
                           x: torch.Tensor) -> torch.Tensor:
        """Get per-sample, per-component log-joint probabilities.
        
        Parameters
        ----------
        x : :class:`torch.Tensor`
            The input data points. This is a 2D tensor where:

            * The first dimension has a length equal to the number
              of data points.

            * The second dimension has a length equal to the
              dimensionality of the data points.
        
        Returns
        -------
        log_prob_comp : :class:`torch.Tensor`
            The per-sample, per-component log-joint probabilities.

            This is a 2D tensor where:

            * The first dimension has a length equal to the number
              of data points in the input tensor.

            * The second dimension has a length equal to the
              number of components in the Gaussian mixture model.

        """

        # Get the per-sample, per-component responsibilities and the
        # log of the normalizing constant of the log-probabilities.
        resp, log_prob_norm = self._e_step(x.to(self.device))

        # Return the log of the responsibilities plus the log of the
        # normalizing constant of the log-probabilities.
        return torch.log(resp + 1.0e-20) + log_prob_norm.unsqueeze(1)


    ######################## PUBLIC METHODS ###########################


    def get_mixture_probs(self) -> torch.Tensor:
        """Get the mixture probabilities.
        
        Returns
        -------
        mixture_probs : :class:`torch.Tensor`
            The mixture probabilities.
        """

        # If the model has no weights yet
        if self.weights_ is None:

            # Raise an error.
            errstr = "The GMM has no weights yet. Fit the model first."
            raise RuntimeError(errstr)

        # Return the mixture probabilities.
        return self.weights_


    def forward(self,
                x: torch.Tensor) -> torch.Tensor:
        """Return negative log-density.
        
        Parameters
        ----------
        x : :class:`torch.Tensor`
            The input data points. This is a 2D tensor where:

            * The first dimension has a length equal to the number
              of data points.

            * The second dimension has a length equal to the
              dimensionality of the data points.
            
        Returns
        -------
        y : :class:`torch.Tensor`
            A 1D tensor whose size is equal to the number of input
            data points.

            Each element of the tensor is the negative log-probability
            density of a data point.
        """

        # Return the negative log-probability density.
        return -self.score_samples(x)


    def log_prob(self,
                 x: torch.Tensor) -> torch.Tensor:
        """Return the log-density.
        
        Parameters
        ----------
        x : :class:`torch.Tensor`
            The input data points. This is a 2D tensor where:

            * The first dimension has a length equal to the number
              of data points.

            * The second dimension has a length equal to the
              dimensionality of the data points.
        
        Returns
        -------
        log_prob : :class:`torch.Tensor`
            A 1D tensor whose size is equal to the number of input
            data points.
        """

        # Return the log-probability density.
        return self.score_samples(x)


    def sample_probs(self,
                     x: torch.Tensor) -> torch.Tensor:
        """Sample probabilities.
        
        Parameters
        ----------
        x : :class:`torch.Tensor`
            The input data points. This is a 2D tensor where:

            * The first dimension has a length equal to the number
              of data points.

            * The second dimension has a length equal to the
              dimensionality of the data points.
        
        Returns
        -------
        probs : :class:`torch.Tensor`
            The probability densities.

            This is a 2D tensor where:

            * The first dimension has a length equal to the number
              of input data points.

            * The second dimension has a length equal to the
              number of components in the Gaussian mixture.

            Each element of the tensor stores a per-data point,
            per-component probability density.
        """

        # Return the probabilities.
        return self.predict_proba(X = x)


    def sample_new_points(self,
                          n_points: int,
                          n_samples_per_comp: int = 1,
                          sampling_method: str = "mean") -> \
                            torch.Tensor:
        """Draw samples from each component.
        
        Parameters
        ----------
        n_points : :class:`int`
            The number of data points for which samples should be
            drawn.
        
        n_samples_per_comp : :class:`int`, ``1``
            The number of samples to draw per data point per component
            of the Gaussian mixture.
        
        sampling_method : :class:`str`, {``"mean"``}, ``"mean"``
            How to draw the samples for the given data points.
        
            Available options are:

            * ``"mean"`` means taking the mean of each component as
              the value of each ``n_samples_per_comp`` sample taken
              for each data point.
            
        Returns
        -------
        new_points : :class:`torch.Tensor`
            The samples drawn.
        """

        # Set the available sampling methods.
        sampling_methods = ["mean"]

        #-------------------------------------------------------------#

        # Get the total number of samples to be drawn from the
        n_samples = n_points * n_samples_per_comp

        #-------------------------------------------------------------#

        # If the user selected the option to take the mean of each
        # component as initial representation for each data point
        if sampling_method == "mean":

            # Disable gradient calculation.
            with torch.no_grad():

                # Get the representations.
                out = \
                    torch.repeat_interleave(
                        self.means_.clone().cpu().detach(
                            ).unsqueeze(0),
                        n_samples,
                        dim = 0)
        
        # Otherwise
        else:

            # Raise an error.
            errstr = \
                "Please specify how to correctly initialize new " \
                "representations. The supported methods are: " \
                f"{', '.join(sampling_methods)}."
            raise ValueError(errstr)

        #-------------------------------------------------------------#

        # Return the representations for the new points as a 2D tensor.
        return out.view(n_samples * self.n_components,
                        self.n_features)


    def to(self,
           *args,
           **kwargs):
        """Move the model parameters and internal tensors to the target
        device.
        """

        # Call the parent to() method.
        super().to(*args, **kwargs)

        # Set the target device to None.
        device = None

        # If there are arguments
        if args:

            # If the first argument is a torch.device or a string
            if isinstance(args[0], (torch.device, str)):
                device = args[0]

            # If the first argument is a torch.Tensor
            elif isinstance(args[0], torch.Tensor):
                device = args[0].device

        # If a device was specified in the keyword arguments
        if "device" in kwargs:

            # Get the device.
            device = kwargs["device"]

        # If a device was specified
        if device is not None:

            # Update the device property.
            self.device = torch.device(device)

            # Move all GMM-specific tensors to the target device.
            for attr in ["means_", "weights_", "covariances_",
                         "precisions_cholesky_",
                         "initial_means_",
                         "initial_weights_",
                         "initial_covariances_"]:

                # Get the attribute value.  
                val = getattr(self, attr, None)

                # If the attribute is a tensor, move it to the target
                # device.
                if isinstance(val, torch.Tensor):

                    # Set the attribute.
                    setattr(self, attr, val.to(self.device))

        # Return the model.
        return self


    def _apply(self, fn):
        # Call the parent _apply.
        super()._apply(fn)

        # Detect the target device.
        dummy = torch.empty(0)
        moved_dummy = fn(dummy)
        device = moved_dummy.device

        # Update the device property.
        self.device = device

        # Move all GMM-specific tensors.
        for attr in ["means_", "weights_", "covariances_", "precisions_cholesky_",
                     "initial_means_", "initial_weights_", "initial_covariances_"]:
            val = getattr(self, attr, None)
            if isinstance(val, torch.Tensor):
                setattr(self, attr, fn(val))

        return self


    def state_dict(self, *args, **kwargs):
        """Get the state dictionary containing the parameters of the model.
        """

        return self.save_state_dict()


    def load_state_dict(self, state_dict, strict=False, *args, **kwargs):
        """Load the state dictionary.
        """

        # Call the tgmm load_state_dict.
        super().load_state_dict(state_dict)

        # Return a dummy NamedTuple to satisfy PyTorch's API.
        from collections import namedtuple
        _IncompatibleKeys = namedtuple('_IncompatibleKeys', ['missing_keys', 'unexpected_keys'])
        return _IncompatibleKeys(missing_keys=[], unexpected_keys=[])


#------------------------ Representation layer -----------------------#


class RepresentationLayer(nn.Module):
    
    """
    Class implementing a representation layer accumulating gradients.

    This layer stores learned embeddings for data samples that can be
    optimized during training. It supports various initialization
    methods and utility functions for representation manipulation and
    analysis.
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
            ``"laplace"``, ``"student_t"``,  ``"cauchy"``, \
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

        # Set the device before sampling, as the sampling helpers use
        # 'self.device' to place the created tensors.
        self._device = device

        #-------------------------------------------------------------#
        
        # Initialize the gradients with respect to the representations
        # None.
        self.dz = None

        #-------------------------------------------------------------#

        # If a tensor of values was passed
        if values is not None:

            # Get the number of representations, the
            # dimensionality of the representations, and the values
            # of the representations from the tensor.
            self._n_rep, self._dim, self._z, self._options = \
                self._get_rep_from_values(values = values)      
        
        # Otherwise
        else:

            # If the representations are to be initialized to zero
            # (i.e., no sampling is performed)
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
            # distribution on the unit ball
            elif dist == "uniform_ball":
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
                self._n_rep, self._dim, self._z, self._options = \
                    self._get_rep_from_cauchy(options = dist_options)

            # Otherwise
            else:

                # Raise an error.
                available_dists_str = \
                    ", ".join(f'{d}' for d in self.AVAILABLE_DISTS)
                errstr = \
                    f"Unsupported distribution '{dist}'. The only " \
                    "distributions from which it is possible to " \
                    "sample the representations are: " \
                    f"{available_dists_str}."
                raise ValueError(errstr)
        
        #-------------------------------------------------------------#

        # Move to the specified device and ensure it remains a Parameter
        self._z = nn.Parameter(self._z.to(device),
                               requires_grad = True)

        #-------------------------------------------------------------#

        # Set the device on which the representations are initialized.
        self._device = device


    def _get_rep_from_values(self,
                             values: torch.Tensor) -> \
                                tuple[int, int, torch.nn.Parameter]:
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

        # Fill the tensor with the given values.
        with torch.no_grad():
            z.copy_(values)

        #-------------------------------------------------------------#

        # Return the number of representations, the dimensionality of
        # the representations, and the values of the representations.
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

            The dictionary must contains the following keys,
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
    
        # Return the number of representations, the dimensionality of
        # the representations, the values of the representations,
        # and the options used to generate them.
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

            The dictionary must contains the following keys,
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

        options : :class:`dict`
            A dictionary containing the options used to initialize
            the representations.
        """

        # Get the desired number of representations to be drawn.
        n_rep = options["n_samples"]

        # Get the dimensionality of the desired representations.
        dim = options["dim"]

        # Get the mean of the normal distribution from which the
        # representations should be samples.
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
        
        # Return the number of representations, the dimensionality of
        # the representations, the values of the representations,
        # and the options used to generate them.
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

            The dictionary must contains the following keys,
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
        
        # Return the number of representations, the dimensionality of
        # the representations, the values of the representations,
        # and the options used to generate them.
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

            The dictionary must contains the following keys,
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

        # Generate random directions by sampling from normal
        # distribution and normalizing to unit vectors.
        normal_samples = torch.randn(n_rep,
                                     dim,
                                     device = self.device)
        normal_samples_norm = torch.norm(normal_samples,
                                         dim = 1,
                                         keepdim = True)
        unit_directions = normal_samples / normal_samples_norm
        
        # Generate random radii with a proper distribution for uniform
        # sampling within a ball. For a uniform distribution in a ball,
        # we need r^(dim-1) distributions of distances
        # from the center, which is achieved by taking u^(1/dim)
        # where u is uniform(0,1).
        u = torch.rand(n_rep,
                       1,
                       device = self.device)
        random_radii = radius * u.pow(1.0 / dim)
        
        # Generate the values of the representations by scaling the
        # unit directions by the random radii.
        samples = unit_directions * random_radii

        #-------------------------------------------------------------#
        
        # Create atensor with the values of the representations.
        z = nn.Parameter(samples,
                         requires_grad = True)

        #-------------------------------------------------------------#

        # Return the number of representations, the dimensionality of
        # the representations, the values of the representations,
        # and the options used to generate them.
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
            representations uniformly from a ball.

            The dictionary must contains the following keys,
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
        
        # Convert uniform to Laplace using the inverse CDF.
        sign = torch.sign(uniform - 0.5)
        
        # Get the values for the representations by sampling the
        # Laplace distribution.
        samples = \
            loc - scale * sign * \
                torch.log(1 - 2 * torch.abs(uniform - 0.5))

        #-------------------------------------------------------------#
        
        # Create a tensor with the values of the representations.
        z = nn.Parameter(samples,
                         requires_grad = True)

        #-------------------------------------------------------------#
    
        # Return the number of representations, the dimensionality of
        # the representations, the values of the representations,
        # and the options used to generate them.
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

            The dictionary must contains the following keys,
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
    
        # Return the number of representations, the dimensionality of
        # the representations, the values of the representations,
        # and the options used to generate them.
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
        options : ``dict``
            A dictionary containing the parameters to sample the
            representations from a Cauchy distribution.

            The dictionary must contains the following keys,
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
    
        # Return the number of representations, the dimensionality of
        # the representations, the values of the representations,
        # and the options used to generate them.
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
        """
        
        errstr = \
            "The value of 'n_samples' is set at initialization and " \
            "cannot be changed. If you want to change the number " \
            "of representations in the layer, initialize a new " \
            f"instance of '{self.__class__.__name__}'."
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
        """
        
        errstr = \
            "The value of 'dim' is set at initialization and cannot " \
            "be changed. If you want to change the dimensionality " \
            "of the representations stored in the layer, initialize " \
            f"a new instance of '{self.__class__.__name__}'."
        raise ValueError(errstr)


    @property
    def options(self):
        """The dictionary ot options used to generate the
        representations, if no values were passed when initializing
        the layer.
        """

        return self._options


    @options.setter
    def options(self,
                value):
        """Raise an exception if the user tries to modify the value of
        ``options`` after initialization.
        """
        
        errstr = \
            "The value of 'options' is set at initialization and " \
            "cannot be changed. If you want to change the options " \
            "used to generate the representations, initialize a " \
            f"new instance of '{self.__class__.__name__}'."
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
        """
        
        errstr = \
            "The value of 'z' is set at initialization and cannot " \
            "be changed. If you want to change the values of the " \
            "representations stored in the layer, initialize a new " \
            f"instance of '{self.__class__.__name__}'."
        raise ValueError(errstr)


    @property
    def device(self):
        """The device where the model is.
        """

        return self._device

    @device.setter
    def device(self,
               value):
        """Move the representation layer to the selected device.
        """
        
        # Move the representation layer to the specified device.
        self.to(device = torch.device(value))

        # Update the device the representation layer is on.
        self._device = value


    ######################### PUBLIC METHODS ##########################


    @classmethod
    def load(cls,
             file : str,
             device: Optional[str | torch.device] = None) \
                -> 'RepresentationLayer':
        """Load representations from a file.
        
        Parameters
        ----------
        path : :class:`str`
            The file from which to load the representations.
            
        device : :class:`str` or :class:`torch.device`, optional
            The device where to load the representations.
            
        Returns
        -------
        rep_layer : :class:`core.latent.RepresentationLayer`
            The representation layer.
        """

        # Load the state dictionary from the specified file.
        state_dict = torch.load(file,
                                map_location = "cpu")
        
        # Move the representation layer to the specified device.
        rep_layer = cls(values = state_dict["z"],
                        device = device)

        # Return the representation layer.
        return rep_layer


    def forward(self,
                ixs: Optional[list[int] | torch.Tensor] = None, 
                index_map: Optional[dict[int, int]] = None,
                batch_size: Optional[int] = None) -> torch.Tensor:
        """Forward pass that returns the values of the representations.

        Parameters
        ----------
        ixs : :class:`int` or :class:`torch.Tensor`, optional
            The indexes of the samples whose representations should
            be returned. If not passed, all representations will be
            returned.

        index_map : :class:`dict`, optional
            A mapping from dataset indices to representation indices.
            
        batch_size : :class:`int`, optional
            Process representations in batches of size ``batch_size``
            for memory efficiency.

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
        
        # If we have an index mapping
        if index_map is not None:

            # If the indexes are provided as a tensor
            if isinstance(ixs, torch.Tensor):

                # Map the indices using the provided map.
                mapped_ixs = \
                    torch.tensor(\
                        [index_map.get(idx.item(), 0) for idx in ixs], 
                        device = ixs.device)
            
            # If the indexes are provided as a list
            else:

                # Map the indices using the provided map.
                mapped_ixs = [index_map.get(idx, 0) for idx in ixs]
        
        # Otherwise
        else:
            
            # If there's no mapping and the indices are provided
            if isinstance(ixs, torch.Tensor):
                
                # Clamp the indices to valid range to prevent
                # out-of-bounds errors.
                mapped_ixs = torch.clamp(ixs, 0, len(self.z)-1)
            
            # Otherwise
            else:

                # Ensure the indices are within the valid range.
                mapped_ixs = \
                    [min(max(0, idx), len(self.z)-1) for idx in ixs]

        #-------------------------------------------------------------#

        # If there is a batch size specified and the number of
        # representations to return is greater than the batch size,
        if batch_size is not None and len(mapped_ixs) > batch_size:

            # Initialize a list to hold the result chunks.
            result_chunks = []

            # For each batch of indices
            for i in range(0, len(mapped_ixs), batch_size):

                # Get the current batch of indices.
                batch_ixs = mapped_ixs[i:i+batch_size]

                # Add the representations for the current batch to the
                # result list.
                result_chunks.append(self.z[batch_ixs])

            # Concatenate the result chunks along the first dimension
            # to return a single tensor.
            return torch.cat(result_chunks,
                             dim = 0)

        #-------------------------------------------------------------#

        # Return representations for the specified indices
        return self.z[mapped_ixs]


    def rescale(self) -> None:
        """Rescale the representations by subtracting the mean of
        the representations' values from each of them and dividing
        each of them by the standard deviation of all representations.

        Given :math:`N` samples, we can indicate with :math:`z^{n}`
        the value of the representation of sample :math:`x^{n}`.

        Therefore, the rescaled value of the representation
        :math:`z^{n}_{rescaled}` will be:
        
        .. math::

           z^{n}_{rescaled} = \\frac{z^{n} - \\bar{z}}{s}

        Where :math:`\\bar{z}` is the mean of the representations'
        values and :math:`s` is the standard deviation.
        """
        
        # Flatten the tensor containing the representations' values.
        z_flat = torch.flatten(self.z.cpu().detach())
        
        # Get the mean and the standard deviation of the
        # representations.
        sd, m = torch.std_mean(z_flat)
        
        # Disable the calculation of the gradients.
        with torch.no_grad():

            # Subtract the mean value of all representations' values
            # from each of the representation's value.
            self.z -= m

            # Divide each representation's value by the standard
            # deviation of all representations' values.
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


#######################################################################


# Set the available latent spaces.
LATENT_SPACES = {
    
    # Legacy Gaussian mixture model.
    "lgmm" : GaussianMixtureModelLegacy,

    # Gaussian mixture model implemented using the 'tgmm' package.
    "gmm" : GaussianMixtureModelTGMM,

    }
