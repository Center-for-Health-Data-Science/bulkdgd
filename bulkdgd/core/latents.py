#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-

#    latents.py
#
#    This module contains the classes implementing the latent space
#    of the :class:`core.models.BulkDGD`, namely the Gaussian mixture
#    models (:class:`core.latents.GaussianMixtureModelLegacy` and
#    :class:`core.latents.GaussianMixtureModelTGMM`).
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
    "The latent space of :class:`core.models.BulkDGD`: the Gaussian " \
    "mixture models " \
    "(:class:`core.latents.GaussianMixtureModelLegacy` and " \
    ":class:`core.latents.GaussianMixtureModelTGMM`) and the refit " \
    "of the final mixture (:func:`core.latents.fit_final_gmm`)."


#######################################################################


# Import from the standard library.
import copy
import logging as log
import math
from typing import Optional

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
    A class implementing a mixture of multivariate Gaussian
    distributions (Gaussian mixture model or GMM) with priors over
    its parameters.
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
            The options to set up the prior over the means of the
            components of the Gaussian mixture model.

            If ``means_prior_name`` is ``"softball"``, the
            options that must be provided are:

            * ``"radius"``: the radius of the soft ball.

            * ``"sharpness"``: the sharpness of the soft boundary of
              the ball.

        weights_prior_options : :class:`dict`
            The options to set up the prior over the weights of the
            components of the Gaussian mixture model.

            If ``weights_prior_name`` is ``"dirichlet"``, the options
            that must be provided are:

            * ``"alpha"``: the alpha of the Dirichlet distribution.

        log_var_prior_options : :class:`dict`
            The options to set up the prior over the negative
            log-variance of the components of the Gaussian mixture
            model.

            If ``log_var_prior_name`` is ``"gaussian"``, the
            options that must be provided are:

            * ``"mean"``: the components' standard deviation at the
              prior's mean, which is ``-2 * log(mean)``.

            * ``"stddev"``: the standard deviation of the Gaussian
              distribution.

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

        # Set the log-variance.
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
        """Get the means of the components of the Gaussian mixture
        model, sampled from the prior.

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

        # Get the means of the mixture, with shape
        # ('n_components', 'dim').
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

            This is a 1D tensor having a length equal to the number
            of components in the Gaussian mixture model.
        """

        # Return the weights of the components.
        return nn.Parameter(torch.ones(self.n_components),
                            requires_grad = True)


    def _get_log_var_prior(
            self,
            log_var_prior_name: str,
            log_var_prior_options: dict[str, object]) -> \
                dict[str, object]:
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

            # Get the 'mean' (the components' standard deviation).
            dist_mean = log_var_prior_options.get("mean")

            # Get the standard deviation of the Gaussian distribution.
            dist_stddev = log_var_prior_options.get("stddev")

            # If the 'mean' was not provided
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

                # Set the log-variance factor to 'dim' / 2.
                dist_factor = self.dim * 0.5

                # Use one log-variance per component.
                dist_dim = 1

                # Do not require gradients.
                requires_grad = False

            #---------------------------------------------------------#

            # If the covariance matrix is isotropic
            elif self.covariance_type == "isotropic":

                # Set the log-variance factor to 'dim' / 2.
                dist_factor = self.dim * 0.5

                # Use one log-variance per component.
                dist_dim = 1

                # Require gradients.
                requires_grad = True

            #---------------------------------------------------------#

            # If the covariance matrix is diagonal
            elif self.covariance_type == "diagonal":

                # Set the log-variance factor to 1/2.
                dist_factor = 0.5

                # Use one log-variance per component per dimension.
                dist_dim = self.dim

                # Require gradients.
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

            # Raise an error.
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
        log_var : :class:`torch.nn.Parameter`
            The log-variance of the components.

            It is a 2D tensor where:

            * The first dimension has a length equal to the number
              of components in the Gaussian mixture.

            * The second dimension has a length equal to 1 for a
              fixed or isotropic covariance, and to the
              dimensionality of the Gaussian mixture model for a
              diagonal one.
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

            # Get the 'mean' (the components' standard deviation).
            dist_mean = self.log_var_prior["options"]["mean"]

            # Disable gradient calculation.
            with torch.no_grad():

                # Set each variance to the square of 'mean'.
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

        Parameters
        ----------
        value : :class:`int`
            The new value.
        """

        # Raise an error.
        errstr = \
            "The value of 'n_components' is set at initialization " \
            "and cannot be changed. Initialize a new instance of " \
            f"'{self.__class__.__name__}' instead."
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

        Parameters
        ----------
        value : :class:`str`
            The new value.
        """

        # Raise an error.
        errstr = \
            "The value of 'covariance_type' is set at initialization " \
            "and cannot be changed. Initialize a new instance of " \
            f"'{self.__class__.__name__}' instead."
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

        Parameters
        ----------
        value : :class:`dict`
            The new value.
        """

        # Raise an error.
        errstr = \
            "The value of 'means_prior' is set at initialization " \
            "and cannot be changed. Initialize a new instance of " \
            f"'{self.__class__.__name__}' with a new " \
            "'means_prior_name' and/or 'means_prior_options'."
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

        Parameters
        ----------
        value : :class:`torch.Tensor`
            The new value.
        """

        # Raise an error.
        errstr = \
            "The value of 'means' is set at initialization from the " \
            "prior ('means_prior_name' and 'means_prior_options') " \
            "and cannot be changed."
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

        Parameters
        ----------
        value : :class:`dict`
            The new value.
        """

        # Raise an error.
        errstr = \
            "The value of 'weights_prior' is set at initialization " \
            "and cannot be changed. Initialize a new instance of " \
            f"'{self.__class__.__name__}' with a new " \
            "'weights_prior_name' and/or 'weights_prior_options'."
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

        Parameters
        ----------
        value : :class:`torch.Tensor`
            The new value.
        """

        # Raise an error.
        errstr = \
            "The value of 'weights' is set at initialization from " \
            "the prior ('weights_prior_name' and " \
            "'weights_prior_options') and cannot be changed."
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

        Parameters
        ----------
        value : :class:`dict`
            The new value.
        """

        # Raise an error.
        errstr = \
            "The value of 'log_var_prior' is set at initialization " \
            "and cannot be changed. Initialize a new instance of " \
            f"'{self.__class__.__name__}' with a new " \
            "'log_var_prior_name' and/or 'log_var_prior_options'."
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

        Parameters
        ----------
        value : :class:`torch.Tensor`
            The new value.
        """

        # Raise an error.
        errstr = \
            "The value of 'log_var' is set at initialization from " \
            "the prior ('log_var_prior_name' and " \
            "'log_var_prior_options') and cannot be changed."
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
            The per-sample, per-component log-probability.

            This is a 2D tensor where:

            * The first dimension has a length equal to the number
              of data points in the input tensor.

            * The second dimension has a length equal to the number
              of components in the Gaussian mixture.
        """

        # Get the log-variance factor.
        log_var_factor = self.log_var_prior["options"]["factor"]

        #-------------------------------------------------------------#

        # Get the variance of the components from the log-variance.
        covariance = torch.exp(self.log_var)

        #-------------------------------------------------------------#

        # Get the 'pi' term.
        pi_term = - 0.5 * self.dim * math.log(2 * math.pi)

        #-------------------------------------------------------------#

        # Get the squared-distance term.
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

        # Return the tensor.
        return y


    ######################### PUBLIC METHODS ##########################


    def set_means(self,
                  means: torch.Tensor) -> None:
        """Set the means of the components of the Gaussian mixture
        model.

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

            * The second dimension has a length equal to 1 for a
              fixed or isotropic covariance, and to the
              dimensionality of the Gaussian mixture model for a
              diagonal one.
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


    def get_prior_log_prob(self) -> torch.Tensor:
        """Calculate the log-probability of the prior over the means,
        log-variance, and mixture coefficients.

        Returns
        -------
        p : :class:`torch.Tensor`
            The log-probability of the prior.
        """

        # Initialize the log-probability to 0.0.
        p = 0.0

        #-------------------------------------------------------------#

        # Get the name of the prior over the weights of the
        # components.
        weights_prior_name = self.weights_prior["name"]

        # If the prior over the weights is the Dirichlet prior
        if weights_prior_name == "dirichlet":

            # Get the alpha.
            alpha = self.weights_prior["options"]["alpha"]

            # Start from the Dirichlet constant.
            p = self.weights_prior["options"]["constant"]

            # If the alpha is different from 1
            if alpha != 1:

                # Add the log-probability of the mixture coefficients.
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

        # Return the log-probability.
        return p


    def forward(self,
                x: torch.Tensor) -> torch.Tensor:
        """Compute the negative log-probability density of a set of
        data points (forward pass).

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

            Each element of the tensor is the negative log-probability
            density of a data point, minus the log-probability of the
            priors divided by the number of data points.
        """

        # Get the per-sample, per-component log-probability.
        y = self._get_log_prob_comp(x = x)

        # Sum the probabilities over the components, in log space.
        y = torch.logsumexp(y,
                            dim = -1)

        # Add the log-probability of the priors, divided by the batch
        # size.
        y = y + self.get_prior_log_prob() / x.shape[0]

        # Return the negative log-probability density.
        return -y


    def sample_probs(self,
                     x: torch.Tensor) -> torch.Tensor:
        """Get the joint probability density of each data point and
        each component (the component's weight times its density).

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
            The joint probability densities.

            This is a 2D tensor where:

            * The first dimension has a length equal to the number
              of input data points.

            * The second dimension has a length equal to the number
              of components in the Gaussian mixture.

            The rows are not normalized: each sums to the mixture's
            probability density at the data point, without the priors.
        """

        # Get the per-sample, per-component log-probability.
        y = self._get_log_prob_comp(x = x)

        # Return the probability density.
        return torch.exp(y)


    def log_prob(self,
                 x: torch.Tensor) -> torch.Tensor:
        """Get the log-probability density of a set of data points.

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
            input data point, plus the log-probability of the priors
            divided by the number of data points.

            The tensor has a size equal to the number of data points
            passed.
        """

        # Return the log-probability density.
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

            * ``"mean"``: each sample is the mean of its component.

        Returns
        -------
        new_points : :class:`torch.Tensor`
            The samples drawn.

            This is a 2D tensor where:

            * The first dimension has a length equal to
              ``n_points * n_samples_per_comp * n_components``.

            * The second dimension has a length equal to the
              dimensionality of the Gaussian mixture model.
        """

        # Set the supported sampling methods.
        SAMPLING_METHODS = ["mean"]

        # Get the number of samples to draw from each component.
        n_samples = n_points * n_samples_per_comp

        #-------------------------------------------------------------#

        # If the samples are the means of the components
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
                f"Unsupported sampling method '{sampling_method}'. " \
                "The supported methods are: " \
                f"{', '.join(SAMPLING_METHODS)}."
            raise ValueError(errstr)

        #-------------------------------------------------------------#

        # Return the representations as a 2D tensor of shape
        # ('n_points' * 'n_samples_per_comp' * 'n_components', 'dim').
        return out.view(n_samples * self.n_components,
                        self.dim)


    def save(self,
             file: str) -> None:
        """Save the GMM parameters to a .pth file.

        Parameters
        ----------
        file : :class:`str`
            The .pth file where to save the model parameters.
        """

        # Save the module state dictionary to the specified file.
        torch.save(self.state_dict(),
                   file)


class GaussianMixtureModelTGMM(tgmm.GaussianMixture):

    """
    A class implementing a Gaussian mixture model through
    :class:`tgmm.GaussianMixture`.
    """

    # Set the supported types of covariance matrix.
    COVARIANCE_TYPES = \
        ["full", "spherical", "diag", "tied_full", "tied_spherical",
         "tied_diag"]

    # Set the supported initialization methods for the means of the
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
        """The dimensionality of the Gaussian mixture model (alias for
        ``n_features``).
        """

        return self.n_features


    @dim.setter
    def dim(self,
            value):
        """Set the dimensionality of the Gaussian mixture model.

        Parameters
        ----------
        value : :class:`int`
            The dimensionality.
        """

        # Set the dimensionality.
        self.n_features = value


    #-----------------------------------------------------------------#


    @property
    def means(self):
        """The means of the components of the Gaussian mixture model
        (alias for ``means_``).
        """

        return self.means_


    @means.setter
    def means(self,
              value):
        """Set the means of the components of the Gaussian mixture
        model.

        Parameters
        ----------
        value : :class:`torch.Tensor`
            The means.
        """

        # Set the means.
        self.means_ = value


    #-----------------------------------------------------------------#


    @property
    def weights(self):
        """The weights of the components of the Gaussian mixture model
        (alias for ``weights_``).
        """

        return self.weights_


    @weights.setter
    def weights(self,
                value):
        """Set the weights of the components of the Gaussian mixture
        model.

        Parameters
        ----------
        value : :class:`torch.Tensor`
            The weights.
        """

        # Set the weights.
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
        # log-probability density of each sample.
        resp, log_prob_norm = self._e_step(x.to(self.device))

        # Return the log of the responsibilities plus the
        # log-probability density, avoiding the log of zero.
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
        """Compute the negative log-probability density of a set of
        data points (forward pass).

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
        """Get the log-probability density of a set of data points.

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

            Each element of the tensor is the log-probability density
            of a data point.
        """

        # Return the log-probability density.
        return self.score_samples(x)


    def sample_probs(self,
                     x: torch.Tensor) -> torch.Tensor:
        """Get the posterior probability of each component for each
        data point.

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
            The posterior probabilities.

            This is a 2D tensor where:

            * The first dimension has a length equal to the number
              of input data points.

            * The second dimension has a length equal to the
              number of components in the Gaussian mixture.

            Each row sums to 1.
        """

        # Return the probabilities.
        return self.predict_proba(X = x)


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

            * ``"mean"``: each sample is the mean of its component.

        Returns
        -------
        new_points : :class:`torch.Tensor`
            The samples drawn.

            This is a 2D tensor where:

            * The first dimension has a length equal to
              ``n_points * n_samples_per_comp * n_components``.

            * The second dimension has a length equal to the
              dimensionality of the Gaussian mixture model.
        """

        # Set the available sampling methods.
        sampling_methods = ["mean"]

        #-------------------------------------------------------------#

        # Get the number of samples to draw from each component.
        n_samples = n_points * n_samples_per_comp

        #-------------------------------------------------------------#

        # If the samples are the means of the components
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
                f"Unsupported sampling method '{sampling_method}'. " \
                "The supported methods are: " \
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

        Parameters
        ----------
        *args
            The positional arguments of :meth:`torch.nn.Module.to`.

        **kwargs
            The keyword arguments of :meth:`torch.nn.Module.to`.

        Returns
        -------
        self : :class:`bulkdgd.core.latents.GaussianMixtureModelTGMM`
            The model.
        """

        # Call the parent to() method.
        super().to(*args, **kwargs)

        # Set the target device to None.
        device = None

        # If there are arguments
        if args:

            # If the first argument is a torch.device or a string
            if isinstance(args[0], (torch.device, str)):

                # Get the device.
                device = args[0]

            # If the first argument is a torch.Tensor
            elif isinstance(args[0], torch.Tensor):

                # Get the tensor's device.
                device = args[0].device

        # If a device was specified in the keyword arguments
        if "device" in kwargs:

            # Get the device.
            device = kwargs["device"]

        # If a device was specified
        if device is not None:

            # Update the device property.
            self.device = torch.device(device)

            # For each GMM-specific tensor attribute
            for attr in ["means_", "weights_", "covariances_",
                         "precisions_cholesky_",
                         "initial_means_",
                         "initial_weights_",
                         "initial_covariances_"]:

                # Get the attribute value.
                val = getattr(self,
                              attr,
                              None)

                # If the attribute is a tensor
                if isinstance(val, torch.Tensor):

                    # Move it to the target device.
                    setattr(self,
                            attr,
                            val.to(self.device))

        # Return the model.
        return self


    def _apply(self,
               fn):
        """Apply a function to the model's tensors, including the
        GMM-specific ones.

        Parameters
        ----------
        fn : callable
            The function to apply to each tensor.

        Returns
        -------
        self : :class:`bulkdgd.core.latents.GaussianMixtureModelTGMM`
            The model.
        """

        # Call the parent _apply.
        super()._apply(fn)

        # Get the target device by applying the function to an
        # empty tensor.
        dummy = torch.empty(0)
        moved_dummy = fn(dummy)
        device = moved_dummy.device

        # Update the device property.
        self.device = device

        # For each tensor attribute
        for attr in ["means_",
                     "weights_",
                     "covariances_",
                     "precisions_cholesky_",
                     "initial_means_",
                     "initial_weights_",
                     "initial_covariances_"]:

            # Get the value of the attribute.
            val = getattr(self,
                          attr,
                          None)

            # If it is a tensor
            if isinstance(val, torch.Tensor):

                # Apply the function to it.
                setattr(self,
                        attr,
                        fn(val))

        # Return the model.
        return self


    def state_dict(self):
        """Get the state dictionary containing the parameters of the
        model.

        Returns
        -------
        state_dict : :class:`dict`
            The state dictionary.
        """

        # Get and return the state dictionary.
        return self.save_state_dict()


    def load_state_dict(self,
                        state_dict):
        """Load the state dictionary.

        Parameters
        ----------
        state_dict : :class:`dict`
            The state dictionary.

        Returns
        -------
        incompatible_keys : :class:`collections.namedtuple`
            An empty record of missing and unexpected keys, as
            :meth:`torch.nn.Module.load_state_dict` returns.
        """

        # Call the tgmm load_state_dict.
        super().load_state_dict(state_dict)

        # Import the named tuple factory.
        from collections import namedtuple

        # Create the named tuple type PyTorch's method returns.
        _IncompatibleKeys = \
            namedtuple('_IncompatibleKeys',
                       ['missing_keys', 'unexpected_keys'])

        # Return it, with no missing or unexpected keys.
        return _IncompatibleKeys(missing_keys = [],
                                 unexpected_keys = [])


    #-----------------------------------------------------------------#


    def save(self,
             file):
        """Save the model's parameters to a .pth file.

        Parameters
        ----------
        file : :class:`str`
            The .pth file where to save the model parameters.
        """

        # Save the state dictionary to the specified file.
        torch.save(self.state_dict(),
                   file)


#######################################################################


#--------------------- The final Gaussian mixture --------------------#


def fit_final_gmm(gmm,
                  reps: torch.Tensor,
                  covariance_type: str,
                  shrinkage: float = 0.0,
                  reg_covar: Optional[float] = None,
                  chunk_size: int = 2048):
    """Refit a trained GMM's covariance on the final representations,
    with the means and the weights frozen.

    Parameters
    ----------
    gmm : :class:`bulkdgd.core.latents.GaussianMixtureModelTGMM`
        The trained Gaussian mixture model.

    reps : :class:`torch.Tensor`
        The final representations, as a 2D tensor whose first
        dimension is the number of samples and whose second dimension
        is the dimensionality of the latent space.

    covariance_type : :class:`str`
        The type of covariance matrix the final mixture should have.

    shrinkage : :class:`float`, ``0.0``
        How far (0.0-1.0) each per-component covariance is pulled
        towards the one shared across components. It is ignored by
        the tied covariance types.

    reg_covar : :class:`float`, optional
        The value added to the diagonal of the covariance. If not
        passed, the trained mixture's own value (or 1e-6) is used.

    chunk_size : :class:`int`, ``2048``
        The number of representations processed at a time.

    Returns
    -------
    gmm_final : :class:`bulkdgd.core.latents.GaussianMixtureModelTGMM`
        A copy of the input mixture, with the same means and
        weights and the refitted covariance.
    """

    # If the covariance type is not one the mixture supports
    if covariance_type not in gmm.COVARIANCE_TYPES:

        # Raise an error.
        errstr = \
            f"Unsupported covariance type '{covariance_type}'. The " \
            "supported covariance types are: " \
            f"{', '.join(gmm.COVARIANCE_TYPES)}."
        raise ValueError(errstr)

    #-----------------------------------------------------------------#

    # If the shrinkage is not a proportion
    if not 0.0 <= shrinkage <= 1.0:

        # Raise an error.
        errstr = \
            f"'shrinkage' must be between 0.0 and 1.0, not " \
            f"{shrinkage}."
        raise ValueError(errstr)

    #-----------------------------------------------------------------#

    # Get the means of the components.
    means = gmm.means_

    # Move the representations to the mixture's device and dtype.
    reps = reps.to(device = means.device, dtype = means.dtype)

    # Get the number of components and the dimensionality of the
    # latent space.
    n_components, n_features = means.shape

    # Initialize the responsibility-weighted scatter of each
    # component.
    scatter = torch.zeros(n_components,
                          n_features,
                          n_features,
                          device = means.device,
                          dtype = means.dtype)

    # Initialize the effective number of samples of each component.
    n_k = torch.zeros(n_components,
                      device = means.device,
                      dtype = means.dtype)

    # For each chunk of representations
    for i in range(0,
                   reps.shape[0],
                   chunk_size):

        # Get the chunk.
        chunk = reps[i:i+chunk_size]

        # Get the responsibilities of the trained mixture.
        resp, _ = gmm._e_step(chunk)

        # Cast the responsibilities to the mixture's dtype.
        resp = resp.to(dtype = means.dtype)

        # Get the displacement of every representation from every
        # component's mean (n_samples, n_components, n_features).
        delta = chunk.unsqueeze(1) - means.unsqueeze(0)

        # Add the chunk's responsibility-weighted scatter.
        scatter += torch.einsum("nk,nki,nkj->kij",
                                resp,
                                delta,
                                delta)

        # Add the chunk's responsibilities to the effective numbers
        # of samples.
        n_k += resp.sum(dim = 0)

    #-----------------------------------------------------------------#

    # Get the covariance of each component, clamping to avoid a
    # division by zero for an empty component.
    per_component = scatter / n_k.clamp(min = 1.0e-10)[:, None, None]

    # Get the covariance all the components would share.
    tied = scatter.sum(dim = 0) / n_k.sum()

    # Pull each component's covariance towards the shared one.
    shrunk = (1.0 - shrinkage) * per_component + shrinkage * tied

    #-----------------------------------------------------------------#

    # If no regularization of the covariance was passed
    if reg_covar is None:

        # Use the trained mixture's own, or 1e-6 if it has none.
        reg_covar = getattr(gmm,
                            "reg_covar",
                            1.0e-6) or 1.0e-6

    # Get an identity matrix to regularize the full covariances.
    eye = torch.eye(n_features,
                    device = means.device,
                    dtype = means.dtype)

    # If each component gets its own full covariance matrix
    if covariance_type == "full":

        # Take the shrunk per-component covariance.
        covariances = shrunk + reg_covar * eye

    # If all the components share one full covariance matrix
    elif covariance_type == "tied_full":

        # Take the shared covariance.
        covariances = tied + reg_covar * eye

    # If each component gets its own axis-aligned covariance
    elif covariance_type == "diag":

        # Keep only the diagonal of the shrunk covariance.
        covariances = \
            torch.diagonal(shrunk,
                           dim1 = -2,
                           dim2 = -1) + reg_covar

    # If all the components share one axis-aligned covariance
    elif covariance_type == "tied_diag":

        # Keep only the diagonal of the shared covariance.
        covariances = torch.diagonal(tied) + reg_covar

    # If each component gets its own single variance
    elif covariance_type == "spherical":

        # Average the variances along the axes into one number per
        # component.
        covariances = \
            torch.diagonal(shrunk,
                           dim1 = -2,
                           dim2 = -1).mean(dim = -1) \
            + reg_covar

    # If all the components share one single variance
    elif covariance_type == "tied_spherical":

        # Average the variances along the axes of the shared
        # covariance into one number.
        covariances = torch.diagonal(tied).mean() + reg_covar

    #-----------------------------------------------------------------#

    # Copy the mixture, leaving the trained one untouched.
    gmm_final = copy.deepcopy(gmm)

    # Set the refitted covariance on the copy.
    gmm_final.covariances_ = covariances

    # Set the type of the refitted covariance.
    gmm_final.covariance_type = covariance_type

    # Return the final mixture.
    return gmm_final


#######################################################################


# Set the available latent spaces.
LATENT_SPACES = {

    # Gaussian mixture model with priors over its parameters.
    "lgmm" : GaussianMixtureModelLegacy,

    # Gaussian mixture model implemented using the 'tgmm' package.
    "gmm" : GaussianMixtureModelTGMM,

    }
