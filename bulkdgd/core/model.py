#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-

#    model.py
#
#    This module contains the class implementing the full BulkDGD model
#    (:class:`core.model.BulkDGD`).
#
#    Copyright (C) 2026 Valentina Sora
#                       <sora.valentina1@gmail.com>
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
    "This module contains the class implementing the full BulkDGD " \
    "model (:class:`core.model.BulkDGG`)."


#######################################################################


# Import from the standard library.
import copy
import logging as log
import math
import re
import time
from typing import Optional, Union

# Import from third-party libraries.
import numpy as np
import pandas as pd
from scipy.stats import chi2
import torch
from torch import nn

# Import from 'bulkdgd'.
from bulkdgd import _internals
from . import (
    dataclasses,
    decoders,
    latents,
    metrics,
    outputmodules,
    _util)


#######################################################################


# Get the module's logger.
logger = log.getLogger(__name__)


#######################################################################


def clip_grads(optimizer: torch.optim.Optimizer,
               max_norm: Optional[Union[float, int]]) -> None:
    """Clip the gradients of the parameters an optimizer steps, to a
    maximum norm, before it takes its step.

    Without it, a single batch whose gradients are large enough can take
    the model out in one step - the loss becomes NaN, and stays NaN,
    since NaN propagates through every parameter it touches.

    The negative binomial output modules make that easy to run into. The
    log-probability mass contains ``lgamma(k+r) - lgamma(r)``, and a gene
    whose dispersion ``r`` is driven towards zero drives both terms
    towards infinity, so their difference is the difference of two
    infinities.

    Parameters
    ----------
    optimizer : :class:`torch.optim.Optimizer`
        The optimizer whose parameters' gradients should be clipped.

    max_norm : :class:`float` or :class:`int`, optional
        The norm to clip the gradients to. If :obj:`None`, the gradients
        are not clipped.
    """

    # If no maximum norm was set, the gradients are not clipped.
    if max_norm is None:
        return

    # Get the parameters the optimizer steps that have a gradient.
    params = \
        [p for group in optimizer.param_groups \
         for p in group["params"] if p.grad is not None]

    # If there are none, there is nothing to clip.
    if not params:
        return

    #-----------------------------------------------------------------#

    # Zero out any gradient that is not finite, before clipping.
    #
    # Clipping cannot rescue an infinite gradient - scaling it by
    # 'max_norm / inf' multiplies an infinity by zero, which is NaN, and
    # NaN then propagates into every parameter the optimizer touches. A
    # step whose gradients have already gone non-finite carries no usable
    # information, so drop it rather than let it take the model out.
    for p in params:

        # If the parameter's gradient is not entirely finite
        if not torch.isfinite(p.grad).all():

            # Warn - this is rare, and worth knowing about.
            logger.warning(\
                "A non-finite gradient was produced and has been "
                "zeroed. The step it came from is effectively skipped.")

            # Drop it.
            p.grad = torch.zeros_like(p.grad)

    #-----------------------------------------------------------------#

    # Clip the gradients. What is returned is the norm they had before
    # being clipped, which is what tells you where to set the maximum:
    # too low, and every step is scaled down, not just the rare one that
    # would have blown the model up.
    total_norm = \
        torch.nn.utils.clip_grad_norm_(parameters = params,
                                       max_norm = max_norm)

    # Log the norm the gradients had, for whoever is choosing the
    # maximum. This is a debug message - there is one per step.
    logger.debug(f"Gradient norm before clipping: {float(total_norm):.4f} "
                 f"(clipped to {max_norm}).")


#######################################################################


class BulkDGD(nn.Module):

    """
    Class implementing the BulkDGD model.
    """

    ######################## PUBLIC ATTRIBUTES ########################


    # Set the supported optimizers to find the representations.
    OPTIMIZERS = ["adam", "adamw"]

    # Set the supported Gaussian mixture model types.
    GMM_TYPES = ["lgmm", "tgmm"]


    ##################### INITIALIZATION METHODS ######################


    def __init__(self,
                 latent_dim: int,
                 latent_options: dict[str, object],
                 decoder_options: dict[str, object],
                 latent_type: str = "tgmm",
                 genes_txt_file: Optional[str] = None,
                 device: str = "cpu") -> None:
        """Initialize an instance of the class.

        The model is initialized on the CPU. To move the model to
        another device, modify the ``device`` property.

        Parameters
        ----------
        latent_dim : :class:`int`
            The dimensionality of the latent space.

        latent_type : :class:`str`, {``"lgmm"``, ``"tgmm"``}, \
            ``"tgmm"``
            The type of the latent space to use.

            The available options are:

            - ``"lgmm"``: the legacy Gaussian mixture model
                implementation, which uses the
                :class:`bulkdgd.core.latents.GaussianMixtureModelLegacy`
                class.

            - ``"tgmm"``: the TGMM Gaussian mixture model
                implementation, which uses the
                :class:`bulkdgd.core.latents.GaussianMixtureModelTGMM`
                class.

        latent_options : :class:`dict`
            The options for setting up the latent space.

            For the available options, refer to the
            :ref:`model_config_options` page.

        decoder_options : :class:`dict`
            The options for setting up the decoder.

            For the available options, refer to the
            :ref:`model_config_options` page.

        genes_txt_file : :class:`str`
            A plain text file containing the Ensembl IDs of the genes
            included in the model.

            Training data will be checked to ensure counts are
            reported for all genes.

            The number of output units in the decoder is initialized
            from the number of genes found in this file.
        
        device : :class:`str`, ``"cpu"``
            The device where the model will be initialized. The model
            is initialized on the CPU by default.
        """

        # Run the superclass' initialization.
        super(BulkDGD, self).__init__()

        #-------------------------------------------------------------#

        # Get the genes included in the model.
        genes = \
            self.__class__._load_genes_list(\
                genes_list_file = genes_txt_file)

        #-------------------------------------------------------------#

        # Get the latent space.
        self._latent = \
            self._get_latent(latent_dim = latent_dim,
                             latent_type = latent_type,
                             latent_options = latent_options,
                             device = device)
        
        # Save the options for the latent space in the
        # model's attributes.
        self._latent_initial_options = latent_options

        # Inform the user that the latent space was set.
        info_msg = \
            "The latent space was successfully set " \
            f"(type: '{self._latent.__class__.__name__}')."
        logger.info(info_msg)

        #-------------------------------------------------------------#

        # Get the decoder and the r-values.
        self._decoder, self._r_values = \
            self._get_decoder(latent_dim = latent_dim,
                              genes = genes,
                              decoder_options = decoder_options,
                              device = device)

        # Inform the user that the decoder was set.
        info_msg = "The decoder was successfully set."
        logger.info(info_msg)

        # Save the options for the decoder in the model's attributes.
        self._decoder_initial_options = decoder_options

        #-------------------------------------------------------------#
        
        # By default, the model is initialized on the CPU.
        self._device = torch.device(device)

        # Move the model to the specified device.
        self.to(device = torch.device(device))


    def _get_latent(self,
                    latent_dim: int,
                    latent_type: str,
                    latent_options: dict[str, object],
                    device: str) -> \
                        latents.GaussianMixtureModelLegacy | \
                        latents.GaussianMixtureModelTGMM:
        """Get the latent space.

        Parameters
        ----------
        latent_dim : :obj:`int`
            The number of dimensions of the latent space.

        latent_type : :class:`str`, {``"lgmm"``, ``"tgmm"``}
            The type of latent space to use.

            The available options are:

            - ``"lgmm"``: the legacy Gaussian mixture model
                implementation, which uses the
                :class:`bulkdgd.core.latents.GaussianMixtureModelLegacy`
                class.
            
            - ``"tgmm"``: the TorchGMM Gaussian mixture model
                implementation, which uses the
                :class:`bulkdgd.core.latents.GaussianMixtureModelTGMM`
                class.
        
        latent_options : :class:`dict`
            A dictionary of options for the latent space.
        
        device : :class:`str`
            The device where the Gaussian mixture model will be
            initialized.
        
        Returns
        -------
        latent : :class:`bulkdgd.core.latents.GaussianMixtureModelLegacy` \
                 or :class:`bulkdgd.core.latents.GaussianMixtureModelTGMM`
            The latent space.
        """

        # Build a copy of the latent space's options without the
        # (optional) path to a file with pre-trained parameters --
        # it is not a valid constructor argument for either latent
        # space class, and is used below (after construction) to
        # load the pre-trained parameters, if provided.
        latent_options_init = \
            {k: v for k, v in latent_options.items()
             if k != "latent_pth_file"}

        # If the user wants to use the legacy Gaussian Mixture Model
        if latent_type == "lgmm":

            # Initialize the Gaussian mixture model.
            latent = \
                latents.GaussianMixtureModelLegacy(dim = latent_dim,
                                                   **latent_options_init)

        # If the user wants to use the 'tgmm' Gaussian Mixture Model
        elif latent_type == "tgmm":

            # Initialize the TGMM Gaussian mixture model.
            latent = \
                latents.GaussianMixtureModelTGMM(
                    n_features = latent_dim,
                    device = device,
                    **latent_options_init)
        
        # If the user provided an unsupported latent space type
        else:

            # Raise an error.
            err_msg = \
                f"The latent space type '{latent_type}' is not " \
                "supported. The supported latent space types are: " \
                f"{', '.join(self.__class__.LATENT_TYPES)}."
            raise ValueError(err_msg)
            
        #-------------------------------------------------------------#
        
        # If the user provided a file with the latent space's trained
        # parameters
        if latent_options.get("latent_pth_file") is not None:

            # Load the parameters.
            self.__class__._load_state(
                mod = latent,
                pth_file = latent_options["latent_pth_file"],
                device = device)
        
        #-------------------------------------------------------------#

        # Return the latent space.
        return latent


    def _get_decoder(self,
                     latent_dim: int,
                     genes: list[str],
                     decoder_options: dict[str, object],
                     device: str) -> \
                        tuple[decoders.Decoder, Optional[pd.Series]]:
        """Get the decoder.

        Parameters
        ----------
        latent_dim : :obj:`int`
            The number of dimensions of the latent space.
        
        genes : :class:`list`
            A list of the genes' Ensembl IDs.
        
        decoder_options : :class:`dict`
            A dictionary of options for the decoder.
        
        device : :class:`str`
            The device to load the parameters onto.

        Returns
        -------
        dec : :class:`bulkdgd.core.decoders.Decoder`
            The decoder.
        
        r_values : :class:`pandas.Series` or :obj:`None`
            A series containing the r-values of the negative
            binomials modeling the genes' counts, if the decoder's
            output module is the 'nb_feature_dispersion' one.

            The series' index contains the genes' Ensembl IDs, and
            its values are the r-values.

            If the decoder's output module is not the
            'nb_feature_dispersion' one, ``r_values`` is :obj:`None`.
        """

        # Create a copy of the configuration options for the
        # decoder.
        decoder_options_copy = copy.deepcopy(decoder_options)

        # Update the decoder's options by adding the number of
        # output units.
        decoder_options_copy[
            "output_module_options"]["output_dim"] = len(genes)

        # Remove the (optional) path to a file with pre-trained
        # parameters -- it is not a valid constructor argument for
        # 'decoders.Decoder', and is used below (after construction)
        # to load the pre-trained parameters, if provided.
        decoder_options_copy.pop("decoder_pth_file", None)

        # Get the decoder.
        decoder = \
            decoders.Decoder(n_units_input_layer = latent_dim,
                            **decoder_options_copy)

        # If the user provided a file with the decoder's trained
        # parameters
        if decoder_options.get("decoder_pth_file") is not None:

            # Load the parameters.
            self.__class__._load_state(
                mod = decoder,
                pth_file = decoder_options["decoder_pth_file"],
                device = device)
        
        #-------------------------------------------------------------#

        # Get the output module's name.
        output_module_name = decoder_options_copy["output_module_name"]

        # If the output module is the 'nb_feature_dispersion' one
        if output_module_name == "nb_feature_dispersion":

            # Get the r-values associated with the negative binomials
            # modeling the different genes.
            r_values = \
                torch.exp(decoder.nb.log_r).squeeze().detach()

            # Associate the r-values with the genes.
            r_values = pd.Series(r_values,
                                 index = genes)

        # If the output module is the 'nb_full_dispersion' or the
        # 'poisson' one
        elif output_module_name in ("nb_full_dispersion", "poisson"):

            # The r-values will be None.
            r_values = None
        
        #-------------------------------------------------------------#

        # Return the decoder and the r-values.
        return decoder, r_values


    ######################## STATIC METHODS ###########################


    @staticmethod
    def _load_state(mod: torch.nn.Module,
                    pth_file: str,
                    device: str) -> None:
        """Load a module's trained parameters.

        Parameters
        ----------
        mod : :class:`nn.Module`
            The module.

        pth_file : :class:`str`
            The PyTorch file to load the parameters from.

        device : :class:`str`
            The device to load the parameters onto.
        """

        # Try to load the parameters
        try:

            # Load the parameters.
            mod.load_state_dict(
                torch.load(pth_file,
                           weights_only = True,
                           map_location = device))

        # If something went wrong
        except Exception as e:

            # Raise an error with a more specific exception type.
            err_msg = \
                f"It was not possible to load the parameters " \
                f"from '{pth_file}'. Error: {e}"
            raise RuntimeError(err_msg)

        # Inform the user that the parameters was successfully loaded.
        info_msg = \
            "The parameters were successfully loaded from " \
            f"'{pth_file}'."
        logger.info(info_msg)


    @staticmethod
    def _load_genes_list(genes_list_file: str) -> list[str]:
        """Load a list of newline-separated genes from a plain text
        file.

        Parameters
        ----------
        genes_list_file : :class:`str`
            The plain text file containing the genes of interest.

        Returns
        -------
        list_genes : :class:`list`
            The list of genes.
        """

        # Return the list of genes from the file (exclude blank
        # and comment lines).
        with open(genes_list_file, "r") as file_handle:
            return [line.rstrip("\n") for line in file_handle
                    if (not line.startswith("#")
                        and not re.match(r"^\s*$", line))]
    

    ########################### PROPERTIES ############################


    @property
    def latent(self):
        """The latent space.
        """

        return self._latent


    @latent.setter
    def latent(self,
            value):
        """Raise an exception if the user tries to modify the value
        of ``latent`` after initialization.
        """
        
        err_msg = \
            "The value of 'latent' is set at initialization and  " \
            "cannot be changed. If you want to change the " \
            "latent space, initialize a new instance of " \
            f"'{self.__class__.__name__}'."
        raise ValueError(err_msg)


    @property
    def decoder(self):
        """The decoder.
        """

        return self._decoder


    @decoder.setter
    def decoder(self,
                value):
        """Raise an exception if the user tries to modify the value of
        ``decoder`` after initialization.
        """
        
        err_msg = \
            "The value of 'decoder' is set at initialization and " \
            "cannot be changed. If you want to change the decoder, " \
            "initialize a new instance of " \
            f"'{self.__class__.__name__}'."
        raise ValueError(err_msg)

    @property
    def device(self):
        """The device where the model is.
        """

        return self._device

    @device.setter
    def device(self,
               value):
        """Move the model to the selected device.
        """
        
        # Move the model to the specified device.
        self.to(device = torch.device(value))

        # Update the device the model is on.
        # Store a torch.device for consistency.
        self._device = torch.device(value)
    

    ######################### PRIVATE METHODS #########################


    def _get_optimizer(self,
                       optimizer_type: str,
                       optimizer_options: dict[str, object],
                       optimizer_parameters: torch.nn.Parameter) -> \
                        torch.optim.Optimizer:
        """Get the optimizer.

        Parameters
        ----------
        optimizer_type : :class:`str`
            The type of optimizer to set up.
        
        optimizer_options : :class:`dict`
            A dictionary of options for the optimizer.

        optimizer_parameters : :class:`torch.nn.Parameter`
            The parameters that will be optimized.

        Returns
        -------
        optimizer : :class:`torch.optim.Optimizer`
            The optimizer.
        """

        # If it is the Adam optimizer
        if optimizer_type == "adam":

            # Set up the optimizer.
            optimizer = \
                torch.optim.Adam(optimizer_parameters,
                                 **optimizer_options)

        # If it is the AdamW optimizer
        elif optimizer_type == "adamw":

            # Set up the optimizer.
            optimizer = \
                torch.optim.AdamW(optimizer_parameters,
                                 **optimizer_options)

        #-------------------------------------------------------------#

        # Return the optimizer.
        return optimizer
    
    
    def _get_scheduler(self,
                       lr_scheduler_target: str,
                       lr_scheduler_type: str,
                       lr_scheduler_options: dict[str, object],
                       optimizer: torch.optim.Optimizer,
                       n_epochs: int,
                       data_loader_train: \
                        Optional[torch.utils.data.DataLoader] = None) \
                            -> Optional[
                                torch.optim.lr_scheduler.LRScheduler]:
        """Get the learning rate scheduler.
        
        Parameters
        ----------
        lr_scheduler_target : :class:`str`, {``"decoder"``, \
            ``"representations"``}
            The target for which to set up the learning rate scheduler.

            The available options are:

            - ``"decoder"``: the learning rate scheduler for the
                decoder, which steps per batch.

            - ``"representations"``: the learning rate scheduler for
                the representations, which steps per epoch.

        lr_scheduler_type : :class:`str` or :obj:`None`
            The type of learning rate scheduler to set up,
            or :obj:`None`.
        
        lr_scheduler_options : :class:`dict`
            A dictionary of options for the learning rate scheduler.
        
        optimizer : :class:`torch.optim.Optimizer`
            The optimizer for which to set up the learning rate
            scheduler.

        n_epochs : :class:`int`
            The number of epochs for training.

        data_loader_train : :class:`torch.utils.data.DataLoader`, \
            optional
            The training data loader, required if the scheduler steps
            per batch.
        
        Returns
        -------
        scheduler : :class:`torch.optim.lr_scheduler.LRScheduler` or \
            :obj:`None`
            The learning rate scheduler, if it is enabled in the
            configuration, or :obj:`None` otherwise.
        """

        # If no learning rate scheduler is enabled
        if lr_scheduler_type is None:

            # Return None.
            return None

        #-------------------------------------------------------------#
            
        # If the scheduler is for the decoder or for the latent space
        if lr_scheduler_target in ("decoder", "latent"):
            
            # Set the total steps to the total number of epochs times
            # the number of batches in the training data.
            total_steps = n_epochs * len(data_loader_train)
        
        # If the scheduler is for the representations
        elif lr_scheduler_target == "representations":

            # Set the total steps to the total number of epochs.
            total_steps = n_epochs

        #-------------------------------------------------------------#
        
        # If the type of scheduler is 'one_cycle'
        if lr_scheduler_type == "one_cycle":
        
            # Set the scheduler.
            lr_scheduler_opts = lr_scheduler_options.copy()
            lr_scheduler_opts.pop("enabled", None)
            lr_scheduler = \
                torch.optim.lr_scheduler.OneCycleLR(\
                    optimizer,
                    total_steps = total_steps,
                    **lr_scheduler_opts)

        #-------------------------------------------------------------#
            
        # Return the scheduler.
        return lr_scheduler

    
    def _optimize_rep(self,
                      data_loader: torch.utils.data.DataLoader,
                      rep_layer: latents.RepresentationLayer,
                      optimizer: torch.optim.Optimizer,
                      n_components: int,
                      n_rep_per_comp: int,
                      epochs: int,
                      opt_num: int,
                      loss_reporting_options: dict[str, object],
                      loss_reduction_type: str,
                      latent_lambda: Optional[float] = None) -> \
                        torch.Tensor:
        """Optimize the representation(s) found for each sample.

        Parameters
        ----------
        data_loader : :class:`torch.utils.data.DataLoader`
            The data loader.

        rep_layer : :class:`bulkdgd.core.latents.RepresentationLayer`
            The representation layer containing the initial
            representations.

        optimizer : :class:`torch.optim.Optimizer`
            The optimizer.

        n_components : :class:`int`
            The number of components of the Gaussian mixture model
            for which at least one representation was drawn per
            sample.

        n_rep_per_comp : :class:`int`
            The number of new representations taken per sample
            per component of the Gaussian mixture model.

        epochs : :class:`int`
            The number of epochs to run the optimization for.

        opt_num : :class:`int`
            The number of the optimization round (especially useful
            if multiple rounds are run).
        
        loss_reporting_options : :class:`dict`
            A dictionary containing the options for reporting the loss.
        
        loss_reduction_type : :class:`str`
            The method to reduce the loss across the samples in the
            batch.
        
        latent_lambda : :class:`float`, optional
            The weight of the latent loss term in the total loss.

        Returns
        -------
        rep : :class:`torch.Tensor`
            A tensor containing the optimized representations.

            This is a 2D tensor where:

            - The first dimension has a length equal to the number
              of samples.

            - The second dimension has a length equal to the
              dimensionality of the latent space where the
              representations live.

        pred_means : :class:`torch.Tensor`
            A tensor containing the predicted means of the
            distributions modelling the genes' counts.

            This is a 2D tensor where:

            - The first dimension has a length equal to the number
              of samples.

            - The second dimension has a length equal to the
              dimensionality of the gene space.

            If the genes counts are modelled using negative binomial
            distributions, the predicted means are scaled by the
            corresponding distributions' r-values.

        pred_r_values : :class:`torch.Tensor` or :obj:`None`
            A tensor containing the predicted r-values of the negative
            binomial distributions modelling the genes' counts, if
            the counts are modelled by negative binomial distributions.

            This is a 2D tensor where:

            - The first dimension has a length equal to the number
              of samples.

            - The second dimension has a length equal to the
              dimensionality of the gene space.

            ``pred_r_values`` is :obj:`None` if the counts are modelled
            by Poisson distributions.

        time_opt : :class:`list`
            A list of tuples storing, for each epoch, information
            about the CPU and wall clock time used by the entire
            epoch and by the backpropagation step run within the
            epoch.
        """

        # Get the total number of samples.
        n_samples = len(data_loader.dataset)

        # Get the dimensionality of the latent space.
        dim = self.latent.dim

        # Get the number of genes (= the dimensionality of the
        # decoder's output).
        n_genes = self.decoder.nb.output_dim

        # Get the method that will be used to normalize the total loss.
        loss_norm_type = loss_reporting_options["total"]["norm_type"]

        # Create a list to store the CPU/wall clock time used in each
        # epoch of the optimization.
        time_opt = []

        #-------------------------------------------------------------#

        # Inform the user that the optimization is starting
        info_msg = f"Starting optimization number {opt_num}..."
        logger.info(info_msg)

        # For each epoch
        for epoch in range(1, epochs+1):

            # Mark the CPU start time of the epoch.
            time_start_epoch_cpu = time.process_time()

            # Mark the wall clock start time of the epoch.
            time_start_epoch_wall = time.time()

            # Initialize the total CPU time needed to perform the
            # backward step to zero.
            time_tot_bw_cpu = 0.0

            # Initialize the total wall-clock time needed to perform
            # the backward step to zero.
            time_tot_bw_wall = 0.0

            # Make the optimizer's gradients zero.
            optimizer.zero_grad()

            # Initialize the loss for the current epoch to 0.0.
            rep_avg_loss_epoch = 0.0

            # For each batch of samples, the mean gene expression
            # in the samples in the batch, and the unique indexes
            # of the samples in the batch
            for samples_exp, samples_mean_exp, samples_ixs \
                in data_loader:

                # Get the number of samples in the batch.
                n_samples_in_batch = len(samples_ixs)

                #-----------------------------------------------------#

                # Move the gene expression of the samples to the
                # correct device.
                samples_exp = samples_exp.to(self.device)

                # Move the mean gene expression of the samples to
                # the correct device.
                samples_mean_exp = samples_mean_exp.to(self.device)

                #-----------------------------------------------------#

                # Get the representations' values from the
                # representation layer.
                # 
                # The representations are stored in a 2D tensor with:
                #
                # - 1st dimension:
                #       the total number of samples times the number of
                #       components in the Gaussian mixture model times
                #       the number of representations taken per
                #       component per sample
                #
                # - 2nd dimension:
                #       the dimensionality of the Gaussian mixture
                #       model
                z_all = rep_layer()

                #-----------------------------------------------------#

                # Reshape the tensor containing the representations.
                #
                # The output is a 4D tensor with:
                #
                # - 1st dimension:
                #       the total number of samples
                # 
                # - 2nd dimension:
                #       the number of representations taken per
                #        component per sample
                #
                # - 3rd dimension:
                #       the number of components in the Gaussian
                #       mixture model
                #
                # - 4th dimension:
                #       the dimensionality of the Gaussian mixture
                #       model
                z_4d = z_all.view(n_samples,
                                  n_rep_per_comp,
                                  n_components,
                                  dim)[samples_ixs]

                # Reshape the tensor again.
                #
                # The output is a 2D tensor with:
                #
                # - 1st dimension:
                #       the number of samples in the current
                #       batch times the number of components
                #       in the Gaussian mixture model times
                #       the number of representations taken
                #       per component per sample
                #
                # - 2nd dimension:
                #       the dimensionality of the Gaussian mixture
                #       model
                z = z_4d.view(n_samples_in_batch * \
                                n_rep_per_comp * \
                                n_components,
                              dim)

                #-----------------------------------------------------#

                # If the chosen output module means that the r-values
                # are not learned
                if isinstance(\
                    self.decoder.nb,
                    (outputmodules.OutputModuleNBFeatureDispersion,
                     outputmodules.OutputModulePoisson)):
                    
                    # Get the predicted scaled means of the
                    # distributions modelling the genes' counts.
                    #
                    # The output is a 2D tensor with:
                    #
                    # - 1st dimension:
                    #       the number of samples in the current batch
                    #       times the number of components in the
                    #       Gaussian mixture model times the number of
                    #       representations taken per component per
                    #       sample
                    #
                    # - 2nd dimension:
                    #       the dimensionality of the output (= gene)
                    #       space
                    pred_means = self.decoder(z = z)

                # If the chosen output module means that the r-values
                # are learned
                elif isinstance(\
                    self.decoder.nb,
                    outputmodules.OutputModuleNBFullDispersion):

                    # Get the predicted scaled means and r-values
                    # of the negative binomial distributions modelling
                    # the genes' counts.
                    #
                    # Both outputs are 2D tensors with:
                    #
                    # - 1st dimension:
                    #       the number of samples in the current batch
                    #       times the number of components in the
                    #       Gaussian mixture model times the number of
                    #       representations taken per component per
                    #       sample
                    #
                    # - 2nd dimension:
                    #       the dimensionality of the output (= gene)
                    #       space
                    pred_means, pred_log_r_values = self.decoder(z = z)

                    # Reshape the predicted r-values to match the shape
                    # required to compute the loss.
                    #
                    # The output is a 4D tensor with:   
                    #
                    # - 1st dimension:
                    #       the number of samples in the current batch
                    #
                    # - 2nd dimension:
                    #       the number of representations taken per
                    #       component per sample
                    #
                    # - 3rd dimension:
                    #       the number of components in the Gaussian
                    #       mixture model
                    #
                    # - 4th dimension:
                    #       the dimensionality of the output (= gene)
                    #       space
                    pred_log_r_values = \
                        pred_log_r_values.view(n_samples_in_batch,
                                               n_rep_per_comp,
                                               n_components,
                                               n_genes)

                #-----------------------------------------------------#

                # Get the observed gene expression and "expand" the
                # resulting tensor to match the shape required to
                # compute the reconstruction loss.
                #
                # The output is a 4D tensor with:
                #
                # - 1st dimension:
                #       the number of samples in the current batch
                #
                # - 2nd dimension:
                #       the number of representations taken per
                #       component per sample
                #
                # - 3rd dimension:
                #       the number of components in the Gaussian
                #       mixture model
                #
                # - 4th dimension:
                #       the dimensionality of the output (= gene)
                #       space
                obs_counts = \
                    samples_exp.unsqueeze(1).unsqueeze(1).expand(\
                        -1,
                        n_rep_per_comp,
                        n_components,
                        -1)

                #-----------------------------------------------------#

                # Get the scaling factors for the mean of each negative
                # binomial modelling the expression of a gene and
                # reshape it so that it matches the shape required to
                # compute the reconstruction loss.
                #
                # The output is a 4D tensor with:
                #
                # - 1st dimension:
                #       the number of samples in the current batch
                #
                # - 2nd dimension: 1
                #
                # - 3rd dimension: 1
                #
                # - 4th dimension: 1
                scaling_factors = \
                    decoders.reshape_scaling_factors(samples_mean_exp,
                                                     4)

                #-----------------------------------------------------#

                # Reshape the predicted scaled means to match the
                # shape required to compute the loss.
                #
                # The output is a 4D tensor with:   
                #
                # - 1st dimension:
                #       the number of samples in the current batch
                #
                # - 2nd dimension:
                #       the number of representations taken per
                #       component per sample
                #
                # - 3rd dimension:
                #       the number of components in the Gaussian
                #       mixture model
                #
                # - 4th dimension:
                #       the dimensionality of the output (= gene)
                #       space    
                pred_means = pred_means.view(n_samples_in_batch,
                                             n_rep_per_comp,
                                             n_components,
                                             n_genes)

                #-----------------------------------------------------#

                # If the chosen output module means that the r-values
                # are not learned
                if isinstance(\
                    self.decoder.nb,
                    (outputmodules.OutputModuleNBFeatureDispersion,
                     outputmodules.OutputModulePoisson)):

                    # Set the options to compute the reconstruction
                    # loss.
                    recon_loss_options = \
                        {"obs_counts" : obs_counts,
                         "pred_means" : pred_means,
                         "scaling_factors" : scaling_factors}

                # If the chosen output module means that the r-values
                # are learned
                elif isinstance(\
                    self.decoder.nb,
                    outputmodules.OutputModuleNBFullDispersion):

                    # Set the options to compute the reconstruction
                    # loss.
                    recon_loss_options = \
                        {"obs_counts" : obs_counts,
                         "pred_means" : pred_means,
                         "pred_log_r_values" : pred_log_r_values,
                         "scaling_factors" : scaling_factors}

                # Get the reconstruction loss.
                #
                # The output is a 4D tensor with:
                #
                # - 1st dimension:
                #       the number of samples in the current batch
                # 
                # - 2nd dimension:
                #       the number of representations taken per
                #       component per sample
                #
                # - 3rd dimension:
                #       the number of components in the Gaussian
                #       mixture model
                #
                # - 4th dimension:
                #       the dimensionality of the output (= gene)
                #       space
                recon_loss = self.decoder.nb.loss(**recon_loss_options)

                #-----------------------------------------------------#
                
                # If the reduction method is 'sum'
                if loss_reduction_type == "sum":

                    # Get the total reconstruction loss by summing all
                    # values in the 'recon_loss' tensor.
                    #
                    # The output is a tensor containing a single value.
                    recon_loss_final = recon_loss.sum().clone()
                
                # If the reduction method is 'mean'
                elif loss_reduction_type == "mean":

                    # Get the total reconstruction loss by averaging
                    # all values in the 'recon_loss' tensor.
                    #
                    # The output is a tensor containing a single value.
                    recon_loss_final = recon_loss.mean().clone()

                #-----------------------------------------------------#

                # If the latent space is the legacy Gaussian mixture
                # model
                if isinstance(self.latent,
                              latents.GaussianMixtureModelLegacy):
                    
                    # If the reduction method is 'sum'
                    if loss_reduction_type == "sum":

                        # Get the loss.
                        latent_loss_final = \
                            self.latent(x = z).sum().clone()
                    
                    # If the reduction method is 'mean'
                    elif loss_reduction_type == "mean": 

                        # Get the loss.
                        latent_loss_final = \
                            self.latent(x = z).mean().clone()
                
                # If the latent space is the TorchGMM wrapper
                elif isinstance(self.latent,
                                latents.GaussianMixtureModelTGMM):
                    
                    # If the reduction method is 'sum'
                    if loss_reduction_type == "sum":

                        # Get the loss.
                        latent_loss_final = \
                            - latent_lambda * \
                                torch.sum(\
                                    self.latent.log_prob(z))
                    
                    # If the reduction method is 'mean'
                    elif loss_reduction_type == "mean":

                        # Get the loss.
                        latent_loss_final = \
                            - latent_lambda * \
                                torch.mean(\
                                    self.latent.log_prob(z))

                #-----------------------------------------------------#

                # Get the total loss by summing the reconstruction loss
                # and the loss of the latent space.
                #
                # The output is a tensor containing a single value.
                total_loss = recon_loss_final + latent_loss_final

                #-----------------------------------------------------#

                # Mark the CPU start time of the backward step.
                time_start_bw_cpu = time.process_time()

                # Mark the wall clock start time of the backward step.
                time_start_bw_wall = time.time()

                # Propagate the loss backward.
                total_loss.backward()

                # Mark the end CPU time of the backward step.
                time_end_bw_cpu = time.process_time()

                # Mark the wall clock end time of the backward step.
                time_end_bw_wall = time.time()

                # Get the total CPU time used by the backward step.
                time_tot_bw_cpu += \
                    time_end_bw_cpu - time_start_bw_cpu

                # Get the total wall clock time used by the backward
                # step.
                time_tot_bw_wall += \
                    time_end_bw_wall - time_start_bw_wall

                #-----------------------------------------------------#

                # Update the average loss for the current epoch.
                rep_avg_loss_epoch += \
                    _util.normalize_loss(\
                        loss = total_loss.item(),
                        loss_type = "total",
                        loss_norm_type = loss_norm_type,
                        loss_norm_options = {"n_samples" : n_samples,
                                             "n_genes" : n_genes})

            #---------------------------------------------------------#

            # Take an optimization step.
            optimizer.step()

            #---------------------------------------------------------#

            # Mark the CPU end time of the epoch.
            time_end_epoch_cpu = time.process_time()

            # Mark the wall clock end time of the epoch.
            time_end_epoch_wall = time.time()

            # Get the total CPU time used by the epoch.
            time_tot_epoch_cpu = \
                time_end_epoch_cpu - time_start_epoch_cpu

            # Get the total wall clock time used by the epoch.
            time_tot_epoch_wall = \
                time_end_epoch_wall - time_start_epoch_wall

            # Add all the total times to the list storing them for
            # all epochs.
            time_opt.append(\
                (opt_num, epoch,
                 time_tot_epoch_cpu, time_tot_bw_cpu,
                 time_tot_epoch_wall, time_tot_bw_wall))

            # Inform the user about the loss at the current epoch and
            # the CPU time/wall clock time elapsed.
            info_msg = \
                f"Epoch {epoch}: loss {rep_avg_loss_epoch:.3f}, " \
                f"epoch CPU time {time_tot_epoch_cpu:.3f} s, " \
                f"backward step CPU time {time_tot_bw_cpu:.3f} s, " \
                "epoch wall clock time " \
                f"{time_tot_epoch_wall:.3f} s, " \
                "backward step wall clock time " \
                f"{time_tot_bw_wall:.3f} s."
            logger.info(info_msg)

            #---------------------------------------------------------#

            # If we reached the last epoch
            if epoch == epochs:

                # Get the optimized representations.
                rep_final = rep_layer()

                #-----------------------------------------------------#

                # If the genes' counts are modelled by negative
                # binomial distributions whose r-values are learned
                # per gene (but not per sample)
                if isinstance(\
                    self.decoder.nb,
                    outputmodules.OutputModuleNBFeatureDispersion):

                    # Get the predicted scaled means.
                    means_final = self.decoder(z = rep_final)

                    # Get the r-values.
                    r_values_final = \
                        torch.exp(\
                            self.decoder.nb.log_r).squeeze().detach()

                # If the genes' counts are modelled by negative
                # binomial distributions whose r-values are learned
                # per gene per sample
                elif isinstance(\
                    self.decoder.nb,
                    outputmodules.OutputModuleNBFullDispersion):

                    # Get the predicted scaled means.
                    means_final, log_r_values_final = \
                        self.decoder(z = rep_final)

                    # Get the r-values.
                    r_values_final = \
                        torch.exp(\
                            log_r_values_final).squeeze().detach()

                # If the genes' counts are modelled by Poisson
                # distributions
                elif isinstance(    
                    self.decoder.nb,
                    outputmodules.OutputModulePoisson):

                    # Get the predicted scaled means.
                    means_final = self.decoder(z = rep_final)

                    # The r-values will be None.
                    r_values_final = None

                #-----------------------------------------------------#

                # Return the representations, the predicted scaled
                # means, the predicted r-values, and the time data.
                return rep_final, \
                       means_final, r_values_final, \
                       time_opt


    def _select_best_rep(self,
                         data_loader: torch.utils.data.DataLoader,
                         rep_layer: latents.RepresentationLayer,
                         n_rep_per_comp: int,
                         loss_reduction_type: str,
                         latent_lambda: \
                            Optional[float] = None) -> \
                                torch.Tensor:
        """Select the best representation per sample.

        Parameters
        ----------
        data_loader : :class:`torch.utils.data.DataLoader`
            The data loader.

        rep_layer : :class:`bulkdgd.core.latents.RepresentationLayer`
            The representation layer containing the representations
            found for the samples.

        n_rep_per_comp : :class:`int`
            The number of new representations that were taken per
            sample per component of the Gaussian mixture model.

        loss_reduction_type : :class:`str`
            The method to reduce the loss across the samples in the
            batch.

        latent_lambda : :class:`float`, optional
            The weight of the GMM loss term in the total loss.

        Returns
        -------
        rep : :class:`torch.Tensor`
            A tensor containing the best representations found for the
            given samples (one representation per sample).

            This is a 2D tensor where:

            - The first dimension has a length equal to the number
              of samples.

            - The second dimension has a length equal to the
              dimensionality of the latent space, where the
              representations live.
        """

        # Get the total number of samples.
        n_samples = len(data_loader.dataset)

        # Get the number of components in the Gaussian mixture model.
        n_components = self.latent.n_components

        # Get the dimensionality of the latent space.
        dim = self.latent.dim

        # Get the number of genes (= dimensionality of the decoder's
        # output).
        n_genes = self.decoder.nb.output_dim
        
        #-------------------------------------------------------------#

        # Initialize an empty tensor to store the best representations
        # found for all samples.
        #
        # This is a 2D tensor with:
        #
        # - 1st dimension:
        #       the total number of samples
        #
        # - 2nd dimension:
        #       the dimensionality of the Gaussian mixture model
        best_reps = torch.empty((n_samples, dim)).to(self.device)
        
        #-------------------------------------------------------------#

        # For each batch of samples, the mean gene expression
        # in the samples in the batch, and the unique indexes
        # of the samples in the batch
        for samples_exp, samples_mean_exp, samples_ixs \
            in data_loader:

            # Get the number of samples in the batch.
            n_samples_in_batch = len(samples_ixs)

            #---------------------------------------------------------#

            # Move the gene expression of the samples to the
            # correct device.
            samples_exp = samples_exp.to(self.device)

            # Move the mean gene expression of the samples to
            # the correct device.
            samples_mean_exp = samples_mean_exp.to(self.device)

            #---------------------------------------------------------#

            # Get the representations' values from the
            # representation layer.
            # 
            # The representations are stored in a 2D tensor with:
            #
            # - 1st dimension:
            #       the total number of samples times the number of
            #       components in the Gaussian mixture model times
            #       the number of representations taken per
            #       component per sample
            #
            # - 2nd dimension:
            #       the dimensionality of the Gaussian mixture
            #       model
            z_all = rep_layer()

            # Reshape the tensor containing the representations.
            #
            # The output is a 4D tensor with:
            #
            # - 1st dimension:
            #       the total number of samples
            # 
            # - 2nd dimension:
            #       the number of representations taken per
            #        component per sample
            #
            # - 3rd dimension:
            #       the number of components in the Gaussian
            #       mixture model
            #
            # - 4th dimension:
            #       the dimensionality of the Gaussian mixture
            #       model
            z_4d = z_all.view(n_samples,
                              n_rep_per_comp,
                              n_components,
                              dim)[samples_ixs]

            # Reshape the tensor again.
            #
            # The output is a 2D tensor with:
            #
            # - 1st dimension:
            #       the number of samples in the current
            #       batch times the number of components
            #       in the Gaussian mixture model times
            #       the number of representations taken
            #       per component per sample
            #
            # - 2nd dimension:
            #       the dimensionality of the Gaussian mixture
            #       model
            z = z_4d.view(n_samples_in_batch * \
                            n_rep_per_comp * \
                            n_components,
                          dim)

            #---------------------------------------------------------#

            # If the chosen output module means that the r-values
            # are not learned
            if isinstance(\
                self.decoder.nb,
                (outputmodules.OutputModuleNBFeatureDispersion,
                 outputmodules.OutputModulePoisson)):
                
                # Get the predicted scaled means of the
                # distributions modelling the genes' counts.
                #
                # The output is a 2D tensor with:
                #
                # - 1st dimension:
                #       the number of samples in the current batch
                #       times the number of components in the
                #       Gaussian mixture model times the number of
                #       representations taken per component per
                #       sample
                #
                # - 2nd dimension:
                #       the dimensionality of the output (= gene)
                #       space
                pred_means = self.decoder(z = z)
            
            # If the chosen output module means that the r-values
            # are learned
            elif isinstance(\
                self.decoder.nb,
                outputmodules.OutputModuleNBFullDispersion):

                # Get the predicted scaled means of the
                # distributions modelling the genes' counts.
                #
                # Both outputs are 2D tensors with:
                #
                # - 1st dimension:
                #       the number of samples in the current batch
                #       times the number of components in the
                #       Gaussian mixture model times the number of
                #       representations taken per component per
                #       sample
                #
                # - 2nd dimension:
                #       the dimensionality of the output (= gene)
                #       space
                pred_means, pred_log_r_values = self.decoder(z = z)

                # Reshape the predicted r-values to match the shape
                # required to compute the loss.
                #
                # The output is a 4D tensor with:
                #
                # - 1st dimension:
                #       the number of samples in the current batch
                #
                # - 2nd dimension:
                #       the number of representations taken per
                #       component per sample
                #
                # - 3rd dimension:
                #       the number of components in the Gaussian
                #       mixture model
                #
                # - 4th dimension:
                #       the dimensionality of the output (= gene)
                #       space
                pred_log_r_values = \
                    pred_log_r_values.view(n_samples_in_batch,
                                           n_rep_per_comp,
                                           n_components,
                                           n_genes)

            #---------------------------------------------------------#

            # Get the observed gene expression and "expand" the
            # resulting tensor to match the shape required to
            # compute the reconstruction loss.
            #
            # The output is a 4D tensor with:
            #
            # - 1st dimension:
            #       the number of samples in the current batch
            #
            # - 2nd dimension:
            #       the number of representations taken per
            #       component per sample
            #
            # - 3rd dimension:
            #       the number of components in the Gaussian
            #       mixture model
            #
            # - 4th dimension:
            #       the dimensionality of the output (= gene)
            #       space
            obs_counts = \
                samples_exp.unsqueeze(1).unsqueeze(1).expand(\
                    -1,
                    n_rep_per_comp,
                    n_components,
                    -1)

            #---------------------------------------------------------#

            # Get the scaling factors for the mean of each negative
            # binomial modelling the expression of a gene and
            # reshape it so that it matches the shape required to
            # compute the reconstruction loss.
            #
            # The output is a 4D tensor with:
            #
            # - 1st dimension:
            #       the number of samples in the current batch
            #
            # - 2nd dimension: 1
            #
            # - 3rd dimension: 1
            #
            # - 4th dimension: 1
            scaling_factors = \
                decoders.reshape_scaling_factors(samples_mean_exp,
                                                 4)

            #---------------------------------------------------------#

            # Reshape the predicted scaled means to match the
            # shape required to compute the loss.
            #
            # The output is a 4D tensor with:   
            #
            # - 1st dimension:
            #       the number of samples in the current batch
            #
            # - 2nd dimension:
            #       the number of representations taken per
            #       component per sample
            #
            # - 3rd dimension:
            #       the number of components in the Gaussian
            #       mixture model
            #
            # - 4th dimension:
            #       the dimensionality of the output (= gene)
            #       space
            pred_means = pred_means.view(n_samples_in_batch,
                                         n_rep_per_comp,
                                         n_components,
                                         n_genes)
     
            #---------------------------------------------------------#

            # If the chosen output module means that the r-values
            # are not learned
            if isinstance(\
                self.decoder.nb,
                (outputmodules.OutputModuleNBFeatureDispersion,
                 outputmodules.OutputModulePoisson)):

                # Set the options to compute the reconstruction
                # loss.
                recon_loss_options = \
                    {"obs_counts" : obs_counts,
                     "pred_means" : pred_means,
                     "scaling_factors" : scaling_factors}

            # If the chosen output module means that the r-values
            # are learned
            elif isinstance(\
                self.decoder.nb,
                outputmodules.OutputModuleNBFullDispersion):

                # Set the options to compute the reconstruction
                # loss.
                recon_loss_options = \
                    {"obs_counts" : obs_counts,
                     "pred_means" : pred_means,
                     "pred_log_r_values" : pred_log_r_values,
                     "scaling_factors" : scaling_factors}

            # Get the reconstruction loss.
            #
            # The output is a 4D tensor with:
            #
            # - 1st dimension:
            #       the number of samples in the current batch
            # 
            # - 2nd dimension:
            #       the number of representations taken per
            #       component per sample
            #
            # - 3rd dimension:
            #       the number of components in the Gaussian
            #       mixture model
            #
            # - 4th dimension:
            #       the dimensionality of the output (= gene)
            #       space
            recon_loss = self.decoder.nb.loss(**recon_loss_options)

            # Get the total reconstruction loss by summing or averaging
            # over the last dimension of the 'recon_loss' tensor.
            #
            # This means that the loss is not per-gene anymore, but it
            # is summed over all genes. However, it is still one loss
            # per representation per sample.
            #
            # The output is a 3D tensor with:
            #
            # - 1st dimension: the number of samples in the current
            #                  batch -> 'n_samples_in_batch'
            #
            # - 2nd dimension: the number of representations taken per
            #                  component per sample ->
            #                  'n_rep_per_comp'
            #
            # - 3rd dimension: the number of components in the Gaussian
            #                  mixture model ->
            #                  'n_components'
            
            # If the reduction method is 'sum'
            if loss_reduction_type == "sum":

                # Get the total reconstruction loss by summing over the
                # last dimension of the 'recon_loss' tensor.
                recon_loss_final = recon_loss.sum(-1).clone()
            
            # If the reduction method is 'mean'
            elif loss_reduction_type == "mean":

                # Get the total reconstruction loss by averaging over
                # the last dimension of the 'recon_loss' tensor.
                recon_loss_final = recon_loss.mean(-1).clone()

            # Reshape the reconstruction loss so that it can be summed
            # to the GMM loss (calculated below).
            #
            # The aim is to have one loss per representation per
            # sample.
            #
            # The output is, therefore, a 1D tensor with the number of
            # samples in the current batch times the number of
            # components in the Gaussian mixture model times the number
            # of representations taken per component per sample.
            recon_loss_final_reshaped = \
                recon_loss_final.view(n_samples_in_batch * \
                                      n_rep_per_comp * \
                                      n_components)

            #---------------------------------------------------------#

            # Get the latent space loss. 
            #
            # For Gaussian mixture models, 'latent(z)' computes the
            # negative log density of the probability of the
            # representations 'z' being drawn from the model.
            #
            # The shape of the loss is consistent with the shape of the
            # reconstruction loss in 'recon_loss_final_shaped'.
            #
            # The output is, therefore, a 1D tensor with the number of
            # samples in the current batch times the number of
            # components in the Gaussian mixture model times the number
            # of representations taken per component per sample.

            # If the latent space is the legacy Gaussian mixture model
            if isinstance(self.latent,
                          latents.GaussianMixtureModelLegacy):

                # Get the loss.
                latent_loss = self.latent(x = z).clone()
            
            # If the latent space is the TorchGMM wrapper
            elif isinstance(self.latent,
                            latents.GaussianMixtureModelTGMM):

                # Get the loss.
                latent_loss = \
                    - latent_lambda * self.latent.log_prob(z)

            #---------------------------------------------------------#

            # Get the total loss.
            #
            # The loss has as many components as the total number of
            # representations computed for the current batch of samples
            # ('n_rep_per_comp' * 'n_components' representations for
            # each sample in the batch).
            #
            # The output is, therefore, a 1D tensor with the number of
            # samples in the current batch times the number of
            # components in the Gaussian mixture model times the number
            # of representations taken per component per sample.
            total_loss = recon_loss_final_reshaped + latent_loss

            #---------------------------------------------------------#

            # Reshape the tensor containing the total loss.
            #
            # The output is a 2D tensor with:
            #
            # - 1st dimension:
            #       the number of samples in the current batch
            #
            # - 2nd dimension:
            #       the number of representations taken per component
            #       of the Gaussian mixture model per sample times the
            #       number of components
            total_loss_reshaped = \
                total_loss.view(n_samples_in_batch,
                                n_rep_per_comp * n_components)

            #---------------------------------------------------------#

            # Get the best representation for each sample in the
            # current batch.
            #
            # The output is a 1D tensor with the number of samples in
            # the current batch
            best_rep_per_sample = torch.argmin(total_loss_reshaped,
                                               dim = 1).squeeze(-1)

            #---------------------------------------------------------#

            # Get the best representations for the samples in the batch
            # from the 'n_rep_per_comp' * 'n_components' representations
            # taken for each sample.
            #
            # The output is a 2D tensor with:
            #
            # - 1st dimension:
            #       the number of samples in the current batch
            #
            # - 2nd dimension:
            #       the dimensionality of the Gaussian mixture model
            rep = z.view(n_samples_in_batch,
                         n_rep_per_comp * n_components,
                         dim)[range(n_samples_in_batch),
                              best_rep_per_sample]

            #---------------------------------------------------------#

            # Add the best representations found for the current batch
            # of samples to the tensor containing the best
            # representations for all samples.
            best_reps[samples_ixs] = rep

        #-------------------------------------------------------------#

        # Return the best representations found for the samples.
        return best_reps


    def _get_representations_one_opt(
            self,
            dataset: dataclasses.GeneExpressionDataset,
            config: dict[str, object]) -> torch.Tensor:
        """Get the representations for a set of samples by
        initializing ``n_rep_per_comp`` representations per each
        component of the Gaussian mixture model per sample, selecting
        the best representation for each sample, and optimizing these
        representations.

        Parameters
        ----------
        dataset : \
            :class:`bulkdgd.core.dataclasses.GeneExpressionDataset`
            The dataset from which the data loader should be created.

        config : :class:`dict`
            A dictionary with the options to run the optimization.

        Returns
        -------
        rep : :class:`torch.Tensor`
            A tensor containing the optimized representations.

            This is a 2D tensor where:

            - The first dimension has a length equal to the number
              of samples.

            - The second dimension has a length equal to the
              dimensionality of the latent space where the
              representations live.

        pred_means : :class:`torch.Tensor`
            A tensor containing the predicted means of the
            distributions modelling the genes' counts.

            This is a 2D tensor where:

            - The first dimension has a length equal to the number
              of samples.

            - The second dimension has a length equal to the
              dimensionality of the gene space.

            If the genes counts are modelled using negative binomial
            distributions, the predicted means are scaled by the
            corresponding distributions' r-values.

        pred_r_values : :class:`torch.Tensor` or :obj:`None`
            A tensor containing the predicted r-values of the negative
            binomial distributions modelling the genes' counts, if
            the counts are modelled by negative binomial distributions.

            This is a 2D tensor where:

            - The first dimension has a length equal to the number
              of samples.

            - The second dimension has a length equal to the
              dimensionality of the gene space.

            ``pred_r_values`` is :obj:`None` if the counts are modelled
            by Poisson distributions.

        time_opt : :class:`list`
            A list of tuples storing, for each epoch, information
            about the CPU and wall clock time used by the entire
            epoch and by the backpropagation step run within the
            epoch.
        """

        # Get the number of samples from the length of the dataset.
        n_samples = len(dataset)

        # Get the options for the data loader.
        data_loader_options = config["data_loader_options"]

        # Get the number of representations per component per sample.
        n_rep_per_comp = config["n_rep_per_comp"]

        # Get the options for the loss reporting.
        loss_reporting_options = config["reporting_options"]["loss"]

        # Get the method to reduce the loss across the samples in the
        # batch.
        loss_reduction_type = \
            config["scheme_options"]["loss_reduction_type"]

        # Get the type of optimizer.
        optimizer_type = \
            config["scheme_options"]["optimization"]["optimizer_type"]

        # Get the options for the optimizer.
        optimizer_options = \
            config["scheme_options"]["optimization"][
                "optimizer_options"]

        # Get the number of epochs to run the optimization for.
        epochs = config["scheme_options"]["optimization"]["epochs"]

        #-------------------------------------------------------------#

        # Create the data loader.
        data_loader = \
            _util.get_data_loader(dataset = dataset,
                                  config = data_loader_options)

        #-------------------------------------------------------------#

        # Get the initial values for the representations by sampling
        # from the latent space.

        # If the latent space is the legacy Gaussian mixture model
        if isinstance(self.latent,
                      latents.GaussianMixtureModelLegacy):

            # Sample new points from the GMM.
            rep_init = \
                self.latent.sample_new_points(\
                    n_points = n_samples, 
                    sampling_method = "mean",
                    n_samples_per_comp = n_rep_per_comp)
            
            # Set the latent lambda to None.
            latent_lambda = None

        # If the latent space is the TorchGMM wrapper
        elif isinstance(self.latent,
                        latents.GaussianMixtureModelTGMM):

            # Get the number of components.
            n_components = self.latent.n_components
            
            # Get the dimensionality.
            n_dim = self.latent.dim

            # Get the latent lambda from the configuration.
            latent_lambda = \
                config["scheme_options"][
                    "latent_loss_calculation"]["lambda"]

            # Initialize a list to store the samples from each
            # component.
            component_samples = []
            
            # For each component
            for comp_idx in range(n_components):
                
                # Sample n_samples * n_rep_per_comp points from this
                # component.
                samples_comp, _ = self.latent.sample(
                    n_samples = n_samples * n_rep_per_comp,
                    component = comp_idx)

                # Add the samples to the list.
                component_samples.append(samples_comp)
            
            # Stack all component samples: shape (n_components,
            # n_samples * n_rep_per_comp, n_dim).
            component_samples = torch.stack(component_samples,
                                            dim = 0)
            
            # Reshape to (n_components, n_samples, n_rep_per_comp,
            # n_dim).
            component_samples = component_samples.view(n_components,
                                                       n_samples,
                                                       n_rep_per_comp,
                                                       n_dim)
            
            # Permute to (n_samples, n_rep_per_comp, n_components,
            # n_dim).
            component_samples = component_samples.permute(1, 2, 0, 3)
            
            # Flatten to final shape (n_samples * n_rep_per_comp * 
            # n_components, n_dim).
            rep_init = \
                component_samples.reshape(
                    n_samples * n_rep_per_comp * n_components,
                    n_dim)

        #-------------------------------------------------------------#

        # Create a representation layer containing the initialized
        # representations.
        rep_layer_init = \
            latents.RepresentationLayer(values = rep_init).to(\
                self.device)

        #-------------------------------------------------------------#

        # Select the best representation for each sample among those
        # initialized (we initialized at least one per sample per
        # component).
        rep_best = \
            self._select_best_rep(\
                data_loader = data_loader,
                rep_layer = rep_layer_init,
                n_rep_per_comp = n_rep_per_comp,
                loss_reduction_type = loss_reduction_type,
                latent_lambda = latent_lambda)

        # Create a representation layer containing the best
        # representations found.
        rep_layer_best = \
            latents.RepresentationLayer(values = rep_best).to(\
                self.device)

        #-------------------------------------------------------------#
        
        # Get the optimizer for the optimization.
        optimizer = \
            self._get_optimizer(\
                optimizer_type = optimizer_type,
                optimizer_options = optimizer_options,
                optimizer_parameters = rep_layer_best.parameters())

        #-------------------------------------------------------------#

        # Get the optimized representations, the predicted means of
        # the distributions modelling the counts, the predicted
        # r-values of the distributions modelling the counts (if any),
        # and the time data.
        rep, pred_means, pred_r_values, time = \
            self._optimize_rep(\
                data_loader = data_loader,
                rep_layer = rep_layer_best,
                optimizer = optimizer,
                n_components = 1,
                n_rep_per_comp = 1,
                loss_reporting_options = loss_reporting_options,
                loss_reduction_type = loss_reduction_type,
                epochs = epochs,
                opt_num = 1,
                latent_lambda = latent_lambda)

        #-------------------------------------------------------------#

        # Make the gradients zero.
        optimizer.zero_grad()

        #-------------------------------------------------------------#

        # Return the representations, the predicted means and r-values,
        # the time data.
        return rep, pred_means, pred_r_values, time


    def _get_representations_two_opt(
            self,
            dataset: dataclasses.GeneExpressionDataset,
            config: dict[str, object]) -> \
                tuple[torch.Tensor, torch.Tensor,
                      Optional[torch.Tensor], list[tuple]]:
        """Get the best representations for a set of samples by
        initializing ``n_rep_per_comp`` representations per each
        component of the Gaussian mixture model per sample, optimizing
        these representations, selecting the best representation for
        for each sample, and optimizing these representations further.

        Parameters
        ----------
        dataset : \
            :class:`bulkdgd.core.dataclasses.GeneExpressionDataset`
            The dataset from which the data loader should be created.
    
        config : :class:`dict`
            A dictionary with the options to run the optimization.

        Returns
        -------
        rep : :class:`torch.Tensor`
            A tensor containing the optimized representations.

            This is a 2D tensor where:

            - The first dimension has a length equal to the number
              of samples.

            - The second dimension has a length equal to the
              dimensionality of the latent space where the
              representations live.

        pred_means : :class:`torch.Tensor`
            A tensor containing the predicted means of the
            distributions modelling the genes' counts.

            This is a 2D tensor where:

            - The first dimension has a length equal to the number
              of samples.

            - The second dimension has a length equal to the
              dimensionality of the gene space.

            If the genes counts are modelled using negative binomial
            distributions, the predicted means are scaled by the
            corresponding distributions' r-values.

        pred_r_values : :class:`torch.Tensor` or :obj:`None`
            A tensor containing the predicted r-values of the negative
            binomial distributions modelling the genes' counts, if
            the counts are modelled by negative binomial distributions.

            This is a 2D tensor where:

            - The first dimension has a length equal to the number
              of samples.

            - The second dimension has a length equal to the
              dimensionality of the gene space.

            ``pred_r_values`` is :obj:`None` if the counts are modelled
            by Poisson distributions.

        time_opt : :class:`list`
            A list of tuples storing, for each epoch, information
            about the CPU and wall clock time used by the entire
            epoch and by the backpropagation step run within the
            epoch.
        """

        # Get the number of samples from the length of the dataset.
        n_samples = len(dataset)

        # Get the configuration for the data loader.
        data_loader_options = config["data_loader_options"]

        # Get the number of representations per component per sample.
        n_rep_per_comp = config["n_rep_per_comp"]

        # Get the configuration for reporting the loss.
        loss_reporting_options = config["reporting_options"]["loss"]

        # Get the method to reduce the loss across the samples in the
        # batch.
        loss_reduction_type = \
            config["scheme_options"]["loss_reduction_type"]

        # Get the configuration for the first optimization.
        config_opt_1 = config["scheme_options"]["optimization_1"]

        # Get the type of optimizer for the first optimization.
        optimizer_type_1 = config_opt_1["optimizer_type"]

        # Get the options for the optimizer for the first optimization.
        optimizer_options_1 = config_opt_1["optimizer_options"]

        # Get the number of epochs to run the first optimization for.
        epochs_1 = config_opt_1["epochs"]

        #-------------------------------------------------------------#

        # Get the configuration for the second optimization.
        config_opt_2 = config["scheme_options"]["optimization_2"]

        # Get the type of optimizer for the second optimization.
        optimizer_type_2 = config_opt_2["optimizer_type"]

        # Get the options for the optimizer for the second
        # optimization.
        optimizer_options_2 = config_opt_2["optimizer_options"]

        # Get the number of epochs to run the second optimization for.
        epochs_2 = config_opt_2["epochs"]

        #-------------------------------------------------------------#

        # Create the data loader.
        data_loader = \
            _util.get_data_loader(dataset = dataset,
                                  config = data_loader_options)

        #-------------------------------------------------------------#

        # Get the initial values for the representations by sampling
        # from the latent space.

        # If the latent space is the legacy Gaussian mixture model
        if isinstance(self.latent,
                      latents.GaussianMixtureModelLegacy):

            # Sample new points from the GMM.
            rep_init = \
                self.latent.sample_new_points(\
                    n_points = n_samples, 
                    sampling_method = "mean",
                    n_samples_per_comp = n_rep_per_comp)

            # Set the lambda parameter for the GMM loss to None.
            latent_lambda = None
        
        # If the latent space is the TorchGMM wrapper
        elif isinstance(self.latent,
                        latents.GaussianMixtureModelTGMM):

            # Get the latent lambda from the configuration.
            latent_lambda = \
                config["scheme_options"][
                    "latent_loss_calculation"]["lambda"]

            # Get the number of components.
            n_components = self.latent.n_components
            
            # Get the dimensionality.
            n_dim = self.latent.dim

            # Initialize a list to store the samples from each
            # component.
            component_samples = []
            
            # For each component
            for comp_idx in range(n_components):
                
                # Sample n_samples * n_rep_per_comp points from this
                # component.
                samples_comp, _ = self.latent.sample(
                    n_samples = n_samples * n_rep_per_comp,
                    component = comp_idx)

                # Add the samples to the list.
                component_samples.append(samples_comp)
            
            # Stack all component samples: shape (n_components,
            # n_samples * n_rep_per_comp, n_dim).
            component_samples = torch.stack(component_samples,
                                            dim = 0)
            
            # Reshape to (n_components, n_samples, n_rep_per_comp,
            # n_dim).
            component_samples = component_samples.view(n_components,
                                                       n_samples,
                                                       n_rep_per_comp,
                                                       n_dim)
            
            # Permute to (n_samples, n_rep_per_comp, n_components,
            # n_dim).
            component_samples = component_samples.permute(1, 2, 0, 3)
            
            # Flatten to final shape (n_samples * n_rep_per_comp * 
            # n_components, n_dim).
            rep_init = \
                component_samples.reshape(
                    n_samples * n_rep_per_comp * n_components,
                    n_dim)

        #-------------------------------------------------------------#

        # Create the representation layer.
        rep_layer_init = \
            latents.RepresentationLayer(values = rep_init).to(\
                self.device)

        #-------------------------------------------------------------#

        # Get the optimizer for the first optimization.
        optimizer_1 = \
            self._get_optimizer(\
                optimizer_type = optimizer_type_1,
                optimizer_options = optimizer_options_1,
                optimizer_parameters = rep_layer_init.parameters())

        #-------------------------------------------------------------#

        # Get the optimized representations, the predicted means of
        # the distributions modelling the counts, the predicted
        # r-values of the distributions modelling the counts (if any),
        # and the time data.
        rep_1, _, _, time_1 = \
            self._optimize_rep(\
                data_loader = data_loader,
                rep_layer = rep_layer_init,
                optimizer = optimizer_1,
                n_components = self.latent.n_components,
                n_rep_per_comp = n_rep_per_comp,
                loss_reporting_options = loss_reporting_options,
                loss_reduction_type = loss_reduction_type,
                epochs = epochs_1,
                opt_num = 1,
                latent_lambda = latent_lambda)

        #-------------------------------------------------------------#

        # Create the representation layer.
        rep_layer_1 = \
            latents.RepresentationLayer(values = rep_1, 
                                        device = self.device)

        #-------------------------------------------------------------#

        # Make the first optimizer's gradients zero.
        optimizer_1.zero_grad()

        #-------------------------------------------------------------#

        # Select the best representation for each sample among those
        # initialized (at least one representation per sample per
        # component of the Gaussian mixture model).
        rep_best = \
            self._select_best_rep(\
                data_loader = data_loader,
                rep_layer = rep_layer_1,
                n_rep_per_comp = n_rep_per_comp,
                loss_reduction_type = loss_reduction_type,
                latent_lambda = latent_lambda)

        # Create a representation layer containing the best
        # representations found (one representation per sample).
        rep_layer_best = \
            latents.RepresentationLayer(values = rep_best,
                                        device = self.device)

        #-------------------------------------------------------------#

        # Get the optimizer for the second optimization.
        optimizer_2 = \
            self._get_optimizer(\
                optimizer_type = optimizer_type_2,
                optimizer_options = optimizer_options_2,
                optimizer_parameters = rep_layer_best.parameters())

        #-------------------------------------------------------------#

        # Get the optimized representations, the predicted means of
        # the distributions modelling the counts, the predicted
        # r-values of the distributions modelling the counts (if any),
        # and the time data.
        rep_2, pred_means_2, pred_r_values_2, time_2 = \
            self._optimize_rep(\
                data_loader = data_loader,
                rep_layer = rep_layer_best,
                optimizer = optimizer_2,
                n_components = 1,
                n_rep_per_comp = n_rep_per_comp,
                loss_reporting_options = loss_reporting_options,
                loss_reduction_type = loss_reduction_type,
                epochs = epochs_2,
                opt_num = 2,
                latent_lambda = latent_lambda)

        #-------------------------------------------------------------#

        # Make the second optimizer's gradients zero.
        optimizer_2.zero_grad()

        #-------------------------------------------------------------#

        # Concatenate the two lists storing the time data for both
        # optimizations.
        time = time_1 + time_2

        #-------------------------------------------------------------#

        # Return the representations, the predicted means and r-values,
        # the time information for both rounds of optimization.
        return rep_2, pred_means_2, pred_r_values_2, time


    def _get_saliency_map(self,
                          z: torch.Tensor) -> torch.Tensor:
        """Compute a saliency map showing the importance of each latent
        dimension for each gene.
        
        Parameters
        ----------
        z : :class:`torch.Tensor`
            A tensor containing the representations.
        
        Returns
        -------
        saliency_map : :class:`torch.Tensor`
            A 2D tensor of shape (n_genes, latent_dim) containing
            gradients indicating the importance of each latent
            dimension for each gene's expression.
        """
        
        # Get the representations (detached from the computational
        # graph).
        z_in = z.clone().detach().to(self.device).requires_grad_(True)

        #-------------------------------------------------------------#
        
        # If the output module is NB with full dispersion
        if isinstance(self.decoder.nb,
                      outputmodules.OutputModuleNBFullDispersion):
            
            # Get predicted means and dispersions.
            pred_means, _ = self.decoder(z=z_in)
        
        # Otherwise
        else:
            
            # Get the predicted means.
            pred_means = self.decoder(z = z_in)

        #-------------------------------------------------------------#
        
        # Get the number of genes.
        n_genes = pred_means.shape[1]
        
        # Initialize the saliency map: (n_genes, latent_dim).
        saliency_map = \
            torch.zeros(n_genes, self.latent.dim).to(self.device)

        #-------------------------------------------------------------#
        
        # For each gene, compute gradient of its predicted expression
        # w.r.t. representations
        for gene_idx in range(n_genes):
            
            # If there are gradients
            if z_in.grad is not None:
                
                # Zero out the gradients.
                z_in.grad.zero_()
            
            # Sum the predicted expression for this gene across all
            # samples.
            gene_output = pred_means[:, gene_idx].sum()
            
            # Compute the gradients.
            gene_output.backward(retain_graph = True)
            
            # Store the mean absolute gradient across all samples for
            # this gene.
            saliency_map[gene_idx] = z_in.grad.abs().mean(dim = 0)

        #-------------------------------------------------------------#
        
        # Clean up.
        del z_in

        #-------------------------------------------------------------#

        # Return the saliency map detached from the computational
        # graph.
        return saliency_map.detach()


    def _get_best_latent_tgmm(self,
                      rep_train: torch.Tensor,
                      latent_n_components_target: int,
                      max_iter: int,
                      is_full_refit_epoch: bool,
                      epoch: int,
                      model_selection_metric: str) -> int:
        """Select the best TGMM candidate across nearby component
        counts.

        The configured metric can be any unsupervised metric exposed
        by :mod:`bulkdgd.core.metrics`.

        Parameters
        ----------
        rep_train : :class:`torch.Tensor`
            The current representations used to fit candidate models.

        latent_n_components_target : :class:`int`
            The target (or ceiling, in dynamic mode) number of
            components.

        max_iter : :class:`int`
            The maximum number of EM iterations for each candidate
            fit.

        is_full_refit_epoch : :class:`bool`
            Whether the current epoch corresponds to a full refit.

        epoch : :class:`int`
            The current epoch.

        model_selection_metric : :class:`str`
            The metric used to rank candidates. Supported values are
            any key in
            :const:`bulkdgd.core.metrics.UNSUPERVISED_METRICS`.
        
        Returns
        -------
        best_n_components : :class:`int`
            The number of components of the best candidate model.
        """

        # Cache the latent space's dimensionality.
        latent_dim = self.latent.dim

        #-------------------------------------------------------------#

        # Set the shared arguments for initializing the candidate GMMs.
        tgmm_shared_kwargs = \
            {"covariance_type": \
                self._latent_initial_options["covariance_type"],
             "n_features" : latent_dim,
             **{opt: val for opt, val \
                in self._latent_initial_options.items()
                if opt not in ["covariance_type", "n_components"]}}

        #-------------------------------------------------------------#

        # The configured number of components is interpreted as the
        # ceiling when dynamic mode is enabled.
        gmm_n_components_ceiling = latent_n_components_target

        #-------------------------------------------------------------#

        # Clamp the current number of components between 1 and the
        # ceiling to ensure valid candidate component counts.
        current_n_components = \
            max(1, min(latent_n_components_target, self.latent.n_components))

        #-------------------------------------------------------------#

        # Initialize the candidate number of components to evaluate to
        # the current models' number of components.
        candidates_n_components = set([current_n_components])

        # If the current numner of components is higher than one
        if current_n_components > 1:

            # Add the current number of components minus one to the
            # candidates.
            candidates_n_components.add(current_n_components - 1)
        
        # If the current number of components is lower than the ceiling
        if current_n_components < gmm_n_components_ceiling:

            # Add the current number of components plus one to the
            # candidates.
            candidates_n_components.add(current_n_components + 1)
        
        # Sort the candidate number of components.
        candidates_n_components = sorted(candidates_n_components)

        #-------------------------------------------------------------#

        # Initialize an empty dictionary to store the selection values.
        candidate_selection_values = {}

        # Initialize the best selection value to None.
        best_selection_value = None

        # Initialize the best number of components to None.
        best_n_components = None

        # Initialize the best model to None.
        best_model = None

        #-------------------------------------------------------------#

        # Get the optimization direction for the selected metric.
        optimize_direction = \
            metrics.get_metric_optimization_direction(
                model_selection_metric)

        #-------------------------------------------------------------#

        # For each candidate number of components
        for candidate_n_components in candidates_n_components:

            # Initialize a candidate model.
            candidate_model = \
                latents.GaussianMixtureModelTGMM(
                    n_components = candidate_n_components,
                    device = self.device,
                    **tgmm_shared_kwargs)

            # Fit the candidate model.
            candidate_model.fit(rep_train,
                                max_iter = max_iter)

            # Get the predicted labels for the candidate model.
            predicted_labels = candidate_model.predict(rep_train)

            # Compute the model-selection metric value for the
            # candidate model.
            selection_value = metrics.get_metric_score(
                metric_name = model_selection_metric,
                X = rep_train,
                labels = predicted_labels,
                gmm_model = candidate_model)

            # Save the candidate model-selection metric value.
            candidate_selection_values[candidate_n_components] = \
                selection_value

            #---------------------------------------------------------#

            # Initialize the best model with the first valid
            # candidate as a fallback.
            if best_model is None:
                best_model = candidate_model
                best_n_components = candidate_n_components
                best_selection_value = selection_value

            # If the selection value is NaN
            if np.isnan(selection_value):

                # The current candidate is not better than the best one
                # found so far.
                is_better = False
            
            # If the current best selection value is NaN
            elif (best_selection_value is None) \
                or (np.isnan(best_selection_value)):

                # The current candidate is better than the best one
                # found so far.
                is_better = True
            
            # If both the current candidate and the best one found so
            # far are valid and the optimization direction is 'max'
            elif optimize_direction == "max":

                # The current candidate is better than the best one
                # found so far if its selection value is higher than
                # the best one.
                is_better = selection_value > best_selection_value
            
            # If both the current candidate and the best one found so
            # far are valid and the optimization direction is 'min'
            elif optimize_direction == "min":

                # The current candidate is better than the best one
                # found so far if its selection value is lower than
                # the best one.
                is_better = selection_value < best_selection_value

            #---------------------------------------------------------#

            # If the current candidate is better than the best one
            # found so far.
            if is_better:

                # Update the best number of components with the current
                # candidate.
                best_n_components = candidate_n_components

                # Update the best model with the current model.
                best_model = candidate_model

                # Update the best selection value.
                best_selection_value = selection_value

        #-------------------------------------------------------------#

        # If the best selection value is NaN or None, meaning that
        # all candidates were invalid or no valid candidate was found
        if not best_selection_value:

            # Raise an error.
            err_msg = \
                "All candidate values for '" \
                f"{model_selection_metric}' are invalid during " \
                "dynamic latent space selection."
            raise RuntimeError(err_msg)
        
        #-------------------------------------------------------------#

        # If no best model was found
        if best_model is None:

            # Raise an error.
            err_msg = \
                "No valid latent space candidate was found during " \
                "dyanmic component selection."
            raise RuntimeError(err_msg)

        #-------------------------------------------------------------#

        # Keep the best model.
        self._latent = best_model

        # Set the latent space's number of components to the best one.
        self.latent.n_components = int(best_n_components)

        #-------------------------------------------------------------#

        # Log candidate model-selection metric values and winner.
        candidate_selection_str = \
            ", ".join([
                f"number of components={k}: "
                f"{candidate_selection_values[k]:.6f}"
                for k in candidates_n_components])
        logger.info(
            f"Epoch {epoch}: selection metric " \
            f"'{model_selection_metric}' candidates "
            f"[{candidate_selection_str}] -> selected number of "
            f"components = {best_n_components} "
            f"({model_selection_metric} = {best_selection_value:.6f}, "
            f"max_iter = {max_iter}, "
            f"full_refit = {is_full_refit_epoch}).")


    def _remove_collapsed_latent_components(
            self,
            collapse_weight_threshold: float,
            epoch: int) -> bool:
        """Remove collapsed components from the current latent space.

        A component is considered collapsed if its mixture weight is
        lower than ``collapse_weight_threshold``.

        Parameters
        ----------
        weight_threshold : :class:`float`
            Threshold below which a component is considered collapsed.

        epoch : :class:`int`
            Current training epoch (used for logging).

        Returns
        -------
        removed_components : :class:`bool`
            :obj:`True` if at least one component was removed,
            :obj:`False` otherwise.
        """

        # Get the current number of components.
        n_components_before = int(self.latent.n_components)

        # If there is only one component, no removal is possible.
        if n_components_before <= 1:
            return False

        #-------------------------------------------------------------#

        # If the latent space is the TorchGMM wrapper
        if isinstance(self.latent,
                      latents.GaussianMixtureModelTGMM):

            # Get the components' mixture probabilities.
            weights = self.latent.weights.detach().clone()
        
        # If the latent space is the legacy Gaussian mixture model
        elif isinstance(self.latent,
                        latents.GaussianMixtureModelLegacy):

            # Get the components' mixture probabilities.
            weights = self.latent.get_mixture_probs().detach().clone()

        #-------------------------------------------------------------#

        # Get the mask identifying non-collapsed components.
        keep_mask = weights > float(collapse_weight_threshold)

        # Get the indexes of the components to keep.
        keep_ixs = torch.where(keep_mask)[0]

        # If all components are above threshold
        if keep_ixs.numel() == n_components_before:

            # Return False, because no removal is needed.
            return False

        #-------------------------------------------------------------#

        # If no component is above threshold
        if keep_ixs.numel() == 0:

            # Keep the strongest component.
            keep_ixs = torch.argmax(weights).view(1)

        # Get the new number of components.
        n_components_after = int(keep_ixs.numel())

        # If the effective number of components did not change
        if n_components_after == n_components_before:

            # Return False, because no removal is needed.
            return False

        #-------------------------------------------------------------#

        # If the current latent space is the TorchGMM wrapper
        if isinstance(self.latent,
                      latents.GaussianMixtureModelTGMM):

            # Slice the means for active components.
            means_new = self.latent.means[keep_ixs].detach().clone()

            #---------------------------------------------------------#

            # Slice and re-normalize the weights for active components.
            weights_new = weights[keep_ixs].detach().clone()
            weights_new = weights_new / torch.sum(weights_new)

            #---------------------------------------------------------#

            # Get the covariance type.
            covariance_type = self.latent.covariance_type

            # If the covariance type is 'full', 'diag', or 'spherical'
            if covariance_type in ("full", "diag", "spherical"):

                # Slice the covariances for active components.
                covariances_new = \
                    self.latent.covariances_[keep_ixs].detach().clone()
            
            # Otherwise
            else:

                # Keep the covariances for all components (they will be
                # re-initialized in the new model).
                covariances_new = \
                    self.latent.covariances_.detach().clone()

            #---------------------------------------------------------#

            # Get the weight concentration prior.
            weight_concentration_prior = \
                self.latent.weight_concentration_prior

            # If the weight concentration prior is a 1D tensor with one
            # value per component and the number of components matches
            # the number before removal
            if isinstance(weight_concentration_prior, torch.Tensor) \
                and weight_concentration_prior.ndim == 1 and \
                weight_concentration_prior.numel() == \
                    n_components_before:

                # Slice the weight concentration prior for the active
                # components.
                weight_concentration_prior = \
                    weight_concentration_prior[keep_ixs].detach(
                        ).clone()

            #---------------------------------------------------------#

            # Get the mean prior.
            mean_prior = self.latent.mean_prior

            # If the mean prior is a 2D tensor with one value per
            # component and the number of components matches the number
            # before removal
            if isinstance(mean_prior, torch.Tensor) and \
                mean_prior.ndim == 2 and \
                mean_prior.shape[0] == n_components_before:

                # Slice the mean prior for the active components.
                mean_prior = mean_prior[keep_ixs].detach().clone()

            #---------------------------------------------------------#

            # Get the covariance prior.
            covariance_prior = self.latent.covariance_prior

            # If the covariance prior is a tensor
            if isinstance(covariance_prior, torch.Tensor):

                # If the covariance type is 'spherical' and the
                # covariance prior is a 1D tensor with one value per
                # component and the number of components matches
                # the number before removal
                if covariance_type == "spherical" and \
                    covariance_prior.ndim == 1 and \
                    covariance_prior.numel() == n_components_before:

                    # Slice the covariance prior for the active
                    # components.
                    covariance_prior = \
                        covariance_prior[keep_ixs].detach().clone()

                # If the covariance type is 'full' or 'diag' and the
                # covariance prior is a 3D or 2D tensor with one value
                # per component and the number of components matches
                # the number before removal
                elif covariance_type in ("full", "diag") and \
                    covariance_prior.ndim > 0 and \
                    covariance_prior.shape[0] == n_components_before:

                    # Slice the covariance prior for the active
                    # components.
                    covariance_prior = \
                        covariance_prior[keep_ixs].detach().clone()

            #---------------------------------------------------------#

            # Build a new model.
            latent_new = \
                latents.GaussianMixtureModelTGMM(
                    n_components = n_components_after,
                    n_features = self.latent.n_features,
                    covariance_type = covariance_type,
                    max_iter = self.latent.max_iter,
                    tol = self.latent.tol,
                    reg_covar = self.latent.reg_covar,
                    n_init = self.latent.n_init,
                    init_means = means_new,
                    init_weights = weights_new,
                    init_covariances = covariances_new,
                    random_state = self.latent.random_state,
                    warm_start = True,
                    cem = self.latent.cem,
                    weight_concentration_prior = \
                        weight_concentration_prior,
                    mean_prior = mean_prior,
                    mean_precision_prior = \
                        self.latent.mean_precision_prior,
                    covariance_prior = covariance_prior,
                    degrees_of_freedom_prior = \
                        self.latent.degrees_of_freedom_prior,
                    verbose = self.latent.verbose,
                    verbose_interval = self.latent.verbose_interval,
                    device = self.device)

            #---------------------------------------------------------#

            # Keep fitted parameters.
            latent_new.weights_ = weights_new
            latent_new.means_ = means_new
            latent_new.covariances_ = covariances_new

            #---------------------------------------------------------#

            # If there are initial weights and their number matches the
            # number of components before removal
            if self.latent.initial_weights_ is not None and \
                self.latent.initial_weights_.numel() == \
                    n_components_before:

                # Slice the initial weights for the active components.
                latent_new.initial_weights_ = \
                    self.latent.initial_weights_[keep_ixs].detach(
                        ).clone()
            
            # Otherwise
            else:

                # Keep the initial weights for the all components.
                latent_new.initial_weights_ = \
                    weights_new.detach().clone()

            #---------------------------------------------------------#

            # If there are initial means and their number matches the
            # number of components before removal
            if self.latent.initial_means_ is not None and \
                self.latent.initial_means_.shape[0] == \
                    n_components_before:

                # Slice the initial means for the active components.
                latent_new.initial_means_ = \
                    self.latent.initial_means_[
                        keep_ixs].detach().clone()
            
            # Otherwise
            else:

                # Keep the initial means for the all components.
                latent_new.initial_means_ = means_new.detach().clone()

            #---------------------------------------------------------#

            # If there are initial covariances and their number matches
            # the number of components before removal
            if self.latent.initial_covariances_ is not None:

                # If the covariance type is 'full', 'diag', or
                # 'spherical' and the initial covariances have one
                # value per component and the number of components
                # matches the number before removal
                if covariance_type in ("full", "diag", "spherical") \
                    and self.latent.initial_covariances_.shape[0] \
                        == n_components_before:
                    
                    # Slice the initial covariances for the active
                    # components.
                    latent_new.initial_covariances_ = \
                        self.latent.initial_covariances_[
                            keep_ixs].detach().clone()
                
                # Otherwise
                else:

                    # Keep the initial covariances for the all
                    # components.
                    latent_new.initial_covariances_ = \
                        self.latent.initial_covariances_.detach(
                            ).clone()
            
            # Otherwise
            else:

                # Keep the initial covariances for the all components.
                latent_new.initial_covariances_ = \
                    covariances_new.detach().clone()
            
            #---------------------------------------------------------#

            # Keep the fit status.
            latent_new.fitted_ = self.latent.fitted_

            # Keep the convergence status.
            latent_new.converged_ = self.latent.converged_

            # Keep the number of iterations.
            latent_new.n_iter_ = self.latent.n_iter_

            # Keep the lower bound history.
            latent_new.lower_bound_ = self.latent.lower_bound_

            # Keep the best random state.
            latent_new.best_random_state_ = \
                self.latent.best_random_state_

            #---------------------------------------------------------#

            # Keep compatibility attributes.
            latent_new.dim = self.latent.dim
            latent_new.n_components = n_components_after

            #---------------------------------------------------------#

            # Replace the current latent space.
            self._latent = latent_new

        #-------------------------------------------------------------#

        # If the current latent space is the legacy Gaussian mixture
        # model
        elif isinstance(self.latent,
                        latents.GaussianMixtureModelLegacy):

            # Get the original options used to initialize the legacy
            # GMM.
            initial_options = self.latent._latent_initial_options

            # Build a new legacy GMM with fewer components.
            latent_new = \
                latents.GaussianMixtureModelLegacy(
                    dim = self.latent.dim,
                    n_components = n_components_after,
                    means_prior_name = \
                        initial_options["means_prior_name"],
                    weights_prior_name = \
                        initial_options["weights_prior_name"],
                    log_var_prior_name = \
                        initial_options["log_var_prior_name"],
                    means_prior_options = \
                        initial_options["means_prior_options"],
                    weights_prior_options = \
                        initial_options["weights_prior_options"],
                    log_var_prior_options = \
                        initial_options["log_var_prior_options"],
                    covariance_type = \
                        initial_options["covariance_type"]).to(
                            self.device)

            # Copy means, weights, and log-variance for kept
            # components.
            with torch.no_grad():
                latent_new.set_means(self.latent.means[keep_ixs])
                latent_new.set_weights(self.latent.weights[keep_ixs])
                latent_new.set_log_var(self.latent.log_var[keep_ixs])

            # Replace the current GMM.
            self._latent = latent_new

        #-------------------------------------------------------------#

        # Log the component-removal event.
        info_msg = \
            f"Epoch {epoch}: removed " \
            f"{n_components_before - n_components_after} " \
            "collapsed GMM component(s) " \
            f"(threshold = {collapse_weight_threshold:.3e}, " \
            f"n_components: {n_components_before} -> " \
            f"{n_components_after}, " \
            f"type = '{self.latent.__class__.__name__}')."
        logger.info(info_msg)

        #-------------------------------------------------------------#

        # Return that components were removed.
        return True
    

    def _save_optional_outputs(
            self,
            reporting_options: dict[str, object],
            rep_layer_train: latents.RepresentationLayer,
            rep_layer_test: latents.RepresentationLayer,
            samples_names_train: list[str],
            samples_names_test: list[str],
            epoch: int,
            genes_names: Optional[list[str]] = None,
            pathways: Optional[dict[str, list[str]]] = None,
            pathways_names: Optional[list[str]] = None) -> None:
        """Save optional outputs during training, according to the
        provided configuration.

        Parameters
        ----------
        reporting_options : :class:`dict`
            The configuration for reporting.
        
        rep_layer_train : \
            :class:`bulkdgd.core.latents.RepresentationLayer`
            The representation layer for the training samples.
        
        rep_layer_test : \
            :class:`bulkdgd.core.latents.RepresentationLayer`
            The representation layer for the test samples.
        
        samples_names_train : :class:`list` of :class:`str`
            The names of the training samples, in the same order as the
            training data.
        
        samples_names_test : :class:`list` of :class:`str`
            The names of the test samples, in the same order as the
            test data.
        
        epoch : :class:`int`
            The current epoch number (used for naming the saved
            outputs).
        """

        # Get the configuration for the representations to save at each
        # epoch.
        config_train_outputs_rep_epoch = \
            reporting_options["representations_epoch"]

        # Get whether to save the representations at each epoch.
        save_rep_epoch = config_train_outputs_rep_epoch["enabled"]
        
        # Get the stride for saving the representations at each epoch.
        save_rep_epoch_stride = \
            config_train_outputs_rep_epoch.get("stride", 1)
        
        # Get the directory for saving the representations at each
        # epoch.
        save_rep_epoch_dir = \
            config_train_outputs_rep_epoch.get("dir", None)

        #-------------------------------------------------------------#
        
        # Get the configuration for the latent probabilities to save at
        # each epoch.
        config_train_outputs_latent_probs_epoch = \
            reporting_options["latent_probs_epoch"]
        
        # Get whether to save the latent probabilities at each epoch.
        save_latent_probs_epoch = \
            config_train_outputs_latent_probs_epoch["enabled"]
        
        # Get the stride for saving the latent probabilities at each
        # epoch.
        save_latent_probs_epoch_stride = \
            config_train_outputs_latent_probs_epoch.get("stride", 1)
        
        # Get the directory for saving the latent probabilities at each
        # epoch.
        save_latent_probs_epoch_dir = \
            config_train_outputs_latent_probs_epoch.get("dir", None)

        #-------------------------------------------------------------#

        # Get the configuration for the latent means to save at each
        # epoch.
        config_train_outputs_latent_means_epoch = \
            reporting_options["latent_means_epoch"]
        
        # Get whether to save the latent means at each epoch.
        save_latent_means_epoch = \
            config_train_outputs_latent_means_epoch["enabled"]

        # Get the stride for saving the latent means at each epoch.
        save_latent_means_epoch_stride = \
            config_train_outputs_latent_means_epoch.get("stride", 1)
        
        # Get the directory for saving the latent means at each epoch.
        save_latent_means_epoch_dir = \
            config_train_outputs_latent_means_epoch.get("dir", None)

        #-------------------------------------------------------------#

        # Get the configuration for the genes' saliency maps to save
        # at each epoch.
        config_train_outputs_genes_saliency_maps_epoch = \
            reporting_options["genes_saliency_maps_epoch"]
        
        # Get whether to save the genes' saliency maps at each epoch.
        save_genes_saliency_maps_epoch = \
            config_train_outputs_genes_saliency_maps_epoch["enabled"]
        
        # Get the stride for saving the genes' saliency maps at each
        # epoch.
        save_genes_saliency_maps_epoch_stride = \
            config_train_outputs_genes_saliency_maps_epoch.get(
                "stride",
                1)
        
        # Get the directory for saving the genes' saliency maps at each
        # epoch.
        save_genes_saliency_maps_epoch_dir = \
            config_train_outputs_genes_saliency_maps_epoch.get(
                "dir",
                None)

        #-------------------------------------------------------------#

        # Get the configuration for the pathways' saliency maps to
        # save at each epoch.
        config_train_outputs_pathways_saliency_maps_epoch = \
            reporting_options["pathways_saliency_maps_epoch"]

        # Get whether to save the pathways' saliency maps at each
        # epoch.
        save_pathways_saliency_maps_epoch = \
            config_train_outputs_pathways_saliency_maps_epoch[
                "enabled"]
        
        # Get the stride for saving the pathways' saliency maps at each
        # epoch.
        save_pathways_saliency_maps_epoch_stride = \
            config_train_outputs_pathways_saliency_maps_epoch.get(
                "stride",
                1)
        
        # Get the directory for saving the pathways' saliency maps at
        # each epoch.
        save_pathways_saliency_maps_epoch_dir = \
            config_train_outputs_pathways_saliency_maps_epoch.get(
                "dir",
                None)
    
        #-------------------------------------------------------------#

        # Get the configuration for the model to save at each epoch.
        config_train_outputs_model_epoch = \
            reporting_options["model_epoch"]

        # Get whether to save the model at each epoch.
        save_model_epoch = \
            config_train_outputs_model_epoch["enabled"]

        # Get the stride for saving the model at each epoch.
        save_model_epoch_stride = \
            config_train_outputs_model_epoch.get("stride",
                                                 1)

        # Get the directory for saving the model at each epoch.
        save_model_epoch_dir = \
            config_train_outputs_model_epoch.get("dir",
                                                 None)

        #-------------------------------------------------------------#

        # If the user wants to save the model
        if save_model_epoch and (epoch % save_model_epoch_stride == 0):

            # Save the decoder's weights and the latent space's
            # parameters.
            _util.save_model_epoch(\
                epoch = epoch,
                decoder = self.decoder,
                latent = self.latent,
                save_dir = save_model_epoch_dir)

        #-------------------------------------------------------------#

        # If the user wants to save the representations
        if save_rep_epoch and (epoch % save_rep_epoch_stride == 0):

            # Save the current representations for the training
            # samples.
            _util.save_rep_epoch(\
                epoch = epoch,
                prefix = "train",
                latent_dim = self.latent.dim,
                save_dir = save_rep_epoch_dir,
                rep_layer = rep_layer_train,
                samples_names = samples_names_train)
            
            # Save the current representations for the test
            # samples.
            _util.save_rep_epoch(\
                epoch = epoch,
                prefix = "test",
                latent_dim = self.latent.dim,
                save_dir = save_rep_epoch_dir,
                rep_layer = rep_layer_test,
                samples_names = samples_names_test)

        #-------------------------------------------------------------#

        # If the user wants to save the probability densities
        if save_latent_probs_epoch and \
            (epoch % save_latent_probs_epoch_stride == 0):

            # Get the probability densities for the
            # representations of the training samples.
            probs_train = \
                self.latent.sample_probs(x = rep_layer_train())

            # Save the probability densities for the current
            # representations of the training samples.
            _util.save_latent_probs_epoch(\
                probs = probs_train,
                epoch = epoch,
                prefix = "train",
                n_components = self.latent.n_components,
                save_dir = save_latent_probs_epoch_dir,
                samples_names = samples_names_train)
            
            # Get the probability densities for the
            # representations of the test samples.
            probs_test = \
                self.latent.sample_probs(x = rep_layer_test())
            
            # Save the probability densities for the current
            # representations of the test samples.
            _util.save_latent_probs_epoch(\
                probs = probs_test,
                epoch = epoch,
                prefix = "test",
                n_components = self.latent.n_components,
                save_dir = save_latent_probs_epoch_dir,
                samples_names = samples_names_test)
            
        #-------------------------------------------------------------#

        # If the user wants to save the means of the GMM components
        if save_latent_means_epoch and \
            (epoch % save_latent_means_epoch_stride == 0):

            # Get the means of the GMM components.
            means = self.latent.means.detach().cpu().numpy()

            # Save the means of the GMM components.
            _util.save_latent_means_epoch(\
                epoch = epoch,
                means = means,
                latent_dim = self.latent.dim,
                n_components = self.latent.n_components,
                save_dir = save_latent_means_epoch_dir)

        #-------------------------------------------------------------#

        # If the user wants to save the saliency maps
        if (save_genes_saliency_maps_epoch and \
                (epoch % \
                    save_genes_saliency_maps_epoch_stride == 0)) \
            or (save_pathways_saliency_maps_epoch and \
                (epoch % \
                    save_pathways_saliency_maps_epoch_stride == 0)):
            
            # Get the saliency map for the training samples.
            saliency_map_train = \
                self._get_saliency_map(\
                    z = rep_layer_train())
            
            # Get the saliency map for the test samples.
            saliency_map_test = \
                self._get_saliency_map(\
                    z = rep_layer_test())
            
            # If the user wants to save the saliency maps for
            # the genes
            if save_genes_saliency_maps_epoch and \
                (epoch % \
                    save_genes_saliency_maps_epoch_stride == 0):
                
                # Save the saliency maps for the training samples.
                _util.save_genes_saliency_maps_epoch(\
                    saliency_map = saliency_map_train,
                    epoch = epoch,
                    prefix = "train",
                    genes_names = genes_names,
                    save_dir = save_genes_saliency_maps_epoch_dir)
                
                # Save the saliency maps for the test samples.
                _util.save_genes_saliency_maps_epoch(\
                    saliency_map = saliency_map_test,
                    epoch = epoch,
                    prefix = "test",
                    genes_names = genes_names,
                    save_dir = save_genes_saliency_maps_epoch_dir)
            
            # If the user wants to save the saliency maps for
            # the pathways
            if save_pathways_saliency_maps_epoch and \
                (epoch % \
                    save_pathways_saliency_maps_epoch_stride == 0):
                    
                # Get the saliency maps for the pathways
                # in the training samples.
                saliency_pathways_train = \
                    _util.get_pathways_saliency_map(
                        saliency_map = saliency_map_train,
                        pathways = pathways,
                        genes_names = genes_names)

                # Get the saliency maps for the pathways
                # in the test samples.
                saliency_pathways_test = \
                    _util.get_pathways_saliency_map(
                        saliency_map = saliency_map_test,
                        pathways = pathways,
                        genes_names = genes_names)

                # Save the saliency maps for the training samples.
                _util.save_pathways_saliency_maps_epoch(\
                    saliency_map = saliency_pathways_train,
                    epoch = epoch,
                    prefix = "train",
                    pathways_names = pathways_names,
                    save_dir = \
                        save_pathways_saliency_maps_epoch_dir)

                # Save the saliency maps for the test samples.
                _util.save_pathways_saliency_maps_epoch(\
                    saliency_map = saliency_pathways_test,
                    epoch = epoch,
                    prefix = "test",
                    pathways_names = pathways_names,
                    save_dir = \
                        save_pathways_saliency_maps_epoch_dir)


    def _train(self,
               config_train: dict[str, object],
               samples_names_train: list[str],
               samples_names_test: list[str],
               genes_names: list[str],
               data_loader_train: torch.utils.data.DataLoader,
               data_loader_test: torch.utils.data.DataLoader,
               rep_layer_train: latents.RepresentationLayer,
               rep_layer_test: latents.RepresentationLayer,
               pathways: Optional[dict[str, list[str]]] = None,
               labels_train: Optional[str] = None,
               labels_test: Optional[str] = None) -> \
                tuple[tuple[torch.Tensor, torch.Tensor],
                      tuple[torch.Tensor, torch.Tensor],
                      Optional[torch.Tensor],
                      list[float],
                      dict[str, float],
                      dict[str, float],
                      list[tuple[float, float]]]:
        """Train the model.

        Parameters
        ----------
        config_train : :class:`dict`
            A dictionary of options for the training.

        samples_names_train : :class:`list`
            A list of the training samples' names.
        
        samples_names_test : :class:`list`
            A list of the testing samples' names.

        genes_names : :class:`list`
            A list of the genes' names.
        
        data_loader_train : :class:`torch.utils.data.DataLoader`
            The data loader for the training samples.
        
        data_loader_test : :class:`torch.utils.data.DataLoader`
            The data loader for the test samples.
        
        rep_layer_train : \
            :class:`bulkdgd.core.latents.RepresentationLayer`
            The representation layer for the training samples.
        
        rep_layer_test : \
            :class:`bulkdgd.core.latents.RepresentationLayer`
            The representation layer for the test samples.
        
        pathways : :class:`dict` or :obj:`None`
            A dictionary where the keys are the names of the pathways
            and the values are lists of genes belonging to each
            pathway.
        
        labels_train : :class:`torch.Tensor` or :obj:`None`
            The clusters' labels for the training samples.
        
        labels_test : :class:`torch.Tensor` or :obj:`None`
            The clusters' labels for the test samples.

        Returns
        -------
        A tuple containing:

            - A :class:`tuple` containing:
        
                - A :class:`torch.Tensor` containing the
                  representations for the training samples.

                - A :class:`torch.Tensor` containing the
                  representations for the test samples.

            - A :class:`tuple` containing:

                - A :class:`torch.Tensor` containing the predicted
                means for the training samples.
                
                - A :class:`torch.Tensor` containing the predicted
                means for the test samples.

        If the output module is
        :class:`bulkdgd.core.outputmodules.OutputModuleNBFeatureDispersion`,
        the tuple will also contain:

            - A :class:`torch.Tensor` containing the predicted r-values
              for all samples.

        If the output module is
        :class:`bulkdgd.core.outputmodules.OutputModuleNBFullDispersion`,
        the tuple will instead contain:

        - A :class:`tuple` containing:

            - A :class:`torch.Tensor` containing the predicted r-values
              for the training samples.

            - A :class:`torch.Tensor` containing the predicted r-values
              for the test samples.
        
        The :class:`tuple` will always also contain:

        - A :class:`list` containing the losses for the training
          and test samples (GMM loss, reconstruction loss, and
          total loss) for each epoch.
        
        - A :class:`pandas.DataFrame` containing data about the CPU
          and wall clock time used by each training epoch (and
          backpropagation step within each epoch).
        """

        # Parse and check the configuration.
        config_train, errors, warnings = \
            _util.parse_config_train(config = config_train)
        
        # If there are errors in the configuration
        if errors:

            # Raise an exception with the error messages.
            error_msg = \
                "Errors in the training configuration: " + \
                "|".join(errors)
            raise ValueError(error_msg)
        
        # If there are warnings in the configuration
        if warnings:

            # Log the warning messages.
            warning_msg = \
                "Warnings in the training configuration: " + \
                "|".join(warnings)
            logger.warning(warning_msg)

        #-------------------------------------------------------------#

        # Get the number of training samples.
        n_samples_train = len(samples_names_train)

        # Get the number of testing samples.
        n_samples_test = len(samples_names_test)
        
        # Get the number of genes.
        n_genes = len(genes_names)
        
        # If a dictionary of pathways is provided
        if pathways is not None:
            
            # Get the names of the pathways.
            pathways_names = list(pathways.keys())
        
        # Otherwise
        else:

            # Set the pathways names to None.
            pathways_names = None

        #-------------------------------------------------------------#

        # Get the number of epochs.
        n_epochs = config_train["n_epochs"]

        # Get the method of loss reduction.
        loss_reduction_type = config_train["loss_reduction_type"]

        #-------------------------------------------------------------#

        # Get the options for training the latent space.
        latent_options = config_train["latent_training_options"]

        # Get the options for training the decoder.
        decoder_options = config_train["decoder_training_options"]

        # Get the options for training the representations.
        representations_options = \
            config_train["representations_training_options"]

        # Get the options for reporting.
        reporting_options = config_train["reporting_options"]

        #-------------------------------------------------------------#

        # Get the norms to which the gradients are clipped before each
        # optimizer takes its step. If not set, they are not clipped.
        grad_clip_decoder = \
            decoder_options.get("grad_clipping_max_norm")
        grad_clip_rep = \
            representations_options.get("grad_clipping_max_norm")
        grad_clip_latent = \
            latent_options.get("grad_clipping_max_norm")

        #-------------------------------------------------------------#
        
        # Initialize the learning rate scheduler for the latent space.
        lr_scheduler_latent = None

        # If the latent space is the legacy Gaussian mixture model
        if isinstance(self.latent,
                      latents.GaussianMixtureModelLegacy):
            
            # Get the type of optimizer to use for the latent space.
            optimizer_latent_type = \
                latent_options["optimizer_type"]
            
            # Get the options for the optimizer for the latent space.
            optimizer_latent_options = \
                latent_options["optimizer_options"]
            
            # Get the optimizer for the latent space.
            optimizer_latent = \
                self._get_optimizer(\
                    optimizer_type = optimizer_latent_type,
                    optimizer_options = optimizer_latent_options,
                    optimizer_parameters = self.latent.parameters())
            
            # Get the type of learning rate scheduler to use for the
            # latent space.
            lr_scheduler_latent_type = \
                latent_options["lr_scheduler_type"]
            
            # Get the options for the learning rate scheduler for the
            # latent space.
            lr_scheduler_latent_options = \
                latent_options["lr_scheduler_options"]

            # Get the learning rate scheduler for the latent space.
            lr_scheduler_latent = \
                self._get_scheduler(
                    lr_scheduler_target = "latent",
                    lr_scheduler_type = lr_scheduler_latent_type,
                    lr_scheduler_options = lr_scheduler_latent_options,
                    optimizer = optimizer_latent,
                    n_epochs = n_epochs,
                    data_loader_train = data_loader_train)

        #-------------------------------------------------------------#

        # Get the type of optimizer to use for the decoder.
        optimizer_decoder_type = \
            decoder_options["optimizer_type"]

        # Get the options for the optimizer for the decoder.
        optimizer_decoder_options = \
            decoder_options["optimizer_options"]

        # Get the optimizer for the decoder.
        optimizer_decoder = \
            self._get_optimizer(\
                optimizer_type = optimizer_decoder_type,
                optimizer_options = optimizer_decoder_options,
                optimizer_parameters = self.decoder.parameters())

        # Get the type of learning rate scheduler to use for the
        # decoder.
        lr_scheduler_decoder_type = \
            decoder_options["lr_scheduler_type"]

        # Get the options for the learning rate scheduler for the
        # decoder.
        lr_scheduler_decoder_options = \
            decoder_options["lr_scheduler_options"]
        
        # Get the learning rate scheduler for the decoder.
        lr_scheduler_decoder = \
            self._get_scheduler(
                lr_scheduler_target = "decoder",
                lr_scheduler_type = lr_scheduler_decoder_type,
                lr_scheduler_options = lr_scheduler_decoder_options,
                optimizer = optimizer_decoder,
                n_epochs = n_epochs,
                data_loader_train = data_loader_train)

        #-------------------------------------------------------------#

        # Get the type of optimizer to use for the representations for
        # the training samples.
        optimizer_rep_type = \
            representations_options["optimizer_type"]
        
        # Get the options for the optimizer for the representations for
        # the training samples.
        optimizer_rep_options = \
            representations_options["optimizer_options"]

        # Get the type of learning rate scheduler to use for the
        # training representations.
        lr_scheduler_rep_type = \
            representations_options["lr_scheduler_type"]
       
        # Get the options for the learning rate scheduler for the
        # training representations.
        lr_scheduler_rep_options = \
            representations_options["lr_scheduler_options"]

        #-------------------------------------------------------------#

        # Get the optimizer for the representations for the training
        # samples.
        optimizer_rep_train = \
            self._get_optimizer(\
                optimizer_type = optimizer_rep_type,
                optimizer_options = optimizer_rep_options,
                optimizer_parameters = rep_layer_train.parameters())
        
        # Get the learning rate scheduler for the training
        # representations.
        lr_scheduler_rep_train = \
            self._get_scheduler(
                lr_scheduler_target = "representations",
                lr_scheduler_type = lr_scheduler_rep_type,
                lr_scheduler_options = lr_scheduler_rep_options,
                optimizer = optimizer_rep_train,
                n_epochs = n_epochs)

        #-------------------------------------------------------------#

        # Get the optimizer for the representations for the testing
        # samples.
        optimizer_rep_test = \
            self._get_optimizer(\
                optimizer_type = optimizer_rep_type,
                optimizer_options = optimizer_rep_options,
                optimizer_parameters = rep_layer_test.parameters())

        # Get the learning rate scheduler for the test representations.
        lr_scheduler_rep_test = \
            self._get_scheduler(
                lr_scheduler_target = "representations",
                lr_scheduler_type = lr_scheduler_rep_type,
                lr_scheduler_options = lr_scheduler_rep_options,
                optimizer = optimizer_rep_test,
                n_epochs = n_epochs)
            
        #-------------------------------------------------------------#

        # Get the type of noise to inject in the representations for
        # the training samples.
        train_noise_type = \
            representations_options["train_noise_type"]

        # Get the noise options for the representations for the
        # training samples.
        train_noise_options = \
            representations_options["train_noise_options"]

        # If the noise if Gaussian
        if train_noise_type == "gaussian":

            # Get the noise scale. If it is 0.0 (the default), no noise
            # will be injected.
            train_noise_scale_base = train_noise_options["scale"]

            # Get the starting noise scale (for cosine annealing).
            train_noise_start = train_noise_options["start"]

            # Get the ending noise scale (for cosine annealing).
            train_noise_end = train_noise_options["end"]
            
            # Get the percentage of training samples within 2 standard
            # deviations. 
            train_noise_within_radius_prob = \
                train_noise_options["within_radius_prob"]
            
            # Get the gain factor.
            train_noise_gain = train_noise_options["gain"]    

        #-------------------------------------------------------------#

        # Get the type of early stopping to perform from the
        # configuration.
        early_stopping_type = config_train["early_stopping_type"]

        # Get the early stopping options from the configuration.
        early_stopping_options = config_train["early_stopping_options"]

        # Initialize the early stopping active flag to False.
        early_stopping_active = False

        # If the early stopping is based on the loss
        if early_stopping_type == "loss":

            # The patience (number of epochs without improvement in test
            # loss before stopping).
            early_stopping_patience = \
                early_stopping_options.get("patience", 10)

            # Initialize the best test loss to positive infinity.
            early_stopping_best_test_loss = float("inf")
            
            # Initialize the number of the epoch with the best test
            # loss to zero.
            early_stopping_best_epoch = 0
            
            # Initialize the number of epochs without improvement in
            # the test loss to zero.
            early_stopping_epochs_without_improvement = 0
            
            # Initialize the state of the best modelto None.
            early_stopping_best_model_state = None

            # Early stopping should only be active after the GMM is
            # fitted (to avoid false triggers during the initial
            # epochs).
            early_stopping_active = False

        #-------------------------------------------------------------#

        # Create an empty list to store the loss for the
        # Gaussian mixture model, the reconstruction loss, and the
        # overall loss.
        losses_list = []

        # Create an empty list to store the training time.
        time_train = []

        #-------------------------------------------------------------#

        # Get the type of component removal to perform.
        components_removal_type = \
            latent_options["components_removal_type"]

        # If the component removal is based on a weight threshold  
        if components_removal_type == "weight_threshold":

            # Get the weight threshold for collapsed-component removal.
            component_weight_threshold = \
                latent_options["components_removal_options"][
                        "threshold"]

            # Log the selected behavior.
            info_msg = \
                "Weight-threshold collapsed-component removal is " \
                "enabled " \
                f"(threshold = {component_weight_threshold})."
            logger.info(info_msg)

        #-------------------------------------------------------------#

        # If the Gaussian mixture model is the TorchGMM wrapper
        if isinstance(self.latent,
                      latents.GaussianMixtureModelTGMM):

            # Get the type of model selection to perform for the
            # Gaussian mixture model.
            latent_model_selection_type = \
                latent_options["model_selection_type"]
                
            # Get the options for GMM model selection.
            latent_model_selection_options = \
                latent_options.get("model_selection_options")

            # If the model selection is based on a metric
            if latent_model_selection_type == "metric" and \
                latent_model_selection_options is not None:

                # Get the metric used to select candidate models.
                latent_model_selection_metric = \
                    latent_model_selection_options.get("metric", "bic")
            
            #---------------------------------------------------------#

            # Get the options for fitting.
            latent_options_fitting = latent_options["fitting"]

            # Get the first epoch after which the Gaussian
            # mixture model is fitted.
            latent_first_epoch = latent_options_fitting["first_epoch"]

            # Get the refit interval for the Gaussian mixture
            # model.
            latent_refit_interval = \
                latent_options_fitting["refit_interval"]

            # Get whether to refit the GMM at the end of training.
            gmm_refit_final = \
                latent_options_fitting["refit_final"]
            
            # Get the number of EM iterations to perform at the
            # 'first epoch' for the Gaussian mixture model.
            latent_max_iter_first_epoch = \
                latent_options_fitting["max_iter_first_epoch"]
            
            # Get the maximum number of EM iterations to perform at
            # a full-refit epoch that is not 'first epoch'
            # (i.e., epochs that are multiples of
            # 'refit_interval').
            latent_max_iter_full_refit = \
                latent_options_fitting["max_iter_full_refit"]
            
            # Get the maximum number of EM iterations to perform at
            # a "warm"-refit epoch (i.e., at epochs later than the
            # first epoch that are not full-refit epochs).
            latent_max_iter_warm_refit = \
                latent_options_fitting["max_iter_warm_refit"]
                
            # Get the maximum number of EM iterations to perform at
            # the final refit (if 'final_refit' is True).
            latent_max_iter_final_refit = \
                latent_options_fitting["max_iter_final_refit"]

            # Get the options for the calculation of the loss.
            latent_lambda = \
                latent_options["loss_calculation"]["lambda"]

        #-------------------------------------------------------------#
    
        # Get the methods that will be used to normalize the losses.
        loss_norm_types = {
            "latent": \
                reporting_options["loss"]["latent"]["norm_type"],
            "decoder": \
                reporting_options["loss"]["decoder"]["norm_type"],
            "total": \
                reporting_options["loss"]["total"]["norm_type"],
        }
        
        #-------------------------------------------------------------#

        # Reporting metrics logged and exported per epoch.
        reporting_metrics_latent = \
            reporting_options["metrics"]["latent"]

        # If there are reporting metrics
        if reporting_metrics_latent:
            
            # Initialize a list to store the metrics that will be used.
            filtered_metrics = []

            # For each of the reporting metrics
            for m in reporting_metrics_latent:
                
                # If the metric is a supervised metric and no labels
                # were provided
                if m in metrics.SUPERVISED_METRICS:

                    # If no labels were provided for training or test
                    if labels_train is None or labels_test is None:

                        # Log the warning.
                        logger.warning(
                            f"Supervised metric '{m}' requires "
                            "ground-truth "
                            "labels, which were not provided. This "
                            "metric will not be used for reporting.")

                        # Move to the next metric.
                        continue

                # Add the metric to the filtered list   .
                filtered_metrics.append(m)

            # Replace the reporting metrics with the filtered ones.
            reporting_metrics_latent = filtered_metrics

        # If there are still reporting metrics to use
        if reporting_metrics_latent:

            # Create a string containing the metrics.
            metrics_str = \
                ", ".join([f"'{m}'" for m in reporting_metrics_latent])
            
            # Log the selected reporting metrics. 
            logger.info(
                "Selected reporting metrics for the latent space: " \
                f"{metrics_str}.")

        # Create lists to store per-epoch train/test metrics rows.
        metrics_rows_train = []
        metrics_rows_test = []

        # Set a flag for when to start storing values for the latent
        # metrics.
        latent_metrics_active = False

        # If the latent space is the legacy GMM
        if isinstance(self.latent,
                      latents.GaussianMixtureModelLegacy):

            # The latent metrics will be active from the first epoch.
            latent_metrics_active = True

            # If early stopping is enabled, activate it from the
            # first epoch.
            if early_stopping_type == "loss":
                early_stopping_active = True

        #-------------------------------------------------------------#

        # For each epoch
        for epoch in range(1, n_epochs + 1):

            # Initialize the losses to 0 for the current epoch.
            losses_list.append([epoch, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

            # Initialize per-epoch metrics rows with NaNs.
            metrics_row_train = {"epoch": epoch}
            metrics_row_test = {"epoch": epoch}

            # Pre-populate all configured metrics with NaN so that
            # every epoch has a complete schema even when some metrics
            # are unavailable (for example, supervised metrics without
            # labels).

            # For each metric
            for metric_name in reporting_metrics_latent:

                # Set the metric value to NaN for the train set.
                metrics_row_train[metric_name] = np.nan

                # Set the metric value to NaN for the test set.
                metrics_row_test[metric_name] = np.nan

            #---------------------------------------------------------#

            # Mark the CPU start time of the epoch.
            time_start_epoch_cpu = time.process_time()

            # Mark the wall clock start time of the epoch.
            time_start_epoch_wall = time.time()

            #---------------------------------------------------------#

            # If the Gaussian mixture model is the TorchGMM wrapper
            if isinstance(self.latent,
                          latents.GaussianMixtureModelTGMM):

                # Get the target number of components (when dynamic
                # mode is disabled) and the maximum number of
                # components (when dynamic mode is enabled).
                latent_n_components_target = \
                    int(latent_options.get("n_components",
                                           self.latent.n_components))

                #-----------------------------------------------------#
            
                # Get the representations for the training samples.
                rep_train = rep_layer_train().detach().to(self.device)

                #-----------------------------------------------------#

                # If we are at the first epoch
                if epoch == 1:

                    # Disable gradient computation.
                    with torch.no_grad():

                        # Allocate the parameters of the Gaussian
                        # mixture model.
                        self.latent._allocate_parameters(rep_train)

                        # Set the 'fitted_' attribute of the Gaussian
                        # mixture model to True.
                        self.latent.fitted_ = True

                #-----------------------------------------------------#

                # If we are at or after the first epoch where the GMM
                # should be fitted.
                elif epoch >= latent_first_epoch:

                    # Disable gradient computation.
                    with torch.no_grad():

                        # Determine whether this is a full-refit epoch.
                        is_full_refit_epoch = \
                            epoch == latent_first_epoch or \
                            (latent_refit_interval \
                                and epoch % latent_refit_interval == 0)

                        # Set the maximum number of iterations for
                        # this epoch's fitting step.

                        # If we are at the first epoch where the GMM
                        # should be fitted
                        if epoch == latent_first_epoch:

                            # Use the configured number of iterations
                            # for the first epoch.
                            max_iter = latent_max_iter_first_epoch
                        
                        # If we are at a full refit epoch
                        elif is_full_refit_epoch:

                            # Use the configured number of iterations
                            # for a full refit.
                            max_iter = latent_max_iter_full_refit
                        
                        # Otherwise
                        else:

                            # Use the configured number of iterations
                            # for a "warm" refit.
                            max_iter = latent_max_iter_warm_refit
                        
                        #---------------------------------------------#

                        # If the model selection is based on a metric
                        if latent_model_selection_type == "metric":
                            
                            # Compare k-1, k, and k+1 via the
                            # configured selection metric and keep the
                            # best model.
                            self._get_best_latent_tgmm(
                                rep_train = rep_train,
                                latent_n_components_target = \
                                    latent_n_components_target,
                                max_iter = max_iter,
                                is_full_refit_epoch = \
                                    is_full_refit_epoch,
                                epoch = epoch,
                                model_selection_metric = \
                                    latent_model_selection_metric)

                        #---------------------------------------------#

                        # Otherwise.
                        else:

                            # If it is a full refit epoch
                            if is_full_refit_epoch:

                                # Fit the GMM.
                                self.latent.fit(rep_train,
                                                max_iter = max_iter)
                            
                            # If it is not a full refit epoch
                            else:

                                # Fit the GMM with warm start
                                # to continue from the previous epoch's
                                # solution.
                                self.latent.fit(rep_train,
                                                max_iter = max_iter,
                                                warm_start = True)
    
                        #---------------------------------------------#

                        # If the weight-threshold removal of collapsed
                        # components is enabled
                        if components_removal_type == \
                            "weight_threshold":

                            # Remove collapsed components.
                            self._remove_collapsed_latent_components(
                                collapse_weight_threshold =  \
                                    component_weight_threshold,
                                epoch = epoch)

                    #-------------------------------------------------#

                    # If early stopping is enabled and we are at the
                    # first epoch after which the GMM is fitted
                    if early_stopping_type == "loss" \
                        and epoch == latent_first_epoch:

                        # Enable early stopping.
                        early_stopping_active = True

                        # Set the best test loss to positive infinity.
                        early_stopping_best_test_loss = float("inf")

                        # Set the number of the epochs without
                        # improvement to zero.
                        early_stopping_epochs_without_improvement = 0

                #-----------------------------------------------------#

                # Calculate clustering metrics after the GMM is fitted.
                if epoch >= latent_first_epoch:

                    # Enable the calculation of latent metrics from
                    # this epoch onward.
                    latent_metrics_active = True

            #---------------------------------------------------------#

            # If the noise to inject in the training representations is
            # Gaussian
            if train_noise_type == "gaussian":

                # If noise injection is enabled for the training
                # representations
                if train_noise_scale_base > 0:
                    
                    # Get the noise progress.
                    progress = (epoch - 1) / max(n_epochs - 1, 1)
                    
                    # Compute the noise scale using cosine annealing
                    # between the start and end values.
                    train_noise_scale = \
                        train_noise_end + \
                            (train_noise_start - train_noise_end) * \
                                0.5 * \
                                    (1 + math.cos(math.pi * progress))
                    
                    # Set the noise scale.
                    train_noise_scale = \
                        train_noise_scale * train_noise_scale_base
                
                # Otherwise
                else:
                    
                    # No noise will be injected.
                    train_noise_scale = 0.0

            # Otherwise (the noise type is not Gaussian, e.g. it is
            # disabled/None)
            else:

                # No noise will be injected.
                train_noise_scale = 0.0

            #=========================================================#
            #                      TRAINING PHASE                     #
            #=========================================================#

            # Make the gradients of the representation layer for the
            # training samples zero.
            optimizer_rep_train.zero_grad()

            #---------------------------------------------------------#

            # Set the decoder in train mode.
            self.decoder.train()

            #---------------------------------------------------------#

            # For each batch of training samples
            for batch_data in data_loader_train:

                # Unpack the batch data (the data loader may
                # return 3 or 4 items depending on whether
                # labels are available).
                samples_exp = batch_data[0]
                samples_mean_exp = batch_data[1]
                samples_ixs = batch_data[2]

                # Move the gene expression of the samples to the
                # correct device.
                samples_exp = samples_exp.to(self.device)

                # Move the mean gene expression of the samples to
                # the correct device.
                samples_mean_exp = samples_mean_exp.to(self.device)

                #-----------------------------------------------------#

                # Make the gradients for the decoder zero.
                optimizer_decoder.zero_grad()

                #-----------------------------------------------------#

                # Get the representations for the current samples.
                z = rep_layer_train(ixs = samples_ixs).to(self.device)

                #-----------------------------------------------------#

                # If noise injection is enabled
                if train_noise_scale > 0:
                    
                    # Get the radius of the hypersphere within which
                    # the specified fraction of samples lie.
                    radius = \
                        float(
                            chi2.ppf(train_noise_within_radius_prob,
                                     self.latent.dim)) ** 0.5
                    
                    # Get the base noise.
                    base_noise = torch.randn_like(z) / radius
                    
                    # Scale the base noise to get the final noise to
                    # inject.
                    noise = \
                        train_noise_scale * base_noise * \
                            train_noise_gain
                    
                    # Add the noise.
                    z = z + noise

                #-----------------------------------------------------#

                # If the Gaussian mixture model is the legacy one
                if isinstance(self.latent,
                              latents.GaussianMixtureModelLegacy):

                    # Make the gradients for the Gaussian mixture
                    # model zero.
                    optimizer_latent.zero_grad()
                    
                    # If the loss reduction type is 'sum'
                    if loss_reduction_type == "sum":

                        # Get the Gaussian mixture model's loss.
                        latent_loss = self.latent(x = z).sum()
                    
                    # If the loss reduction type is 'mean'
                    elif loss_reduction_type == "mean":

                        # Get the Gaussian mixture model's loss.
                        latent_loss = self.latent(x = z).mean()
                
                # If the Gaussian mixture model is the TGMM wrapper
                elif isinstance(self.latent,
                                latents.GaussianMixtureModelTGMM):

                    # If we are at any epoch after the
                    # Gaussian mixture model was fitted
                    if epoch >= latent_first_epoch:
                        
                        # If the loss reduction type is 'sum'
                        if loss_reduction_type == "sum":

                            # Get the Gaussian mixture model's loss.
                            latent_loss = \
                                - latent_lambda * \
                                    torch.sum(\
                                        self.latent.log_prob(z))
                        
                        # If the loss reduction type is 'mean'
                        elif loss_reduction_type == "mean":

                            # Get the Gaussian mixture model's loss.
                            latent_loss = \
                                - latent_lambda * \
                                    torch.mean(\
                                        self.latent.log_prob(z))

                    # If we are not at the first epoch after the
                    # Gaussian mixture model was fitted
                    else:
                        
                        # Set the Gaussian mixture model's loss to
                        # zero.
                        latent_loss = \
                            torch.tensor(0.0).to(self.device)

                #-----------------------------------------------------#

                # If the chosen output module means that the
                # r-values are not learned
                if isinstance(\
                    self.decoder.nb,
                    (outputmodules.OutputModuleNBFeatureDispersion,
                     outputmodules.OutputModulePoisson)):
                    
                    # Get the predicted means of the distributions
                    # modelling the genes' counts.
                    #
                    # The output is a 2D tensor with:
                    #
                    # - 1st dimension: the number of samples in the
                    #                  current batch times the
                    #                  number of components in the
                    #                  Gaussian mixture model times
                    #                  the number of
                    #                  representations taken per
                    #                  component per sample ->
                    #                  'n_samples_in_batch' *
                    #                  'n_components' *
                    #                  'n_rep_per_comp'
                    #
                    # - 2nd dimension: the dimensionality of the
                    #                  output (= gene) space ->
                    #                  'n_genes'
                    pred_means = self.decoder(z = z)

                    # Set the options to compute the reconstruction
                    # loss.
                    recon_loss_options = \
                        {"obs_counts" : samples_exp,
                         "pred_means" : pred_means,
                         "scaling_factors" : samples_mean_exp}

                # If the chosen output module means that the
                # r-values are learned
                elif isinstance(\
                    self.decoder.nb,
                    outputmodules.OutputModuleNBFullDispersion):

                    # Get the predicted means and r-values of the
                    # negative binomials.
                    #
                    # Both outputs are 2D tensors with:
                    #
                    # - 1st dimension: the number of samples in the
                    #                  current batch times the
                    #                  number of components in the
                    #                  Gaussian mixture model times
                    #                  the number of
                    #                  representations taken per
                    #                  component per sample ->
                    #                  'n_samples_in_batch' *
                    #                  'n_components' *
                    #                  'n_rep_per_comp'
                    #
                    # - 2nd dimension: the dimensionality of the
                    #                  output (= gene) space ->
                    #                  'n_genes'
                    pred_means, pred_log_r_values = self.decoder(z = z)

                    # Set the options to compute the reconstruction
                    # loss.
                    recon_loss_options = \
                        {"obs_counts" : samples_exp,
                         "pred_means" : pred_means,
                         "pred_log_r_values" : pred_log_r_values,
                         "scaling_factors" : samples_mean_exp}

                #-----------------------------------------------------#
                
                # If the loss reduction type is 'sum'
                if loss_reduction_type == "sum":

                    # Get the reconstruction loss.
                    recon_loss = \
                        self.decoder.nb.loss(
                            **recon_loss_options).sum()
                
                # If the loss reduction type is 'mean'
                elif loss_reduction_type == "mean":

                    # Get the reconstruction loss.
                    recon_loss = \
                        self.decoder.nb.loss(
                            **recon_loss_options).mean()

                #-----------------------------------------------------#

                # If the Gaussian mixture model is the legacy one
                if isinstance(self.latent,
                              latents.GaussianMixtureModelLegacy):

                    # Get the overall loss.
                    loss = latent_loss.clone() + recon_loss.clone()
                
                # If the Gaussian mixture model is the TGMM wrapper
                elif isinstance(self.latent,
                                latents.GaussianMixtureModelTGMM):
                    
                    # If we are at any epoch after the Gaussian mixture
                    # model was fitted
                    if epoch >= latent_first_epoch:
                        
                        # The overall loss is the sum of the
                        # reconstruction loss and the Gaussian
                        # mixture model's loss.
                        loss = latent_loss.clone() + recon_loss.clone()

                    # If we are not at the first epoch after the
                    # Gaussian mixture model was fitted
                    else:
                        
                        # The overall loss is just the
                        # reconstruction loss (the GMM loss is not
                        # active yet).
                        loss = recon_loss.clone()

                #-----------------------------------------------------#

                # Backpropagate the loss.
                loss.backward()

                #-----------------------------------------------------#

                # Clip the gradients, if a maximum norm was set for
                # them, before any optimizer steps on them.
                clip_grads(optimizer = optimizer_decoder,
                           max_norm = grad_clip_decoder)
                clip_grads(optimizer = optimizer_rep_train,
                           max_norm = grad_clip_rep)

                #-----------------------------------------------------#

                # If the Gaussian mixture model is the legacy one
                if isinstance(self.latent,
                              latents.GaussianMixtureModelLegacy):

                    # Clip the Gaussian mixture model's gradients, too.
                    clip_grads(optimizer = optimizer_latent,
                               max_norm = grad_clip_latent)

                    # Take a step with the optimizer for the
                    # Gaussian mixture model.
                    optimizer_latent.step()

                #-----------------------------------------------------#

                # Take a step with the optimizer for the decoder.
                optimizer_decoder.step()

                #-----------------------------------------------------#

                # If the learning rate scheduler for the latent space
                # is defined and the latent space is the legacy GMM
                if lr_scheduler_latent is not None and \
                    isinstance(self.latent,
                               latents.GaussianMixtureModelLegacy):
                    
                    # Take a step with the scheduler.
                    lr_scheduler_latent.step()

                # If the learning rate scheduler for the decoder is
                # defined
                if lr_scheduler_decoder is not None:

                    # Take a step with the scheduler.
                    lr_scheduler_decoder.step()

                #-----------------------------------------------------#

                # Get the loss for the Gaussian mixture model for
                # the current epoch.
                latent_loss_epoch = \
                    _util.normalize_loss(\
                        loss = latent_loss.item(),
                        loss_type = "latent",
                        loss_norm_type = \
                            loss_norm_types["latent"],
                        loss_norm_options = \
                            {"n_samples" : n_samples_train,
                             "latent_dim" : self.latent.dim})

                # Get the reconstruction loss for the current
                # epoch.
                recon_loss_epoch = \
                    _util.normalize_loss(\
                        loss = recon_loss.item(),
                        loss_type = "decoder",
                        loss_norm_type = \
                            loss_norm_types["decoder"],
                        loss_norm_options = \
                            {"n_samples" : n_samples_train,
                             "n_genes" : n_genes})

                # Get the overall loss for the current epoch.
                loss_epoch = \
                    _util.normalize_loss(\
                        loss = loss.item(),
                        loss_type = "total",
                        loss_norm_type = \
                            loss_norm_types["total"],
                        loss_norm_options = \
                            {"n_samples" : n_samples_train,
                             "n_genes" : n_genes})

                #-----------------------------------------------------#

                # If the Gaussian mixture model is the legacy one
                if isinstance(self.latent,
                              latents.GaussianMixtureModelLegacy):

                    # Update the losses list.
                    losses_list[-1][1] += latent_loss_epoch

                # If the Gaussian mixture model is the TGMM wrapper
                elif isinstance(self.latent,
                                latents.GaussianMixtureModelTGMM):
                    
                    # If we are at any epoch after the Gaussian mixture
                    # model was fitted
                    if epoch >= latent_first_epoch:
                        
                        # Update the losses list.
                        losses_list[-1][1] += latent_loss_epoch
                
                # Update the losses list with the reconstruction loss.
                losses_list[-1][2] += recon_loss_epoch
                
                # Update the losses list with the overall loss.
                losses_list[-1][3] += loss_epoch

            #---------------------------------------------------------#

            # Take a step with the optimizer for the
            # representations.
            optimizer_rep_train.step()

            #---------------------------------------------------------#

            # If the learning rate scheduler for the representations is
            # defined
            if lr_scheduler_rep_train is not None:

                # Take a step with the scheduler.
                lr_scheduler_rep_train.step()

            #=========================================================#
            #                      TESTING PHASE                      #
            #=========================================================#

            # Make the gradients of the representation layer for the
            # testing samples zero.
            optimizer_rep_test.zero_grad()

            #---------------------------------------------------------#

            # Set the decoder in eval mode.
            self.decoder.eval()

            # Store the gradient computation status for all parameters
            # of the decoder.
            dec_requires_grad = \
                [p.requires_grad for p in self.decoder.parameters()]
            
            # For each parameter of the decoder
            for p in self.decoder.parameters():

                # Disable gradient computation.
                p.requires_grad_(False)

            #---------------------------------------------------------#

            # If the latent space is the legacy GMM
            if isinstance(self.latent,
                          latents.GaussianMixtureModelLegacy):

                # Store the gradient computation status for all
                # parameters.
                latent_requires_grad = \
                    [p.requires_grad for p in self.latent.parameters()]
                
                # For each parameter
                for p in self.latent.parameters():

                    # Disable gradient computation.
                    p.requires_grad_(False)

            # If the Gaussian mixture model is the TGMM wrapper
            else:

                # No GMM parameter gradients are tracked in this
                # branch.
                latent_requires_grad = None

            #---------------------------------------------------------#

            # For each batch of testing samples
            for batch_data in data_loader_test:

                # Unpack the batch data (the data loader may
                # return 3 or 4 items depending on whether
                # labels are available).
                samples_exp = batch_data[0]
                samples_mean_exp = batch_data[1]
                samples_ixs = batch_data[2]

                # Move the gene expression of the samples to the
                # correct device.
                samples_exp = samples_exp.to(self.device)

                # Move the mean gene expression of the samples to
                # the correct device.
                samples_mean_exp = samples_mean_exp.to(self.device)

                #-----------------------------------------------------#

                # Get the representations for the current samples.
                z = rep_layer_test(ixs = samples_ixs).to(self.device)

                #-----------------------------------------------------#

                # If the Gaussian mixture model is the legacy one
                if isinstance(self.latent,
                              latents.GaussianMixtureModelLegacy):
                    
                    # If the loss reduction type is 'sum'
                    if loss_reduction_type == "sum":

                        # Get the Gaussian mixture model's loss.
                        latent_loss = self.latent(x = z).sum()
                    
                    # If the loss reduction type is 'mean'
                    elif loss_reduction_type == "mean":
                        
                        # Get the Gaussian mixture model's loss.
                        latent_loss = self.latent(x = z).mean()
                
                # If the Gaussian mixture model is the TGMM wrapper
                elif isinstance(self.latent,
                                latents.GaussianMixtureModelTGMM):

                    # If we are at the first epoch after the
                    # Gaussian mixture model was fitted
                    if epoch >= latent_first_epoch:
                        
                        # If the loss reduction type is 'sum'
                        if loss_reduction_type == "sum":

                            # Get the Gaussian mixture model's loss.
                            latent_loss = \
                                - latent_lambda * \
                                    torch.sum(\
                                        self.latent.log_prob(z))
                        
                        # If the loss reduction type is 'mean'
                        elif loss_reduction_type == "mean":

                            # Get the Gaussian mixture model's loss.
                            latent_loss = \
                                - latent_lambda * \
                                    torch.mean(\
                                        self.latent.log_prob(z))

                    # If we are not at the first epoch after the
                    # Gaussian mixture model was fitted
                    else:
                        
                        # Set the Gaussian mixture model's loss to
                        # zero.
                        latent_loss = \
                            torch.tensor(0.0).to(self.device)

                #-----------------------------------------------------#

                # If the chosen output module means that the
                # r-values are not learned
                if isinstance(\
                    self.decoder.nb,
                    (outputmodules.OutputModuleNBFeatureDispersion,
                     outputmodules.OutputModulePoisson)):
                    
                    # Get the predicted means of the distributions
                    # modelling the genes' counts.
                    #
                    # The output is a 2D tensor with:
                    #
                    # - 1st dimension: the number of samples in the
                    #                  current batch times the
                    #                  number of components in the
                    #                  Gaussian mixture model times
                    #                  the number of
                    #                  representations taken per
                    #                  component per sample ->
                    #                  'n_samples_in_batch' *
                    #                  'n_components' *
                    #                  'n_rep_per_comp'
                    #
                    # - 2nd dimension: the dimensionality of the
                    #                  output (= gene) space ->
                    #                  'n_genes'
                    pred_means = self.decoder(z = z)

                    # Set the options to compute the reconstruction
                    # loss.
                    recon_loss_options = \
                        {"obs_counts" : samples_exp,
                         "pred_means" : pred_means,
                         "scaling_factors" : samples_mean_exp}

                # If the chosen output module means that the
                # r-values are learned
                elif isinstance(\
                    self.decoder.nb,
                    outputmodules.OutputModuleNBFullDispersion):

                    # Get the predicted means and r-values of the
                    # negative binomials.
                    #
                    # Both outputs are 2D tensors with:
                    #
                    # - 1st dimension: the number of samples in the
                    #                  current batch times the
                    #                  number of components in the
                    #                  Gaussian mixture model times
                    #                  the number of
                    #                  representations taken per
                    #                  component per sample ->
                    #                  'n_samples_in_batch' *
                    #                  'n_components' *
                    #                  'n_rep_per_comp'
                    #
                    # - 2nd dimension: the dimensionality of the
                    #                  output (= gene) space ->
                    #                  'n_genes'
                    pred_means, pred_log_r_values = self.decoder(z = z)

                    # Set the options to compute the reconstruction
                    # loss.
                    recon_loss_options = \
                        {"obs_counts" : samples_exp,
                         "pred_means" : pred_means,
                         "pred_log_r_values" : pred_log_r_values,
                         "scaling_factors" : samples_mean_exp}

                #-----------------------------------------------------#
                
                # If the loss reduction type is 'sum'
                if loss_reduction_type == "sum":

                    # Get the reconstruction loss.
                    recon_loss = \
                        self.decoder.nb.loss(
                            **recon_loss_options).sum()
                
                # If the loss reduction type is 'mean'
                elif loss_reduction_type == "mean":
                    
                    # Get the reconstruction loss.
                    recon_loss = \
                        self.decoder.nb.loss(
                            **recon_loss_options).mean()

                #-----------------------------------------------------#

                # If the Gaussian mixture model is the legacy one
                if isinstance(self.latent,
                              latents.GaussianMixtureModelLegacy):

                    # Get the overall loss.
                    loss = latent_loss.clone() + recon_loss.clone()
                
                # If the Gaussian mixture model is the TGMM wrapper
                elif isinstance(self.latent,
                                latents.GaussianMixtureModelTGMM):
                    
                    # If we are at any epoch after the Gaussian mixture
                    # model was fitted
                    if epoch >= latent_first_epoch:
                        
                        # The overall loss is the sum of the
                        # reconstruction loss and the Gaussian
                        # mixture model's loss.
                        loss = latent_loss.clone() + recon_loss.clone()

                    # If we are not at the first epoch after the
                    # Gaussian mixture model was fitted
                    else:
                        
                        # The overall loss is just the
                        # reconstruction loss (the GMM loss is not
                        # active yet).
                        loss = recon_loss.clone()

                #-----------------------------------------------------#

                # Backpropagate the loss.
                loss.backward()

                #-----------------------------------------------------#

                # Get the loss for the Gaussian mixture model for
                # the current epoch.
                latent_loss_epoch = \
                    _util.normalize_loss(\
                        loss = latent_loss.item(),
                        loss_type = "latent",
                        loss_norm_type = \
                            loss_norm_types["latent"],
                        loss_norm_options = \
                            {"n_samples" : n_samples_test,
                             "latent_dim" : self.latent.dim})

                # Get the reconstruction loss for the current
                # epoch.
                recon_loss_epoch = \
                    _util.normalize_loss(\
                        loss = recon_loss.item(),
                        loss_type = "decoder",
                        loss_norm_type = \
                            loss_norm_types["decoder"],
                        loss_norm_options = \
                            {"n_samples" : n_samples_test,
                             "n_genes" : n_genes})

                # Get the overall loss for the current epoch.
                loss_epoch = \
                    _util.normalize_loss(\
                        loss = loss.item(),
                        loss_type = "total",
                        loss_norm_type = \
                            loss_norm_types["total"],
                        loss_norm_options = \
                            {"n_samples" : n_samples_test,
                             "n_genes" : n_genes})

                #-----------------------------------------------------#

                # If the Gaussian mixture model is the legacy one
                if isinstance(self.latent,
                              latents.GaussianMixtureModelLegacy):

                    # Update the losses list.
                    losses_list[-1][4] += latent_loss_epoch

                # If the Gaussian mixture model is the TGMM wrapper
                elif isinstance(self.latent,
                                latents.GaussianMixtureModelTGMM):
                    
                    # If we are at any epoch after the Gaussian mixture
                    # model was fitted
                    if epoch >= latent_first_epoch:
                        
                        # Update the losses list.
                        losses_list[-1][4] += latent_loss_epoch
                
                # Update the losses list with the reconstruction loss.
                losses_list[-1][5] += recon_loss_epoch
                
                # Update the losses list with the overall loss.
                losses_list[-1][6] += loss_epoch

            #---------------------------------------------------------#

            # For each parameter of the decoder
            for p, req_grad in zip(self.decoder.parameters(),
                                   dec_requires_grad):
                
                # Restore the original 'requires_grad' setting.
                p.requires_grad_(req_grad)

            # If the latent space is the legacy GMM
            if latent_requires_grad is not None:

                # For each parameter of the latent space
                for p, req_grad in zip(self.latent.parameters(),
                                       latent_requires_grad):
                    
                    # Restore the original 'requires_grad' setting.
                    p.requires_grad_(req_grad)

            #---------------------------------------------------------#

            # Take a step with the optimizer for the test 
            # representations.
            optimizer_rep_test.step()

            #---------------------------------------------------------#

            # If the test representations were optimized and
            # the learning rate scheduler is defined
            if lr_scheduler_rep_test is not None:

                # Take a step with the scheduler.
                lr_scheduler_rep_test.step()

            #---------------------------------------------------------#

            # If the current GMM is the legacy one and
            # collapsed-component weight-threshold removal is enabled
            if isinstance(self.latent,
                          latents.GaussianMixtureModelLegacy) and \
                components_removal_type == "weight_threshold":

                # Remove collapsed components, if any.
                removed_components = \
                    self._remove_collapsed_latent_components(
                        collapse_weight_threshold = component_weight_threshold,
                        epoch = epoch)

                # If some components were removed, recreate the
                # optimizer so it points to the current GMM
                # parameters.
                if removed_components:
                    optimizer_latent = \
                        self._get_optimizer(
                            optimizer_type = optimizer_latent_type,
                            optimizer_options = \
                                optimizer_latent_options,
                            optimizer_parameters = \
                                self.latent.parameters())

            #=========================================================#
            #                      LOGGING PHASE                      #
            #=========================================================#

            # Mark the CPU end time of the epoch.
            time_end_epoch_cpu = time.process_time()

            # Mark the wall clock end time of the epoch.
            time_end_epoch_wall = time.time()

            # Get the total CPU time used by the epoch.
            time_tot_epoch_cpu = \
                time_end_epoch_cpu - time_start_epoch_cpu

            # Get the total wall clock time used by the epoch.
            time_tot_epoch_wall = \
                time_end_epoch_wall - time_start_epoch_wall

            # Add all the total times to the list storing them for
            # all epochs.
            time_train.append(\
                (epoch, time_tot_epoch_cpu, time_tot_epoch_wall))

            # Inform the user about the loss at the current epoch
            # and the CPU time/wall clock time elapsed.
            info_msg = \
                f"Epoch {epoch}: loss train " \
                f"{losses_list[-1][3]:.3f}, loss test " \
                f"{losses_list[-1][6]:.3f}, epoch total CPU time " \
                f"{time_tot_epoch_cpu:.3f} s, epoch " \
                f"total wall clock time {time_tot_epoch_wall:.3f} s"
            
            # If the learning rate scheduler for the latent space is
            # enabled and the latent space is the legacy GMM
            if lr_scheduler_latent is not None and \
                isinstance(self.latent,
                           latents.GaussianMixtureModelLegacy):
                
                # Get the learning rate for the latent space.
                lr_latent = optimizer_latent.param_groups[0]["lr"]

                # Add it to the log string.
                info_msg += f", LR: latent={lr_latent:.2e}"

            # If the learning rate scheduler for the decoder is
            # enabled
            if lr_scheduler_decoder is not None:
                
                # Get the learning rate for the decoder.
                lr_decoder = optimizer_decoder.param_groups[0]["lr"]
                
                # Add it to the log string.
                info_msg += f", LR: decoder={lr_decoder:.2e}"
            
            # If the learning rate scheduler for the representations
            # for the training samples
            if lr_scheduler_rep_train is not None:
                
                # Get the learning rate for the representations for the
                # training samples.
                lr_rep = optimizer_rep_train.param_groups[0]["lr"]
                
                # Add it to the log string.
                info_msg += f", LR: rep_train={lr_rep:.2e}"

            # If the noise is Gaussian
            if train_noise_type == "gaussian":

                # If noise injection is enabled
                if train_noise_scale_base > 0:

                    # Add the noise scale to the log string.
                    info_msg += \
                        f", noise scale {train_noise_scale:.6f}"

            # Add a period at the end of the log string and log it.
            info_msg += "."
            logger.info(info_msg)

            #---------------------------------------------------------# 

            # If latent metrics are active, compute and log them.
            if latent_metrics_active:
                
                # Log the clustering metrics.
                metrics_parts = [f"Epoch {epoch}:"]

                # Disable gradient computation.
                with torch.no_grad():

                    # Get the representations for the training
                    # samples.
                    rep_train = \
                        rep_layer_train().detach().to(self.device)

                    # Get the predicted labels for the training
                    # samples.
                    predicted_labels_train = \
                        self.latent.predict(
                            rep_train).detach().cpu().numpy()

                    # Get the representations for the test
                    # samples.
                    rep_test = \
                        rep_layer_test().detach().to(self.device)

                    # Get the predicted labels for the test
                    # samples.
                    predicted_labels_test = \
                        self.latent.predict(
                            rep_test).detach().cpu().numpy()

                    # Compute configured reporting metrics.
                    for metric_name in reporting_metrics_latent:

                        # Unsupervised metrics use representations
                        # and predicted labels.
                        if metric_name in \
                            metrics.UNSUPERVISED_METRICS:

                            # Compute the metric for the train data.
                            value_train = \
                                metrics.get_metric_score(
                                    metric_name = metric_name,
                                    X = rep_train,
                                    gmm_model = self.latent,
                                    labels = \
                                        predicted_labels_train)
                            
                            # Store the metric value for the train
                            # data.
                            metrics_row_train[metric_name] = \
                                value_train

                            # Compute the metric for the test data.
                            value_test = \
                                metrics.get_metric_score(
                                    metric_name = metric_name,
                                    X = rep_test,
                                    gmm_model = self.latent,
                                    labels = \
                                        predicted_labels_test)
                            
                            # Store the metric value for the test
                            # data.
                            metrics_row_test[metric_name] = \
                                value_test

                        # Supervised metrics require ground-truth
                        # labels; leave NaN if unavailable.
                        elif metric_name in \
                            metrics.SUPERVISED_METRICS:

                            # If labels are available
                            if labels_train is not None and \
                                labels_test is not None:

                                # Encode the ground-truth labels as
                                # integers.
                                enc_true_labels_train, \
                                    enc_true_labels_test = \
                                        metrics.encode_labels(
                                            [labels_train,
                                             labels_test])
                                
                                # Encode the predicted labels as
                                # integers.
                                enc_predicted_labels_train, \
                                    enc_predicted_labels_test = \
                                        metrics.encode_labels(
                                            [predicted_labels_train,
                                             predicted_labels_test])

                                # Compute the metric for the train
                                # data.
                                value_train = \
                                    metrics.get_metric_score(
                                        metric_name = metric_name,
                                        y_true = \
                                            enc_true_labels_train,
                                        y_pred = \
                                            enc_predicted_labels_train)

                                # Store the metric value for the train
                                # data.
                                metrics_row_train[metric_name] = \
                                    value_train

                                # Compute the metric for the test
                                # data.
                                value_test = \
                                    metrics.get_metric_score(
                                        metric_name = metric_name,
                                        y_true = \
                                            enc_true_labels_test,
                                        y_pred = \
                                            enc_predicted_labels_test)
                                
                                # Store the metric value for the test
                                # data.
                                metrics_row_test[metric_name] = \
                                    value_test

                        # Convert the train value to a pretty string,
                        # keeping 'nan' for invalid values.
                        train_value = metrics_row_train[metric_name]
                        try:
                            train_str = \
                                f"{train_value:.4f}" \
                                if train_value is not None and \
                                    np.isfinite(float(train_value)) \
                                else "nan"
                        except (TypeError, ValueError):
                            train_str = "nan"

                        # Convert the test value to a pretty string,
                        # keeping 'nan' for invalid values.
                        test_value = metrics_row_test[metric_name]
                        try:
                            test_str = \
                                f"{test_value:.4f}" \
                                if test_value is not None \
                                    and np.isfinite(float(test_value)) \
                                else "nan"
                        except (TypeError, ValueError):
                            test_str = "nan"

                        # Append metric summary.
                        metrics_parts.append(
                            f"{metric_name}: train={train_str}, "
                            f"test={test_str}")
            
                #-----------------------------------------------------#

                # Emit a single log line per epoch.
                logger.info(" ".join(metrics_parts) + ".")

            #=========================================================#
            #                      SAVING PHASE                       #
            #=========================================================#

            # Save the optional outputs.
            self._save_optional_outputs(
                reporting_options = \
                    reporting_options["optional_outputs"],
                rep_layer_train = rep_layer_train,
                rep_layer_test = rep_layer_test,
                samples_names_train = samples_names_train,
                samples_names_test = samples_names_test,
                epoch = epoch,
                genes_names = genes_names,
                pathways = pathways,
                pathways_names = pathways_names)

            #=========================================================#
            #                 EARLY STOPPING PHASE                    #
            #=========================================================#

            # Save per-epoch metrics rows.
            metrics_rows_train.append(metrics_row_train)
            metrics_rows_test.append(metrics_row_test)

            # If early stopping is active
            if early_stopping_active:

                # Get the current test loss.
                current_test_loss = losses_list[-1][6]

                # If the current test loss is better than the best
                # test loss so far
                if current_test_loss < early_stopping_best_test_loss:

                    # Update the best test loss.
                    early_stopping_best_test_loss = current_test_loss

                    # Update the best epoch.
                    early_stopping_best_epoch = epoch

                    # Reset the counter for epochs without
                    # improvement.
                    early_stopping_epochs_without_improvement = 0

                    # Save the best model state.
                    early_stopping_best_model_state = {
                        "decoder": \
                            copy.deepcopy(\
                                self.decoder.state_dict()),
                        "rep_layer_train": \
                            copy.deepcopy(\
                                rep_layer_train.state_dict()),
                        "rep_layer_test": \
                            copy.deepcopy(\
                                rep_layer_test.state_dict()),
                        "latent": \
                            copy.deepcopy(\
                                self.latent.state_dict()),
                    }

                # Otherwise
                else:

                    # Increment the counter.
                    early_stopping_epochs_without_improvement += 1

                # If the patience has been exhausted
                if early_stopping_epochs_without_improvement \
                    >= early_stopping_patience:

                    # Log the event.
                    info_msg = \
                        f"Early stopping triggered at epoch " \
                        f"{epoch}. Best test loss " \
                        f"{early_stopping_best_test_loss:.3f} " \
                        "was at epoch " \
                        f"{early_stopping_best_epoch}."
                    logger.info(info_msg)

                    # Stop training.
                    break

        #-------------------------------------------------------------#

        # If early stopping was used and a best model state was saved
        if early_stopping_type == "loss" \
            and early_stopping_best_model_state is not None:

            # Restore the best model state.
            self.decoder.load_state_dict(\
                early_stopping_best_model_state["decoder"])
            rep_layer_train.load_state_dict(\
                early_stopping_best_model_state["rep_layer_train"])
            rep_layer_test.load_state_dict(\
                early_stopping_best_model_state["rep_layer_test"])
            self.latent.load_state_dict(\
                early_stopping_best_model_state["latent"])

            # Inform the user.
            info_msg = \
                f"Restored best model state from epoch " \
                f"{early_stopping_best_epoch}."
            logger.info(info_msg)

        #=============================================================#
        #                        RETURN PHASE                         #
        #=============================================================#
            
        # If the Gaussian mixture model is the TGMM wrapper and a final
        # refitting of the Gaussian mixture model is needed after
        # training (it is going to happen even if there was early
        # stopping, since the refitting is done after training)
        if isinstance(self.latent,
                      latents.GaussianMixtureModelTGMM) \
            and gmm_refit_final:
            
            # Disable gradient computation.
            with torch.no_grad():

                # Get the representations for all the training samples.
                rep_train = rep_layer_train().detach().to(self.device)

                # Fit the Gaussian mixture model to the final
                # representations of the training samples.
                self.latent.fit(rep_train,
                                max_iter = latent_max_iter_final_refit)

                # If the weight-threshold removal of collapsed
                # components is enabled
                if components_removal_type == "weight_threshold":

                    # Remove the collapsed components, if any.
                    self._remove_collapsed_latent_components(
                        collapse_weight_threshold = \
                            component_weight_threshold,
                        epoch = epoch)

        #-------------------------------------------------------------#

        # Get the final representations for the training samples.
        rep_train = rep_layer_train()

        # Get the final representations for the test samples.
        rep_test = rep_layer_test()

        #-------------------------------------------------------------#

        # If the genes' counts are modelled by negative binomial
        # distributions whose r-values are learned per gene (but not
        # per sample)
        if isinstance(self.decoder.nb,
                      outputmodules.OutputModuleNBFeatureDispersion):

            # Get the predicted scaled means for the training samples.
            means_final_train = self.decoder(z = rep_train)

            # Get the predicted scaled means for the test samples.
            means_final_test = self.decoder(z = rep_test)

            # Get the r-values for the training samples.
            r_values_final = \
                torch.exp(self.decoder.nb.log_r).squeeze().detach()

            # Return the representations, decoder's outputs, losses,
            # and training time.
            return ((rep_train, rep_test),
                    (means_final_train, means_final_test),
                    r_values_final,
                    losses_list,
                    (metrics_rows_train, metrics_rows_test),
                    time_train)

        #-------------------------------------------------------------#
        
        # If the genes' counts are modelled by negative binomial
        # distributions whose r-values are learned per gene per sample
        elif isinstance(self.decoder.nb,
                        outputmodules.OutputModuleNBFullDispersion):

            # Get the predicted scaled means and log r-values for the
            # training samples.
            means_final_train, log_r_values_final_train = \
                self.decoder(z = rep_train)

            # Get the r-values for the training samples.
            r_values_final_train = \
                torch.exp(\
                    log_r_values_final_train).squeeze().detach()

            # Get the predicted scaled means and log r-values for the
            # test samples.
            means_final_test, log_r_values_final_test = \
                self.decoder(z = rep_test)
            
            # Get the r-values for the test samples.
            r_values_final_test = \
                torch.exp(\
                    log_r_values_final_test).squeeze().detach()

            # Return the representations, decoder's outputs, losses,
            # and training time.
            return ((rep_train, rep_test),
                    (means_final_train, means_final_test),
                    (r_values_final_train, r_values_final_test),
                    losses_list,
                    (metrics_rows_train, metrics_rows_test),
                    time_train)

        #-------------------------------------------------------------#

        # If the genes' counts are modelled by Poisson distributions
        elif isinstance(self.decoder.nb,
                        outputmodules.OutputModulePoisson):

            # Get the predicted scaled means for the training samples.
            means_final_train = self.decoder(z = rep_train)

            # Get the predicted scaled means for the test samples.
            means_final_test = self.decoder(z = rep_test)

            # The r-values will be None.
            r_values_final = None

            # Return the representations, decoder's outputs, losses,
            # and training time.
            return ((rep_train, rep_test),
                    (means_final_train, means_final_test),
                    r_values_final,
                    losses_list,
                    (metrics_rows_train, metrics_rows_test),
                    time_train)


    ######################### PUBLIC METHODS #########################


    @staticmethod
    def rescale_pred_means(df_pred_means: pd.DataFrame,
                           df_pred_r_values: pd.DataFrame) -> \
                            pd.DataFrame:
        """Rescale the means of the negative binomials modeling
        the genes' counts.

        Parameters
        ----------
        df_pred_means : :class:`pandas.DataFrame`
            A data frame containing the predicted scaled means of
            the negative binomials modeling the genes' counts.

            Here, each row contains the scaled mean for a given
            representation/sample, and the columns contain either the
            values of the scaled means or additional information.

            The columns containing the scaled means must be
            named after the corresponding genes' Ensembl IDs.

        df_pred_r_values : :class:`pandas.DataFrame`
            A data frame containing the predicted r-values of
            the negative binomials modeling the genes' counts.

            Here, each row contains the r-value for a given
            representation/sample, and the columns contain either the
            r-values or additional information.

            The columns containing the r-values must be
            named after the corresponding genes' Ensembl IDs.
        
        Returns
        -------
        df_scaled : :class:`pandas.DataFrame`
            A data frame containing the predicted means.

            It contains the same columns of the ``df_pred_means`` data
            frame, in the same order they appear in the
            ``df_pred_means`` data frame.

            However, the values in the columns containing the
            predicted means are scaled back by the corresponding
            r-values.
        """

        # Get whether the rows' names of the two input data frames
        # are identical.
        index_equal = \
            (df_pred_means.index == df_pred_r_values.index).all()

        # If they are not identical
        if not index_equal:

            # Raise an error.
            err_msg = \
                "The names of the rows of the 'df_pred_means' and " \
                "'df_pred_r_values' data frames must be identical."
            raise ValueError(err_msg)

        #-------------------------------------------------------------#

        # Get whether the columns' names of the two input data frames
        # are identical.
        columns_equal = \
            (df_pred_means.columns == df_pred_r_values.columns).all()

        # If they are not identical
        if not columns_equal:

            # Raise an error.
            err_msg = \
                "The names of the columns of the 'df_pred_means' " \
                "and 'df_pred_r_values' data frames must be identical."
            raise ValueError(err_msg)

        #-------------------------------------------------------------#

        # Get the names of the columns containing gene expression
        # data from the data frame with the means.
        genes_columns = \
            [col for col in df_pred_means.columns \
             if col.startswith("ENSG")]

        # Create a data frame with only those columns containing gene
        # expression data.
        df_pred_means_data = df_pred_means.loc[:,genes_columns]

        # Create a data frame with only those columns containing gene
        # expression data.
        df_pred_r_values_data = df_pred_r_values.loc[:,genes_columns]

        #-------------------------------------------------------------#

        # Get the names of the other columns.
        other_columns = \
            [col for col in df_pred_means.columns \
             if col not in genes_columns]

        # Create a data frame with only those columns containing
        # additional information.
        df_other_data = df_pred_means.loc[:,other_columns]

        #-------------------------------------------------------------#

        # Rescale the means.
        df_final_means_data = \
            df_pred_means_data * df_pred_r_values_data

        #-------------------------------------------------------------#

        # Make a new data frame with the scaled means.
        df_final_means = \
            pd.concat([df_final_means_data, df_other_data],
                      axis = 1)

        # Re-order the columns in the original order.
        df_final_means = df_final_means[df_pred_means.columns.tolist()]

        #-------------------------------------------------------------#

        # Return the new data frame
        return df_final_means


    def get_representations(self,
                            df_samples: pd.DataFrame,
                            config_rep: dict[str, object],
                            get_saliency_map: bool = False) -> \
                                tuple[pd.DataFrame, pd.DataFrame,
                                      Optional[pd.DataFrame],
                                      pd.DataFrame]:
        """Find the best representations for a set of samples.

        Parameters
        ----------
        df_samples : :class:`pandas.DataFrame`
            A data frame containing the samples.

        config_rep : :class:`dict`
            A dictionary of options for the optimization(s). It varies
            according to the selected ``method``.

            The supported options for all available methods can be
            found :doc:`here <../rep_config_options>`.

        get_saliency_map : :class:`bool`, optional
            Whether to also compute and return the saliency maps
            showing the importance of each latent dimension for each
            gene's expression based on the obtained representations.
            Default: ``False``.

        Returns
        -------
        df_rep : :class:`pandas.DataFrame`
            A data frame containing the representations.

            Here, each row contains a representation and the
            columns contain either the values of the representations'
            along the latent space's dimensions or additional
            information about the input samples found in the
            input data frame. Columns containing additional
            information, if present in the input data frame, will
            appear last in the data frame.

        df_pred_means : :class:`pandas.DataFrame`
            A data frame containing the predicted means of the
            distributions modelling the genes' counts for the
            representations found.

            Here, each row contains the predicted means for a
            given representation, and the columns contain either the
            mean of a distribution or additional information about the
            input samples found in the input data frame. Columns
            containing additional information, if present in the input
            data frame, will appear last in the data frame.

            If the genes counts are modelled using negative binomial
            distributions, the predicted means are scaled by the
            corresponding distributions' r-values.

        df_pred_r_values : :class:`pandas.DataFrame`, optional
            A data frame containing the predicted r-values of the
            negative binomials for the representations found, if the
            genes' counts are modelled by negative binomial
            distributions

            Here, each row contains the predicted r-values for a given
            representation, and the columns contain either the
            r-value of a negative binomial or additional information
            about the input samples found in the input
            data frame. Columns containing additional
            information, if present in the input data frame, will
            appear last in the data frame.

            ``df_pred_r_values`` is :obj:`None` if the genes' counts
            are modelled by Poisson distributions.

        df_time : :class:`pandas.DataFrame`
            A data frame containing data about the CPU and wall
            clock time used by each epoch (and backpropagation
            step within each epoch) in each optimization step.

            Here, each row represents an epoch of an optimization
            step, and the columns contain data about the platform
            where the calculation was run, the number of CPU threads
            used by the computation, and the CPU and wall clock
            time used by the entire epoch and by the backpropagation
            step run inside it.

        df_saliency_map : :class:`pandas.DataFrame`, optional
            A data frame containing the gradients indicating the
            importance of each latent dimension for each gene's
            expression.
            
            Here, each row is a gene (indexed by ENSG naming) and
            columns  correspond to each latent dimension.
            Returned as an element of a tuple uniquely when
            ``get_saliency_map`` is ``True``.
        """

        # Get the columns containing gene expression data.
        genes_columns = \
            [col for col in df_samples.columns \
             if col.startswith("ENSG")]

        # Get a data frame with only the columns containing gene
        # expression data.
        df_expr_data = df_samples[genes_columns]

        #-------------------------------------------------------------#

        # Get the other columns.
        other_columns = \
            [col for col in df_samples.columns \
             if col not in genes_columns]

        # Get a data frame with only the columns containing additional
        # data.
        df_other_data = df_samples[other_columns]

        #-------------------------------------------------------------#

        # Create the dataset.
        dataset = dataclasses.GeneExpressionDataset(df = df_expr_data)

        #-------------------------------------------------------------#

        # Get the names/IDs/indexes of the samples from the data
        # frame's rows' names.
        samples_names = df_expr_data.index.tolist()

        # Get the names of the genes from the expression data frame's
        # columns' names.
        genes_names = df_expr_data.columns

        #-------------------------------------------------------------#

        # Check the configuration for finding representations.
        config_rep, errors, warnings = \
            _util.parse_config_rep(config = config_rep)

        # If there were errors while checking the configuration
        if errors:

            # Make a string containing all the errors.
            errors_str = "\n".join([str(e) for e in errors])

            # Raise an error.
            raise ValueError(
                f"Configuration errors found: {errors_str}")

        #-------------------------------------------------------------#

        # Get the optimization scheme from the configuration.
        opt_scheme = config_rep["scheme_type"]

        #-------------------------------------------------------------#

        # If the user selected the one-optimization scheme
        if opt_scheme == "one_opt":

            # Select the corresponding method.
            opt_method = self._get_representations_one_opt

        # If the user selected the two-optimizations scheme
        elif opt_scheme == "two_opt":

            # Select the corresponding method.
            opt_method = self._get_representations_two_opt

        #-------------------------------------------------------------#
            
        # Get the representations, the corresponding predicted means
        # of the distributions, the r-values of the distributions (if
        # any), and the time data.
        rep, pred_means, pred_r_values, time_opt = \
            opt_method(dataset = dataset,
                       config = config_rep)

        #-------------------------------------------------------------#

        # Generate the final data frames.
        df_rep, df_pred_means, df_pred_r_values, df_time = \
            _util.get_final_data_frames_rep(\
                rep = rep,
                pred_means = pred_means,
                pred_r_values = pred_r_values,
                time_opt = time_opt,
                samples_names = samples_names,
                genes_names = genes_names)

        #-------------------------------------------------------------#

        # Add the extra data found in the input data frame to the
        # representations' data frame.
        df_rep = pd.concat([df_rep, df_other_data],
                           axis = 1)

        # Add the extra data found in the input data frame to the
        # predicted scaled means' data frame.
        df_pred_means = pd.concat([df_pred_means, df_other_data],
                                  axis = 1)

        # If there is a data frame containing the predicted r-values
        if df_pred_r_values is not None:

            # Add the extra data found in the input data frame to the
            # predicted r-values' data frame.
            df_pred_r_values = \
                pd.concat([df_pred_r_values, df_other_data],
                          axis = 1)

        #-------------------------------------------------------------#
        
        # If saliency maps are requested
        if get_saliency_map:
            
            # Compute the saliency map.
            saliency_tensor = self._get_saliency_map(z = rep)
            
            # Create a dataframe.
            latent_cols = \
                [f"latent_dim_{i}" for i in range(self.latent.dim)]
            df_saliency_map = \
                pd.DataFrame(saliency_tensor.cpu().numpy(),
                             index = genes_names,
                             columns = latent_cols)
            
            # Return the data frames along with the saliency map.
            return (df_rep, df_pred_means,
                    df_pred_r_values, df_time,
                    df_saliency_map)

        # Return the data frames.
        return df_rep, df_pred_means, df_pred_r_values, df_time


    def get_probability_density(self,
                                df_rep: pd.DataFrame) -> pd.DataFrame:
        """Given a set of representations, get the probability density
        of each component of the Gaussian mixture model for each
        representation and the representation(s) having the maximum
        probability density for each component.

        Parameters
        ----------
        df_rep : :class:`pandas.DataFrame`
            A data frame containing the representations.

        Returns
        -------
        df_prob_rep : :class:`pandas.DataFrame`
            A data frame containing the probability densities for each
            representation, together with an indication of what the
            maximum probability density found is and for which
            component it is found.

        df_prob_comp : :class:`pandas.DataFrame`
            A data frame containing, for each component, the
            representation(s) having the maximum probability density
            for the component, together with the probability density
            for that(those) representation(s).
        """

        # Set the name of the column that will contain the maximum
        # probability density found per sample.
        MAX_PROB_COL = "max_prob_density"

        # Set the name of the column that will contain the component
        # for which the maximum probability density was found per
        # sample.
        MAX_PROB_COMP_COL = "max_prob_density_comp"

        # Set the name of the column that will contain the unique
        # index of the sample having the maximum probability for a
        # component.
        SAMPLE_IDX_COL = "sample_idx"

        #-------------------------------------------------------------#

        # Get the names of the columns containing the values of the
        # representations along the latent space's dimensions.
        latent_dims_columns = \
            [col for col in df_rep.columns \
             if col.startswith("latent_dim_")]

        # Get the names of the other columns.
        other_columns = \
            [col for col in df_rep.columns \
             if col not in latent_dims_columns]

        #-------------------------------------------------------------#

        # Split the data frame in two.
        df_rep_data, _ = \
            df_rep[latent_dims_columns], df_rep[other_columns]

        #-------------------------------------------------------------#
        
        # Get the probability densities of the representations for
        # each component.
        probs_values = \
            self.latent.sample_probs(\
                x = torch.Tensor(df_rep_data.values).to(
                    self.device))

        #-------------------------------------------------------------#

        # Convert the result into a data frame.
        df_prob_rep = pd.DataFrame(probs_values.detach().cpu().numpy())

        # Add a column storing the highest probability density per
        # representation.
        df_prob_rep[MAX_PROB_COL] = df_prob_rep.max(axis = 1)

        # Add a column storing which component has the highest
        # probability density per representation.
        df_prob_rep[MAX_PROB_COMP_COL] = df_prob_rep.idxmax(axis = 1)

        # Initialize an empty list to store the rows containing
        # the representations/samples that have the highest
        # probability density for each component .
        rows_with_max = []

        # For each component for which at least one representation
        # had maximum probability density
        for comp in df_prob_rep[MAX_PROB_COMP_COL].unique():

            # Get only those rows corresponding to the current
            # component under consideration.
            sub_df = \
                df_prob_rep.loc[df_prob_rep[MAX_PROB_COMP_COL] == comp]

            # Get the sample with maximum probability for the
            # component (we use 'max()' instead of 'idxmax()' because
            # it does not preserve numbers in scientific notation,
            # possibly because it returns a Series with a different
            # data type).
            max_for_comp = \
                sub_df.loc[sub_df[MAX_PROB_COL] == \
                           sub_df[MAX_PROB_COL].max()].copy()
            
            # Add a column storing the representation/sample unique
            # index.
            max_for_comp[SAMPLE_IDX_COL] = max_for_comp.index

            # The new index will be the component number.
            max_for_comp = max_for_comp.set_index(MAX_PROB_COMP_COL)
            
            # Append the data frame to the list of data frames.
            rows_with_max.append(max_for_comp)

        #-------------------------------------------------------------#

        # Concatenate the data frames.
        df_prob_comp = pd.concat(rows_with_max, axis = 0)

        #-------------------------------------------------------------#

        # Return the two data frames.
        return df_prob_rep, df_prob_comp


    def train(self,
              df_samples: pd.DataFrame,
              names_train: list,
              names_test: list,
              config_train: dict[str, object],
              gmm_pth_file: str = "gmm.pth",
              dec_pth_file: str = "dec.pth",
              pathways: Optional[pd.DataFrame] = None,
              labels_train: Optional[object] = None,
              labels_test: Optional[object] = None) -> \
                tuple[tuple[pd.DataFrame, pd.DataFrame],
                  tuple[pd.DataFrame, pd.DataFrame],
                  Optional[tuple[pd.DataFrame, pd.DataFrame]],
                  pd.DataFrame,
                  Optional[tuple[pd.DataFrame, pd.DataFrame]],
                  pd.DataFrame]:
        """Train the model.

        Parameters
        ----------
        df_samples : :class:`pandas.DataFrame`
            A data frame containing the samples.

            Each row should contain a unique sample, and each
            column should either contain a gene's expression for that
            sample (if the column is named after the gene's Ensembl
            ID) or additional information about the sample.

        names_train : :class:`list`
            A list of the names of the training samples, which should
            be a subset of the names of the samples in the input data
            frame.
        
        names_test : :class:`list`
            A list of the names of the test samples, which should be a
            subset of the names of the samples in the input data frame.

        config_train : :class:`dict`
            A dictionary of options for the training.

        gmm_pth_file : :class:`str`, ``"gmm.pth"``
            The .pth file where to save the GMM's trained parameters
            (means of the components, weights of the components,
            and log-variance of the components).

        dec_pth_file : :class:`str`, ``"dec.pth"``
            The .pth file where to save the decoder's trained
            parameters (weights and biases).
        
        pathways : :class:`dict`, optional
            A dictionary where the keys are pathway names and the
            values are lists of genes' Ensembl IDs belonging to
            each pathway.
            
            It is needed if ``save_pathways_saliency_maps_epoch`` is
            set to :obj:`True`.
        
        labels_train : :class:`numpy.ndarray`, optional
            The ground-truth labels for the training samples.
        
        labels_test : :class:`numpy.ndarray`, optional
            The ground-truth labels for the test samples.

        Returns
        -------
        dfs_rep : :class:`tuple`
            A tuple ``(df_rep_train, df_rep_test)`` with the optimized
            latent representations for training and testing samples.

        dfs_pred_means : :class:`tuple`
            A tuple ``(df_pred_means_train, df_pred_means_test)`` with
            the predicted decoder means for training and testing
            samples.

        dfs_pred_r_values  : :obj:`None` or \
            :class:`pandas.DataFrame` or :class:`tuple`
            The predicted r-values, depending on the output module:

            - :obj:`None` for
              :class:`bulkdgd.core.outputmodules.OutputModulePoisson`.
            - A single :class:`pandas.DataFrame` for
              :class:`bulkdgd.core.outputmodules.OutputModuleNBFeatureDispersion`.
            - A tuple ``(df_pred_r_values_train,
              df_pred_r_values_test)`` for
              :class:`bulkdgd.core.outputmodules.OutputModuleNBFullDispersion`.

        df_loss : :class:`pandas.DataFrame`
            A data frame containing the losses calculated during
            training.
        
        dfs_metrics : :obj:`None` or :class:`tuple`
            The per-epoch metrics rows for training and testing
            samples, depending on whether the user requested to
            calculate metrics during training:

            - :obj:`None` if the user did not request to calculate
              metrics during training.
            - A tuple ``(df_metrics_train, df_metrics_test)`` of
              data frames, where each data frame contains the
              metrics calculated for the training or test samples in a
              given epoch, if the user requested to calculate metrics
              during training.

        df_time : :class:`pandas.DataFrame`
            A data frame containing the training-time metrics.
        """

        #-------------------------------------------------------------#

        # Check the configuration that will be used for training.
        config_train, errors, warnings = \
            _util.parse_config_train(config = config_train)

        # If there are errors in the configuration
        if errors:

            # Raise an exception.
            err_msg = \
                "The configuration is not valid. Errors: " + \
                " ".join(errors)
            raise ValueError(err_msg)

        #-------------------------------------------------------------#
        
        # Get the scale factor for the initialization of the
        # representations from the configuration.
        init_rep_scale = \
            config_train["representations_training_options"].get(
                "init_rep_scale", 0.0)

        #-------------------------------------------------------------#

        # Get the training samples.
        df_train = df_samples.loc[names_train]

        # Get the test samples.
        df_test = df_samples.loc[names_test]

        #-------------------------------------------------------------#

        # Get the names of the columns containing gene expression data.
        genes_columns = \
            [col for col in df_samples.columns \
             if col.startswith("ENSG")]
        
        # Get the names of the columns not containing gene expression
        # data.
        other_columns = \
            [col for col in df_samples.columns \
             if col not in genes_columns]

        #-------------------------------------------------------------#
        
        # Check if the user wants to save pathways' saliency maps
        # during training.
        _opt_outputs = config_train.get(
            "reporting_options", {}).get(
                "optional_outputs", {})
        _pathways_config = _opt_outputs.get(
            "pathways_saliency_maps_epoch", {})
        _save_pathways = _pathways_config.get("enabled", False)

        # If the user wants to save pathways' saliency maps but
        # did not provide the 'pathways' option
        if _save_pathways and pathways is None:
            
            # Raise an error.
            err_msg = \
                "The 'pathways' option must be provided if " \
                "'save_pathways_saliency_maps_epoch' is set to True."
            raise ValueError(err_msg)

        #-------------------------------------------------------------#

        # Get a data frame with the training data and only the columns
        # containing gene expression data.
        df_expr_data_train = df_train[genes_columns]

        # Get a data frame with the training data and only the columns
        # containing additional information.
        df_other_data_train = df_train[other_columns]

        # Get the number of samples and genes in the data frame
        # containing the training data.
        n_samples_train, n_genes = df_expr_data_train.shape

        # Get the training samples' names.
        samples_names_train = df_expr_data_train.index.tolist()

        # Create the dataset with the training samples.
        dataset_train = \
            dataclasses.GeneExpressionDataset(\
                df = df_expr_data_train,
                labels = labels_train)

        # Create the data loader with the training samples.
        data_loader_train = \
            _util.get_data_loader(
                dataset = dataset_train,
                config = config_train["data_loader_options"]["train"])

        # Create the representation layer for the training samples.
        rep_layer_train = \
            latents.RepresentationLayer(values = \
                init_rep_scale * torch.randn(\
                    size = (n_samples_train, self.latent.dim))).to(\
                        self.device)

        #-------------------------------------------------------------#

        # Get a data frame with the testing data and only the columns
        # containing gene expression data.
        df_expr_data_test = df_test[genes_columns]

        # Get a data frame with the testing data and only the columns
        # containing additional information.
        df_other_data_test = df_test[other_columns]

        # Get the number of test samples.
        n_samples_test = df_expr_data_test.shape[0]

        # Get the testing samples' names.
        samples_names_test = df_expr_data_test.index.tolist()

        # Create the dataset with the test samples.
        dataset_test = \
            dataclasses.GeneExpressionDataset(\
                df = df_expr_data_test,
                labels = labels_test)

        # Create the data loader with the testing samples.
        data_loader_test = \
            _util.get_data_loader(
                dataset = dataset_test,
                config = config_train["data_loader_options"]["test"])

        # Create the representation layer for the testing samples.
        rep_layer_test = \
            latents.RepresentationLayer(values = \
                init_rep_scale * torch.randn(\
                    size = (n_samples_test, self.latent.dim))).to(\
                        self.device)

        #-------------------------------------------------------------#

        # Train the model.
        reps, pred_means, pred_r_values, losses_list, \
            metrics_rows, time_train = \
            self._train(\
                config_train = config_train,
                samples_names_train = samples_names_train,
                samples_names_test = samples_names_test,
                genes_names = genes_columns,
                data_loader_train = data_loader_train,
                data_loader_test = data_loader_test,
                rep_layer_train = rep_layer_train,
                rep_layer_test = rep_layer_test,
                pathways = pathways,
                labels_train = dataset_train.labels,
                labels_test = dataset_test.labels)

        # Unpack the metrics rows tuple.
        metrics_rows_train, metrics_rows_test = metrics_rows

        #-------------------------------------------------------------#

        # Save the GMM's parameters.
        self.latent.save(_internals.uniquify_file_path(gmm_pth_file))

        # Inform the user that the parameters were saved.
        info_msg = \
            "The trained Gaussian mixture model's parameters were " \
            f"successfully saved in '{gmm_pth_file}'."
        logger.info(info_msg)

        #-------------------------------------------------------------#

        # Save the decoder's parameters.
        torch.save(self.decoder.state_dict(),
                   _internals.uniquify_file_path(dec_pth_file))

        # Inform the user that the parameters were saved.
        info_msg = \
            "The trained decoder's parameters were successfully " \
            f"saved in '{dec_pth_file}'."
        logger.info(info_msg)

        #-------------------------------------------------------------#

        # Create and return the final data frames.
        return _util.get_final_data_frames_train(\
                    reps = reps,
                    pred_means = pred_means,
                    pred_r_values = pred_r_values,
                    losses_list = losses_list,
                    metrics_rows_train = metrics_rows_train,
                    metrics_rows_test = metrics_rows_test,
                    time_train = time_train,
                    samples_names_train = samples_names_train,
                    samples_names_test = samples_names_test,
                    df_other_data_train = df_other_data_train,
                    df_other_data_test = df_other_data_test,
                    genes_names = genes_columns)

