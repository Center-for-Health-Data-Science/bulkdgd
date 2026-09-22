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
    "model (:class:`core.model.BulkDGD`)."


#######################################################################


# Import from the standard library.
import contextlib
import copy
import hashlib
import logging as log
import math
import os
import random
import re
import string
import time
from typing import Optional, Union

# Import from third-party libraries.
import numpy as np
import pandas as pd
from scipy.stats import chi2, nbinom, poisson
import torch
from torch import nn
import yaml

# Import from 'bulkdgd'.
from bulkdgd import _internals, defaults
from . import (
    dataclasses,
    decoders,
    latents,
    metrics,
    outputmodules,
    warmstart,
    _util)


#######################################################################


# Get the module's logger.
logger = log.getLogger(__name__)


#######################################################################


def clip_grads(optimizer: torch.optim.Optimizer,
               max_norm: Optional[Union[float, int]]) -> None:
    """Clip an optimizer's gradients to a maximum norm.

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

    # Zero out non-finite gradients, since clipping turns them into NaN.
    for p in params:

        # If the parameter's gradient is not entirely finite
        if not torch.isfinite(p.grad).all():

            # Warn the user.
            logger.warning(\
                "A non-finite gradient was produced and has been "
                "zeroed. The step it came from is effectively skipped.")

            # Zero the gradient.
            p.grad = torch.zeros_like(p.grad)

    #-----------------------------------------------------------------#

    # Clip the gradients and get their norm before clipping.
    total_norm = \
        torch.nn.utils.clip_grad_norm_(parameters = params,
                                       max_norm = max_norm)

    # Log the gradient norm before clipping.
    logger.debug(
        f"Gradient norm before clipping: {float(total_norm):.4f} "
        f"(clipped to {max_norm}).")


#######################################################################


class BulkDGD(nn.Module):

    """
    Class implementing the BulkDGD model.
    """

    ######################## PUBLIC ATTRIBUTES ########################


    # Set the supported optimizers to find the representations.
    OPTIMIZERS = ["adam", "adamw", "lbfgs"]

    # Set the supported Gaussian mixture model types.
    GMM_TYPES = ["lgmm", "tgmm"]

    # Set the supported precisions.
    DTYPES = ["float32", "float64"]

    # Map each precision to its torch data type.
    _DTYPES_TORCH = {"float32" : torch.float32,
                     "float64" : torch.float64}


    ##################### INITIALIZATION METHODS ######################


    def __init__(self,
                 latent_dim: Optional[int] = None,
                 latent_options: Optional[dict[str, object]] = None,
                 decoder_options: Optional[dict[str, object]] = None,
                 latent_type: Optional[str] = None,
                 gmm_final: Optional[dict[str, object]] = None,
                 genes_txt_file: Optional[str] = None,
                 scaling_factor: Optional[str] = None,
                 dtype: Optional[str] = None,
                 device: str = "cpu",
                 seed: Optional[str] = None) -> None:
        """Initialize an instance of the class. If no architecture is
        given, load the trained model shipped with the package.

        Parameters
        ----------
        latent_dim : :class:`int`, optional
            The dimensionality of the latent space.

        latent_options : :class:`dict`, optional
            The options for setting up the latent space.

        decoder_options : :class:`dict`, optional
            The options for setting up the decoder.

        latent_type : :class:`str`, {``"lgmm"``, ``"tgmm"``}, \
            ``"tgmm"``
            The type of the latent space to use.

        gmm_final : :class:`dict`, optional
            The options for the Gaussian mixture fitted to the
            representations after training. If not given, no such
            mixture is fitted.

        genes_txt_file : :class:`str`, optional
            A plain text file containing the Ensembl IDs of the genes
            included in the model.

        scaling_factor : :class:`str`, \
            {``"mean"``, ``"median"``}, ``"mean"``
            How to compute a sample's scaling factor, used to rescale
            the decoder's predicted means to the sample's counts.

        dtype : :class:`str`, \
            {``"float32"``, ``"float64"``}, ``"float32"``
            The precision of the model's parameters.

        device : :class:`str`, ``"cpu"``
            The device where the model will be initialized.

        seed : :class:`str`, optional
            The seed of the shipped ensemble member to load. Only valid
            when no architecture is given.
        """

        # Run the superclass' initialization.
        super(BulkDGD, self).__init__()

        #-------------------------------------------------------------#

        # If no architecture is given
        if (latent_dim is None and latent_options is None
                and decoder_options is None):

            # Load the shipped model's settings.
            shipped = _util.load_shipped_model(seed = seed)

            # Get the shipped architecture.
            latent_dim = shipped["latent_dim"]
            latent_options = shipped["latent_options"]
            decoder_options = shipped["decoder_options"]

            # If no latent space type was given, use the shipped one.
            if latent_type is None:
                latent_type = shipped["latent_type"]

            # If no genes' file was given, use the shipped one.
            if genes_txt_file is None:
                genes_txt_file = shipped["genes_txt_file"]

            # If no scaling factor was given, use the shipped one.
            if scaling_factor is None:
                scaling_factor = shipped["scaling_factor"]

            # If no precision was given, use the shipped one.
            if dtype is None:
                dtype = shipped["dtype"]

        # If a seed was given together with an architecture
        elif seed is not None:

            # Raise an error.
            errstr = \
                "'seed' selects which shipped model to load and so " \
                "cannot be combined with an explicit architecture. " \
                "Pass either 'seed', or 'latent_dim', " \
                "'latent_options' and 'decoder_options'."
            raise ValueError(errstr)

        # If no latent space type was given, use the default one.
        if latent_type is None:
            latent_type = "tgmm"

        # If no scaling factor was given, use the default one.
        if scaling_factor is None:
            scaling_factor = "mean"

        # If no precision was given, use the default one.
        if dtype is None:
            dtype = "float32"

        # If the architecture was only partly given
        if latent_dim is None or latent_options is None \
                or decoder_options is None:

            # Raise an error.
            errstr = \
                "'latent_dim', 'latent_options' and " \
                "'decoder_options' describe the architecture and " \
                "must be given together. Give none of them to load " \
                "the trained model that ships with the package."
            raise ValueError(errstr)

        #-------------------------------------------------------------#

        # If the scaling factor is not one that is supported.
        if scaling_factor not in dataclasses.GeneExpressionDataset.\
                                    SCALING_FACTORS:

            # Raise an error.
            raise ValueError(
                f"Unsupported scaling factor '{scaling_factor}'. The "
                "supported scaling factors are: "
    f"{', '.join(dataclasses.GeneExpressionDataset.SCALING_FACTORS)}.")

        # Save the scaling factor.
        self._scaling_factor = scaling_factor

        # Inform the user about the scaling factor.
        logger.info(
            f"The scaling factor is the '{scaling_factor}' of a "
            "sample's counts.")

        #-------------------------------------------------------------#

        # If the precision is not one that is supported.
        if dtype not in self.DTYPES:

            # Raise an error.
            raise ValueError(
                f"Unsupported dtype '{dtype}'. The supported dtypes "
                f"are: {', '.join(self.DTYPES)}.")

        # Save the precision.
        self._dtype = dtype

        #-------------------------------------------------------------#

        # If no genes' file or the "default" one was given, use the
        # default genes' file.
        if genes_txt_file in (None, "default"):
            genes_txt_file = defaults.DATA_FILES_MODEL["genes"]

        # Get the genes included in the model.
        genes = \
            self.__class__._load_genes_list(\
                genes_list_file = genes_txt_file)

        #-------------------------------------------------------------#

        # Build the model in its own precision (casting it afterwards
        # would keep the already-rounded values).
        with self._default_dtype(dtype):

            # Get the latent space.
            self._latent = \
                self._get_latent(latent_dim = latent_dim,
                                 latent_type = latent_type,
                                 latent_options = latent_options,
                                 device = device)

            # Save the options for the latent space in the
            # model's attributes.
            self._latent_initial_options = latent_options

            # Save the type of latent space.
            self._latent_type = latent_type

            # Save the options for the final Gaussian mixture.
            self._gmm_final_options = gmm_final

            # Set the final Gaussian mixture (fitted after training).
            self._latent_final = None

            # Inform the user that the latent space was set.
            info_msg = \
                "The latent space was successfully set " \
                f"(type: '{self._latent.__class__.__name__}')."
            logger.info(info_msg)

            #---------------------------------------------------------#

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

        # Set whether the model is trained (has a trained decoder).
        self._is_trained = \
            decoder_options.get("decoder_pth_file") is not None

        # Save the genes, in the decoder's output order.
        self._genes = genes

        # Save the genes' file.
        self._genes_txt_file = genes_txt_file

        #-------------------------------------------------------------#

        # By default, do not keep the details of the selection of the
        # best representations.
        self._keep_selection_details = False

        # Initialize the list to hold the selection details.
        self._selection_details = []

        #-------------------------------------------------------------#

        # Save the device.
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

        latent_options : :class:`dict`
            A dictionary of options for the latent space.

        device : :class:`str`
            The device where the Gaussian mixture model will be
            initialized.

        Returns
        -------
        latent : \
            :class:`bulkdgd.core.latents.GaussianMixtureModelLegacy` \
            or :class:`bulkdgd.core.latents.GaussianMixtureModelTGMM`
            The latent space.
        """

        # Copy the latent space's options without the path to the
        # file with the trained parameters (loaded after construction).
        latent_options_init = \
            {k: v for k, v in latent_options.items()
             if k != "latent_pth_file"}

        # If the user wants to use the legacy Gaussian Mixture Model
        if latent_type == "lgmm":

            # Initialize the Gaussian mixture model.
            latent = \
                latents.GaussianMixtureModelLegacy(
                    dim = latent_dim,
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
                f"{', '.join(self.__class__.GMM_TYPES)}."
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
            The negative binomials' r-values indexed by gene, or
            :obj:`None` if the output module is not
            ``"nb_feature_dispersion"``.
        """

        # Create a copy of the configuration options for the
        # decoder.
        decoder_options_copy = copy.deepcopy(decoder_options)

        # Update the decoder's options by adding the number of
        # output units.
        decoder_options_copy[
            "output_module_options"]["output_dim"] = len(genes)

        # Remove the path to the file with the trained parameters
        # (loaded after construction).
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

        # Otherwise (the r-values are not stored per gene)
        else:

            # There are no per-gene r-values.
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

        # Try to load the parameters.
        try:

            # Load the parameters.
            mod.load_state_dict(
                torch.load(pth_file,
                           weights_only = True,
                           map_location = device))

        # If something went wrong
        except Exception as e:

            # Raise an error.
            err_msg = \
                "It was not possible to load the parameters " \
                f"from '{pth_file}'. Error: {e}"
            raise RuntimeError(err_msg)

        # Inform the user that the parameters were loaded.
        info_msg = \
            "The parameters were successfully loaded from " \
            f"'{pth_file}'."
        logger.info(info_msg)


    @classmethod
    @contextlib.contextmanager
    def _default_dtype(cls,
                       dtype: str):
        """Set torch's default data type for the duration of a block.

        Parameters
        ----------
        dtype : :class:`str`, {``"float32"``, ``"float64"``}
            The precision.

        Returns
        -------
        context : :class:`contextlib.AbstractContextManager`
            A context manager restoring the previous default data type
            on exit.
        """

        # Get the current default data type.
        previous = torch.get_default_dtype()

        # Set the requested one.
        torch.set_default_dtype(cls._DTYPES_TORCH[dtype])

        # Run the block.
        try:

            yield

        # Afterwards
        finally:

            # Restore the previous default data type.
            torch.set_default_dtype(previous)


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
    def scaling_factor(self):
        """How the scaling factor of a sample is computed
        (``"mean"`` or ``"median"``).
        """

        return self._scaling_factor


    @scaling_factor.setter
    def scaling_factor(self,
                       value) -> None:
        """Raise an exception if the user tries to modify the value of
        ``scaling_factor`` after initialization.

        Parameters
        ----------
        value : :class:`str`
            The value.
        """

        # Raise an error.
        errstr = \
            "The value of 'scaling_factor' is set at initialization " \
            "and cannot be changed. Set it in the model's " \
            "configuration file instead."
        raise ValueError(errstr)


    @property
    def dtype(self):
        """The precision of the model's parameters (``"float32"`` or
        ``"float64"``).
        """

        return self._dtype


    @dtype.setter
    def dtype(self,
              value) -> None:
        """Raise an exception if the user tries to modify the value of
        ``dtype`` after initialization.

        Parameters
        ----------
        value : :class:`str`
            The value.
        """

        # Raise an error.
        errstr = \
            "The value of 'dtype' is set at initialization and " \
            "cannot be changed. Set it in the model's configuration " \
            "file instead."
        raise ValueError(errstr)


    @property
    def genes(self):
        """The genes included in the model, in the decoder's output
        order.
        """

        return self._genes


    @genes.setter
    def genes(self,
              value) -> None:
        """Raise an exception if the user tries to modify the value of
        ``genes`` after initialization.

        Parameters
        ----------
        value : :class:`list`
            The value.
        """

        # Raise an error.
        errstr = \
            "The value of 'genes' is set at initialization and " \
            "cannot be changed. If you want to change the genes, " \
            "initialize a new instance of " \
            f"'{self.__class__.__name__}'."
        raise ValueError(errstr)


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

        Parameters
        ----------
        value : :class:`bulkdgd.core.latents.GaussianMixtureModelTGMM`
            The value.
        """

        # Raise an error.
        err_msg = \
            "The value of 'latent' is set at initialization and " \
            "cannot be changed. If you want to change the " \
            "latent space, initialize a new instance of " \
            f"'{self.__class__.__name__}'."
        raise ValueError(err_msg)


    @property
    def latent_final(self):
        """The Gaussian mixture fitted to the representations after
        training, or ``None`` if none was fitted.
        """

        return self._latent_final


    @latent_final.setter
    def latent_final(self,
                     value):
        """Raise an exception if the user tries to set the final
        Gaussian mixture model by hand.

        Parameters
        ----------
        value : :class:`bulkdgd.core.latents.GaussianMixtureModelTGMM`
            The value.
        """

        # Raise an error.
        err_msg = \
            "The value of 'latent_final' is set when the model is " \
            "trained, from the 'gmm_final' section of its " \
            "configuration, and cannot be set by hand."
        raise ValueError(err_msg)


    @property
    def gmm_final_options(self):
        """The options for the Gaussian mixture model fitted after
        training, or ``None`` if none is fitted.
        """

        return self._gmm_final_options


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

        Parameters
        ----------
        value : :class:`bulkdgd.core.decoders.Decoder`
            The value.
        """

        # Raise an error.
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

        Parameters
        ----------
        value : :class:`str`
            The device to move the model to.
        """

        # Move the model to the specified device.
        self.to(device = torch.device(value))

        # Update the device the model is on.
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

        # If it is the L-BFGS optimizer
        elif optimizer_type == "lbfgs":

            # Set up the optimizer (it steps with a closure).
            optimizer = \
                torch.optim.LBFGS(optimizer_parameters,
                                  **optimizer_options)

        # If the optimizer is not supported
        else:

            # Raise an error.
            errstr = \
                f"Unsupported optimizer '{optimizer_type}'. The " \
                "supported optimizers are: " \
                f"{', '.join(self.OPTIMIZERS)}."
            raise ValueError(errstr)

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
            ``"latent"``, ``"representations"``}
            The target for the scheduler: the decoder or the latent
            space (steps per batch) or the representations (steps per
            epoch).

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

        # If the target is not supported
        else:

            # Raise an error.
            errstr = \
                "Unsupported learning rate scheduler target " \
                f"'{lr_scheduler_target}'. The supported targets " \
                "are: decoder, latent, representations."
            raise ValueError(errstr)

        #-------------------------------------------------------------#

        # If the type of scheduler is 'one_cycle'
        if lr_scheduler_type == "one_cycle":

            # Copy the scheduler's options without the 'enabled' flag.
            lr_scheduler_opts = lr_scheduler_options.copy()
            lr_scheduler_opts.pop("enabled", None)

            # Set the scheduler.
            lr_scheduler = \
                torch.optim.lr_scheduler.OneCycleLR(\
                    optimizer,
                    total_steps = total_steps,
                    **lr_scheduler_opts)

        #-------------------------------------------------------------#

        # If the type of scheduler is 'cosine'
        elif lr_scheduler_type == "cosine":

            # Copy the scheduler's options without the 'enabled' flag.
            lr_scheduler_opts = lr_scheduler_options.copy()
            lr_scheduler_opts.pop("enabled", None)

            # Set the scheduler, annealing over all steps.
            lr_scheduler = \
                torch.optim.lr_scheduler.CosineAnnealingLR(\
                    optimizer,
                    T_max = total_steps,
                    **lr_scheduler_opts)

        #-------------------------------------------------------------#

        # If the type of scheduler is not supported
        else:

            # Raise an error.
            errstr = \
                "Unsupported learning rate scheduler type " \
                f"'{lr_scheduler_type}'. The supported types are: " \
                "one_cycle, cosine."
            raise ValueError(errstr)

        #-------------------------------------------------------------#

        # Return the scheduler.
        return lr_scheduler


    @staticmethod
    def _get_masked_scaling_factors(obs_counts: torch.Tensor,
                                    pred_means: torch.Tensor,
                                    mask: torch.Tensor,
                                    n_genes: int,
                                    scaling_factor: str = "mean") \
                                        -> torch.Tensor:
        """Get the scaling factors of samples with only some genes
        measured, from the measured counts and the predictions.

        Parameters
        ----------
        obs_counts : :class:`torch.Tensor`
            The observed counts for the sample(s).

        pred_means : :class:`torch.Tensor`
            The model's predicted, unscaled means for the sample(s).

        mask : :class:`torch.Tensor`
            A 0/1 mask marking which genes were measured.

        n_genes : :class:`int`
            The total number of genes the model was built for.

        scaling_factor : :class:`str`, {``"mean"``, ``"median"``}, \
            optional
            Which scaling factor to compute.

        Returns
        -------
        scaling_factors : :class:`torch.Tensor`
            The scaling factor(s).
        """

        # If computing the median
        if scaling_factor == "median":

            # Mark the unmeasured genes.
            unmeasured = mask == 0.0

            # Count the measured genes per sample.
            n_measured = mask.sum(dim = -1, keepdim = True).long()

            # Take the lower middle value, as 'torch.median' does.
            idx = ((n_measured - 1) // 2).clamp(min = 0)

            # Get the median of the measured counts (the unmeasured
            # genes are sorted past the end).
            obs_median = \
                obs_counts.masked_fill(unmeasured,
                                       float("inf")).sort(\
                    dim = -1).values.gather(-1, idx)

            # Get the predicted median over the measured genes.
            pred_measured = \
                pred_means.masked_fill(unmeasured,
                                       float("inf")).sort(\
                    dim = -1).values.gather(\
                        -1,
                        idx.expand(*pred_means.shape[:-1], 1))

            # Get the predicted median over every gene.
            pred_all = pred_means.sort(dim = -1).values[\
                ..., [(n_genes - 1) // 2]]

            # Return the measured median, corrected by the ratio of the
            # predicted medians over all and measured genes.
            return obs_median * pred_all / pred_measured

        #-------------------------------------------------------------#

        # Otherwise, sum the observed counts over the measured genes.
        obs_measured = (obs_counts * mask).sum(dim = -1, keepdim = True)

        # Sum the model's predicted means over the unmeasured genes.
        pred_unmeasured = \
            (pred_means * (1.0 - mask)).sum(dim = -1, keepdim = True)

        # Get the denominator (about the number of measured genes),
        # clamped to avoid a division by zero.
        denominator = (n_genes - pred_unmeasured).clamp(min = 1e-6)

        # Return the scaling factors (the plain mean when every gene
        # is measured).
        return obs_measured / denominator


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
                      latent_lambda: Optional[float] = None,
                      genes_mask: Optional[torch.Tensor] = None,
                      contamination: float = 0.0,
                      contamination_r: float = 0.05,
                      noise_type: Optional[str] = None,
                      noise_options: Optional[dict] = None) -> \
                        tuple[torch.Tensor, torch.Tensor,
                              Optional[torch.Tensor], list[tuple]]:
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
            The number of mixture components with at least one drawn
            representation per sample.

        n_rep_per_comp : :class:`int`
            The number of new representations per sample per
            component.

        epochs : :class:`int`
            The number of epochs to run the optimization for.

        opt_num : :class:`int`
            The number of the optimization round.

        loss_reporting_options : :class:`dict`
            A dictionary containing the options for reporting the loss.

        loss_reduction_type : :class:`str`
            The method to reduce the loss across the samples in the
            batch.

        latent_lambda : :class:`float`, optional
            The weight of the latent loss term in the total loss.

        genes_mask : :class:`torch.Tensor`, optional
            A 2D 1.0/0.0 mask (samples x genes) of measured genes.
            Unmeasured genes are excluded from the loss and the
            scaling factor. By default, all genes are measured.

        contamination : :class:`float`, ``0.0``
            How much of a sample the model may give up on. Zero
            disables it.

        contamination_r : :class:`float`, ``0.05``
            The dispersion used for the contamination model.

        noise_type : :class:`str`, optional
            The type of noise to inject into the representations
            during optimization (only ``"gaussian"``). :obj:`None`
            disables it.

        noise_options : :class:`dict`, optional
            The noise options: ``scale``, ``start``/``end`` (the
            cosine-annealed scale multipliers), ``within_radius_prob``
            and ``gain``.

        Returns
        -------
        rep : :class:`torch.Tensor`
            The optimized representations, shaped (samples, latent
            dimensionality).

        pred_means : :class:`torch.Tensor`
            The predicted gene-count means, shaped (samples, genes).

        pred_r_values : :class:`torch.Tensor` or :obj:`None`
            The predicted negative-binomial r-values, same shape as
            ``pred_means``, or :obj:`None` for Poisson counts.

        time_opt : :class:`list`
            Per-epoch CPU/wall-clock timing for the epoch and its
            backpropagation step.
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

        # Get the noise options.
        noise_options = noise_options or {}

        # Enable the noise only if it is Gaussian.
        noise_enabled = (noise_type == "gaussian")

        # Get the base noise scale (zero disables the noise).
        noise_scale_base = \
            float(noise_options.get("scale", 0.0)) if noise_enabled \
            else 0.0

        # If the noise is enabled
        if noise_scale_base > 0:

            # Get the annealing schedule and the gain.
            noise_start = float(noise_options["start"])
            noise_end = float(noise_options["end"])
            noise_gain = float(noise_options["gain"])

            # Get the radius holding the given fraction of the mass,
            # making the noise's size independent of the dimensionality.
            noise_radius = \
                float(chi2.ppf(
                    float(noise_options["within_radius_prob"]),
                    self.latent.dim)) ** 0.5

            # Inform the user about the noise.
            logger.info(
                f"Optimization number {opt_num} will inject Gaussian "
                "noise into the representations (scale "
                f"{noise_scale_base}, annealed from {noise_start} to "
                f"{noise_end}).")

        #-------------------------------------------------------------#

        # Inform the user that the optimization is starting.
        info_msg = f"Starting optimization number {opt_num}..."
        logger.info(info_msg)

        # For each epoch
        for epoch in range(1, epochs+1):

            # If the noise is enabled
            if noise_scale_base > 0:

                # Get the fraction of the epochs elapsed.
                progress = (epoch - 1) / max(epochs - 1, 1)

                # Get the noise scale for this epoch, cosine-annealed
                # between 'start' and 'end'.
                noise_scale = \
                    noise_end + (noise_start - noise_end) * 0.5 * \
                        (1 + math.cos(math.pi * progress))

                # Multiply it by the base scale.
                noise_scale = noise_scale * noise_scale_base

            # Otherwise
            else:

                # Set no noise.
                noise_scale = 0.0

            # Mark the CPU start time of the epoch.
            time_start_epoch_cpu = time.process_time()

            # Mark the wall clock start time of the epoch.
            time_start_epoch_wall = time.time()

            # Initialize the totals the closure rebinds.
            time_tot_bw_cpu = 0.0
            time_tot_bw_wall = 0.0
            rep_avg_loss_epoch = 0.0

            # Define the epoch's loss computation as a closure (L-BFGS
            # evaluates it several times per step).
            def closure():
                """Compute the epoch's loss and its gradients.

                Returns
                -------
                rep_avg_loss_epoch : :class:`float`
                    The epoch's loss.
                """

                # Rebind the epoch's totals.
                nonlocal time_tot_bw_cpu, time_tot_bw_wall
                nonlocal rep_avg_loss_epoch

                # Initialize the total CPU time needed to perform the
                # backward step to zero.
                time_tot_bw_cpu = 0.0

                # Initialize the total wall-clock time needed to
                # perform the backward step to zero.
                time_tot_bw_wall = 0.0

                # Make the optimizer's gradients zero.
                optimizer.zero_grad()

                # Initialize the loss for the current epoch to 0.0.
                rep_avg_loss_epoch = 0.0

                # For each batch: gene expression, mean gene expression,
                # and unique sample indexes.
                for samples_exp, samples_mean_exp, samples_ixs \
                    in data_loader:

                    # Get the number of samples in the batch.
                    n_samples_in_batch = len(samples_ixs)

                    #-------------------------------------------------#

                    # Move the gene expression of the samples to the
                    # correct device.
                    samples_exp = samples_exp.to(self.device)

                    # Move the mean gene expression of the samples to
                    # the correct device.
                    samples_mean_exp = samples_mean_exp.to(self.device)

                    #-------------------------------------------------#

                    # Get the representations from the representation
                    # layer: shape (samples * components * reps, dim).
                    z_all = rep_layer()

                    #-------------------------------------------------#

                    # Reshape to (samples, reps, components, dim) and
                    # select this batch's samples.
                    z_4d = z_all.view(n_samples,
                                      n_rep_per_comp,
                                      n_components,
                                      dim)[samples_ixs]

                    # Flatten back to (batch * reps * components, dim)
                    # for the decoder.
                    z = z_4d.view(n_samples_in_batch * \
                                    n_rep_per_comp * \
                                    n_components,
                                  dim)

                    #-------------------------------------------------#

                    # If the noise is enabled
                    if noise_scale > 0:

                        # Add it to the decoder's input (not to the
                        # optimized representations).
                        z = z + noise_scale * noise_gain * \
                                torch.randn_like(z) / noise_radius

                    #-------------------------------------------------#

                    # If the output module means that the r-values are
                    # not learned
                    if isinstance(\
                        self.decoder.nb,
                        (outputmodules.OutputModuleNBFeatureDispersion,
                         outputmodules.OutputModulePoisson)):

                        # Get the predicted scaled means: shape
                        # (batch * reps * components, genes).
                        pred_means = self.decoder(z = z)

                        # There are no predicted r-values.
                        pred_log_r_values = None

                    # If the output module means that the r-values are
                    # learned
                    elif isinstance(\
                        self.decoder.nb,
                        outputmodules.OutputModuleNBFullDispersion):

                        # Get the predicted scaled means and r-values,
                        # shaped (batch * reps * components, genes).
                        pred_means, pred_log_r_values = \
                            self.decoder(z = z)

                        # Reshape the r-values to (batch, reps,
                        # components, genes) to compute the loss.
                        pred_log_r_values = \
                            pred_log_r_values.view(n_samples_in_batch,
                                                   n_rep_per_comp,
                                                   n_components,
                                                   n_genes)

                    #-------------------------------------------------#

                    # Expand the observed gene expression to (batch,
                    # reps, components, genes).
                    obs_counts = \
                        samples_exp.unsqueeze(1).unsqueeze(1).expand(\
                            -1,
                            n_rep_per_comp,
                            n_components,
                            -1)

                    #-------------------------------------------------#

                    # Reshape the scaling factors to (batch, 1, 1, 1),
                    # broadcastable over the loss.
                    scaling_factors = \
                        decoders.reshape_scaling_factors(
                            samples_mean_exp,
                            4)

                    #-------------------------------------------------#

                    # Get the batch's mask, if any, shaped to broadcast
                    # over the representations and the components.
                    mask_batch = \
                        genes_mask[samples_ixs].unsqueeze(1).\
                            unsqueeze(1) \
                            if genes_mask is not None else None

                    #-------------------------------------------------#

                    # Reshape the predicted means to (batch, reps,
                    # components, genes) to compute the loss.
                    pred_means = pred_means.view(n_samples_in_batch,
                                                 n_rep_per_comp,
                                                 n_components,
                                                 n_genes)

                    #-------------------------------------------------#

                    # If only some genes were measured
                    if mask_batch is not None:

                        # Re-estimate the scaling factors from the
                        # measured genes.
                        scaling_factors = \
                            self.__class__._get_masked_scaling_factors(
                                obs_counts = obs_counts,
                                pred_means = pred_means,
                                mask = mask_batch,
                                n_genes = n_genes,
                                scaling_factor = self._scaling_factor)

                    #-------------------------------------------------#

                    # If the output module means that the r-values are
                    # not learned
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

                    # If the output module means that the r-values are
                    # learned
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

                    # If contamination is enabled
                    if contamination:

                        # Add the contamination options.
                        recon_loss_options["contamination"] = \
                            contamination
                        recon_loss_options["contamination_r"] = \
                            contamination_r

                    # Get the reconstruction loss: shape (batch, reps,
                    # components, genes).
                    recon_loss = \
                        self.decoder.nb.loss(**recon_loss_options)

                    #-------------------------------------------------#

                    # If only some genes were measured
                    if mask_batch is not None:

                        # Zero out the unmeasured genes' loss terms.
                        recon_loss = recon_loss * mask_batch

                    #-------------------------------------------------#

                    # If the reduction method is 'sum'
                    if loss_reduction_type == "sum":

                        # Get the total reconstruction loss by summing
                        # all the values (a single-value tensor).
                        recon_loss_final = recon_loss.sum().clone()

                    # If the reduction method is 'mean'
                    elif loss_reduction_type == "mean":

                        # Get the total reconstruction loss by averaging
                        # all the values (a single-value tensor).
                        recon_loss_final = recon_loss.mean().clone()

                    #-------------------------------------------------#

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

                    #-------------------------------------------------#

                    # Get the output module's dispersion regularization
                    # (zero unless it shrinks the dispersion).
                    dispersion_reg = \
                        self.decoder.nb.dispersion_regularization(
                            pred_means = pred_means,
                            pred_log_r_values = pred_log_r_values,
                            reduction = loss_reduction_type)

                    #-------------------------------------------------#

                    # Get the total loss (a single-value tensor).
                    total_loss = \
                        recon_loss_final + latent_loss_final + \
                            dispersion_reg

                    #-------------------------------------------------#

                    # Mark the CPU start time of the backward step.
                    time_start_bw_cpu = time.process_time()

                    # Mark the wall clock start time of the backward
                    # step.
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

                    #-------------------------------------------------#

                    # Update the average loss for the current epoch.
                    rep_avg_loss_epoch += \
                        _util.normalize_loss(\
                            loss = total_loss.item(),
                            loss_type = "total",
                            loss_norm_type = loss_norm_type,
                            loss_norm_options = \
                                {"n_samples" : n_samples,
                                 "n_genes" : n_genes})

                # Return the epoch's loss.
                return rep_avg_loss_epoch

            #---------------------------------------------------------#

            # If the optimizer is L-BFGS
            if isinstance(optimizer, torch.optim.LBFGS):

                # Take a step (the optimizer evaluates the closure).
                optimizer.step(closure)

            # If it is any other optimizer
            else:

                # Compute the loss and its gradients.
                closure()

                # Take a step.
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
                # binomials with per-gene r-values
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
                # binomials with per-gene, per-sample r-values
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
                            Optional[float] = None,
                         genes_mask: \
                            Optional[torch.Tensor] = None,
                         contamination: float = 0.0,
                         contamination_r: float = 0.05,
                         n_components: \
                            Optional[int] = None) -> \
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

        genes_mask : :class:`torch.Tensor`, optional
            A 2D mask of the measured genes, as used in the
            optimization. Unmeasured genes are excluded from the
            candidates' losses.

        contamination : :class:`float`, ``0.0``
            How much of a sample the model may give up on, as used in
            the optimization.

        contamination_r : :class:`float`, ``0.05``
            The dispersion used for the contamination model.

        n_components : :class:`int`, optional
            The number of components to lay the candidates out over.
            By default, the mixture's number of components.

        Returns
        -------
        rep : :class:`torch.Tensor`
            The best representation per sample, shaped (samples,
            latent dimensionality).
        """

        # Get the total number of samples.
        n_samples = len(data_loader.dataset)

        # Get the number of components (by default, the mixture's).
        n_components = \
            self.latent.n_components if n_components is None \
            else int(n_components)

        # Get the dimensionality of the latent space.
        dim = self.latent.dim

        # Get the number of genes (= dimensionality of the decoder's
        # output).
        n_genes = self.decoder.nb.output_dim

        #-------------------------------------------------------------#

        # Initialize an empty tensor, shape (samples, dim), to store
        # the best representations found for all samples.
        best_reps = torch.empty((n_samples, dim)).to(self.device)

        #-------------------------------------------------------------#

        # For each batch: gene expression, mean gene expression, and
        # unique sample indexes.
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

            # Get the representations from the representation layer:
            # shape (samples * components * reps, dim).
            z_all = rep_layer()

            # Reshape to (samples, reps, components, dim) and select
            # this batch's samples.
            z_4d = z_all.view(n_samples,
                              n_rep_per_comp,
                              n_components,
                              dim)[samples_ixs]

            # Flatten back to (batch * reps * components, dim) for
            # the decoder.
            z = z_4d.view(n_samples_in_batch * \
                            n_rep_per_comp * \
                            n_components,
                          dim)

            #---------------------------------------------------------#

            # If the output module means that the r-values are not
            # learned
            if isinstance(\
                self.decoder.nb,
                (outputmodules.OutputModuleNBFeatureDispersion,
                 outputmodules.OutputModulePoisson)):

                # Get the predicted scaled means: shape
                # (batch * reps * components, genes).
                pred_means = self.decoder(z = z)

            # If the output module means that the r-values are learned
            elif isinstance(\
                self.decoder.nb,
                outputmodules.OutputModuleNBFullDispersion):

                # Get the predicted scaled means and r-values: both
                # shaped (batch * reps * components, genes).
                pred_means, pred_log_r_values = self.decoder(z = z)

                # Reshape the r-values to (batch, reps, components,
                # genes) to compute the loss.
                pred_log_r_values = \
                    pred_log_r_values.view(n_samples_in_batch,
                                           n_rep_per_comp,
                                           n_components,
                                           n_genes)

            #---------------------------------------------------------#

            # Expand the observed gene expression to (batch, reps,
            # components, genes).
            obs_counts = \
                samples_exp.unsqueeze(1).unsqueeze(1).expand(\
                    -1,
                    n_rep_per_comp,
                    n_components,
                    -1)

            #---------------------------------------------------------#

            # Reshape the scaling factors to (batch, 1, 1, 1),
            # broadcastable over the loss.
            scaling_factors = \
                decoders.reshape_scaling_factors(
                    samples_mean_exp,
                    4)

            #---------------------------------------------------------#

            # Get the batch's mask, if any, shaped to broadcast over the
            # representations and the components.
            mask_batch = \
                genes_mask[samples_ixs].unsqueeze(1).unsqueeze(1) \
                    if genes_mask is not None else None

            #---------------------------------------------------------#

            # Reshape the predicted means to (batch, reps, components,
            # genes) to compute the loss.
            pred_means = pred_means.view(n_samples_in_batch,
                                         n_rep_per_comp,
                                         n_components,
                                         n_genes)

            #---------------------------------------------------------#

            # If only some genes were measured
            if mask_batch is not None:

                # Re-estimate the scaling factors from the measured
                # genes, as in the optimization.
                scaling_factors = \
                    self.__class__._get_masked_scaling_factors(
                        obs_counts = obs_counts,
                        pred_means = pred_means,
                        mask = mask_batch,
                        n_genes = n_genes,
                        scaling_factor = self._scaling_factor)

            #---------------------------------------------------------#

            # If the output module means that the r-values are not
            # learned
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

            # If the output module means that the r-values are learned
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

            # If contamination is enabled (as in the optimization)
            if contamination:

                # Add the contamination options.
                recon_loss_options["contamination"] = contamination
                recon_loss_options["contamination_r"] = \
                    contamination_r

            # Get the reconstruction loss: shape (batch, reps,
            # components, genes).
            recon_loss = self.decoder.nb.loss(**recon_loss_options)

            # If only some genes were measured
            if mask_batch is not None:

                # Zero out the unmeasured genes' loss terms.
                recon_loss = recon_loss * mask_batch

            # If the reduction method is 'sum'
            if loss_reduction_type == "sum":

                # Get the reconstruction loss per representation by
                # summing over the genes: shape (batch, reps, comps).
                recon_loss_final = recon_loss.sum(-1).clone()

            # If the reduction method is 'mean'
            elif loss_reduction_type == "mean":

                # Get the reconstruction loss per representation by
                # averaging over the genes: shape (batch, reps, comps).
                recon_loss_final = recon_loss.mean(-1).clone()

            # Flatten it to (batch * reps * components).
            recon_loss_final_reshaped = \
                recon_loss_final.view(n_samples_in_batch * \
                                      n_rep_per_comp * \
                                      n_components)

            #---------------------------------------------------------#

            # If the latent space is the legacy Gaussian mixture model
            if isinstance(self.latent,
                          latents.GaussianMixtureModelLegacy):

                # Get the loss (the negative log density of 'z').
                latent_loss = self.latent(x = z).clone()

            # If the latent space is the TorchGMM wrapper
            elif isinstance(self.latent,
                            latents.GaussianMixtureModelTGMM):

                # Get the loss.
                latent_loss = \
                    - latent_lambda * self.latent.log_prob(z)

            #---------------------------------------------------------#

            # Get the total loss per representation: shape
            # (batch * reps * components).
            total_loss = recon_loss_final_reshaped + latent_loss

            #---------------------------------------------------------#

            # Reshape the total loss to (batch, reps * components).
            total_loss_reshaped = \
                total_loss.view(n_samples_in_batch,
                                n_rep_per_comp * n_components)

            #---------------------------------------------------------#

            # Get the index of the best candidate for each sample:
            # shape (batch,).
            best_rep_per_sample = torch.argmin(total_loss_reshaped,
                                               dim = 1).squeeze(-1)

            #---------------------------------------------------------#

            # Select the winning candidate's representation for each
            # sample: shape (batch, dim).
            rep = z.view(n_samples_in_batch,
                         n_rep_per_comp * n_components,
                         dim)[range(n_samples_in_batch),
                              best_rep_per_sample]

            #---------------------------------------------------------#

            # Add the batch's best representations to those of all
            # samples.
            best_reps[samples_ixs] = rep

            #---------------------------------------------------------#

            # If the selection details should be kept
            if self._keep_selection_details:

                # Without tracking gradients
                with torch.no_grad():

                    # Get the candidates: shape (batch, reps *
                    # components, dim).
                    z_cand = z.view(n_samples_in_batch,
                                    n_rep_per_comp * n_components,
                                    dim)

                    # Get the log-probability of each candidate under
                    # each component.
                    log_prob_comp = \
                        self.latent._get_log_prob_comp(
                            z_cand.reshape(-1, dim))

                    # Get the component each candidate ended in.
                    arrived = log_prob_comp.argmax(dim = 1).view(
                        n_samples_in_batch,
                        n_rep_per_comp * n_components)

                    # Get the component each candidate started in
                    # (candidates are laid out as (sample, rep, comp)).
                    born = torch.arange(
                        n_rep_per_comp * n_components,
                        device = arrived.device) % n_components

                    # Repeat it for each sample in the batch.
                    born = born.unsqueeze(0).expand(
                        n_samples_in_batch, -1)

                    # Save the batch's selection details.
                    self._selection_details.append(
                        {"samples_ixs" :
                            samples_ixs.detach().cpu().clone(),
                         "total_loss" :
                            total_loss_reshaped.detach().cpu().clone(),
                         "recon_loss" :
                            recon_loss_final_reshaped.detach().cpu(
                                ).view(n_samples_in_batch,
                                       n_rep_per_comp *
                                       n_components).clone(),
                         "latent_loss" :
                            latent_loss.detach().cpu().view(
                                n_samples_in_batch,
                                n_rep_per_comp *
                                n_components).clone(),
                         "born_in" : born.detach().cpu().clone(),
                         "arrived_in" : arrived.detach().cpu().clone(),
                         "winner" :
                            best_rep_per_sample.detach().cpu().clone()})

        #-------------------------------------------------------------#

        # Return the best representations found for the samples.
        return best_reps


    def _draw_rep_init(self,
                       n_samples: int,
                       n_rep_per_comp: int,
                       seed: Optional[int],
                       samples_names: Optional[list[str]] = None,
                       mode: str = "sample_keyed",
                       index_file: Optional[str] = None,
                       original_n_samples: Optional[int] = None,
                       chunk_size: Optional[int] = None) -> \
            torch.Tensor:
        """Draw the candidate representations for one seed.

        Parameters
        ----------
        n_samples : :class:`int`
            The number of samples to initialize.

        n_rep_per_comp : :class:`int`
            The number of candidate representations to draw from each
            mixture component for each sample.

        seed : :class:`int`, optional
            The initialization seed, required by ``sample_keyed`` and
            ``legacy_indexed``.

        samples_names : :class:`list` of :class:`str`, optional
            The sample names, required by ``sample_keyed`` and
            ``legacy_indexed``.

        mode : :class:`str`, {``"sample_keyed"``, \
            ``"legacy_positional"``, ``"legacy_indexed"``}
            The initialization mode: ``sample_keyed`` draws an
            independent stream per sample ID, ``legacy_positional``
            one global stream in chunk order, and ``legacy_indexed``
            the ``legacy_positional`` draws of the positions in an
            index table.

        index_file : :class:`str`, optional
            The CSV file mapping sample names to historical absolute
            positions for ``legacy_indexed``.

        original_n_samples : :class:`int`, optional
            The number of samples in the historical input for
            ``legacy_indexed``.

        chunk_size : :class:`int`, optional
            The outer chunk size used by the historical run for
            ``legacy_indexed``.

        Returns
        -------
        rep_init : :class:`torch.Tensor`
            The initialized candidate representations.
        """

        # Set the supported initialization modes.
        modes = {"sample_keyed", "legacy_positional", "legacy_indexed"}

        # If the user provided an unsupported initialization mode
        if mode not in modes:

            # Raise an error.
            raise ValueError(
                "Unsupported representation initialization mode "
                f"'{mode}'. The supported modes are: "
                f"{', '.join(sorted(modes))}.")

        # Get the number of components in the Gaussian mixture model.
        n_components = self.latent.n_components

        # Get the dimensionality of the latent space.
        n_dim = self.latent.dim

        #-------------------------------------------------------------#

        def draw_legacy_chunk(chunk_n_samples: int) -> torch.Tensor:
            """Draw candidates for one chunk, using the legacy
            positional draw order.

            Parameters
            ----------
            chunk_n_samples : :class:`int`
                The number of samples in this chunk.

            Returns
            -------
            component_samples : :class:`torch.Tensor`
                The drawn candidates for the chunk, shaped (samples,
                reps, components, dim).
            """

            # Initialize the list of per-component draws.
            component_samples = []

            # In a context that may hold a forked RNG
            with contextlib.ExitStack() as stack:

                # If a seed was given
                if seed is not None:

                    # Fork the RNG to leave the global stream alone.
                    stack.enter_context(
                        torch.random.fork_rng(devices = []))

                    # Seed the forked RNG.
                    torch.manual_seed(int(seed))

                # For each component, in order
                for comp_idx in range(n_components):

                    # Draw this component's samples for the chunk.
                    samples_comp, _ = self.latent.sample(
                        n_samples = chunk_n_samples * n_rep_per_comp,
                        component = comp_idx)

                    # Add them to the list.
                    component_samples.append(samples_comp)

            # Stack the per-component draws into one tensor.
            component_samples = torch.stack(component_samples, dim = 0)

            # Reshape into the legacy positional layout.
            component_samples = component_samples.view(
                n_components,
                chunk_n_samples,
                n_rep_per_comp,
                n_dim)

            # Return the draws as (samples, reps, components, dim).
            return component_samples.permute(1,
                                             2,
                                             0,
                                             3)

        #-------------------------------------------------------------#

        def draw_sample_keyed(sample_ids: list[str]) -> torch.Tensor:
            """Draw candidates keyed by sample identity, independent
            of row, chunk and order.

            Parameters
            ----------
            sample_ids : :class:`list`
                The sample IDs to draw candidates for, in the order
                the result is returned in.

            Returns
            -------
            samples : :class:`torch.Tensor`
                The drawn candidates, shaped (samples, reps,
                components, dim).
            """

            # Initialize the list of per-sample standard normal draws.
            standard_normal = []

            # For each sample ID
            for sample_id in sample_ids:

                # Build the per-sample payload to seed the digest with.
                payload = \
                    f"{int(seed)}\0{sample_id}".encode("utf-8")

                # Hash the payload into a fixed-size digest (Python's
                # 'hash' is randomized between processes).
                digest = hashlib.blake2b(
                    payload,
                    digest_size = 8,
                    person = b"BulkDGD.init.v1").digest()

                # Turn the digest into a valid torch seed.
                sample_seed = \
                    int.from_bytes(digest,
                                   byteorder = "big",
                                   signed = False) & ((1 << 63) - 1)

                # Create a dedicated CPU generator for this sample.
                generator = torch.Generator(device = "cpu")

                # Seed the generator.
                generator.manual_seed(sample_seed)

                # Draw the standard normal samples for this sample ID.
                standard_normal.append(
                    torch.randn((n_rep_per_comp, n_components, n_dim),
                                generator = generator,
                                device = "cpu",
                                dtype = self.latent.means.dtype))

            # Stack the per-sample draws into one tensor.
            standard_normal = torch.stack(standard_normal, dim = 0)

            # Get the components' means.
            means = self.latent.means

            # Get the component indices.
            components = torch.arange(n_components,
                                      dtype = torch.long,
                                      device = means.device)

            # Build the covariances used for sampling.
            covariances = \
                self.latent._build_covariances_for_sampling(
                    components, n_components)

            # Get the Cholesky factor of the covariances.
            scale_tril = torch.linalg.cholesky(covariances)

            # Move the standard normals to the means' device.
            standard_normal = standard_normal.to(device = means.device)

            # Shift and scale the standard normals by each component's
            # mean and covariance, and return them.
            return means.view(1,
                              1,
                              n_components,
                              n_dim) + \
                torch.einsum("srcj,cij->srci",
                             standard_normal,
                             scale_tril)

        #-------------------------------------------------------------#

        # If the legacy positional initialization mode was requested
        if mode == "legacy_positional":

            # Draw, flatten and return the candidates.
            return draw_legacy_chunk(n_samples).reshape(
                n_samples * n_rep_per_comp * n_components, n_dim)

        #-------------------------------------------------------------#

        # If the legacy-indexed initialization mode was requested
        if mode == "legacy_indexed":

            # If no seed was given
            if seed is None:

                # Raise an error.
                raise ValueError(
                    "The 'legacy_indexed' representation "
                    "initialization mode requires "
                    "'scheme_options.initialization.seed'.")

            # If no sample names were given
            if samples_names is None:

                # Raise an error.
                raise ValueError(
                    "The 'legacy_indexed' representation "
                    "initialization mode requires the samples' IDs.")

            # If no index file was given
            if index_file is None:

                # Raise an error.
                raise ValueError(
                    "The 'legacy_indexed' representation "
                    "initialization mode requires 'index_file'.")

            # If the original number of samples is missing or invalid
            if original_n_samples is None or \
                    int(original_n_samples) <= 0:

                # Raise an error.
                raise ValueError(
                    "The 'legacy_indexed' representation "
                    "initialization mode requires a positive "
                    "'original_n_samples'.")

            # If the chunk size is missing or invalid
            if chunk_size is None or int(chunk_size) <= 0:

                # Raise an error.
                raise ValueError(
                    "The 'legacy_indexed' representation "
                    "initialization mode requires a positive "
                    "'chunk_size'.")

            # Convert the original number of samples to an integer.
            original_n_samples = int(original_n_samples)

            # Convert the chunk size to an integer.
            chunk_size = int(chunk_size)

            # Convert the sample names to strings.
            samples_names = \
                [str(sample_id) for sample_id in samples_names]

            # If the number of sample names does not match the number
            # of samples to initialize
            if len(samples_names) != n_samples:

                # Raise an error.
                raise ValueError(
                    "The indexed legacy initializer received "
                    f"{len(samples_names)} sample IDs for {n_samples} "
                    "samples.")

            # If the sample names are not unique
            if len(set(samples_names)) != len(samples_names):

                # Raise an error.
                raise ValueError(
                    "Sample IDs must be unique after conversion to "
                    "strings when 'legacy_indexed' initialization is "
                    "used.")

            # Load the table of historical positions.
            df_positions = pd.read_csv(index_file,
                                       sep = ",",
                                       index_col = 0)

            # If the table does not have exactly one data column
            if df_positions.shape[1] != 1:

                # Raise an error.
                raise ValueError(
                    f"The legacy index file '{index_file}' must have "
                    "exactly one data column; found "
                    f"{df_positions.shape[1]}.")

            # If the table has duplicate sample IDs
            if df_positions.index.has_duplicates:

                # Raise an error.
                raise ValueError(
                    f"The legacy index file '{index_file}' contains "
                    "duplicate sample IDs.")

            # Convert the table's index to strings.
            df_positions.index = df_positions.index.map(str)

            # Get the sample IDs missing a position in the table.
            missing = \
                sorted(set(samples_names) - set(df_positions.index))

            # If any sample IDs are missing
            if missing:

                # Raise an error.
                raise ValueError(
                    f"The legacy index file '{index_file}' has no "
                    "position for the following sample IDs: "
                    f"{missing}.")

            # Get the positions as strings (a row holding the sample's
            # own ID marks a sample-keyed sample).
            positions_raw = df_positions.iloc[:, 0].astype(str)

            # Get the sample-keyed samples.
            keyed_ids = \
                [sample_id for sample_id in samples_names
                 if positions_raw.loc[sample_id] == sample_id]

            # Get the legacy positional samples.
            legacy_ids = \
                [sample_id for sample_id in samples_names
                 if sample_id not in keyed_ids]

            # Initialize the candidates drawn per sample.
            drawn = {}

            # If there are sample-keyed samples
            if keyed_ids:

                # Draw their candidates.
                keyed_samples = draw_sample_keyed(keyed_ids)

                # For each sample-keyed sample
                for i, sample_id in enumerate(keyed_ids):

                    # Save its candidates.
                    drawn[sample_id] = keyed_samples[i]

            #---------------------------------------------------------#

            # If there are non-keyed (legacy-positional) samples
            if legacy_ids:

                # Get their raw, string-typed positions.
                legacy_positions_raw = positions_raw.loc[legacy_ids]

                # Convert the positions to numbers.
                positions_numeric = pd.to_numeric(
                    legacy_positions_raw,
                    errors = "coerce")

                # If any position failed to convert
                if positions_numeric.isna().any():

                    # Raise an error.
                    raise ValueError(
                        "Every non-keyed position in the legacy "
                        f"index file '{index_file}' must be an "
                        "integer.")

                # Convert the positions to a float array.
                positions_float = \
                    positions_numeric.to_numpy(dtype = np.float64)

                # If any position is not an integer value
                if not np.equal(positions_float,
                                np.floor(positions_float)).all():

                    # Raise an error.
                    raise ValueError(
                        "Every non-keyed position in the legacy "
                        f"index file '{index_file}' must be an "
                        "integer.")

                # Map each sample ID to its integer position.
                positions = \
                    {sample_id : int(positions_numeric.loc[sample_id])
                     for sample_id in legacy_ids}

                # If two sample IDs map to the same position
                if len(set(positions.values())) != len(positions):

                    # Raise an error.
                    raise ValueError(
                        "The requested samples map to duplicate "
                        "historical positions in the legacy index "
                        "file.")

                # Get any position outside the historical sample range.
                invalid = \
                    {sample_id : position
                     for sample_id, position in positions.items()
                     if position < 0 or position >= original_n_samples}

                # If any position is invalid
                if invalid:

                    # Raise an error.
                    raise ValueError(
                        "Legacy positions must be in [0, "
                        f"{original_n_samples - 1}]; got {invalid}.")

                # Cache one draw per chunk length (the seed is reset
                # for each chunk, so equal lengths give equal draws).
                chunks_by_length = {}

                # For each non-keyed sample
                for sample_id in legacy_ids:

                    # Get its absolute historical position.
                    absolute_position = positions[sample_id]

                    # Get the start of the historical chunk it fell in.
                    chunk_start = \
                        (absolute_position // chunk_size) * chunk_size

                    # Get the size of that historical chunk.
                    old_chunk_n_samples = \
                        min(chunk_size,
                            original_n_samples - chunk_start)

                    # Get its position within that chunk.
                    position_in_chunk = \
                        absolute_position - chunk_start

                    # If this chunk length has not been drawn yet
                    if old_chunk_n_samples not in chunks_by_length:

                        # Draw and cache it.
                        chunks_by_length[old_chunk_n_samples] = \
                            draw_legacy_chunk(old_chunk_n_samples)

                    # Pick out this sample's candidates.
                    drawn[sample_id] = \
                        chunks_by_length[old_chunk_n_samples][
                            position_in_chunk]

            #---------------------------------------------------------#

            # Reassemble in the caller's requested order.
            selected = [drawn[sample_id] for sample_id in samples_names]

            # Flatten and return the candidates.
            return torch.stack(selected, dim = 0).reshape(
                n_samples * n_rep_per_comp * n_components, n_dim)

        #-------------------------------------------------------------#

        # If no seed was given
        if seed is None:

            # Raise an error.
            raise ValueError(
                "The 'sample_keyed' representation initialization mode "
                "requires 'scheme_options.initialization.seed'.")

        # If no sample names were given
        if samples_names is None:

            # Raise an error.
            raise ValueError(
                "The 'sample_keyed' representation initialization mode "
                "requires the samples' IDs.")

        # Convert the sample names to strings.
        samples_names = [str(sample_id) for sample_id in samples_names]

        # If the number of sample names does not match the number of
        # samples to initialize
        if len(samples_names) != n_samples:

            # Raise an error.
            raise ValueError(
                "The sample-keyed initializer received "
                f"{len(samples_names)} sample IDs for {n_samples} "
                "samples.")

        # If the sample names are not unique
        if len(set(samples_names)) != len(samples_names):

            # Raise an error.
            raise ValueError(
                "Sample IDs must be unique after conversion to strings "
                "when 'sample_keyed' initialization is used.")

        # Draw and return the candidates.
        return draw_sample_keyed(samples_names).reshape(
            n_samples * n_rep_per_comp * n_components, n_dim)


    def _get_representations_two_opt(
            self,
            dataset: dataclasses.GeneExpressionDataset,
            config: dict[str, object],
            genes_mask: Optional[torch.Tensor] = None) -> \
                tuple[torch.Tensor, torch.Tensor,
                      Optional[torch.Tensor], list[tuple]]:
        """Get the best representations via the two-optimization
        scheme: initialize candidates, optimize, select the best per
        sample, then optimize those further.

        Parameters
        ----------
        dataset : \
            :class:`bulkdgd.core.dataclasses.GeneExpressionDataset`
            The dataset from which the data loader should be created.

        config : :class:`dict`
            A dictionary with the options to run the optimization.

        genes_mask : :class:`torch.Tensor`, optional
            A 2D mask of which genes were measured for each sample.
            By default, all genes are measured.

        Returns
        -------
        rep : :class:`torch.Tensor`
            The optimized representations, shaped (samples, latent
            dimensionality).

        pred_means : :class:`torch.Tensor`
            The predicted gene-count means, shaped (samples, genes).

        pred_r_values : :class:`torch.Tensor` or :obj:`None`
            The predicted negative-binomial r-values, same shape as
            ``pred_means``, or :obj:`None` for Poisson counts.

        time_opt : :class:`list`
            Per-epoch CPU/wall-clock timing for the epoch and its
            backpropagation step.
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

        # Reset the selection details.
        self._selection_details = []

        # Reset the record of which component each sample settled in.
        self._settled_in = None

        #-------------------------------------------------------------#

        # Get the configuration for the first optimization.
        config_opt_1 = config["scheme_options"]["optimization_1"]

        # Get the type of optimizer for the first optimization.
        optimizer_type_1 = config_opt_1["optimizer_type"]

        # Get the options for the optimizer for the first optimization.
        optimizer_options_1 = config_opt_1["optimizer_options"]

        # Get the number of epochs to run the first optimization for.
        epochs_1 = config_opt_1["epochs"]

        # Get the noise to inject during the first optimization, if
        # any.
        noise_type_1 = config_opt_1.get("noise_type")
        noise_options_1 = config_opt_1.get("noise_options")

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

        # Get the noise to inject during the second optimization, if
        # any.
        noise_type_2 = config_opt_2.get("noise_type")
        noise_options_2 = config_opt_2.get("noise_options")

        #-------------------------------------------------------------#

        # Create the data loader.
        data_loader = \
            _util.get_data_loader(dataset = dataset,
                                  config = data_loader_options)

        #-------------------------------------------------------------#

        # Get the number of components.
        n_components = self.latent.n_components

        # Get the dimensionality.
        n_dim = self.latent.dim

        #-------------------------------------------------------------#

        # If the latent space is the legacy Gaussian mixture model
        if isinstance(self.latent,
                      latents.GaussianMixtureModelLegacy):

            # Get the initial representations by sampling new points
            # from the GMM.
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

            #---------------------------------------------------------#

            # Get the initialization options.
            init_options = \
                config["scheme_options"].get("initialization", {})

            # Get the initialization seed.
            seed = init_options.get("seed")

            # Get the initialization mode (sample-keyed by default).
            mode = init_options.get("mode", "sample_keyed")

            #---------------------------------------------------------#

            # Draw the initial candidates.
            rep_init = \
                self._draw_rep_init(
                    n_samples = n_samples,
                    n_rep_per_comp = n_rep_per_comp,
                    seed = seed,
                    samples_names = dataset.samples,
                    mode = mode,
                    index_file = init_options.get("index_file"),
                    original_n_samples = \
                        init_options.get("original_n_samples"),
                    chunk_size = init_options.get("chunk_size"))

        #-------------------------------------------------------------#

        # Get the warm-start options, if any.
        warm_start_cfg = \
            config.get("scheme_options", {}).get("warm_start") or {}

        # If a warm-start checkpoint was given
        if warm_start_cfg.get("pth_file"):

            # Load the ridge warm-start model.
            ws = warmstart.RidgeWarmStart.from_file(
                warm_start_cfg["pth_file"])

            # Predict a starting representation for each sample.
            z_ws = ws.predict(
                dataset.data_exp.cpu().numpy(),
                device = rep_init.device).to(rep_init.dtype)

            # Get the total number of candidates per sample.
            n_cand = n_rep_per_comp * n_components

            # Reshape the candidates to (samples, candidates, dim).
            rep_init = rep_init.view(n_samples,
                                     n_cand,
                                     n_dim)

            # Replace each sample's first candidate with the warm-start
            # prediction.
            rep_init[:, 0, :] = z_ws

            # Flatten the candidates back.
            rep_init = rep_init.reshape(n_samples * n_cand, n_dim)

            # Inform the user that the warm start was applied.
            logger.info(
                "The ridge warm start from "
                f"'{warm_start_cfg['pth_file']}' replaced one of the "
                f"{n_cand} candidates of each sample. The other "
                f"{n_cand - 1} are drawn from the mixture as usual.")

        #-------------------------------------------------------------#

        # Create the representation layer.
        rep_layer_init = \
            latents.RepresentationLayer(values = rep_init).to(\
                self.device)

        #-------------------------------------------------------------#

        # Get how much of a sample the model may give up on (zero
        # disables the contamination model).
        contamination = \
            float(config["scheme_options"].get("contamination", 0.0))

        # Get the dispersion used for the contamination model.
        contamination_r = \
            float(config["scheme_options"].get("contamination_r", 0.05))

        # If contamination is enabled
        if contamination:

            # Inform the user.
            logger.info(
                "Representations will be found with a contaminated "
                "output distribution (contamination "
                f"{contamination:.2e}, r {contamination_r:g}).")

        # Get the optimizer for the first optimization.
        optimizer_1 = \
            self._get_optimizer(\
                optimizer_type = optimizer_type_1,
                optimizer_options = optimizer_options_1,
                optimizer_parameters = rep_layer_init.parameters())

        #-------------------------------------------------------------#

        # Get the optimized representations and the time data.
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
                latent_lambda = latent_lambda,
                genes_mask = genes_mask,
                contamination = contamination,
                contamination_r = contamination_r,
                noise_type = noise_type_1,
                noise_options = noise_options_1)

        #-------------------------------------------------------------#

        # Create the representation layer.
        rep_layer_1 = \
            latents.RepresentationLayer(values = rep_1,
                                        device = self.device)

        #-------------------------------------------------------------#

        # Make the first optimizer's gradients zero.
        optimizer_1.zero_grad()

        #-------------------------------------------------------------#

        # Select the best representation for each sample among
        # those initialized.
        rep_best = \
            self._select_best_rep(\
                data_loader = data_loader,
                rep_layer = rep_layer_1,
                n_rep_per_comp = n_rep_per_comp,
                loss_reduction_type = loss_reduction_type,
                latent_lambda = latent_lambda,
                genes_mask = genes_mask,
                contamination = contamination,
                contamination_r = contamination_r)

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

        # Get the optimized representations (one per sample),
        # predicted means and r-values (if any), and time data.
        rep_2, pred_means_2, pred_r_values_2, time_2 = \
            self._optimize_rep(\
                data_loader = data_loader,
                rep_layer = rep_layer_best,
                optimizer = optimizer_2,
                n_components = 1,
                n_rep_per_comp = 1,
                loss_reporting_options = loss_reporting_options,
                loss_reduction_type = loss_reduction_type,
                epochs = epochs_2,
                opt_num = 2,
                latent_lambda = latent_lambda,
                genes_mask = genes_mask,
                contamination = contamination,
                contamination_r = contamination_r,
                noise_type = noise_type_2,
                noise_options = noise_options_2)

        #-------------------------------------------------------------#

        # Make the second optimizer's gradients zero.
        optimizer_2.zero_grad()

        #-------------------------------------------------------------#

        # Concatenate the two lists storing the time data for both
        # optimizations.
        time = time_1 + time_2

        #-------------------------------------------------------------#

        # If the selection details should be kept
        if self._keep_selection_details \
                and isinstance(self.latent,
                               latents.GaussianMixtureModelTGMM):

            # Without tracking gradients
            with torch.no_grad():

                # Get the component each sample ended in after the
                # second optimization.
                self._settled_in = \
                    self.latent._get_log_prob_comp(
                        rep_2.to(self.device)).argmax(
                            dim = 1).detach().cpu().clone()

        #-------------------------------------------------------------#

        # Return the representations, the predicted means and r-values,
        # and the time data.
        return rep_2, pred_means_2, pred_r_values_2, time


    def _final_losses(self,
                      data_loader: torch.utils.data.DataLoader,
                      rep: torch.Tensor,
                      loss_reduction_type: str,
                      latent_lambda: Optional[float],
                      genes_mask: Optional[torch.Tensor],
                      contamination: float,
                      contamination_r: float) -> np.ndarray:
        """Get the current loss of each sample's representation.

        Parameters
        ----------
        data_loader : :class:`torch.utils.data.DataLoader`
            The data loader.

        rep : :class:`torch.Tensor`
            The current representations, one per sample.

        loss_reduction_type : :class:`str`
            The method to reduce the loss across the samples in the
            batch.

        latent_lambda : :class:`float` or :obj:`None`
            The weight of the latent loss term in the total loss.

        genes_mask : :class:`torch.Tensor` or :obj:`None`
            A 2D mask of which genes were measured for each sample.

        contamination : :class:`float`
            How much of a sample the model may give up on.

        contamination_r : :class:`float`
            The dispersion used for the contamination model.

        Returns
        -------
        losses : :class:`numpy.ndarray`
            The per-sample loss.
        """

        # Get the number of samples.
        n_samples = rep.shape[0]

        # Save the selection details' state to restore afterwards.
        keep_before = self._keep_selection_details
        details_before = self._selection_details

        # Record the selection details for this call.
        self._keep_selection_details = True
        self._selection_details = []

        # Try to compute the losses.
        try:

            # Run the selection with a single candidate per sample to
            # compute its loss.
            self._select_best_rep(
                data_loader = data_loader,
                rep_layer = latents.RepresentationLayer(
                    values = rep, device = self.device),
                n_rep_per_comp = 1,
                loss_reduction_type = loss_reduction_type,
                latent_lambda = latent_lambda,
                genes_mask = genes_mask,
                contamination = contamination,
                contamination_r = contamination_r,
                n_components = 1)

            # Initialize the per-sample loss array to NaN.
            out = np.full(n_samples,
                          np.nan,
                          dtype = np.float64)

            # For each batch's recorded selection details
            for d in self._selection_details:

                # Get the batch's sample indices.
                ixs = d["samples_ixs"].numpy()

                # Fill in their losses.
                out[ixs] = d["total_loss"].squeeze(-1).numpy()

            # Return the per-sample losses.
            return out

        # Afterwards
        finally:

            # Restore the selection details' state.
            self._keep_selection_details = keep_before
            self._selection_details = details_before


    def _get_representations_two_opt_multiseed(
            self,
            dataset: dataclasses.GeneExpressionDataset,
            config: dict[str, object],
            genes_mask: Optional[torch.Tensor] = None) -> \
                tuple[torch.Tensor, torch.Tensor,
                      Optional[torch.Tensor], list[tuple]]:
        """Run the two-optimization scheme once per seed, and keep
        every seed's results in ``multiseed_results``.

        Parameters
        ----------
        dataset : \
            :class:`bulkdgd.core.dataclasses.GeneExpressionDataset`
            The dataset from which the data loader should be created.

        config : :class:`dict`
            A dictionary with the options to run the optimization.

        genes_mask : :class:`torch.Tensor`, optional
            A 2D mask of which genes were measured for each sample.
            By default, all genes are measured.

        Returns
        -------
        rep : :class:`torch.Tensor`
            The optimized representations for the first seed.

        pred_means : :class:`torch.Tensor`
            The predicted means for the first seed.

        pred_r_values : :class:`torch.Tensor` or :obj:`None`
            The predicted r-values for the first seed, or :obj:`None`
            if the counts are modelled by Poisson distributions.

        time_opt : :class:`list`
            A list of tuples storing, for each epoch, information
            about the CPU and wall clock time used.
        """

        # Get the seeds to run the scheme with.
        seeds = \
            config["scheme_options"]["initialization"]["seeds"]

        # Convert the seeds to integers.
        seeds = [int(s) for s in seeds]

        # If the seeds are not distinct
        if len(set(seeds)) != len(seeds):

            # Raise an error.
            raise ValueError(
                f"The seeds must be distinct; got {seeds}.")

        # Inform the user.
        logger.info(
            f"Representations will be found from {len(seeds)} "
            "independent initializations (seeds "
            f"{', '.join(map(str, seeds))}).")

        #-------------------------------------------------------------#

        # Initialize the per-seed results.
        reps, pred_means, pred_r_values = {}, {}, {}

        # Initialize the per-seed losses.
        losses = {}

        # Initialize the combined timing data.
        time_all = []

        # Create the data loader for computing the final losses.
        data_loader = \
            _util.get_data_loader(
                dataset = dataset,
                config = config["data_loader_options"])

        # Get the latent loss weight (TGMM only).
        latent_lambda = \
            config["scheme_options"]["latent_loss_calculation"][\
                "lambda"] \
            if isinstance(self.latent,
                          latents.GaussianMixtureModelTGMM) else None

        # Get the contamination options.
        contamination = \
            float(config["scheme_options"].get("contamination", 0.0))
        contamination_r = \
            float(config["scheme_options"].get("contamination_r", 0.05))

        # Get the method to reduce the loss across a batch's samples.
        loss_reduction_type = \
            config["scheme_options"]["loss_reduction_type"]

        # Get the total number of samples.
        n_samples = len(dataset.samples)

        # For each seed (run separately to reproduce a single-seed
        # run)
        for seed in seeds:

            # Copy the configuration.
            cfg = copy.deepcopy(config)

            # Set this seed as the single initialization seed.
            cfg["scheme_options"]["initialization"] = \
                {**cfg["scheme_options"].get("initialization", {}),
                 "seed" : seed}

            # Remove the multiseed 'seeds' key from the copy.
            cfg["scheme_options"]["initialization"].pop("seeds", None)

            # Inform the user.
            logger.info(f"Optimizing from seed {seed}...")

            # Save the selection details' state to restore afterwards.
            keep_before = self._keep_selection_details
            details_before = self._selection_details

            # Record the selection details for this seed.
            self._keep_selection_details = True
            self._selection_details = []

            # Try to find the representations.
            try:

                # Run the two-optimization scheme for this seed.
                rep, pm, prv, t = \
                    self._get_representations_two_opt(
                        dataset = dataset,
                        config = cfg,
                        genes_mask = genes_mask)

                # Initialize the winners' losses at the end of the
                # first optimization.
                won = np.full(n_samples,
                              np.nan,
                              dtype = np.float64)

                # For each batch's recorded selection details
                for d in self._selection_details:

                    # Get the batch's sample indices.
                    ixs = d["samples_ixs"].numpy()

                    # Get the batch's total losses.
                    tl = d["total_loss"].numpy()

                    # Fill in the winner's loss for each sample.
                    won[ixs] = tl[np.arange(len(ixs)),
                                  d["winner"].numpy()]

            # Afterwards
            finally:

                # Restore the selection details' state.
                self._keep_selection_details = keep_before
                self._selection_details = details_before

            # Store this seed's representations and predicted means.
            reps[seed], pred_means[seed] = rep, pm

            # Store this seed's predicted r-values.
            pred_r_values[seed] = prv

            # Accumulate this seed's timing data.
            time_all.extend(t)

            # Store this seed's first-optimization losses.
            losses[f"loss_opt1_seed{seed}"] = won

            # Store this seed's second-optimization losses.
            losses[f"loss_opt2_seed{seed}"] = \
                self._final_losses(
                    data_loader = data_loader,
                    rep = rep,
                    loss_reduction_type = loss_reduction_type,
                    latent_lambda = latent_lambda,
                    genes_mask = genes_mask,
                    contamination = contamination,
                    contamination_r = contamination_r)

        #-------------------------------------------------------------#

        # Initialize the loss columns (two per seed, in seed order).
        cols = []

        # For each seed
        for seed in seeds:

            # Add its two loss columns.
            cols += [f"loss_opt1_seed{seed}", f"loss_opt2_seed{seed}"]

        # Assemble the losses data frame.
        df_losses = pd.DataFrame({c: losses[c] for c in cols},
                                 index = dataset.samples)

        # Name the index.
        df_losses.index.name = "sample"

        # Save every seed's results.
        self.multiseed_results = \
            {"seeds" : seeds,
             "representations" : reps,
             "pred_means" : pred_means,
             "pred_r_values" : pred_r_values,
             "losses" : df_losses}

        #-------------------------------------------------------------#

        # Get the first seed.
        first = seeds[0]

        # Return the first seed's results.
        return (reps[first], pred_means[first], pred_r_values[first],
                time_all)


    def keep_selection_details(self,
                               keep: bool = True) -> None:
        """Set whether to record the details of the selection of the
        best representations, not only the winners.

        Parameters
        ----------
        keep : :class:`bool`, ``True``
            Whether to record them.
        """

        # Set whether to record the details.
        self._keep_selection_details = bool(keep)


    def get_selection_details(
            self,
            samples_names: Optional[list] = None) -> pd.DataFrame:
        """Get the details of the selection of the best
        representations, one row per candidate per sample.

        Parameters
        ----------
        samples_names : :class:`list`, optional
            The samples' names, in the order the model was given them.

        Returns
        -------
        df : :class:`pandas.DataFrame`
            One row per candidate per sample. Columns: ``sample``,
            ``candidate``; ``born_in``/``arrived_in``/``settled_in``,
            the component at start, at selection, and (winner only)
            after the second optimization; ``recon_loss``,
            ``latent_loss``, ``total_loss`` in nats; ``is_winner``;
            ``margin`` (loss above the winner's); ``softmax``
            (unreliable at this loss scale).
        """

        # If there is nothing recorded
        if not self._selection_details:

            # Raise an error.
            raise RuntimeError(
                "There is nothing recorded. Call "
                "'keep_selection_details()' before "
                "'get_representations()'.")

        #-------------------------------------------------------------#

        # Initialize the list of output rows.
        rows = []

        # For each batch's recorded selection details
        for batch in self._selection_details:

            # Get the batch's sample indices.
            ixs = batch["samples_ixs"]

            # Get the batch's total losses.
            total = batch["total_loss"]

            # Get each sample's best (lowest) loss.
            best = total.min(dim = 1, keepdim = True).values

            # Get the softmax of the negated losses.
            soft = torch.softmax(-total.double(), dim = 1)

            # Get the number of candidates per sample.
            n_cand = total.shape[1]

            # For each sample in the batch
            for i, ix in enumerate(ixs.tolist()):

                # Get the sample's name, if given.
                name = samples_names[ix] \
                    if samples_names is not None else ix

                # Get the component the sample settled in, if recorded.
                settled = int(self._settled_in[ix]) \
                    if self._settled_in is not None else None

                # For each candidate
                for c in range(n_cand):

                    # Add the candidate's row.
                    rows.append(
                        {"sample" : name,
                         "candidate" : c,
                         "born_in" : int(batch["born_in"][i, c]),
                         "arrived_in" : int(batch["arrived_in"][i, c]),
                         "settled_in" : settled,
                         "recon_loss" :
                            float(batch["recon_loss"][i, c]),
                         "latent_loss" :
                            float(batch["latent_loss"][i, c]),
                         "total_loss" : float(total[i, c]),
                         "is_winner" :
                            bool(c == int(batch["winner"][i])),
                         "margin" : float(total[i, c] - best[i, 0]),
                         "softmax" : float(soft[i, c])})

        #-------------------------------------------------------------#

        # Assemble and return the rows as a data frame.
        return pd.DataFrame(rows)


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
            A 2D CPU tensor of shape (n_genes, latent_dim) containing
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

        # For each gene
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
        # graph, on the CPU.
        return saliency_map.detach().cpu()


    def _get_best_latent_tgmm(self,
                              rep_train: torch.Tensor,
                              latent_n_components_target: int,
                              max_iter: int,
                              is_full_refit_epoch: bool,
                              epoch: int,
                              model_selection_metric: str,
                              model_selection_step: int = 1) -> None:
        """Replace the latent space with the best TGMM candidate across
        nearby numbers of components, ranked by a model-selection
        metric.

        Parameters
        ----------
        rep_train : :class:`torch.Tensor`
            The current representations used to fit candidate models.

        latent_n_components_target : :class:`int`
            The maximum number of components.

        max_iter : :class:`int`
            The maximum number of EM iterations for each candidate
            fit.

        is_full_refit_epoch : :class:`bool`
            Whether the current epoch corresponds to a full refit.

        epoch : :class:`int`
            The current epoch.

        model_selection_metric : :class:`str`
            The metric used to rank candidates.

        model_selection_step : :class:`int`, ``1``
            How many components either side of the current number to
            try as candidates.
        """

        # Get the latent space's dimensionality.
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

        # Get the maximum number of components.
        gmm_n_components_ceiling = latent_n_components_target

        #-------------------------------------------------------------#

        # Clamp the current number of components between 1 and the
        # maximum.
        current_n_components = \
            max(1,
                min(latent_n_components_target,
                    self.latent.n_components))

        #-------------------------------------------------------------#

        # Initialize the candidate number of components to evaluate to
        # the current models' number of components.
        candidates_n_components = set([current_n_components])

        # Get how many components either side of the current number to
        # try.
        step = max(1, int(model_selection_step))

        # If the lower candidate has at least one component
        if current_n_components - step >= 1:

            # Add it.
            candidates_n_components.add(current_n_components - step)

        # If the upper candidate does not exceed the maximum
        if current_n_components + step <= gmm_n_components_ceiling:

            # Add it.
            candidates_n_components.add(current_n_components + step)

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

            # If there is no best model yet, use the current candidate
            # as a fallback.
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

                # It is better if its selection value is higher than
                # the best one.
                is_better = selection_value > best_selection_value

            # If both the current candidate and the best one found so
            # far are valid and the optimization direction is 'min'
            elif optimize_direction == "min":

                # It is better if its selection value is lower than
                # the best one.
                is_better = selection_value < best_selection_value

            #---------------------------------------------------------#

            # If the current candidate is better than the best one
            # found so far
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
        if best_selection_value is None \
                or np.isnan(best_selection_value):

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
                "dynamic component selection."
            raise RuntimeError(err_msg)

        #-------------------------------------------------------------#

        # Keep the best model.
        self._latent = best_model

        # Set the latent space's number of components to the best one.
        self.latent.n_components = int(best_n_components)

        #-------------------------------------------------------------#

        # Get the candidates' metric values as a string.
        candidate_selection_str = \
            ", ".join([
                f"number of components={k}: "
                f"{candidate_selection_values[k]:.6f}"
                for k in candidates_n_components])

        # Inform the user about the selection.
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
        """Remove the collapsed components (with a mixture weight below
        ``collapse_weight_threshold``) from the latent space.

        Parameters
        ----------
        collapse_weight_threshold : :class:`float`
            The weight below which a component is considered collapsed.

        epoch : :class:`int`
            The current training epoch (used for logging).

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

                # Keep the covariances for all components (they are
                # re-initialized in the new model).
                covariances_new = \
                    self.latent.covariances_.detach().clone()

            #---------------------------------------------------------#

            # Get the weight concentration prior.
            weight_concentration_prior = \
                self.latent.weight_concentration_prior

            # If the weight concentration prior has one value per
            # component before removal
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

            # If the mean prior has one row per component before
            # removal
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
                # prior has one value per component before removal
                if covariance_type == "spherical" and \
                    covariance_prior.ndim == 1 and \
                    covariance_prior.numel() == n_components_before:

                    # Slice the covariance prior for the active
                    # components.
                    covariance_prior = \
                        covariance_prior[keep_ixs].detach().clone()

                # If the covariance type is 'full' or 'diag' and the
                # prior has one entry per component before removal
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

            # Keep the fitted parameters.
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

                # Use the new weights as the initial weights.
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

                # Use the new means as the initial means.
                latent_new.initial_means_ = means_new.detach().clone()

            #---------------------------------------------------------#

            # If there are initial covariances
            if self.latent.initial_covariances_ is not None:

                # If the covariance type is 'full', 'diag', or
                # 'spherical' and there is one per component
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

                    # Keep the initial covariances for all the
                    # components.
                    latent_new.initial_covariances_ = \
                        self.latent.initial_covariances_.detach(
                            ).clone()

            # Otherwise
            else:

                # Use the new covariances as the initial covariances.
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

            # Keep the dimensionality and the number of components.
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
            initial_options = self._latent_initial_options

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

            # Copy the means, weights, and log-variances of the kept
            # components.
            with torch.no_grad():
                latent_new.means.copy_(self.latent.means[keep_ixs])
                latent_new.weights.copy_(self.latent.weights[keep_ixs])
                latent_new.log_var.copy_(self.latent.log_var[keep_ixs])

            # Replace the current GMM.
            self._latent = latent_new

        #-------------------------------------------------------------#

        # Inform the user about the removed components.
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

        genes_names : :class:`list` of :class:`str`, optional
            The names of the genes, needed to save per-gene saliency
            maps.

        pathways : :class:`dict`, optional
            A dictionary where the keys are pathway names and the
            values are lists of gene IDs belonging to each pathway,
            needed to save per-pathway saliency maps.

        pathways_names : :class:`list` of :class:`str`, optional
            The names of the pathways, needed to save per-pathway
            saliency maps.
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
            The parsed options for the training.

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
        results : :class:`tuple`
            ``((rep_train, rep_test), (pred_means_train,
            pred_means_test), pred_r_values, losses,
            (metrics_train, metrics_test), time_train)``.
            ``pred_r_values`` is a single tensor for per-gene
            r-values, a ``(train, test)`` tuple for per-sample
            r-values, or :obj:`None` for Poisson counts. ``losses``
            holds each epoch's GMM, reconstruction and total losses,
            ``metrics_train``/``metrics_test`` each epoch's
            reporting metrics, and ``time_train`` the per-epoch
            CPU/wall-clock timing.
        """

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

        # Get the norms to clip the gradients to, if any.
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
                latent_options.get("lr_scheduler_options")

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
            decoder_options.get("lr_scheduler_options")

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
            representations_options.get("lr_scheduler_options")

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
            representations_options.get("train_noise_options")

        # If the noise is Gaussian
        if train_noise_type == "gaussian":

            # Get the noise scale (zero disables the noise).
            train_noise_scale_base = train_noise_options["scale"]

            # Get the starting noise scale (for cosine annealing).
            train_noise_start = train_noise_options["start"]

            # Get the ending noise scale (for cosine annealing).
            train_noise_end = train_noise_options["end"]

            # Get the fraction of the mass within the radius the noise
            # is scaled by.
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

            # Get the patience (the number of epochs without improvement
            # in the test loss before stopping).
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

            # Initialize the state of the best model to None.
            early_stopping_best_model_state = None

            # Keep early stopping inactive until the GMM is fitted.
            early_stopping_active = False

        #-------------------------------------------------------------#

        # Create an empty list to store the GMM's loss, the
        # reconstruction loss, and the overall loss.
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

            # Inform the user.
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

            # Set how many components either side of the current number
            # to try (by default, one).
            latent_model_selection_step = 1

            # If the model selection is based on a metric
            if latent_model_selection_type == "metric" and \
                latent_model_selection_options is not None:

                # If the options are not a dictionary
                if not isinstance(latent_model_selection_options, dict):

                    # Raise an error.
                    errstr = \
                        "'model_selection_options' must be a " \
                        "mapping, with a 'metric' and, optionally, " \
                        "a 'step' - for instance " \
                        "'{'metric' : 'bic', 'step' : 4}'. It is a " \
                f"'{type(latent_model_selection_options).__name__}'."
                    raise TypeError(errstr)

                # Get the metric used to select candidate models.
                latent_model_selection_metric = \
                    latent_model_selection_options.get("metric", "bic")

                # Get how many components either side to try.
                latent_model_selection_step = \
                    int(latent_model_selection_options.get("step", 1))

                # Inform the user.
                logger.info(
                    "The number of components of the Gaussian "
                    "mixture model is selected dynamically, on the "
                    f"'{latent_model_selection_metric}', looking "
                    f"{latent_model_selection_step} component(s) "
                    "either side of the current number at every "
                    "refit.")

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

            # Get the maximum number of EM iterations at a
            # full-refit epoch (a multiple of 'refit_interval').
            latent_max_iter_full_refit = \
                latent_options_fitting["max_iter_full_refit"]

            # Get the maximum number of EM iterations at a
            # "warm"-refit epoch (any other epoch).
            latent_max_iter_warm_refit = \
                latent_options_fitting["max_iter_warm_refit"]

            # Get the maximum number of EM iterations to perform at
            # the final refit (if 'final_refit' is True).
            latent_max_iter_final_refit = \
                latent_options_fitting["max_iter_final_refit"]

            # Get the weight of the latent loss.
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

        # Get the latent space's reporting metrics.
        reporting_metrics_latent = \
            reporting_options["metrics"]["latent"]

        # If there are reporting metrics
        if reporting_metrics_latent:

            # Initialize a list to store the metrics that will be used.
            filtered_metrics = []

            # For each of the reporting metrics
            for m in reporting_metrics_latent:

                # If the metric is supervised
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

                # Add the metric to the filtered list.
                filtered_metrics.append(m)

            # Replace the reporting metrics with the filtered ones.
            reporting_metrics_latent = filtered_metrics

        # If there are still reporting metrics to use
        if reporting_metrics_latent:

            # Create a string containing the metrics.
            metrics_str = \
                ", ".join([f"'{m}'" for m in reporting_metrics_latent])

            # Inform the user about the reporting metrics.
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

            # If there are reporting metrics
            if reporting_metrics_latent:

                # Warn that they are not computed.
                logger.warning(
                    "The latent metrics are only computed for the "
                    "'tgmm' latent type, so they will be NaN.")

            # If early stopping is enabled
            if early_stopping_type == "loss":

                # Activate it from the first epoch.
                early_stopping_active = True

        #-------------------------------------------------------------#

        # For each epoch
        for epoch in range(1, n_epochs + 1):

            # Initialize the losses to 0 for the current epoch.
            losses_list.append([epoch, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

            # Initialize the epoch's metrics rows.
            metrics_row_train = {"epoch": epoch}
            metrics_row_test = {"epoch": epoch}

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

                # Get the target (or, with dynamic selection, maximum)
                # number of components.
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
                # should be fitted
                elif epoch >= latent_first_epoch:

                    # Disable gradient computation.
                    with torch.no_grad():

                        # Determine whether this is a full-refit epoch.
                        is_full_refit_epoch = \
                            epoch == latent_first_epoch or \
                            (latent_refit_interval \
                                and epoch % latent_refit_interval == 0)

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

                            # Keep the best of the models with k,
                            # k - step and k + step components.
                            self._get_best_latent_tgmm(
                                rep_train = rep_train,
                                latent_n_components_target = \
                                    latent_n_components_target,
                                max_iter = max_iter,
                                is_full_refit_epoch = \
                                    is_full_refit_epoch,
                                epoch = epoch,
                                model_selection_step = \
                                    latent_model_selection_step,
                                model_selection_metric = \
                                    latent_model_selection_metric)

                        #---------------------------------------------#

                        # Otherwise
                        else:

                            # If it is a full refit epoch
                            if is_full_refit_epoch:

                                # Fit the GMM.
                                self.latent.fit(rep_train,
                                                max_iter = max_iter)

                            # If it is not a full refit epoch
                            else:

                                # Fit the GMM starting from the previous
                                # epoch's solution.
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
                                collapse_weight_threshold = \
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

                # If the GMM has been fitted
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

            # Otherwise (no noise)
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

                # Unpack the batch data (3 or 4 items, depending on
                # whether labels are available).
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

                    # If the Gaussian mixture model was not fitted yet
                    else:

                        # Set the Gaussian mixture model's loss to
                        # zero.
                        latent_loss = \
                            torch.tensor(0.0).to(self.device)

                #-----------------------------------------------------#

                # If the output module means that the r-values are not
                # learned
                if isinstance(\
                    self.decoder.nb,
                    (outputmodules.OutputModuleNBFeatureDispersion,
                     outputmodules.OutputModulePoisson)):

                    # Get the predicted means: shape (batch, genes).
                    pred_means = self.decoder(z = z)

                    # There are no predicted r-values.
                    pred_log_r_values = None

                    # Set the options to compute the reconstruction
                    # loss.
                    recon_loss_options = \
                        {"obs_counts" : samples_exp,
                         "pred_means" : pred_means,
                         "scaling_factors" : samples_mean_exp}

                # If the output module means that the r-values are
                # learned
                elif isinstance(\
                    self.decoder.nb,
                    outputmodules.OutputModuleNBFullDispersion):

                    # Get the predicted means and r-values, both shaped
                    # (batch, genes).
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

                        # Get the overall loss (reconstruction loss plus
                        # the Gaussian mixture model's loss).
                        loss = latent_loss.clone() + recon_loss.clone()

                    # If the Gaussian mixture model was not fitted yet
                    else:

                        # Get the overall loss (the reconstruction loss
                        # only).
                        loss = recon_loss.clone()

                #-----------------------------------------------------#

                # Add the output module's dispersion regularization
                # (zero unless it shrinks the dispersion).
                loss = loss + \
                    self.decoder.nb.dispersion_regularization(
                        pred_means = pred_means,
                        pred_log_r_values = pred_log_r_values,
                        reduction = loss_reduction_type)

                # Backpropagate the loss.
                loss.backward()

                #-----------------------------------------------------#

                # Clip the gradients, if a maximum norm was set.
                clip_grads(optimizer = optimizer_decoder,
                           max_norm = grad_clip_decoder)
                clip_grads(optimizer = optimizer_rep_train,
                           max_norm = grad_clip_rep)

                #-----------------------------------------------------#

                # If the Gaussian mixture model is the legacy one
                if isinstance(self.latent,
                              latents.GaussianMixtureModelLegacy):

                    # Clip the Gaussian mixture model's gradients.
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

                # Set no gradient computation status to restore.
                latent_requires_grad = None

            #---------------------------------------------------------#

            # For each batch of testing samples
            for batch_data in data_loader_test:

                # Unpack the batch data (3 or 4 items, depending on
                # whether labels are available).
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

                    # If we are at any epoch after the Gaussian mixture
                    # model was fitted
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

                    # If the Gaussian mixture model was not fitted yet
                    else:

                        # Set the Gaussian mixture model's loss to
                        # zero.
                        latent_loss = \
                            torch.tensor(0.0).to(self.device)

                #-----------------------------------------------------#

                # If the output module means that the r-values are not
                # learned
                if isinstance(\
                    self.decoder.nb,
                    (outputmodules.OutputModuleNBFeatureDispersion,
                     outputmodules.OutputModulePoisson)):

                    # Get the predicted means: shape (batch, genes).
                    pred_means = self.decoder(z = z)

                    # There are no predicted r-values.
                    pred_log_r_values = None

                    # Set the options to compute the reconstruction
                    # loss.
                    recon_loss_options = \
                        {"obs_counts" : samples_exp,
                         "pred_means" : pred_means,
                         "scaling_factors" : samples_mean_exp}

                # If the output module means that the r-values are
                # learned
                elif isinstance(\
                    self.decoder.nb,
                    outputmodules.OutputModuleNBFullDispersion):

                    # Get the predicted means and r-values, both shaped
                    # (batch, genes).
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

                        # Get the overall loss (reconstruction loss plus
                        # the Gaussian mixture model's loss).
                        loss = latent_loss.clone() + recon_loss.clone()

                    # If the Gaussian mixture model was not fitted yet
                    else:

                        # Get the overall loss (the reconstruction loss
                        # only).
                        loss = recon_loss.clone()

                #-----------------------------------------------------#

                # Add the output module's dispersion regularization
                # (with the decoder frozen).
                loss = loss + \
                    self.decoder.nb.dispersion_regularization(
                        pred_means = pred_means,
                        pred_log_r_values = pred_log_r_values,
                        reduction = loss_reduction_type)

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

            # If the learning rate scheduler for the test
            # representations is defined
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
                        collapse_weight_threshold = \
                            component_weight_threshold,
                        epoch = epoch)

                # If some components were removed
                if removed_components:

                    # Get the old optimizer's parameter groups.
                    param_groups_old = optimizer_latent.param_groups

                    # Recreate the optimizer for the new GMM's
                    # parameters.
                    optimizer_latent = \
                        self._get_optimizer(
                            optimizer_type = optimizer_latent_type,
                            optimizer_options = \
                                optimizer_latent_options,
                            optimizer_parameters = \
                                self.latent.parameters())

                    # If there is a scheduler for the latent space
                    if lr_scheduler_latent is not None:

                        # Get the scheduler's state.
                        lr_scheduler_latent_state = \
                            lr_scheduler_latent.state_dict()

                        # Recreate the scheduler for the new optimizer.
                        lr_scheduler_latent = \
                            self._get_scheduler(
                                lr_scheduler_target = "latent",
                                lr_scheduler_type = \
                                    lr_scheduler_latent_type,
                                lr_scheduler_options = \
                                    lr_scheduler_latent_options,
                                optimizer = optimizer_latent,
                                n_epochs = n_epochs,
                                data_loader_train = data_loader_train)

                        # Resume the schedule where it was.
                        lr_scheduler_latent.load_state_dict(
                            lr_scheduler_latent_state)

                        # For each new and old parameter group
                        for group, group_old in \
                            zip(optimizer_latent.param_groups,
                                param_groups_old):

                            # Keep the old group's current settings.
                            group.update(
                                {k : v for k, v in group_old.items()
                                 if k != "params"})

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

            #---------------------------------------------------------#

            # Get the output module's diagnostics.
            diagnostics = self.decoder.nb.diagnostics()

            # If there are any
            if diagnostics:

                # Inform the user about them.
                logger.info(
                    f"Epoch {epoch} [output module]: "
                    + ", ".join(f"{k}={v:.4g}"
                                for k, v in diagnostics.items()))

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
            # for the training samples is enabled
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

            # End the log string and inform the user.
            info_msg += "."
            logger.info(info_msg)

            #---------------------------------------------------------#

            # If the latent metrics are active and the latent space is
            # the TorchGMM wrapper
            if latent_metrics_active \
                and isinstance(self.latent,
                               latents.GaussianMixtureModelTGMM):

                # Initialize the parts of the metrics' log line.
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

                    # For each reporting metric
                    for metric_name in reporting_metrics_latent:

                        # If the metric is unsupervised
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

                        # If the metric is supervised (it stays NaN
                        # without labels)
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

                        # Get the train value.
                        train_value = metrics_row_train[metric_name]

                        # Try to format it ('nan' if invalid).
                        try:
                            train_str = \
                                f"{train_value:.4f}" \
                                if train_value is not None and \
                                    np.isfinite(float(train_value)) \
                                else "nan"

                        # If it is not a number
                        except (TypeError, ValueError):

                            # Use 'nan'.
                            train_str = "nan"

                        # Get the test value.
                        test_value = metrics_row_test[metric_name]

                        # Try to format it ('nan' if invalid).
                        try:
                            test_str = \
                                f"{test_value:.4f}" \
                                if test_value is not None \
                                    and np.isfinite(float(test_value)) \
                                else "nan"

                        # If it is not a number
                        except (TypeError, ValueError):

                            # Use 'nan'.
                            test_str = "nan"

                        # Add the metric's summary.
                        metrics_parts.append(
                            f"{metric_name}: train={train_str}, "
                            f"test={test_str}")

                #-----------------------------------------------------#

                # Inform the user about the metrics.
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

            # Save the epoch's metrics rows.
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

                    # Inform the user.
                    info_msg = \
                        "Early stopping triggered at epoch " \
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
                "Restored best model state from epoch " \
                f"{early_stopping_best_epoch}."
            logger.info(info_msg)

        #=============================================================#
        #                        RETURN PHASE                         #
        #=============================================================#

        # If the Gaussian mixture model is the TGMM wrapper and a final
        # refit is needed (also after early stopping)
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

        # If the genes' counts are modelled by negative binomials
        # with per-gene r-values
        if isinstance(self.decoder.nb,
                      outputmodules.OutputModuleNBFeatureDispersion):

            # Get the predicted scaled means for the training samples.
            means_final_train = self.decoder(z = rep_train)

            # Get the predicted scaled means for the test samples.
            means_final_test = self.decoder(z = rep_test)

            # Get the r-values.
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


    ######################### PUBLIC METHODS ##########################


    @staticmethod
    def rescale_pred_means(df_pred_means: pd.DataFrame,
                           df_pred_r_values: pd.DataFrame) -> \
                            pd.DataFrame:
        """Rescale the means of the negative binomials modeling
        the genes' counts.

        Parameters
        ----------
        df_pred_means : :class:`pandas.DataFrame`
            One row per representation/sample; columns named after
            genes' Ensembl IDs hold the scaled means.

        df_pred_r_values : :class:`pandas.DataFrame`
            One row per representation/sample, same shape as
            ``df_pred_means``; columns named after genes' Ensembl IDs
            hold the r-values.

        Returns
        -------
        df_scaled : :class:`pandas.DataFrame`
            Same columns and order as ``df_pred_means``, with the
            gene columns' values scaled back by the r-values.
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

        # Return the new data frame.
        return df_final_means


    def get_representations(self,
                            df_samples: pd.DataFrame,
                            config_rep: dict[str, object],
                            get_saliency_map: bool = False,
                            genes_mask: \
                                Optional[torch.Tensor] = None) -> \
                                tuple[pd.DataFrame, pd.DataFrame,
                                      Optional[pd.DataFrame],
                                      pd.DataFrame]:
        """Find the best representations for a set of samples.

        Parameters
        ----------
        df_samples : :class:`pandas.DataFrame`
            A data frame containing the samples.

        config_rep : :class:`dict`
            A dictionary of options for the optimization(s), varying
            by the selected scheme.

        get_saliency_map : :class:`bool`, ``False``
            Whether to also compute and return the saliency maps.

        genes_mask : :class:`torch.Tensor`, optional
            A 2D 1.0/0.0 mask (samples x genes) of which genes were
            measured. By default, all genes are measured.

        Returns
        -------
        df_rep : :class:`pandas.DataFrame`
            One row per representation; columns are the latent
            dimensions, followed by any additional input columns.

        df_pred_means : :class:`pandas.DataFrame`
            One row per representation; columns are the predicted
            gene-count means (scaled by the r-values for negative
            binomial counts), followed by any additional input
            columns.

        df_pred_r_values : :class:`pandas.DataFrame`, optional
            One row per representation; columns are the predicted
            negative-binomial r-values, followed by any additional
            input columns. :obj:`None` for Poisson counts.

        df_time : :class:`pandas.DataFrame`
            One row per optimization epoch, with the platform, CPU
            thread count, and CPU/wall-clock time for the epoch and
            its backpropagation step.

        df_saliency_map : :class:`pandas.DataFrame`, optional
            One row per gene (ENSG-indexed), one column per latent
            dimension. Only returned when ``get_saliency_map`` is
            ``True``.
        """

        # Get the batch size the data loader will use, if any.
        batch_size = config_rep.get(
            "data_loader_options", {}).get("batch_size")

        # Get whether to pad the samples out to whole batches.
        use_padding = config_rep.get("use_batch_size_padding", True)

        # Get how many samples are missing from the last batch (GPU
        # results depend on the batch shape).
        n_padding = \
            -len(df_samples) % batch_size \
            if use_padding and batch_size else 0

        # If the last batch would be a short one
        if n_padding:

            # Get the sample identifiers already taken.
            taken = set(df_samples.index.astype(str))

            # Initialize the padding samples' own identifiers.
            padding_names = []

            # Until there is one for every padding sample
            while len(padding_names) < n_padding:

                # Draw an identifier at random.
                name = "".join(
                    random.choices(
                        string.ascii_letters + string.digits,
                        k = 16))

                # If no sample already has it
                if name not in taken:

                    # Keep it.
                    taken.add(name)
                    padding_names.append(name)

            # Repeat the first sample under those identifiers.
            df_padding = df_samples.iloc[[0] * n_padding].copy()
            df_padding.index = padding_names

            # Add the padding to the samples.
            df_samples = pd.concat([df_samples, df_padding])

            # If the measured genes were given as a mask
            if genes_mask is not None:

                # Repeat its first row for the padding samples.
                genes_mask = \
                    torch.cat([genes_mask,
                               genes_mask[[0]].repeat(n_padding, 1)],
                              dim = 0)

        #-------------------------------------------------------------#

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

        # Create the dataset (with the mask, keeping the unmeasured
        # genes out of the scaling factors).
        dataset = \
            dataclasses.GeneExpressionDataset(\
                df = df_expr_data,
                scaling_factor = self._scaling_factor,
                mask = genes_mask)

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

        # If the user selected the two-optimizations scheme
        if opt_scheme == "two_opt":

            # Select the corresponding method.
            opt_method = self._get_representations_two_opt

        # If the user selected the multi-seed two-optimizations scheme
        elif opt_scheme == "two_opt_multiseed":

            # Select the corresponding method.
            opt_method = self._get_representations_two_opt_multiseed

        # If the scheme is not supported
        else:

            # Raise an error.
            raise ValueError(
                f"Unsupported optimization scheme '{opt_scheme}'. "
                "The schemes are 'two_opt' and "
                "'two_opt_multiseed'.")

        #-------------------------------------------------------------#

        # In the model's own precision
        with self._default_dtype(self._dtype):

            # Get the representations, predicted means, r-values (if
            # any), and time data.
            rep, pred_means, pred_r_values, time_opt = \
                opt_method(dataset = dataset,
                           config = config_rep,
                           genes_mask = genes_mask)

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

        # If padding was added to fill the last batch
        if n_padding:

            # Drop it from the representations.
            df_rep = df_rep.iloc[:-n_padding]

            # Drop it from the predicted scaled means.
            df_pred_means = df_pred_means.iloc[:-n_padding]

            # If there are predicted r-values
            if df_pred_r_values is not None:

                # Drop it from the predicted r-values.
                df_pred_r_values = df_pred_r_values.iloc[:-n_padding]

        #-------------------------------------------------------------#

        # If saliency maps are requested
        if get_saliency_map:

            # Compute the saliency map.
            saliency_tensor = self._get_saliency_map(z = rep)

            # Get the names of the latent dimensions' columns.
            latent_cols = \
                [f"latent_dim_{i}" for i in range(self.latent.dim)]

            # Create a data frame for the saliency map.
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


    def impute(self,
               df_samples: pd.DataFrame,
               config_rep: dict[str, object],
               genes_measured: Optional[list[str]] = None,
               quantiles: tuple[float, float] = (0.025, 0.975)) -> \
                tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
                      Optional[pd.DataFrame], pd.DataFrame]:
        """Predict the counts of the genes a sample does not have,
        from the genes it does. Missing genes must be
        :obj:`numpy.nan`, not zero.

        Parameters
        ----------
        df_samples : :class:`pandas.DataFrame`
            The samples, with :obj:`numpy.nan` in the genes that were
            not measured. A gene column missing entirely is taken as
            unmeasured in every sample.

        config_rep : :class:`dict`
            The options for finding the representations, as for
            :meth:`get_representations`.

        genes_measured : :class:`list`, optional
            The genes that were measured, as an alternative to writing
            :obj:`numpy.nan` everywhere else. Every other gene is
            taken to be unmeasured in every sample.

        quantiles : :class:`tuple`, ``(0.025, 0.975)``
            The quantiles of the predicted distribution to report.

        Returns
        -------
        df_imputed : :class:`pandas.DataFrame`
            The expected count of every gene of every sample (the
            predicted distribution's mean), including the measured
            genes.

        df_lower, df_upper : :class:`pandas.DataFrame`
            The requested quantiles of the predicted distribution.

        df_pred_r_values : :class:`pandas.DataFrame` or :obj:`None`
            The r-values of the predicted negative binomials, or
            :obj:`None` for Poisson counts.

        df_rep : :class:`pandas.DataFrame`
            The representations found from the measured genes.
        """

        # If the model is median-scaled and was given a panel (whose
        # median, unlike a random subset's, is biased)
        if self._scaling_factor == "median" \
           and genes_measured is not None:

            # Raise an error.
            raise NotImplementedError(
                "Imputation from a fixed panel of measured genes is "
                "only implemented for a model whose scaling factor is "
                "the mean, and this one's is the "
                f"'{self._scaling_factor}'. Mask genes at random - "
                "pass the unmeasured counts as NaN and leave "
                "'genes_measured' unset - or use a mean-scaled model.")

        #-------------------------------------------------------------#

        # Get the genes the model knows.
        genes_model = list(self.genes)

        # Reindex the samples to the model's own gene order.
        df = df_samples.reindex(columns = genes_model)

        # Get what was measured (absent and NaN genes are not).
        measured = df.notna().to_numpy()

        # If an explicit list of measured genes was given
        if genes_measured is not None:

            # Get which of the model's genes are in that list.
            keep = np.isin(np.array(genes_model),
                           np.array(list(genes_measured)))

            # Combine it with the NaN-derived mask.
            measured = measured & keep[np.newaxis, :]

        # If any sample has no measured gene at all
        if not measured.any(axis = 1).all():

            # Raise an error.
            raise ValueError(
                "At least one sample has no measured gene at all. "
                "There is nothing to find a representation from.")

        # Get the number of measured genes per sample.
        n_measured = measured.sum(axis = 1)

        # Inform the user.
        logger.info(
            "The samples have a median of "
            f"{int(np.median(n_measured)):,} measured gene(s), of the "
            f"{len(genes_model):,} the model knows.")

        #-------------------------------------------------------------#

        # Fill the unmeasured genes with zeros (masked out of the loss,
        # while NaNs would still turn the loss into NaN).
        df_filled = df.fillna(0.0)

        # Build the genes mask tensor.
        genes_mask = \
            torch.tensor(measured.astype("float64"),
                         dtype = torch.float64,
                         device = self.device)

        #-------------------------------------------------------------#

        # Find the representations from the measured genes.
        df_rep, df_pred_means, df_pred_r_values, _ = \
            self.get_representations(df_samples = df_filled,
                                     config_rep = config_rep,
                                     genes_mask = genes_mask)

        #-------------------------------------------------------------#

        # Get the predicted means as a plain array.
        pred = df_pred_means[genes_model].to_numpy(dtype = "float64")

        # Get the predicted r-values as a plain array (none for
        # Poisson counts).
        r = df_pred_r_values[genes_model].to_numpy(dtype = "float64") \
            if df_pred_r_values is not None else None

        # Get the (zero-filled) observed counts as a plain array.
        obs = df_filled[genes_model].to_numpy(dtype = "float64")

        # Re-estimate the scaling factors from the measured genes, as
        # in the optimization (the predicted means are unscaled).
        scale = \
            self.__class__._get_masked_scaling_factors(
                obs_counts = torch.from_numpy(obs),
                pred_means = torch.from_numpy(pred),
                mask = torch.from_numpy(
                    measured.astype("float64")),
                n_genes = len(genes_model),
                scaling_factor = self._scaling_factor).numpy()

        # Scale the predicted means.
        means = pred * scale

        #-------------------------------------------------------------#

        # If the counts are modelled by Poisson distributions
        if r is None:

            # Get the lower quantile of the predicted distribution.
            lower = poisson.ppf(quantiles[0],
                                means)

            # Get the upper quantile of the predicted distribution.
            upper = poisson.ppf(quantiles[1],
                                means)

        # Otherwise
        else:

            # Get the negative binomials' probability of success
            # (scipy's parameterization).
            p = r / (r + means)

            # Get the lower quantile of the predicted distribution.
            lower = nbinom.ppf(quantiles[0],
                               r,
                               p)

            # Get the upper quantile of the predicted distribution.
            upper = nbinom.ppf(quantiles[1],
                               r,
                               p)

        #-------------------------------------------------------------#

        # Get the samples' index.
        index = df.index

        # Assemble the imputed-means data frame.
        df_imputed = pd.DataFrame(means,
                                  index = index,
                                  columns = genes_model)

        # Assemble the lower-quantile data frame.
        df_lower = pd.DataFrame(lower,
                                index = index,
                                columns = genes_model)

        # Assemble the upper-quantile data frame.
        df_upper = pd.DataFrame(upper,
                                index = index,
                                columns = genes_model)

        # Assemble the r-values data frame (none for Poisson counts).
        df_r = pd.DataFrame(r,
                            index = index,
                            columns = genes_model) \
            if r is not None else None

        # Return the imputed means, quantiles, r-values, and
        # representations.
        return df_imputed, df_lower, df_upper, df_r, df_rep


    def get_probability_density(self,
                                df_rep: pd.DataFrame) -> \
                                    tuple[pd.DataFrame, pd.DataFrame]:
        """Get each component's probability density for each
        representation, and the representation(s) of maximum density
        per component.

        Parameters
        ----------
        df_rep : :class:`pandas.DataFrame`
            A data frame containing the representations.

        Returns
        -------
        df_prob_rep : :class:`pandas.DataFrame`
            The per-component probability densities for each
            representation, plus its maximum density and the
            component it belongs to.

        df_prob_comp : :class:`pandas.DataFrame`
            For each component, the representation(s) of maximum
            density and that density.
        """

        # Set the name of the column that will contain the maximum
        # probability density found per sample.
        MAX_PROB_COL = "max_prob_density"

        # Set the name of the column that will contain the
        # component of maximum probability density per sample.
        MAX_PROB_COMP_COL = "max_prob_density_comp"

        # Set the name of the column that will contain the index
        # of the sample of maximum probability per component.
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

        # Initialize the list of the rows of the samples with
        # the highest probability density for each component.
        rows_with_max = []

        # For each component for which at least one representation
        # had maximum probability density
        for comp in df_prob_rep[MAX_PROB_COMP_COL].unique():

            # Get only those rows corresponding to the current
            # component under consideration.
            sub_df = \
                df_prob_rep.loc[df_prob_rep[MAX_PROB_COMP_COL] == comp]

            # Get the sample with maximum probability for the
            # component ('idxmax()' does not preserve the values).
            max_for_comp = \
                sub_df.loc[sub_df[MAX_PROB_COL] == \
                           sub_df[MAX_PROB_COL].max()].copy()

            # Add a column storing the representation/sample unique
            # index.
            max_for_comp[SAMPLE_IDX_COL] = max_for_comp.index

            # Index the data frame by the component.
            max_for_comp = max_for_comp.set_index(MAX_PROB_COMP_COL)

            # Append the data frame to the list of data frames.
            rows_with_max.append(max_for_comp)

        #-------------------------------------------------------------#

        # Concatenate the data frames.
        df_prob_comp = pd.concat(rows_with_max, axis = 0)

        #-------------------------------------------------------------#

        # Return the two data frames.
        return df_prob_rep, df_prob_comp


    def _fit_final_gmm(self,
                       reps_train: torch.Tensor,
                       config_final: dict[str, object],
                       gmm_final_pth_file: str) -> None:
        """Fit the Gaussian mixture model that describes the latent
        space after training, and save it.

        Parameters
        ----------
        reps_train : :class:`torch.Tensor`
            The representations found for the training samples.

        config_final : :class:`dict`
            The configuration for fitting the final Gaussian mixture
            model.

        gmm_final_pth_file : :class:`str`
            The path where the final Gaussian mixture model's
            parameters will be saved.
        """

        # Copy the options for the final mixture.
        options = dict(self._gmm_final_options)

        # Use the fitting configuration for the options not set.
        options.setdefault("fit",
                           config_final.get("fit", "covariance_only"))
        options.setdefault("max_iter",
                           config_final.get("max_iter", 1000))

        # Fit the mixture to the training representations.
        self._latent_final = \
            self._fit_gmm_to_reps(reps = reps_train,
                                  options = options)

        # Save the final mixture's parameters to its own file.
        self._latent_final.save(\
            _internals.uniquify_file_path(gmm_final_pth_file))

        # Inform the user that the parameters were saved.
        logger.info(
            "The final Gaussian mixture model (covariance type: "
            f"'{options.get('covariance_type')}', shrinkage: "
            f"{options.get('shrinkage', 0.0)}, fit: "
            f"'{options['fit']}') was successfully saved in "
            f"'{gmm_final_pth_file}'. The prior the model was "
            "trained with is unchanged.")


    def _fit_gmm_to_reps(self,
                         reps: torch.Tensor,
                         options: dict[str, object]):
        """Fit a Gaussian mixture model to a set of representations.

        Parameters
        ----------
        reps : :class:`torch.Tensor`
            The representations to fit the mixture to.

        options : :class:`dict`
            The options for the fit: the covariance's type and
            shrinkage, the regularization added to its diagonal, what
            the fit may move, and the maximum number of iterations.

        Returns
        -------
        gmm : :class:`bulkdgd.core.latents.GaussianMixtureModelTGMM`
            The fitted Gaussian mixture model.
        """

        # Get the type of covariance the mixture should have.
        covariance_type = options.get("covariance_type")

        # If no covariance type was given
        if covariance_type is None:

            # Raise an error.
            errstr = \
                "The 'gmm_final' section of the model's " \
                "configuration must specify a 'covariance_type'."
            raise ValueError(errstr)

        # Get how far each component's covariance is pulled towards
        # the one shared by all of them.
        shrinkage = options.get("shrinkage", 0.0)

        # If a full covariance was requested without shrinkage
        if covariance_type == "full" and not shrinkage:

            # Raise an error.
            errstr = \
                "A 'full' covariance for the final Gaussian mixture " \
                "model needs a non-zero 'shrinkage'."
            raise ValueError(errstr)

        #-------------------------------------------------------------#

        # Get what the fit is allowed to move.
        fit = options.get("fit", "covariance_only")

        # If only the covariance is refitted
        if fit == "covariance_only":

            # Refit it, with the means and the weights frozen, and
            # return the mixture.
            return \
                latents.fit_final_gmm(\
                    gmm = self.latent,
                    reps = reps,
                    covariance_type = covariance_type,
                    shrinkage = shrinkage,
                    reg_covar = options.get("reg_covar"))

        # If the whole mixture is refitted
        elif fit == "full_em":

            # Warn the user that the components no longer match the
            # prior's.
            warn_msg = \
                "The final Gaussian mixture model is fitted with " \
                "'fit: full_em', so its components are not the " \
                "prior's. Use 'fit: covariance_only' to keep the " \
                "prior's components."
            logger.warning(warn_msg)

            # Copy the trained mixture (the prior stays untouched).
            gmm_final = copy.deepcopy(self.latent)

            # Set the requested covariance type.
            gmm_final.covariance_type = covariance_type

            # Fit the copy to the final representations.
            gmm_final.fit(\
                reps,
                max_iter = options.get("max_iter", 1000))

            # Return it.
            return gmm_final

        # Otherwise
        else:

            # Raise an error.
            errstr = \
                f"Unsupported 'fit' option '{fit}' for the final " \
                "Gaussian mixture model. The supported options are: " \
                "covariance_only, full_em."
            raise ValueError(errstr)


    def fit_gmm(self,
                input_reps: Union[str, pd.DataFrame,
                                  list[Union[str, pd.DataFrame]]],
                config_fit: dict[str, object]) -> "BulkDGD":
        """Fit a new Gaussian mixture to a trained model's training
        representations, and return a new model using it.

        Parameters
        ----------
        input_reps : :class:`str`, :class:`pandas.DataFrame`, or a \
            :class:`list` of either
            The representations to fit the mixture to. Each may be a
            path to a CSV file (samples on the rows, the latent
            dimensions in columns named ``latent_dim_*``) or an
            in-memory :class:`pandas.DataFrame` in the same format.

        config_fit : :class:`dict`
            The configuration for the fit: ``gmm_new_pth_file`` (the
            file the fitted mixture is written to),
            ``config_model_new`` (the YAML file the new model's
            configuration is written to), ``dec_pth_file`` (optional,
            the trained decoder's parameters, by default the model's
            own), and ``gmm_options`` (``covariance_type``, required;
            ``shrinkage``, non-zero for ``"full"`` covariances;
            ``reg_covar``; ``fit``, either ``"covariance_only"``
            (default, keeping the prior's means and weights) or
            ``"full_em"`` (re-estimating everything); and
            ``max_iter`` for ``"full_em"``).

        Returns
        -------
        model : :class:`BulkDGD`
            A new model with the fitted mixture and the trained
            decoder, built from the files written.
        """

        # Get the required paths from the configuration.
        gmm_new_pth_file = config_fit["gmm_new_pth_file"]
        config_model_new = config_fit["config_model_new"]

        # Get the options for the fit.
        options = dict(config_fit.get("gmm_options") or {})

        # Get the file with the trained decoder's parameters.
        dec_pth_file = \
            config_fit.get(
                "dec_pth_file",
                self._decoder_initial_options.get("decoder_pth_file"))

        # If there are none
        if dec_pth_file is None:

            # Raise an error.
            raise ValueError(
                "No trained decoder's parameters were given, and the "
                "model's configuration points at none. Pass "
                "'dec_pth_file' in the configuration for the fit.")

        #-------------------------------------------------------------#

        # Load the representations the mixture is to be fitted to.
        reps = self._load_reps(input_reps = input_reps,
                               latent_dim = self.latent.dim)

        # Inform the user.
        logger.info(
            f"The mixture will be fitted to {len(reps):,} "
            "representations.")

        # Convert them to a tensor on the mixture's device and in its
        # precision.
        reps = torch.tensor(reps,
                            device = self.device,
                            dtype = self.latent.means.dtype)

        #-------------------------------------------------------------#

        # Fit the mixture.
        gmm_new = self._fit_gmm_to_reps(reps = reps,
                                        options = options)

        #-------------------------------------------------------------#

        # Create the directory of the mixture's file, if needed.
        os.makedirs(
            os.path.dirname(os.path.abspath(gmm_new_pth_file)),
            exist_ok = True)

        # Write the fitted mixture's parameters.
        gmm_new.save(gmm_new_pth_file)

        #-------------------------------------------------------------#

        # Get the configuration to rebuild the model.
        config = self._get_config_for_rebuilding()

        # Set the fitted covariance type.
        config["latent_options"]["covariance_type"] = \
            options.get("covariance_type")

        # Point at the fitted mixture's parameters.
        config["latent_options"]["latent_pth_file"] = gmm_new_pth_file

        # Point at the trained decoder's parameters.
        config["decoder_options"]["decoder_pth_file"] = dec_pth_file

        # Create the directory of the configuration file, if needed.
        os.makedirs(
            os.path.dirname(os.path.abspath(config_model_new)),
            exist_ok = True)

        # Write the new model's configuration.
        with open(config_model_new, "w") as fh:
            yaml.safe_dump(config,
                           fh,
                           sort_keys = False)

        # Inform the user.
        logger.info(
            f"The fitted mixture was written to '{gmm_new_pth_file}' "
            f"and '{config_model_new}'. The prior the model was "
            "trained with is unchanged.")

        #-------------------------------------------------------------#

        # Return a model built from the files written.
        return self.__class__(**config, device = str(self.device))


    def _get_config_for_rebuilding(self) -> dict[str, object]:
        """Get a configuration that rebuilds this model, apart from
        the files its trained parameters live in.

        Returns
        -------
        config : :class:`dict`
            The configuration.
        """

        # Assemble what the model was built from.
        config = {
            "latent_dim" : int(self.latent.dim),
            "latent_type" : self._latent_type,
            "latent_options" : \
                copy.deepcopy(self._latent_initial_options),
            "decoder_options" : \
                copy.deepcopy(self._decoder_initial_options),
            "scaling_factor" : self._scaling_factor,
            "dtype" : self._dtype}

        #-------------------------------------------------------------#

        # If the model has options for a final mixture
        if self._gmm_final_options is not None:

            # Carry them over.
            config["gmm_final"] = \
                copy.deepcopy(self._gmm_final_options)

        # If the model was built from a genes' file
        if self._genes_txt_file is not None:

            # Point at it.
            config["genes_txt_file"] = self._genes_txt_file

        #-------------------------------------------------------------#

        # Return the configuration.
        return config


    def prune(self,
              input_reps: Union[str, pd.DataFrame,
                                list[Union[str, pd.DataFrame]]],
              config_prune: dict[str, object]) -> "BulkDGD":
        """Prune a trained decoder to the hidden units it uses, without
        retraining, and return a new, smaller model computing the same
        function (with the same latent space).

        Parameters
        ----------
        input_reps : :class:`str`, :class:`pandas.DataFrame`, or a \
            :class:`list` of either
            The probe-set representations, which should include every
            representation the model has produced, in and out of
            distribution. Each may be a path to a CSV file (samples on
            the rows, the latent dimensions in columns named
            ``latent_dim_*``) or a :class:`pandas.DataFrame` in the
            same format.

        config_prune : :class:`dict`
            The configuration for the pruning: ``gmm_pth_file`` (the
            latent space, reused unchanged), ``dec_pth_file`` (the
            decoder to prune), ``dec_pruned_pth_file`` and
            ``config_model_pruned`` (where the results are written),
            and optional ``pruning_options``: ``rel_tol`` (the
            footprint threshold, default ``1e-7``),
            ``verification_tol`` (the maximum relative error against
            the unpruned decoder, default ``1e-4``),
            ``n_jitter_copies`` (default ``2``), ``jitter_sd``
            (default ``0.25``), and ``seed`` (default ``0``).

        Returns
        -------
        pruned_model : :class:`BulkDGD`
            A new model with the pruned decoder, built from the files
            written.
        """

        # Get the required paths from the configuration.
        gmm_pth_file = config_prune["gmm_pth_file"]
        dec_pth_file = config_prune["dec_pth_file"]
        dec_pruned_pth_file = config_prune["dec_pruned_pth_file"]
        config_model_pruned = config_prune["config_model_pruned"]

        # Get the pruning options.
        p_opts = config_prune.get("pruning_options") or {}

        # Get the footprint and verification tolerances.
        rel_tol = float(p_opts.get("rel_tol", 1e-7))
        verification_tol = float(p_opts.get("verification_tol", 1e-4))

        # Get the jittering options and the seed.
        n_jitter_copies = int(p_opts.get("n_jitter_copies", 2))
        jitter_sd = float(p_opts.get("jitter_sd", 0.25))
        seed = int(p_opts.get("seed", 0))

        #-------------------------------------------------------------#

        # Get the latent dimensionality, the genes, and the device.
        latent_dim = self.latent.dim
        genes = list(self._genes)
        device = str(self.device)

        # Get the full decoder's options without the path to its
        # parameters.
        full_options = \
            {k: v for k, v in self._decoder_initial_options.items()
             if k != "decoder_pth_file"}

        # In the model's own precision
        with self._default_dtype(self._dtype):

            # Build the full decoder.
            dec_full, _ = \
                self._get_decoder(latent_dim = latent_dim,
                                  genes = genes,
                                  decoder_options = full_options,
                                  device = device)

        # Set the full decoder in eval mode.
        dec_full = dec_full.eval()

        # Load its trained parameters.
        dec_full.load_state_dict(
            torch.load(dec_pth_file, map_location = self.device))

        # Get the device and the precision of the decoder's parameters.
        dev = next(dec_full.parameters()).device
        pdtype = next(dec_full.parameters()).dtype

        #-------------------------------------------------------------#

        # Load the representations.
        z_real = self._load_reps(input_reps = input_reps,
                                 latent_dim = latent_dim)

        # Get a random number generator.
        rng = np.random.default_rng(seed)

        # Get the representations' standard deviation per dimension.
        sd_per_dim = z_real.std(axis = 0, keepdims = True)

        # Initialize the probe set with the representations.
        probes = [z_real]

        # For each jittered copy
        for _ in range(n_jitter_copies):

            # Add a jittered copy of the representations.
            probes.append(
                z_real + rng.normal(size = z_real.shape) \
                    * sd_per_dim * jitter_sd)

        # Concatenate the probes.
        z_probe = np.concatenate(probes, axis = 0)

        # Convert the probes and the representations to tensors.
        z_probe = torch.tensor(z_probe,
                               device = dev,
                               dtype = pdtype)
        z_real = torch.tensor(z_real,
                              device = dev,
                              dtype = pdtype)

        # Inform the user about the probes.
        logger.info(
            f"Pruning against {len(z_probe):,} probes "
            f"({len(z_real):,} real representations plus "
            f"{n_jitter_copies} jittered copies at {jitter_sd} sd).")

        #-------------------------------------------------------------#

        # Get the decoder's parameters.
        sd = {k: v.detach() for k, v in dec_full.state_dict().items()}

        # Get the weights and biases of the two hidden layers and the
        # weights of the two output heads.
        w0, b0 = sd["main.0.weight"], sd["main.0.bias"]
        w2, b2 = sd["main.2.weight"], sd["main.2.bias"]
        w_means = sd["nb._layer_means.weight"]
        w_r_values = sd["nb._layer_r_values.weight"]

        # Set the batch size for the pass over the probes.
        batch_size = 4096

        # Get the number of probes.
        n_probes = len(z_probe)

        # Initialize the sums and the sums of squares of the hidden
        # units' activations.
        s1 = torch.zeros(w0.shape[0],
                         device = dev,
                         dtype = pdtype)
        ss1 = torch.zeros_like(s1)
        s2 = torch.zeros(w2.shape[0],
                         device = dev,
                         dtype = pdtype)
        ss2 = torch.zeros_like(s2)

        # Without tracking gradients
        with torch.no_grad():

            # For each batch of probes
            for i in range(0,
                           n_probes,
                           batch_size):

                # Get the batch.
                z = z_probe[i:i + batch_size]

                # Get the hidden layers' activations (after the ReLU).
                h1 = torch.clamp(z @ w0.T + b0, min = 0.0)
                h2 = torch.clamp(h1 @ w2.T + b2, min = 0.0)

                # Accumulate the sums and the sums of squares.
                s1 += h1.sum(0)
                ss1 += (h1 * h1).sum(0)
                s2 += h2.sum(0)
                ss2 += (h2 * h2).sum(0)

        # Get the mean and standard deviation of each unit in the first
        # hidden layer.
        mean1 = s1 / n_probes
        std1 = torch.sqrt(torch.clamp(ss1 / n_probes - mean1 ** 2,
                                      min = 0.0))

        # Get the mean and standard deviation of each unit in the
        # second hidden layer.
        mean2 = s2 / n_probes
        std2 = torch.sqrt(torch.clamp(ss2 / n_probes - mean2 ** 2,
                                      min = 0.0))

        # Get each unit's footprint on the next layer (how much its
        # output varies times how strongly it is read out).
        foot1 = std1 * torch.linalg.norm(w2, dim = 0)
        foot2 = torch.maximum(
            std2 * torch.linalg.norm(w_means, dim = 0),
            std2 * torch.linalg.norm(w_r_values, dim = 0))

        # Keep the units whose footprint is at least 'rel_tol' times the
        # largest in the layer.
        keep1 = foot1 >= rel_tol * foot1.max()
        keep2 = foot2 >= rel_tol * foot2.max()

        # Get the indexes of the kept and dropped units.
        k1 = torch.where(keep1)[0]
        d1 = torch.where(~keep1)[0]
        k2 = torch.where(keep2)[0]
        d2 = torch.where(~keep2)[0]

        # Inform the user about the kept units.
        logger.info(
            f"Layer 1: keeping {len(k1)}/{w0.shape[0]} units "
            f"(dropping {len(d1)}). "
            f"Layer 2: keeping {len(k2)}/{w2.shape[0]} units "
            f"(dropping {len(d2)}).")

        #-------------------------------------------------------------#

        # Fold the first layer's dropped units' constant contributions
        # (mean activation times outgoing weights) into the next bias.
        b2_folded = b2 + w2[:, d1] @ mean1[d1]

        # Keep the surviving rows and columns of the hidden layers.
        w0_pruned = w0[k1, :].contiguous()
        b0_pruned = b0[k1].contiguous()
        w2_pruned = w2[:, k1][k2, :].contiguous()
        b2_pruned = b2_folded[k2].contiguous()

        # Fold the second layer's dropped units' constant contributions
        # into the output heads' biases.
        bm_folded = sd["nb._layer_means.bias"] \
            + w_means[:, d2] @ mean2[d2]
        br_folded = sd["nb._layer_r_values.bias"] \
            + w_r_values[:, d2] @ mean2[d2]

        # Keep the surviving columns of the output heads.
        wm_pruned = w_means[:, k2].contiguous()
        wr_pruned = w_r_values[:, k2].contiguous()

        # Assemble the pruned parameters.
        pruned_sd = {
            "main.0.weight": w0_pruned,
            "main.0.bias": b0_pruned,
            "main.2.weight": w2_pruned,
            "main.2.bias": b2_pruned,
            "nb._layer_means.weight": wm_pruned,
            "nb._layer_means.bias": bm_folded,
            "nb._layer_r_values.weight": wr_pruned,
            "nb._layer_r_values.bias": br_folded}

        #-------------------------------------------------------------#

        # Get the pruned decoder's options (narrower hidden layers).
        pruned_options = copy.deepcopy(self._decoder_initial_options)
        pruned_options["n_units_hidden_layers"] = \
            [int(len(k1)), int(len(k2))]
        pruned_options.pop("decoder_pth_file", None)

        # In the model's own precision
        with self._default_dtype(self._dtype):

            # Build the pruned decoder.
            dec_pruned, _ = \
                self._get_decoder(latent_dim = latent_dim,
                                  genes = genes,
                                  decoder_options = pruned_options,
                                  device = device)

        # Set the pruned decoder in eval mode.
        dec_pruned = dec_pruned.eval()

        # Load the pruned parameters.
        dec_pruned.load_state_dict(
            {k: v.clone() for k, v in pruned_sd.items()})

        #-------------------------------------------------------------#

        # Without tracking gradients
        with torch.no_grad():

            # Get the full and pruned decoders' outputs for the real
            # representations.
            means_full, log_r_full = dec_full(z_real)
            means_pruned, log_r_pruned = dec_pruned(z_real)

        # Get the maximum relative error of the means.
        rel_means = \
            (means_full - means_pruned).abs().max().item() \
            / (means_full.abs().max().item() + 1e-30)

        # Get the maximum relative error of the log r-values.
        rel_log_r = \
            (log_r_full - log_r_pruned).abs().max().item() \
            / (log_r_full.abs().max().item() + 1e-30)

        # If the pruned decoder does not reproduce the full one
        if max(rel_means, rel_log_r) > verification_tol:

            # Raise an error.
            raise RuntimeError(
                "The pruned decoder does not reproduce the full one "
                f"(maximum relative error: means {rel_means:.3e}, "
                f"log-r-values {rel_log_r:.3e}; tolerance "
                f"{verification_tol:.1e}). Lower 'rel_tol' or widen "
                "the probe set.")

        # Inform the user about the verification.
        logger.info(
            "The pruned decoder reproduces the full one "
            f"(maximum relative error: means {rel_means:.3e}, "
            f"log-r-values {rel_log_r:.3e}).")

        #-------------------------------------------------------------#

        # Get the directory of the pruned decoder's file.
        dec_pruned_dir = \
            os.path.dirname(os.path.abspath(dec_pruned_pth_file))

        # Create it, if needed.
        os.makedirs(dec_pruned_dir, exist_ok = True)

        # Write the pruned decoder's parameters.
        torch.save(dec_pruned.state_dict(), dec_pruned_pth_file)

        # Build the pruned model's configuration.
        pruned_config = \
            self._build_pruned_config(
                gmm_pth_file = gmm_pth_file,
                dec_pruned_pth_file = dec_pruned_pth_file,
                n_units_hidden_layers = [int(len(k1)), int(len(k2))],
                config_model_pruned = config_model_pruned)

        # Get the directory of the configuration file.
        config_dir = \
            os.path.dirname(os.path.abspath(config_model_pruned))

        # Create it, if needed.
        os.makedirs(config_dir, exist_ok = True)

        # Write the configuration.
        with open(config_model_pruned, "w") as fh:
            yaml.safe_dump(pruned_config,
                           fh,
                           sort_keys = False)

        # Inform the user.
        logger.info(
            f"The pruned model was written to '{dec_pruned_pth_file}' "
            f"and '{config_model_pruned}'.")

        #-------------------------------------------------------------#

        # Return a new instance of the pruned model, on the same
        # device.
        return self.__class__(**pruned_config, device = device)


    def _load_reps(
            self,
            input_reps: Union[str, pd.DataFrame,
                              list[Union[str, pd.DataFrame]]],
            latent_dim: int) -> np.ndarray:
        """Load representations from CSV files and/or data frames into
        one array.

        Parameters
        ----------
        input_reps : :class:`str`, :class:`pandas.DataFrame`, or a \
            :class:`list` of either
            The representations, as passed to :meth:`prune` or
            :meth:`fit_gmm`.

        latent_dim : :class:`int`
            The expected number of latent dimensions.

        Returns
        -------
        z : :class:`numpy.ndarray`
            The representations, of shape ``(n_points, latent_dim)``,
            in double precision.
        """

        # If a single item was given
        if isinstance(input_reps, (str, pd.DataFrame)):

            # Make it a list.
            input_reps = [input_reps]

        # Initialize the list of the representations' arrays.
        frames = []

        # For each item
        for item in input_reps:

            # Read the item if it is a path, otherwise use it directly.
            df = item if isinstance(item, pd.DataFrame) \
                else pd.read_csv(item, index_col = 0)

            # Get the latent dimensions' columns.
            cols = [c for c in df.columns
                    if str(c).startswith("latent_dim_")]

            # If none of the expected columns were found
            if not cols:

                # Raise an error.
                raise ValueError(
                    "No 'latent_dim_*' columns were found in one of "
                    "the representations passed. Pass the "
                    "representations as written by "
                    "'get_representations'.")

            # Add this item's representations to the collected ones.
            frames.append(df[cols].to_numpy(dtype = "float64"))

        # Concatenate the representations.
        z = np.concatenate(frames, axis = 0)

        # If the representations do not match the latent space
        if z.shape[1] != latent_dim:

            # Raise an error.
            raise ValueError(
                f"The representations have {z.shape[1]} latent "
                "dimensions, but the model's latent space has "
                f"{latent_dim}.")

        # Return the representations.
        return z


    def _build_pruned_config(
            self,
            gmm_pth_file: str,
            dec_pruned_pth_file: str,
            n_units_hidden_layers: list[int],
            config_model_pruned: str) -> dict[str, object]:
        """Build the configuration of a pruned version of the model.

        Parameters
        ----------
        gmm_pth_file : :class:`str`
            The trained latent space's parameters, reused unchanged.

        dec_pruned_pth_file : :class:`str`
            The pruned decoder's parameters.

        n_units_hidden_layers : :class:`list`
            The number of units in the pruned decoder's hidden layers.

        config_model_pruned : :class:`str`
            The configuration file to be written (a genes' file is
            written next to it if the model has none).

        Returns
        -------
        config : :class:`dict`
            The pruned model's configuration.
        """

        # Copy the latent space's options, pointing them at the
        # trained parameters.
        latent_options = copy.deepcopy(self._latent_initial_options)
        latent_options["latent_pth_file"] = gmm_pth_file

        # Copy the decoder's options, with the pruned hidden layers
        # and parameters.
        decoder_options = copy.deepcopy(self._decoder_initial_options)
        decoder_options["n_units_hidden_layers"] = n_units_hidden_layers
        decoder_options["decoder_pth_file"] = dec_pruned_pth_file

        # Assemble the pruned model's configuration.
        config = {
            "latent_dim": int(self.latent.dim),
            "latent_type": self._latent_type,
            "latent_options": latent_options,
            "decoder_options": decoder_options,
            "scaling_factor": self._scaling_factor,
            "dtype": self._dtype}

        # If the model has options for a final mixture
        if self._gmm_final_options is not None:

            # Carry them over.
            config["gmm_final"] = self._gmm_final_options

        # If the model was built from a genes' file
        if self._genes_txt_file is not None:

            # Point at it.
            config["genes_txt_file"] = self._genes_txt_file

        # Otherwise
        else:

            # Get the path of a genes' file next to the configuration.
            genes_txt_file = \
                os.path.join(
                    os.path.dirname(
                        os.path.abspath(config_model_pruned)),
                    "genes_pruned.txt")

            # Write the genes to it.
            with open(genes_txt_file, "w") as fh:
                fh.write("\n".join(self._genes) + "\n")

            # Point at it.
            config["genes_txt_file"] = genes_txt_file

        # Return the configuration.
        return config


    def train(self,
              df_samples: pd.DataFrame,
              names_train: list,
              names_test: list,
              config_train: Optional[dict[str, object]] = None,
              gmm_pth_file: str = "gmm.pth",
              dec_pth_file: str = "dec.pth",
              gmm_final_pth_file: str = "gmm_final.pth",
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
            A data frame containing the samples, one per row. Each
            column contains either a gene's expression (if named
            after the gene's Ensembl ID) or additional information.

        names_train : :class:`list`
            A list of the names of the training samples, which should
            be a subset of the names of the samples in the input data
            frame.

        names_test : :class:`list`
            A list of the names of the test samples, which should be a
            subset of the names of the samples in the input data frame.

        config_train : :class:`dict`, optional
            A dictionary of options for the training. By default, the
            configuration the shipped models were trained with.

        gmm_pth_file : :class:`str`, ``"gmm.pth"``
            The .pth file where to save the GMM's trained parameters
            (means of the components, weights of the components,
            and log-variance of the components).

        dec_pth_file : :class:`str`, ``"dec.pth"``
            The .pth file where to save the decoder's trained
            parameters (weights and biases).

        gmm_final_pth_file : :class:`str`, ``"gmm_final.pth"``
            The .pth file for the Gaussian mixture fitted to the
            representations after training, written only if the
            model has ``gmm_final`` options.

        pathways : :class:`dict`, optional
            A dictionary where the keys are pathway names and the
            values are lists of genes' Ensembl IDs belonging to
            each pathway. Needed to save the pathways' saliency maps.

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
            The predicted r-values: :obj:`None` for Poisson counts, a
            single data frame for per-gene r-values, or a
            ``(train, test)`` tuple for per-sample r-values.

        df_loss : :class:`pandas.DataFrame`
            The losses calculated during training.

        dfs_metrics : :obj:`None` or :class:`tuple`
            :obj:`None` if metrics were not requested, else a
            ``(df_metrics_train, df_metrics_test)`` tuple with each
            epoch's metrics.

        df_time : :class:`pandas.DataFrame`
            The training-time metrics.
        """

        # If no configuration was given
        if config_train is None:

            # Import the configuration module (here, to avoid a
            # circular import).
            from bulkdgd.ioutil import configio

            # Load the configuration the shipped models used.
            config_train = \
                configio.load_config_train(config_file = None)

            # Inform the user.
            logger.info(
                "No training configuration was given, so the one the "
                "published ensemble was trained with is used.")

        #-------------------------------------------------------------#

        # Get whether an already-trained model may be trained further.
        continue_training = bool(config_train.get("continue_training",
                                                  False)) \
            if hasattr(config_train, "get") else False

        # If the model is trained and may not be trained further
        if self._is_trained and not continue_training:

            # Raise an error.
            errstr = \
                "This model was built from trained parameters, and " \
                "training it would move it away from them. To " \
                "continue training it, set 'continue_training: " \
                "true' at the top level of the training " \
                "configuration. To train a new model instead, build " \
                "one from an architecture rather than from a " \
                "checkpoint."
            raise RuntimeError(errstr)

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

        # If there are warnings in the configuration
        if warnings:

            # Log the warning messages.
            warning_msg = \
                "Warnings in the training configuration: " + \
                "|".join(warnings)
            logger.warning(warning_msg)

        #-------------------------------------------------------------#

        # Get the scale factor for the initialization of the
        # representations from the configuration.
        init_rep_scale = \
            config_train["representations_training_options"].get(
                "init_rep_scale", 0.0)

        # Get the distribution the initial representations are drawn
        # from (by default, a scaled normal).
        init_rep_dist = \
            config_train["representations_training_options"].get(
                "init_rep_dist", None)

        # Get the options for the distribution.
        init_rep_dist_options = \
            dict(config_train["representations_training_options"].get(
                "init_rep_dist_options", {}))

        #-------------------------------------------------------------#

        def _make_rep_layer(n_samples):
            """Build a representation layer for ``n_samples`` samples.

            Parameters
            ----------
            n_samples : :class:`int`
                The number of samples to build representations for.

            Returns
            -------
            rep_layer : \
                :class:`bulkdgd.core.latents.RepresentationLayer`
                The representation layer.
            """

            # Get the model's precision (the samplers would otherwise
            # use torch's default data type).
            dtype = self._DTYPES_TORCH[self._dtype]

            # If no distribution was given
            if init_rep_dist is None:

                # Return a layer of scaled standard normal draws.
                return latents.RepresentationLayer(\
                    values = init_rep_scale * torch.randn(\
                        size = (n_samples, self.latent.dim),
                        dtype = dtype)).to(self.device)

            # Copy the distribution's options, setting the shape from
            # the data and the model.
            options = dict(init_rep_dist_options)
            options["n_samples"] = n_samples
            options["dim"] = self.latent.dim

            # Return the representation layer.
            return latents.RepresentationLayer(\
                dist = init_rep_dist,
                dist_options = options,
                device = self.device).to(dtype = dtype)

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

        # Get whether to save the pathways' saliency maps during
        # training.
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
                "'pathways_saliency_maps_epoch.enabled' is set to " \
                "True."
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
                labels = labels_train,
                scaling_factor = self._scaling_factor)

        # Create the data loader with the training samples.
        data_loader_train = \
            _util.get_data_loader(
                dataset = dataset_train,
                config = config_train["data_loader_options"]["train"])

        # Create the representation layer for the training samples.
        rep_layer_train = _make_rep_layer(n_samples_train)

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
                labels = labels_test,
                scaling_factor = self._scaling_factor)

        # Create the data loader with the testing samples.
        data_loader_test = \
            _util.get_data_loader(
                dataset = dataset_test,
                config = config_train["data_loader_options"]["test"])

        # Create the representation layer for the testing samples.
        rep_layer_test = _make_rep_layer(n_samples_test)

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

        # If the model has options for a final Gaussian mixture model
        if self._gmm_final_options is not None:

            # Fit and save it (separately from the prior).
            self._fit_final_gmm(\
                reps_train = reps[0],
                config_final = \
                    config_train.get("gmm_final_training_options", {}),
                gmm_final_pth_file = gmm_final_pth_file)

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
