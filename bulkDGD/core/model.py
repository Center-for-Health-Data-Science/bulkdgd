#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-

#    model.py
#
#    This module contains the class implementing the full bulkDGD model
#    (:class:`core.model.BulkDGDModel`).
#
#    Copyright (C) 2024 Valentina Sora 
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
    "This module contains the class implementing the full bulkDGD " \
    "model (:class:`core.model.BulkDGDModel`)."


#######################################################################


# Import from the standard library.
import logging as log
import platform
import re
import time
# Import from third-party packages.
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
# Import from 'bulkDGD'.
from bulkDGD import util
from . import (
    dataclasses,
    decoder,
    latent,
    outputmodules,
    )


#######################################################################


# Get the module's logger.
logger = log.getLogger(__name__)


#######################################################################


class BulkDGDModel(nn.Module):

    """
    Class implementing the full bulkDGD model.
    """

    ####################### PRIVATE ATTRIBUTES ########################


    # Set the available normalization methods for the loss.
    _LOSS_NORM_METHODS = \
        {"gmm" : ["none", "n_samples"],
         "recon" : ["none", "n_samples"],
         "total" : ["none", "n_samples"]}


    ######################## PUBLIC ATTRIBUTES ########################


    # Set the supported optimizers to find the representations.
    OPTIMIZERS = ["adam"]


    ######################### INITIALIZATION ##########################


    def __init__(self,
                 input_dim,
                 gmm_options,
                 dec_options,
                 genes_txt_file = None,
                 gmm_pth_file = None,
                 dec_pth_file = None,):
        """Initialize an instance of the class.

        The model is initialized on the CPU. To move the model to
        another device, modify the ``device`` property.

        Parameters
        ----------
        input_dim : :class:`int`
            The dimensionality of the input (= the dimensionality of
            the representations, of the Gaussian mixture model, and of
            the first layer of the decoder.

        gmm_options : :class:`dict`
            The options for setting up the Gaussian mixture model.

            For the available options, refer to the
            :ref:`model_config_options` page.

        dec_options : :class:`dict`
            The options for setting up the decoder.

            For the available options, refer to the
            :ref:`model_config_options` page.

        genes_txt_file : :class:`str`
            A .txt file containing the Ensembl IDs of the genes
            included in the model.

            Training data will be checked to ensure counts are
            reported for all genes.

            The number of output units in the decoder is initialized
            from the number of genes found in this file.

        gmm_pth_file : :class:`str`, optional
            A .pth file with the GMM's trained parameters
            (means, weights, and log-variance of the components).

            Please ensure that the parameters match the Gaussian
            mixture model's structure.

            Omit it if the model needs training.

        dec_pth_file : :class:`str`, optional
            A .pth file containing the decoder's trained parameters
            (weights and biases).

            Please ensure that the parameters match the decoder's
            architecture.

            Omit it if the model needs training.
        """

        # Run the superclass' initialization.
        super(BulkDGDModel, self).__init__()

        #-------------------------------------------------------------#

        # Get the genes included in the model.
        genes = \
            self.__class__._load_genes_list(\
                genes_list_file = genes_txt_file)

        #-------------------------------------------------------------#

        # Initialize the Gaussian mixture model.
        self._gmm = \
            latent.GaussianMixtureModel(dim = input_dim,
                                        **gmm_options)

        # If the user provided a file with the GMM's trained
        # parameters
        if gmm_pth_file is not None:

            # Load the parameters.
            self.__class__._load_state(mod = self._gmm,
                                       pth_file = gmm_pth_file)

        # Inform the user that the GMM was set.
        infostr = "The Gaussian mixture model was successfully set."
        logger.info(infostr)

        #-------------------------------------------------------------#

        # Update the decoder's options.
        dec_options["output_module_options"]["output_dim"] = len(genes)

        # Get the decoder.
        self._dec = \
            decoder.Decoder(n_units_input_layer = input_dim,
                            **dec_options)

        # If the user provided a file with the decoder's trained
        # parameters
        if dec_pth_file is not None:

            # Load the parameters.
            self.__class__._load_state(mod = self._dec,
                                       pth_file = dec_pth_file)

        # Inform the user that the decoder was set.
        infostr = "The decoder was successfully set."
        logger.info(infostr)

        #-------------------------------------------------------------#

        # Get the output module's name.
        output_module_name = dec_options["output_module_name"]

        # If the output module is the 'nb_feature_dispersion' one
        if output_module_name == "nb_feature_dispersion":

            # Get the r-values associated with the negative binomials
            # modeling the different genes.
            r_values = \
                torch.exp(self._dec.nb.log_r).squeeze().detach()

            # Associate the r-values with the genes.
            self._r_values = pd.Series(r_values,
                                       index = genes)

        # If the output module is the 'nb_full_dispersion' or the
        # 'poisson' one
        elif output_module_name in ("nb_full_dispersion", "poisson"):

            # The r-values will be None.
            self._r_values = None

        #-------------------------------------------------------------#

        # By default, the model is initialized on the CPU.
        self._device = torch.device("cpu")


    @staticmethod
    def _load_state(mod,
                    pth_file):
        """Load a module's trained parameters.

        Parameters
        ----------
        mod : ``nn.Module``
            The module.

        pth_file : :class:`str`
            The PyTorch file to load the parameters from.
        """

        # Try to load the parameters
        try:

            mod.load_state_dict(torch.load(pth_file))

        # If something went wrong
        except Exception as e:

            # Raise an error.
            errstr = \
                f"It was not possible to load the parameters " \
                f"from '{pth_file}'. Error: {e}"
            raise Exception(errstr)

        # Inform the user that the parameters was successfully loaded.
        infostr = \
            "The parameters were successfully loaded from " \
            f"'{pth_file}'."
        logger.info(infostr)


    @staticmethod
    def _load_genes_list(genes_list_file):
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
        return \
            [line.rstrip("\n") for line in open(genes_list_file, "r") \
             if (not line.startswith("#") \
                 and not re.match(r"^\s*$", line))]


    ########################### PROPERTIES ############################


    @property
    def gmm(self):
        """The Gaussian mixture model.
        """

        return self._gmm


    @gmm.setter
    def gmm(self,
            value):
        """Raise an exception if the user tries to modify the value
        of ``gmm`` after initialization.
        """
        
        errstr = \
            "The value of 'gmm' is set at initialization and  " \
            "cannot be changed. If you want to change the " \
            "Gaussian mixture model, initialize a new instance of " \
            f"'{self.__class__.__name__}'."
        raise ValueError(errstr)


    @property
    def dec(self):
        """The decoder.
        """

        return self._dec


    @dec.setter
    def dec(self,
            value):
        """Raise an exception if the user tries to modify the value of
        ``dec`` after initialization.
        """
        
        errstr = \
            "The value of 'dec' is set at initialization and " \
            "cannot be changed. If you want to change the decoder, " \
            "initialize a new instance of " \
            f"'{self.__class__.__name__}'."
        raise ValueError(errstr)

    @property
    def device(self):
        """The device where the model is.
        """

        return self._device

    @device.setter
    def device(self,
               value):
        """Raise an exception if the user tries to modify the value of
        ``device`` directly.
        """
        
        # Move the model to the specified device.
        self.to(device = torch.device(value))

        # Update the device the model is on.
        self._device = value
    

    ######################### PRIVATE METHODS #########################


    def _get_optimizer(self,
                       optimizer_config,
                       optimizer_parameters):
        """Get the optimizer.

        Parameters
        ----------
        optimizer_config : :class:`dict`
            The configuration for the optimizer.

        optimizer_parameters : :class:`torch.nn.Parameter`
            The parameters that will be optimized.

        Returns
        -------
        optimizer : :class:`torch.optim.Optimizer`
            The optimizer.
        """

        # Get the name of the optimizer.
        optimizer_name = optimizer_config.get("name", None)

        # If the optimizer is not supported
        if optimizer_name not in self.OPTIMIZERS:

            # Raise an error.
            errstr = \
                f"The optimizer '{optimizer_name}' is not " \
                "supported. The supported optimizers are: " \
                f"{', '.join(self.OPTIMIZERS)}."
            raise ValueError(errstr)

        #-------------------------------------------------------------#

        # If it is the Adam optimizer
        if optimizer_name == "adam":

            # Set up the optimizer.
            optimizer = \
                torch.optim.Adam(optimizer_parameters,
                                 **optimizer_config.get("options", {}))

        #-------------------------------------------------------------#

        # Return the optimizer.
        return optimizer


    def _get_data_loader(self,
                         dataset,
                         data_loader_config):
        """Get the data loader.

        Parameters
        ----------
        dataset : \
            :class:`bulkDGD.core.dataclasses.GeneExpressionDataset`
            The dataset from which the data loader should be created.

        data_loader_config : :class:`dict`
            The configuration for the data loader.

        Returns
        -------
        data_loader : :class:`torch.utils.data.DataLoader`
            The data loader.
        """

        # Get the data loader.
        data_loader = DataLoader(dataset = dataset,
                                 **data_loader_config)

        # Return the data loader.
        return data_loader


    def _optimize_rep(self,
                      data_loader,
                      rep_layer,
                      optimizer,
                      n_comp,
                      n_rep_per_comp,
                      config_loss,
                      epochs,
                      opt_num):
        """Optimize the representation(s) found for each sample.

        Parameters
        ----------
        data_loader : :class:`torch.utils.data.DataLoader`
            The data loader.

        rep_layer : :class:`bulkDGD.core.latent.RepresentationLayer`
            The representation layer containing the initial
            representations.

        optimizer : :class:`torch.optim.Optimizer`
            The optimizer.

        n_comp : :class:`int`
            The number of components of the Gaussian mixture model
            for which at least one representation was drawn per
            sample.

        n_rep_per_comp : :class:`int`
            The number of new representations taken per sample
            per component of the Gaussian mixture model.

        config_loss : :class:`dict`
            A dictionary containing the options to output the loss.

        epochs : :class:`int`
            The number of epochs to run the optimization for.

        opt_num : :class:`int`
            The number of the optimization round (especially useful
            if multiple rounds are run).

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

        pred_r_values : :class:`torch.Tensor` or ``None``
            A tensor containing the predicted r-values of the negative
            binomial distributions modelling the genes' counts, if
            the counts are modelled by negative binomial distributions.

            This is a 2D tensor where:

            - The first dimension has a length equal to the number
              of samples.

            - The second dimension has a length equal to the
              dimensionality of the gene space.

            ``pred_r_values`` is ``None`` if the counts are modelled
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
        dim = self.gmm.dim

        # Get the number of genes (= the dimensionality of the
        # decoder's output).
        n_genes = self.dec.nb.output_dim

        # Get the method that will be used to normalize the total loss.
        loss_norm_method = config_loss.get("total").get("norm_method")

        # Create a list to store the CPU/wall clock time used in each
        # epoch of the optimization.
        time_opt = []       

        #-------------------------------------------------------------#

        # Inform the user that the optimization is starting
        infostr = f"Starting optimization number {opt_num}..."
        logger.info(infostr)

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
                # - 1st dimension: the total number of samples times
                #                  the number of components in the
                #                  Gaussian mixture model times the
                #                  number of representations taken per
                #                  component per sample ->
                #                  'n_samples' * 
                #                  'n_comp' *
                #                  'n_rep_per_comp'
                #
                # - 2nd dimension: the dimensionality of the Gaussian
                #                  mixture model ->
                #                  'dim'
                z_all = rep_layer()

                #-----------------------------------------------------#

                # Reshape the tensor containing the representations.
                #
                # The output is a 4D tensor with:
                #
                # - 1st dimension: the total number of samples ->
                #                  'n_samples'
                # 
                # - 2nd dimension: the number of representations taken
                #                  per component per sample ->
                #                  'n_rep_per_comp'
                #
                # - 3rd dimension: the number of components in the
                #                  Gaussian mixture model ->
                #                  'n_comp'
                #
                # - 4th dimension: the dimensionality of the Gaussian
                #                  mixture model ->
                #                  'dim'
                z_4d = z_all.view(n_samples,
                                  n_rep_per_comp,
                                  n_comp,
                                  dim)[samples_ixs]

                # Reshape the tensor again.
                #
                # The output is a 2D tensor with:
                #
                # - 1st dimension: the number of samples in the current
                #                  batch times the number of components
                #                  in the Gaussian mixture model times
                #                  the number of representations taken
                #                  per component per sample ->
                #                  'n_samples_in_batch' * 
                #                  'n_comp' *
                #                  'n_rep_per_comp'
                #
                # - 2nd dimension: the dimensionality of the Gaussian
                #                  mixture model ->
                #                  'dim'
                z = z_4d.view(n_samples_in_batch * \
                                n_rep_per_comp * \
                                n_comp,
                              dim)

                #-----------------------------------------------------#

                # If the chosen output module means that the r-values
                # are not learned
                if isinstance(\
                    self.dec.nb,
                    (outputmodules.OutputModuleNBFeatureDispersion,
                     outputmodules.OutputModulePoisson)):
                    
                    # Get the predicted scaled means of the
                    # distributions modelling the genes' counts.
                    #
                    # The output is a 2D tensor with:
                    #
                    # - 1st dimension: the number of samples in the
                    #                  current batch times the number
                    #                  of components in the Gaussian
                    #                  mixture model times the number
                    #                  of representations taken per
                    #                  component per sample ->
                    #                  'n_samples_in_batch' * 
                    #                  'n_comp' *
                    #                  'n_rep_per_comp'
                    #
                    # - 2nd dimension: the dimensionality of the output
                    #                  (= gene) space ->
                    #                  'n_genes'
                    pred_means = self.dec(z = z)

                # If the chosen output module means that the r-values
                # are learned
                elif isinstance(\
                    self.dec.nb,
                    outputmodules.OutputModuleNBFullDispersion):

                    # Get the predicted scaled means and r-values
                    # of the negative binomial distributions modelling
                    # the genes' counts.
                    #
                    # Both outputs are 2D tensors with:
                    #
                    # - 1st dimension: the number of samples in the
                    #                  current batch times the number
                    #                  of components in the Gaussian
                    #                  mixture model times the number
                    #                  of representations taken per
                    #                  component per sample ->
                    #                  'n_samples_in_batch' * 
                    #                  'n_comp' *
                    #                  'n_rep_per_comp'
                    #
                    # - 2nd dimension: the dimensionality of the output
                    #                  (= gene) space ->
                    #                  'n_genes'
                    pred_means, pred_log_r_values = self.dec(z = z)

                    # Reshape the predicted r-values to match the shape
                    # required to compute the loss.
                    #
                    # The output is a 4D tensor with:   
                    #
                    # - 1st dimension: the number of samples in the
                    #                  current batch ->
                    #                  'n_samples_in_batch'
                    #
                    # - 2nd dimension: the number of representations
                    #                  taken per component per
                    #                  sample -> 'n_rep_per_comp'
                    #
                    # -3rd dimension: the number of components in the
                    #                 Gaussian mixture model ->
                    #                 'n_comp'
                    #
                    # - 4th dimension: the dimensionality of the output
                    #                  (= gene) space -> 'n_genes'      
                    pred_log_r_values = \
                        pred_log_r_values.view(n_samples_in_batch,
                                               n_rep_per_comp,
                                               n_comp,
                                               n_genes)

                #-----------------------------------------------------#

                # Get the observed gene expression and "expand" the
                # resulting tensor to match the shape required to
                # compute the reconstruction loss.
                #
                # The output is a 4D tensor with:
                #
                # - 1st dimension: the number of samples in the current
                #                  batch -> 'n_samples_in_batch'
                #
                # - 2nd dimension: the number of representations taken
                #                  per component per sample ->
                #                  'n_rep_per_comp'
                #
                # - 3rd dimension: the number of components in the
                #                  Gaussian mixture model -> 'n_comp'
                #
                # - 4th dimension: the dimensionality of the output
                #                  (= gene) space -> 'n_genes'
                obs_counts = \
                    samples_exp.unsqueeze(1).unsqueeze(1).expand(\
                        -1,
                        n_rep_per_comp,
                        n_comp,
                        -1)

                #-----------------------------------------------------#

                # Get the scaling factors for the mean of each negative
                # binomial modelling the expression of a gene and
                # reshape it so that it matches the shape required to
                # compute the reconstruction loss.
                #
                # The output is a 4D tensor with:
                #
                # - 1st dimension: the number of samples in the current
                #                  batch -> 'n_samples_in_batch'
                #
                # - 2nd dimension: 1
                #
                # - 3rd dimension: 1
                #
                # - 4th dimension: 1
                scaling_factors = \
                    decoder.reshape_scaling_factors(samples_mean_exp,
                                                    4)

                #-----------------------------------------------------#

                # Reshape the predicted scaled means to match the
                # shape required to compute the loss.
                #
                # The output is a 4D tensor with:   
                #
                # - 1st dimension: the number of samples in the current
                #                  batch -> 'n_samples_in_batch'
                #
                # - 2nd dimension: the number of representations taken
                #                  per component per sample ->
                #                  'n_rep_per_comp'
                #
                # -3rd dimension: the number of components in the
                #                 Gaussian mixture model -> 'n_comp'
                #
                # - 4th dimension: the dimensionality of the output
                #                  (= gene) space -> 'n_genes'      
                pred_means = pred_means.view(n_samples_in_batch,
                                             n_rep_per_comp,
                                             n_comp,
                                             n_genes)

                #-----------------------------------------------------#

                # If the chosen output module means that the r-values
                # are not learned
                if isinstance(\
                    self.dec.nb,
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
                    self.dec.nb,
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
                # - 1st dimension: the number of samples in the current
                #                  batch -> 'n_samples_in_batch'
                # 
                # - 2nd dimension: the number of representations taken
                #                  per component per sample ->
                #                  'n_rep_per_comp'
                #
                # - 3rd dimension: the number of components in the
                #                  Gaussian mixture model ->
                #                  'n_comp'
                #
                # - 4th dimension: the dimensionality of the output
                #                  (= gene) space ->
                #                  'n_genes'
                recon_loss = self.dec.nb.loss(**recon_loss_options)

                # Get the total reconstruction loss by summing all
                # values in the 'recon_loss' tensor.
                #
                # The output is a tensor containing a single value.
                recon_loss_sum = recon_loss.sum().clone()

                #-----------------------------------------------------#

                # Get the loss for the Gaussian mixture model.
                #
                # 'gmm(z)' computes the negative log density of the
                # probability of the representations 'z' being drawn
                # from the Gaussian mixture model.
                #
                # The output is a 1D tensor with:
                #
                # - 1st dimension: the number of samples in the current
                #                  batch times the number of components
                #                  in the Gaussian mixture model times
                #                  the number of representations taken
                #                  per component per sample ->
                #                  'n_samples_in_batch' * 
                #                  'n_comp' *
                #                  'n_rep_per_comp'
                gmm_loss = self.gmm(x = z)

                # Get the total loss for the Gaussian mixture model by
                # summing over all values in the 'gmm_loss' tensor.
                # 
                # The output is a tensor containing a single value.
                gmm_loss_sum = gmm_loss.sum().clone()

                #-----------------------------------------------------#

                # Get the total loss by summing the reconstruction loss
                # and the loss of the Gaussian mixture model.
                #
                # The output is a tensor containing a single value.
                total_loss = recon_loss_sum + gmm_loss_sum

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
                    self._normalize_loss(\
                        loss = total_loss.item(),
                        loss_type = "total",
                        loss_norm_method = loss_norm_method,
                        loss_norm_options = {"n_samples" : n_samples})

            #---------------------------------------------------------#

            # Take an optimization step.
            optimizer.step()

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
            infostr = \
                f"Epoch {epoch}: loss {rep_avg_loss_epoch:.3f}, " \
                f"epoch CPU time {time_tot_epoch_cpu:.3f} s, " \
                f"backward step CPU time {time_tot_bw_cpu:.3f} s, " \
                "epoch wall clock time " \
                f"{time_tot_epoch_wall:.3f} s, " \
                "backward step wall clock time " \
                f"{time_tot_bw_wall:.3f} s."
            logger.info(infostr)

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
                    self.dec.nb,
                    outputmodules.OutputModuleNBFeatureDispersion):

                    # Get the predicted scaled means.
                    means_final = self.dec(z = rep_final)

                    # Get the r-values.
                    r_values_final = \
                        torch.exp(\
                            self.dec.nb.log_r).squeeze().detach()

                # If the genes' counts are modelled by negative
                # binomial distributions whose r-values are learned
                # per gene per sample
                elif isinstance(\
                    self.dec.nb,
                    outputmodules.OutputModuleNBFullDispersion):

                    # Get the predicted scaled means.
                    means_final, log_r_values_final = \
                        self.dec(z = rep_final)

                    # Get the r-values.
                    r_values_final = \
                        torch.exp(\
                            log_r_values_final).squeeze().detach()

                # If the genes' counts are modelled by Poisson
                # distributions
                elif isinstance(    
                    self.dec.nb,
                    outputmodules.OutputModulePoisson):

                    # Get the predicted scaled means.
                    means_final = self.dec(z = rep_final)

                    # The r-values will be None.
                    r_values_final = None

                #-----------------------------------------------------#

                # Return the representations, the predicted scaled
                # means, the predicted r-values, and the time data.
                return rep_final, \
                       means_final, r_values_final, \
                       time_opt


    def _select_best_rep(self,
                         data_loader,
                         rep_layer,
                         n_rep_per_comp):
        """Select the best representation per sample.

        Parameters
        ----------
        data_loader : :class:`torch.utils.data.DataLoader`
            The data loader.

        rep_layer : :class:`bulkDGD.core.latent.RepresentationLayer`
            The representation layer containing the representations
            found for the samples.

        n_rep_per_comp : :class:`int`
            The number of new representations that were taken per
            sample per component of the Gaussian mixture model.

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
        n_comp = self.gmm.n_comp

        # Get the dimensionality of the latent space.
        dim = self.gmm.dim

        # Get the number of genes (= dimensionality of the decoder's
        # output).
        n_genes = self.dec.nb.output_dim

        # Initialize an empty tensor to store the best representations
        # found for all samples.
        #
        # This is a 2D tensor with:
        #
        # - 1st dimension: the total number of samples -> 'n_samples'
        #
        # - 2nd dimension: the dimensionality of the Gaussian mixture
        #                  model -> 'dim'
        best_reps = torch.empty((n_samples, dim))

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

            # Get the representations' values for the samples in the
            # batch from the representation layer.
            # 
            # The representations are stored in a 2D tensor with:
            #
            # - 1st dimension: the total number of samples times the
            #                  number of components in the Gaussian
            #                  mixture model times the number of
            #                  representations taken per component
            #                  per sample ->
            #                  'n_samples' * 
            #                  'n_comp' *
            #                  'n_rep_per_comp'
            #
            # - 2nd dimension: the dimensionality of the Gaussian
            #                  mixture model ->
            #                  'dim'
            z_all = rep_layer()

            # Reshape the tensor containing the representations.
            #
            # The output is a 4D tensor with:
            #
            # - 1st dimension: the total number of samples ->
            #                  'n_samples'
            # 
            # - 2nd dimension: the number of representations taken per
            #                  component per sample ->
            #                  'n_rep_per_comp'
            #
            # - 3rd dimension: the number of components in the Gaussian
            #                  mixture model ->
            #                  'n_comp'
            #
            # - 4th dimension: the dimensionality of the Gaussian
            #                  mixture model ->
            #                  'dim'
            z_4d = \
                z_all.view(n_samples,
                           n_rep_per_comp,
                           n_comp,
                           dim)[samples_ixs]

            # Reshape the tensor again.
            #
            # The output is a 2D tensor with:
            #
            # - 1st dimension: the number of samples in the current
            #                  batch times the number of components
            #                  in the Gaussian mixture model times the
            #                  number of representations taken per
            #                  component per sample ->
            #                  'n_samples_in_batch' * 
            #                  'n_comp' *
            #                  'n_rep_per_comp'
            #
            # - 2nd dimension: the dimensionality of the Gaussian
            #                  mixture model ->
            #                  'dim'
            z = z_4d.view(n_samples_in_batch * \
                            n_rep_per_comp * \
                            n_comp,
                          dim)

            #---------------------------------------------------------#

            # If the chosen output module means that the r-values
            # are not learned
            if isinstance(\
                self.dec.nb,
                (outputmodules.OutputModuleNBFeatureDispersion,
                 outputmodules.OutputModulePoisson)):
                
                # Get the predicted scaled means of the distributions
                # modelling the genes' counts.
                #
                # The output is a 2D tensor with:
                #
                # - 1st dimension: the number of samples in the
                #                  current batch times the number
                #                  of components in the Gaussian
                #                  mixture model times the number
                #                  of representations taken per
                #                  component per sample ->
                #                  'n_samples_in_batch' * 
                #                  'n_comp' *
                #                  'n_rep_per_comp'
                #
                # - 2nd dimension: the dimensionality of the output
                #                  (= gene) space ->
                #                  'n_genes'
                pred_means = self.dec(z = z)

            # If the chosen output module means that the r-values
            # are learned
            elif isinstance(\
                self.dec.nb,
                outputmodules.OutputModuleNBFullDispersion):

                # Get the predicted scaled means and r-values of the
                # negative binomial distributions.
                #
                # Both outputs are 2D tensors with:
                #
                # - 1st dimension: the number of samples in the
                #                  current batch times the number
                #                  of components in the Gaussian
                #                  mixture model times the number
                #                  of representations taken per
                #                  component per sample ->
                #                  'n_samples_in_batch' * 
                #                  'n_comp' *
                #                  'n_rep_per_comp'
                #
                # - 2nd dimension: the dimensionality of the output
                #                  (= gene) space ->
                #                  'n_genes'
                pred_means, pred_log_r_values = self.dec(z = z)

                # Reshape the predicted r-values to match the shape
                # required to compute the loss.
                #
                # The output is a 4D tensor with:   
                #
                # - 1st dimension: the number of samples in the
                #                  current batch ->
                #                  'n_samples_in_batch'
                #
                # - 2nd dimension: the number of representations
                #                  taken per component per
                #                  sample -> 'n_rep_per_comp'
                #
                # -3rd dimension: the number of components in the
                #                 Gaussian mixture model ->
                #                 'n_comp'
                #
                # - 4th dimension: the dimensionality of the output
                #                  (= gene) space -> 'n_genes'      
                pred_log_r_values = \
                    pred_log_r_values.view(n_samples_in_batch,
                                           n_rep_per_comp,
                                           n_comp,
                                           n_genes)

            #---------------------------------------------------------#

            # Get the observed counts for the expression of each gene
            # in each sample, and "expand" the resulting tensor to
            # match the shape required to compute the reconstruction
            # loss.
            #
            # The output is a 4D tensor with:
            #
            # - 1st dimension: the number of samples in the current
            #                  batch -> 'n_samples_in_batch'
            #
            # - 2nd dimension: the number of representations taken
            #                  per component per sample ->
            #                  'n_rep_per_comp'
            #
            # - 3rd dimension: the number of components in the Gaussian
            #                  mixture model -> 'n_comp'
            #
            # - 4th dimension: the dimensionality of the output
            #                  (= gene) space -> 'n_genes'
            obs_counts = \
                samples_exp.unsqueeze(1).unsqueeze(1).expand(\
                    -1,
                    n_rep_per_comp,
                    n_comp,
                    -1)

            #---------------------------------------------------------#

            # Get the scaling factors for the mean of each negative
            # binomial used to model the expression of a gene and
            # reshape it so that it matches the shape required to
            # compute the reconstruction loss.
            #
            # The output is a 4D tensor with:
            #
            # - 1st dimension: the number of samples in the current
            #                  batch -> 'n_samples_in_batch'
            #
            # - 2nd dimension: 1
            #
            # - 3rd dimension: 1
            #
            # - 4th dimension: 1
            scaling_factors = \
                decoder.reshape_scaling_factors(samples_mean_exp,
                                                4)

            #---------------------------------------------------------#

            # Reshape the predicted scaled means to match the shape
            # required to compute the reconstruction loss.
            #
            # The output is a 4D tensor with:   
            #
            # - 1st dimension: the number of samples in the current
            #                  batch -> 'n_samples_in_batch'
            #
            # - 2nd dimension: the number of representations taken
            #                  per component per sample ->
            #                  'n_rep_per_comp'
            #
            # - 3rd dimension: the number of components in the Gaussian
            #                  mixture model -> 'n_comp'
            #
            # - 4th dimension: the dimensionality of the output
            #                  (= gene) space -> 'n_genes'      
            pred_means = pred_means.view(n_samples_in_batch,
                                         n_rep_per_comp,
                                         n_comp,
                                         n_genes)
     
            #---------------------------------------------------------#

            # If the chosen output module means that the r-values
            # are not learned
            if isinstance(\
                self.dec.nb,
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
                self.dec.nb,
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
            # - 1st dimension: the number of samples in the current
            #                  batch -> 'n_samples_in_batch'
            # 
            # - 2nd dimension: the number of representations taken
            #                  per component per sample ->
            #                  'n_rep_per_comp'
            #
            # - 3rd dimension: the number of components in the
            #                  Gaussian mixture model ->
            #                  'n_comp'
            #
            # - 4th dimension: the dimensionality of the output
            #                  (= gene) space ->
            #                  'n_genes'
            recon_loss = self.dec.nb.loss(**recon_loss_options)

            # Get the total reconstruction loss by summing over the
            # last dimension of the 'recon_loss' tensor.
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
            #                  'n_comp'
            recon_loss_sum = recon_loss.sum(-1).clone()

            # Reshape the reconstruction loss so that it can be summed
            # to the GMM loss (calculated below).
            #
            # The aim is to have one loss per representation per
            # sample.
            #
            # The output is, therefore, a 1D tensor with:
            #
            # - 1st dimension: the number of samples in the current
            #                  batch times the number of components
            #                  in the Gaussian mixture model times the
            #                  number of representations taken per
            #                  component per sample ->
            #                  'n_samples_in_batch' * 
            #                  'n_comp' *
            #                  'n_rep_per_comp'
            recon_loss_sum_reshaped = \
                recon_loss_sum.view(n_samples_in_batch * \
                                    n_rep_per_comp * \
                                    n_comp)

            #---------------------------------------------------------#

            # Get the GMM error. 
            #
            # 'gmm(z)' computes the negative log density of the
            # probability of the representations 'z' being drawn from
            # the Gaussian mixture model.
            #
            # The shape of the loss is consistent with the shape of the
            # reconstruction loss in 'recon_loss_sum_shaped'.
            #
            # The output is, therefore, a 1D tensor with:
            #
            # - 1st dimension: the number of samples in the current
            #                  batch times the number of components
            #                  in the Gaussian mixture model times the
            #                  number of representations taken per
            #                  component per sample ->
            #                  'n_samples_in_batch' * 
            #                  'n_comp' *
            #                  'n_rep_per_comp'
            gmm_loss = self.gmm(z).clone()

            #---------------------------------------------------------#

            # Get the total loss.
            #
            # The loss has as many components as the total number of
            # representations computed for the current batch of samples
            # ('n_rep_per_comp' * 'n_comp' representations for each
            # sample in the batch).
            #
            # The output is a 1D tensor with:
            #
            # - 1st dimension: the number of samples in the current
            #                  batch times the number of components
            #                  in the Gaussian mixture model times the
            #                  number of representations taken per
            #                  component per sample ->
            #                  'n_samples_in_batch' * 
            #                  'n_comp' *
            #                  'n_rep_per_comp'
            total_loss = recon_loss_sum_reshaped + gmm_loss

            #---------------------------------------------------------#

            # Reshape the tensor containing the total loss.
            #
            # The output is a 2D tensor with:
            #
            # - 1st dimension: the number of samples in the current
            #                  batch  -> 'n_samples_in_batch'
            #
            # - 2nd dimension: the number of representations taken
            #                  per component of the Gaussian mixture
            #                  model per sample times the number of
            #                  components -> 
            #                  'n_rep_per_comp' * 'n_comp'
            total_loss_reshaped = \
                total_loss.view(n_samples_in_batch,
                                n_rep_per_comp * n_comp)

            #---------------------------------------------------------#

            # Get the best representation for each sample in the
            # current batch.
            #
            # The output is a 1D tensor with:
            #
            # - 1st dimension: the number of samples in the current
            #                  batch -> 'n_samples_in_batch'
            best_rep_per_sample = torch.argmin(total_loss_reshaped,
                                               dim = 1).squeeze(-1)

            #---------------------------------------------------------#

            # Get the best representations for the samples in the batch
            # from the 'n_rep_per_comp' * 'n_comp' representations
            # taken for each sample.
            #
            # The output is a 2D tensor with:
            #
            # - 1st dimension: the number of samples in the current
            #                  batch-> 'n_samples_in_batch'
            #
            # - 2nd dimension: the dimensionality of the Gaussian
            #                  mixture model ->
            #                  'dim'
            rep = z.view(n_samples_in_batch,
                         n_rep_per_comp * n_comp,
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


    def _get_representations_one_opt(self,
                                     dataset,
                                     n_rep_per_comp,
                                     config):
        """Get the representations for a set of samples by
        initializing ``n_rep_per_comp`` representations per each
        component of the Gaussian mixture model per sample, selecting
        the best representation for each sample, and optimizing these
        representations.

        Parameters
        ----------
        dataset : \
            :class:`bulkDGD.core.dataclasses.GeneExpressionDataset`
            The dataset from which the data loader should be created.

        config : :class:`dict`
            A dictionary with the following keys:

            * ``"n_rep_per_comp", whose associated value should be
              the he number of new representations to be taken per
              component per sample.

            * ``"loss"``, whose associated value should be a dictionary
              with the following keys:

                * ``"gmm"``, whose associated value should be a
                  dictionary with the following keys:

                    * ``"norm_method"``, whose associated value should
                      be the name of the method that will be used to
                      normalize the GMM loss reported for each epoch.

                * ``"recon"``, whose associated value should be a
                  dictionary with the following keys:

                    * ``"norm_method"``, whose associated value should
                      be the name of the method that will be used to
                      normalize the reconstruction loss reported for
                      each epoch.

                * ``"total"``, whose associated value should be a
                  dictionary with the following keys:

                    * ``"norm_method"``, whose associated value should
                      be the name of the method that will be used to
                      normalize the total loss reported for each epoch.
            
            * ``"opt"``, whose associated value should be a dictionary
              with the following keys:

                * ``"epochs"``, whose associated value should be the
                  number of epochs the optimization is run for.

                * ``"optimizer"``, whose associated value should be a
                  dictionary with the following keys:

                    * ``"name"``, whose associated value should be the
                      name of the optimizer to be used. The names of
                      the available optimizers are stored into the
                      ``OPTIMIZERS`` class attribute.

                    The following optional key may also be present:

                    * ``"options`", whose associated value should be a
                      dictionary whose key-value pairs correspond to
                      the options needed to set up the optimizer. The
                      options available depend on the chosen optimizer.

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

        pred_r_values : :class:`torch.Tensor` or ``None``
            A tensor containing the predicted r-values of the negative
            binomial distributions modelling the genes' counts, if
            the counts are modelled by negative binomial distributions.

            This is a 2D tensor where:

            - The first dimension has a length equal to the number
              of samples.

            - The second dimension has a length equal to the
              dimensionality of the gene space.

            ``pred_r_values`` is ``None`` if the counts are modelled
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
        data_loader_config = config["data_loader"]

        # Get the number of representations per component per sample.
        n_rep_per_comp = config["n_rep_per_comp"]

        # Get the configuration for the reporting of the loss.
        config_loss = config["loss"]

        #-------------------------------------------------------------#

        # Get the configuration for the optimizer.
        config_optimizer = config["opt"]["optimizer"]

        # Get the number of epochs to run the optimization for.
        epochs = config["opt"]["epochs"]

        #-------------------------------------------------------------#

        # Create the data loader.
        data_loader = \
            self._get_data_loader(\
                dataset = dataset,
                data_loader_config = data_loader_config)

        #-------------------------------------------------------------#

        # Get the initial values for the representations by sampling
        # from the Gaussian mixture model.
        # 
        # The representations will be the sampled from the means of
        # the mixture components (since 'sampling_method' is set to
        # '"mean"').
        #
        # The output is a 2D tensor with:
        #
        # - 1st dimension: the total number of samples times the number
        #                  of components in the Gaussian mixture model
        #                  times the number of representations taken
        #                  per component per sample ->
        #                  'n_samples' *
        #                  'n_comp' * 
        #                  'n_rep_per_comp'
        #
        # - 2nd dimension: the dimensionality of the Gaussian mixture
        #                  model ->
        #                  'dim'
        rep_init = \
            self.gmm.sample_new_points(\
                n_points = n_samples, 
                sampling_method = "mean",
                n_samples_per_comp = n_rep_per_comp)

        # Create a representation layer containing the initialized
        # representations.
        rep_layer_init = \
            latent.RepresentationLayer(values = rep_init).to(\
                self.device)

        #-------------------------------------------------------------#

        # Select the best representation for each sample among those
        # initialized (we initialized at least one per sample per
        # component of the Gaussian mixture model).
        rep_best = \
            self._select_best_rep(\
                data_loader = data_loader,
                rep_layer = rep_layer_init,
                n_rep_per_comp = n_rep_per_comp)

        # Create a representation layer containing the best
        # representations found.
        rep_layer_best = \
            latent.RepresentationLayer(values = rep_best).to(\
                self.device)

        #-------------------------------------------------------------#
        
        # Get the optimizer for the optimization.
        optimizer = \
            self._get_optimizer(\
                optimizer_config = config_optimizer,
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
                n_comp = 1,
                n_rep_per_comp = 1,
                config_loss = config_loss,
                epochs = epochs,
                opt_num = 1)

        #-------------------------------------------------------------#

        # Make the gradients zero.
        optimizer.zero_grad()

        #-------------------------------------------------------------#

        # Return the representations, the predicted means and r-values,
        # the time data.
        return rep, pred_means, pred_r_values, time


    def _get_representations_two_opt(self,
                                     dataset,
                                     config):
        """Get the best representations for a set of samples by
        initializing ``n_rep_per_comp`` representations per each
        component of the Gaussian mixture model per sample, optimizing
        these representations, selecting the best representation for
        for each sample, and optimizing these representations further.

        Parameters
        ----------
        dataset : \
            :class:`bulkDGD.core.dataclasses.GeneExpressionDataset`
            The dataset from which the data loader should be created.
    
        config : :class:`dict`
            A dictionary with the following keys:

            * ``"n_rep_per_comp", whose associated value should be
              the he number of new representations to be taken per
              component per sample.

            * ``"loss"``, whose associated value should be a dictionary
              with the following keys:

                * ``"gmm"``, whose associated value should be a
                  dictionary with the following keys:

                    * ``"norm_method"``, whose associated value should
                      be the name of the method that will be used to
                      normalize the GMM loss reported for each epoch.

                * ``"recon"``, whose associated value should be a
                  dictionary with the following keys:

                    * ``"norm_method"``, whose associated value should
                      be the name of the method that will be used to
                      normalize the reconstruction loss reported for
                      each epoch.

                * ``"total"``, whose associated value should be a
                  dictionary with the following keys:

                    * ``"norm_method"``, whose associated value should
                      be the name of the method that will be used to
                      normalize the total loss reported for each epoch.
            
            * ``"opt1"``, whose associated value should be a dictionary
              with the following keys:

                * ``"epochs"``, whose associated value should be the
                  number of epochs the optimization is run for.

                * ``"optimizer"``, whose associated value should be a
                  dictionary with the following keys:

                    * ``"name"``, whose associated value should be the
                      name of the optimizer to be used. The names of
                      the available optimizers are stored into the
                      ``OPTIMIZERS`` class attribute.

                    The following optional key may also be present:

                    * ``"options`", whose associated value should be a
                      dictionary whose key-value pairs correspond to
                      the options needed to set up the optimizer. The
                      options available depend on the chosen optimizer.

            * ``"opt2"``, whose associated value should be a dictionary
              with the following keys:

                * ``"epochs"``, whose associated value should be the
                  number of epochs the optimization is run for.

                * ``"optimizer"``, whose associated value should be a
                  dictionary with the following keys:

                    * ``"name"``, whose associated value should be the
                      name of the optimizer to be used. The names of
                      the available optimizers are stored into the
                      ``OPTIMIZERS`` class attribute.

                    The following optional key may also be present:

                    * ``"options`", whose associated value should be a
                      dictionary whose key-value pairs correspond to
                      the options needed to set up the optimizer. The
                      options available depend on the chosen optimizer.

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

        pred_r_values : :class:`torch.Tensor` or ``None``
            A tensor containing the predicted r-values of the negative
            binomial distributions modelling the genes' counts, if
            the counts are modelled by negative binomial distributions.

            This is a 2D tensor where:

            - The first dimension has a length equal to the number
              of samples.

            - The second dimension has a length equal to the
              dimensionality of the gene space.

            ``pred_r_values`` is ``None`` if the counts are modelled
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
        data_loader_config = config["data_loader"]

        # Get the number of representations per component per sample.
        n_rep_per_comp = config["n_rep_per_comp"]

        # Get the configuration for reporting the loss.
        config_loss = config["loss"]

        #-------------------------------------------------------------#

        # Get the configuration for the first optimization.
        config_opt_1 = config["opt1"]

        # Get the configuration for the optimizer for the first
        # optimization.
        config_optimizer_1 = config_opt_1["optimizer"]

        # Get the number of epochs to run the first optimization for.
        epochs_1 = config_opt_1["epochs"]

        #-------------------------------------------------------------#

        # Get the configuration for the second optimization.
        config_opt_2 = config["opt2"]

        # Get the configuration for the optimizer for the second
        # optimization.
        config_optimizer_2 = config_opt_2["optimizer"]

        # Get the number of epochs to run the second optimization for.
        epochs_2 = config_opt_2["epochs"]

        #-------------------------------------------------------------#

        # Create the data loader.
        data_loader = \
            self._get_data_loader(\
                dataset = dataset,
                data_loader_config = data_loader_config)

        #-------------------------------------------------------------#

        # Get the initial values for the representations by sampling
        # from the Gaussian mixture model.
        #
        # The representations will be the sampled from the scaled
        # means of the mixture components (since 'sampling_method'
        # is set to '"mean"').
        #
        # The output is a 2D tensor with:
        #
        # - 1st dimension: the total number of samples times the number
        #                  of components in the Gaussian mixture model
        #                  times the number of representations taken
        #                  per component per sample ->
        #                  'n_samples' *
        #                  'n_comp' * 
        #                  'n_rep_per_comp'
        #
        # - 2nd dimension: the dimensionality of the Gaussian mixture
        #                  model ->
        #                  'dim'
        rep_init = \
            self.gmm.sample_new_points(\
                n_points = n_samples, 
                sampling_method = "mean",
                n_samples_per_comp = n_rep_per_comp)

        #-------------------------------------------------------------#

        # Create the representation layer.
        rep_layer_init = \
            latent.RepresentationLayer(values = rep_init).to(\
                self.device)

        #-------------------------------------------------------------#

        # Get the optimizer for the first optimization.
        optimizer_1 = \
            self._get_optimizer(\
                optimizer_config = config_optimizer_1,
                optimizer_parameters = rep_layer_init.parameters())

        #-------------------------------------------------------------#

        # Get the optimized representations, the predicted means of
        # the distributions modelling the counts, the predicted
        # r-values of the distributions modelling the counts (if any),
        # and the time data.
        rep_1, pred_means_1, pred_r_values_1, time_1 = \
            self._optimize_rep(\
                data_loader = data_loader,
                rep_layer = rep_layer_init,
                optimizer = optimizer_1,
                n_comp = self.gmm.n_comp,
                n_rep_per_comp = n_rep_per_comp,
                config_loss = config_loss,
                epochs = epochs_1,
                opt_num = 1)

        #-------------------------------------------------------------#

        # Create the representation layer.
        rep_layer_1 = \
            latent.RepresentationLayer(values = rep_1).to(\
                self.device)

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
                n_rep_per_comp = n_rep_per_comp)

        # Create a representation layer containing the best
        # representations found (one representation per sample).
        rep_layer_best = \
            latent.RepresentationLayer(values = rep_best).to(\
                self.device)

        #-------------------------------------------------------------#

        # Get the optimizer for the second optimization.
        optimizer_2 = \
            self._get_optimizer(\
                optimizer_config = config_optimizer_2,
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
                n_comp = 1,
                n_rep_per_comp = n_rep_per_comp,
                config_loss = config_loss,
                epochs = epochs_2,
                opt_num = 2)

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


    def _get_time_dataframe(self,
                            time_list):
        """Get the data frame containing the information about the
        computing time.

        Parameters
        ----------
        time_list : :class:`list`
            A list of tuples storing, for each epoch, information
            about the CPU and wall clock time used by the entire
            epoch and by the backpropagation step run within the
            epoch.

        Returns
        -------
        df_time : ``pandas.DataFrame``
            A data frame containing data about the computing time.
        """

        # Crate a data frame for the CPU/wall clock time.
        df_time = pd.DataFrame(time_list)

        # Get the platform on which we are running.
        curr_platform = platform.platform()

        # Get the name of the processor.
        curr_processor = platform.processor()

        # Get the number of threads used for running.
        num_threads = torch.get_num_threads()

        # Add a column defining the platform to the data frame.
        df_time.insert(loc = 0,
                       column = "platform",
                       value = curr_platform)

        # Add a column defining the processor to the data frame.
        df_time.insert(loc = 1,
                       column = "processor",
                       value = curr_processor)

        # Add a column defining the number of threads that were used
        # to the data frame.
        df_time.insert(loc = 2,
                       column = "num_threads",
                       value = num_threads)

        # Return the data frame.
        return df_time


    def _get_final_dataframes_rep(self,
                                  rep,
                                  pred_means,
                                  pred_r_values,
                                  time_opt,
                                  samples_names,
                                  genes_names):
        """Get the final data frames containing the representations,
        the decoder outputs corresponding to the representations,
        and the time needed for the optimizations.

        Parameters
        ----------
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

        pred_r_values : :class:`torch.Tensor` or ``None``
            A tensor containing the predicted r-values of the negative
            binomial distributions modelling the genes' counts, if
            the counts are modelled by negative binomial distributions.

            This is a 2D tensor where:

            - The first dimension has a length equal to the number
              of samples.

            - The second dimension has a length equal to the
              dimensionality of the gene space.

            ``pred_r_values`` is ``None`` if the counts are modelled
            by Poisson distributions.

        time_opt : :class:`list`
            A list of tuples storing, for each epoch, information
            about the CPU and wall clock time used by the entire
            epoch and by the backpropagation step run within the
            epoch.

        samples_names : :class:`list`
            A list containing the samples' names.

        genes_names : :class:`list`
            A list containing the genes' names.
        
        Returns
        -------
        df_rep : ``pandas.DataFrame``
            A data frame containing the representations.

        df_pred_means : ``pandas.DataFrame``
            A data frame containing the predicted means of the
            distributions modelling the genes' counts.

            If the genes counts are modelled using negative binomial
            distributions, the predicted means are scaled by the
            corresponding distributions' r-values.

        df_pred_r_values : ``pandas.DataFrame`` or ``None``
            A data frame containing the predicted r-values of the
            negative binomials. It is ``None`` if the genes' counts
            are modelled by Poisson distributions.

        df_time_opt : ``pandas.DataFrame``
            A data frame containing data about the optimization time.
        """

        #-------------------------------------------------------------#

        # Convert the tensor containing the predicted scaled means
        # into an array.
        pred_means_array = pred_means.detach().cpu().numpy()

        # Get a data frame containing the predicted scaled means for
        # all samples.
        df_pred_means = pd.DataFrame(pred_means_array.tolist())

        # Set the names of the rows of the data frame to be the
        # names/IDs/indexes of the samples.
        df_pred_means.index = samples_names

        # Set the names of the columns of the data frame to be the
        # names of the genes.
        df_pred_means.columns = genes_names

        #-------------------------------------------------------------#

        # If the predicted r-values were passed
        if pred_r_values is not None:
            
            # Convert the tensor containing the predicted r-values into
            # an array.
            pred_r_values_array = pred_r_values.detach().cpu().numpy()

            # If the array is one-dimensional (one r-value per gene for
            # all samples)
            if len(pred_r_values_array.shape) == 1:

                # Convert it into a two-dimensional array by repeating
                # the r-values for as many samples we have.
                pred_r_values_array = np.tile(pred_r_values_array,
                                              (len(samples_names), 1))

            # Get a data frame containing the predicted r-values for
            # all samples.
            df_pred_r_values = \
                pd.DataFrame(pred_r_values_array.tolist())

            # Set the names of the rows of the data frame to be the
            # names/IDs/indexes of the samples.
            df_pred_r_values.index = samples_names

            # Set the names of the columns of the data frame to be the
            # names of the genes.
            df_pred_r_values.columns = genes_names

        # Otherwise
        else:

            # The data frame containing the r-values will be None.
            df_pred_r_values = None

        #-------------------------------------------------------------#

        # Convert the tensor containing the representations into a
        # list.
        rep_list = rep.detach().cpu().numpy().tolist()

        # Create a data frame for the representations.
        df_rep = pd.DataFrame(rep_list)

        # Set the names of the rows of the data frame to be the
        # names/IDs/indexes of the samples.
        df_rep.index = samples_names

        # Name the columns of the data frame as the dimensions of the
        # latent space.
        df_rep.columns = \
            [f"latent_dim_{i}" for i \
             in range(1, df_rep.shape[1]+1)]

        #-------------------------------------------------------------#

        # Get the data frame containing the information about computing
        # time.
        df_time = self._get_time_dataframe(time_list = time_opt)

        # Name and sort the columns.
        df_time.columns = \
            ["platform", "processor", "num_threads",
             "opt_round", "epoch",
             "time_tot_epoch_cpu", "time_tot_bw_cpu",
             "time_tot_epoch_wall", "time_tot_bw_wall"]

        #-------------------------------------------------------------#

        # Return the data frames.
        return df_rep, df_pred_means, df_pred_r_values, df_time


    def _get_final_dataframes_train(self,
                                    rep_train,
                                    rep_test,
                                    losses_list,
                                    time_train,
                                    samples_names_train,
                                    samples_names_test,
                                    df_other_data_train,
                                    df_other_data_test):
        """Get the final data frames containing the losses calculated
        during training and the time needed to train the model.

        Parameters
        ----------
        rep_train : :class:`torch.Tensor`
            A tensor containing the optimized representations for the
            training samples.

            This is a 2D tensor where:

            * The first dimension has a length equal to the number
              of samples.

            * The second dimension has a length equal to the
              dimensionality of the latent space, where the
              representations live.

        rep_test : :class:`torch.Tensor`
            A tensor containing the optimized representations for the
            testing samples.

            This is a 2D tensor where:

            * The first dimension has a length equal to the number
              of samples.

            * The second dimension has a length equal to the
              dimensionality of the latent space, where the
              representations live.

        losses_list : :class:`list`
            A list containing the losses calculated during each
            training epoch.
        
        time_train : :class:`list`
            A list of tuples storing, for each epoch, information
            about the CPU and wall clock time used by the entire
            epoch and by the backpropagation step run within the
            epoch.

        samples_names_train : :class:`list`
            A list containing the training samples' names.

        samples_names_test : :class:`list`
            A list containing the testing samples' names.
        
        df_other_data_train : :class:`pandas.DataFrame`
            A data frame containing the additional data about the
            training samples.
        
        df_other_data_test : :class:`pandas.DataFrame`
            A data frame containing the additional data about the
            test samples.

        Returns
        -------
        df_rep_train : ``pandas.DataFrame``
            A data frame containing the optimized representations
            for the training samples.

        df_rep_test : ``pandas.DataFrame``
            A data frame containing the optimized representations
            for the testing samples.

        df_train : ``pandas.DataFrame``
            A data frame containing the losses for the training
            (such as loss for each model's components per epoch,
            overall loss per epoch, etc.).

        df_time : ``pandas.DataFrame``
            A data frame containing data about the CPU and wall
            clock time used by each training epoch (and backpropagation
            step within each epoch).

            Here, each row represents a training epoch, and the columns
            contain data about the platform where the calculation was
            run, the number of CPU threads used by the computation,
            and the CPU and wall clock time used by the entire epoch
            and by the backpropagation step run inside it.
        """

        # Convert the tensor containing the representations for the
        # training samples into a list.
        rep_list_train = rep_train.detach().cpu().numpy().tolist()

        # Create a data frame for the representations.
        df_rep_train = pd.DataFrame(rep_list_train)

        # Set the names of the rows of the data frame to be the
        # names/IDs/indexes of the samples.
        df_rep_train.index = samples_names_train

        # Name the columns of the data frame as the dimensions of the
        # latent space.
        df_rep_train.columns = \
            [f"latent_dim_{i}" for i \
             in range(1, df_rep_train.shape[1]+1)]

        # Concatenate the data frame with the one containing additional
        # information about the samples.
        df_rep_train = pd.concat([df_rep_train, df_other_data_train],
                                 axis = 1)

        #-------------------------------------------------------------#

        # Convert the tensor containing the representations for the
        # testing samples into a list.
        rep_list_test = rep_test.detach().cpu().numpy().tolist()

        # Create a data frame for the representations.
        df_rep_test = pd.DataFrame(rep_list_test)

        # Set the names of the rows of the data frame to be the
        # names/IDs/indexes of the samples.
        df_rep_test.index = samples_names_test

        # Name the columns of the data frame as the dimensions of the
        # latent space.
        df_rep_test.columns = \
            [f"latent_dim_{i}" for i \
             in range(1, df_rep_test.shape[1]+1)]

        # Concatenate the data frame with the one containing additional
        # information about the samples.
        df_rep_test = pd.concat([df_rep_test, df_other_data_test],
                                axis = 1)

        #-------------------------------------------------------------#

        # Create a data frame storing the training losses.
        df_loss = pd.DataFrame(losses_list)

        # Set the data frame's columns.
        df_loss.columns = \
            ["epoch", 
             "-log p_dens(z_train)",
             "-log p_dens(x_train|z_train)",
             "loss_train",
             "-log p_dens(z_test)",
             "-log p_dens(x_test|z_test)",
             "loss_test"]

        #-------------------------------------------------------------#

        # Get the data frame containing the information about the
        # computing time.
        df_time = self._get_time_dataframe(time_list = time_train)

        #-------------------------------------------------------------#

        # Return the data frames.
        return df_rep_train, df_rep_test, df_loss, df_time


    def _normalize_loss(self,
                        loss,
                        loss_type,
                        loss_norm_method,
                        loss_norm_options):
        """Normalize the loss per epoch.

        Parameters
        ----------
        loss : :class:`torch.Tensor`
            The loss to be normalized.

        loss_type : :class:`str`, {``"gmm"``, ``"recon"``, ``"total"``}
            The type of loss.

        loss_norm_method : :class:`str`
            The name of the normalization method.

        loss_norm_options : :class:`dict`
            A dictionary of options for the normalization method.
        """

        # If the normalization method is not supported
        if loss_norm_method not in self._LOSS_NORM_METHODS[loss_type]:

            # Get the available normalization methods for the current
            # type of loss.
            supported_methods = \
                ", ".join([f"'{m}'" for m \
                           in self._LOSS_NORM_METHODS[loss_type]])

            # Raise an error.
            errstr = \
                "Unsupported normalization method " \
                f"'{loss_norm_method}' for the '{loss_type}' loss. " \
                "Available normalization methods are: " \
                f"{supported_methods}."
            raise ValueError(errstr)

        #-------------------------------------------------------------#

        # If the normalization method is 'none'
        if loss_norm_method == "none":

            # Simply return the loss.
            return loss

        # If the loss should be normalized by the total number of
        # samples
        elif loss_norm_method == "n_samples":

            # Return the loss normalized by the total number of
            # samples.
            return loss / loss_norm_options["n_samples"]


    ######################### PUBLIC METHODS #########################


    @staticmethod
    def rescale_pred_means(df_pred_means,
                           df_pred_r_values):
        """Rescale the means of the negative binomials modeling
        the genes' counts.

        Parameters
        ----------
        df_pred_means : ``pandas.DataFrame``
            A data frame containing the predicted scaled means of
            the negative binomials modeling the genes' counts.

            Here, each row contains the scaled mean for a given
            representation/sample, and the columns contain either the
            values of the scaled means or additional information.

            The columns containing the scaled means must be
            named after the corresponding genes' Ensembl IDs.

        df_pred_r_values : ``pandas.DataFrame``
            A data frame containing the predicted r-values of
            the negative binomials modeling the genes' counts.

            Here, each row contains the r-value for a given
            representation/sample, and the columns contain either the
            r-values or additional information.

            The columns containing the r-values must be
            named after the corresponding genes' Ensembl IDs.
        
        Returns
        -------
        df_scaled : ``pandas.DataFrame``
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
            errstr = \
                "The names of the rows of the 'df_pred_means' and " \
                "'df_pred_r_values' data frames must be identical."
            raise ValueError(errstr)

        #-------------------------------------------------------------#

        # Get whether the columns' names of the two input data frames
        # are identical.
        columns_equal = \
            (df_pred_means.columns == df_pred_r_values.columns).all()

        # If they are not identical
        if not columns_equal:

            # Raise an error.
            errstr = \
                "The names of the columns of the 'df_pred_means' " \
                "and 'df_pred_r_values' data frames must be identical."
            raise ValueError(errstr)

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
                            df_samples,
                            config_rep):
        """Find the best representations for a set of samples.

        Parameters
        ----------
        df_samples : ``pandas.DataFrame``
            A data frame containing the samples.

        config_rep : :class:`dict`
            A dictionary of options for the optimization(s). It varies
            according to the selected ``method``.

            The supported options for all available methods can be
            found :doc:`here <../rep_config_options>`.

        Returns
        -------
        df_rep : ``pandas.DataFrame``
            A data frame containing the representations.

            Here, each row contains a representation and the
            columns contain either the values of the representations'
            along the latent space's dimensions or additional
            information about the input samples found in the
            input data frame. Columns containing additional
            information, if present in the input data frame, will
            appear last in the data frame.

        df_pred_means : ``pandas.DataFrame``
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

        df_pred_r_values : ``pandas.DataFrame`` or ``None``
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

            ``df_pred_r_values`` is ``None`` if the genes' counts are
            modelled by Poisson distributions.

        df_time : ``pandas.DataFrame``
            A data frame containing data about the CPU and wall
            clock time used by each epoch (and backpropagation
            step within each epoch) in each optimization step.

            Here, each row represents an epoch of an optimization
            step, and the columns contain data about the platform
            where the calculation was run, the number of CPU threads
            used by the computation, and the CPU and wall clock
            time used by the entire epoch and by the backpropagation
            step run inside it.
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

        # Get the number of samples and genes in the dataset from the
        # input data frame's shape.
        n_samples, n_genes = df_expr_data.shape

        #-------------------------------------------------------------#

        # Create the dataset.
        dataset = dataclasses.GeneExpressionDataset(df = df_expr_data)

        #-------------------------------------------------------------#

        # Get the names/IDs/indexes of the samples from the data
        # frame's rows' names.
        samples_names = df_expr_data.index

        # Get the names of the genes from the expression data frame's
        # columns' names.
        genes_names = df_expr_data.columns

        #-------------------------------------------------------------#

        # Check the configuration of the model.
        config_rep = util.check_config_rep(config = config_rep)

        #-------------------------------------------------------------#

        # Get the optimization scheme from the configuration.
        opt_scheme = config_rep["scheme"]

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
            self._get_final_dataframes_rep(\
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

        # Return the data frames.
        return df_rep, df_pred_means, df_pred_r_values, df_time


    def get_probability_density(self,
                                df_rep):
        """Given a set of representations, get the probability density
        of each component of the Gaussian mixture model for each
        representation and the representation(s) having the maximum
        probability density for each component.

        Parameters
        ----------
        df_rep : ``pandas.DataFrame``
            A data frame containing the representations.

        Returns
        -------
        df_prob_rep : ``pandas.DataFrame``
            A data frame containing the probability densities for each
            representation, together with an indication of what the
            maximum probability density found is and for which
            component it is found.

        df_prob_comp : ``pandas.DataFrame``
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
        
        # Get the probability densities of the representations for each
        # component.
        probs_values = \
            self.gmm.sample_probs(x = torch.Tensor(df_rep_data.values))

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

            # If more than one representation/sample has the highest
            # probability density
            if len(max_for_comp) > 1:

                # Warn the user.
                max_for_comp_rep = \
                    [str(i) for i in max_for_comp.index.tolist()]
                warnstr = \
                    "Multiple representations have the highest " \
                    f"probability density for component {comp} " \
                    f"({sub_df[MAX_PROB_COL].max()}): " \
                    f"{', '.join(max_for_comp_rep)}."
                logger.warning(warnstr)
            
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
              df_train,
              df_test,
              config_train,
              gmm_pth_file = "gmm.pth",
              dec_pth_file = "dec.pth",
              rep_pth_file = "rep.pth"):
        """Train the model.

        Parameters
        ----------
        df_train : ``pandas.DataFrame``
            A data frame containing the training data.

            Each row should contain a unique sample, and each
            column should either contain a gene's expression for that
            sample (if the column is named after the gene's Ensembl
            ID) or additional information about the sample.

        df_test : ``pandas.DataFrame``
            A data frame containing the testing data.

            Each row should contain a unique sample, and each
            column should either contain a gene's expression for that
            sample (if the column is named after the gene's Ensembl
            ID) or additional information about the sample.

        config_train : :class:`dict`
            A dictionary of options for the training.

        gmm_pth_file : :class:`str`, ``"gmm.pth"``
            The .pth file where to save the GMM's trained parameters
            (means of the components, weights of the components,
            and log-variance of the components).

        dec_pth_file : :class:`str`, ``"gmm.pth"``
            The .pth file where to save the decoder's trained
            parameters (weights and biases).

        rep_pth_file : :class:`str`, ``"rep.pth"``
            The .pth file where to save the representations found
            for the training samples. 

        Returns
        -------
        df_loss : ``pandas.DataFrame``
            A data frame containing the losses for the training
            (such as loss for each model's components per epoch,
            overall loss per epoch, etc.).

        df_time : ``pandas.DataFrame``
            A data frame containing data about the CPU and wall
            clock time used by each training epoch (and backpropagation
            step within each epoch).

            Here, each row represents a training epoch, and the columns
            contain data about the platform where the calculation was
            run, the number of CPU threads used by the computation,
            and the CPU and wall clock time used by the entire epoch
            and by the backpropagation step run inside it.
        """

        # Get the columns containing gene expression data for the
        # training data.
        genes_columns_train = \
            [col for col in df_train.columns \
             if col.startswith("ENSG")]

        # Get the columns containing gene expression data for the
        # testing data.
        genes_columns_test = \
            [col for col in df_test.columns \
             if col.startswith("ENSG")]

        #-------------------------------------------------------------#

        # If the genes are not the same
        if genes_columns_train != genes_columns_test:

            # Raise an error.
            errstr = \
                "Training and testing data must contain the " \
                "same genes in the same order."
            raise ValueError(errstr)

        #-------------------------------------------------------------#

        # Get a data frame with the training data and only the columns
        # containing gene expression data.
        df_expr_data_train = df_train[genes_columns_train]

        # Get the other columns for the training data.
        other_columns_train = \
            [col for col in df_train.columns \
             if col not in genes_columns_train]

        # Get a data frame with the training data and only the columns
        # containing additional information.
        df_other_data_train = df_train[other_columns_train]

        # Get the number of samples and genes in the data frame
        # containing the training data.
        n_samples_train, n_genes = df_expr_data_train.shape

        # Get the training samples' names.
        samples_names_train = df_expr_data_train.index

        # Create the dataset with the training samples.
        dataset_train = \
            dataclasses.GeneExpressionDataset(df = df_expr_data_train)

        # Create the data loader with the training samples.
        data_loader_train = \
            self._get_data_loader(\
                dataset = dataset_train,
                data_loader_config = config_train["data_loader"])

        # Create the representation layer for the training samples.
        rep_layer_train = \
            latent.RepresentationLayer(values = \
                torch.zeros(\
                    size = (n_samples_train, self.gmm.dim))).to(\
                        self.device)

        #-------------------------------------------------------------#

        # Get a data frame with the testing data and only the columns
        # containing gene expression data.
        df_expr_data_test = df_test[genes_columns_test]

        # Get the other columns for the testing data.
        other_columns_test = \
            [col for col in df_test.columns \
             if col not in genes_columns_test]

        # Get a data frame with the testing data and only the columns
        # containing additional information.
        df_other_data_test = df_test[other_columns_test]

        # Get the number of test samples.
        n_samples_test = df_expr_data_test.shape[0]

        # Get the testing samples' names.
        samples_names_test = df_expr_data_test.index

        # Create the dataset with the test samples.
        dataset_test = \
            dataclasses.GeneExpressionDataset(df = df_expr_data_test)

        # Create the data loader with the testing samples.
        data_loader_test = \
            self._get_data_loader(\
                dataset = dataset_test,
                data_loader_config = config_train["data_loader"])

        # Create the representation layer for the testing samples.
        rep_layer_test = \
            latent.RepresentationLayer(values = \
                torch.zeros(\
                    size = (n_samples_test, self.gmm.dim))).to(\
                        self.device)

        #-------------------------------------------------------------#

        # Create an empty list to store the loss for the
        # Gaussian mixture model, the reconstruction loss, and the
        # overall loss.
        losses_list = []

        # Create an empty list to store the training time.
        time_train = []

        #-------------------------------------------------------------#

        # Check the configuration that will be used for training.
        config_train = util.check_config_train(config = config_train)

        #-------------------------------------------------------------#

        # Get the number of epochs.
        n_epochs = config_train["epochs"]

        # Get the methods that will be used to normalize the losses.
        loss_norm_methods = \
            {item : config_train["loss"].get(item).get("norm_method") \
             for item in ["gmm", "recon", "total"]}
        
        # Get the optimizer for the Gaussian mixture model.
        optimizer_gmm = \
            self._get_optimizer(\
                optimizer_config = config_train["gmm"]["optimizer"],
                optimizer_parameters = self.gmm.parameters())

        # Get the optimizer for the decoder.
        optimizer_dec = \
            self._get_optimizer(\
                optimizer_config = config_train["dec"]["optimizer"],
                optimizer_parameters = self.dec.parameters())

        # Get the optimizer for the representations for the training
        # samples.
        optimizer_rep_train = \
            self._get_optimizer(\
                optimizer_config = config_train["rep"]["optimizer"],
                optimizer_parameters = rep_layer_train.parameters())

        # Get the optimizer for the representations for the testing
        # samples.
        optimizer_rep_test = \
            self._get_optimizer(\
                optimizer_config = config_train["rep"]["optimizer"],
                optimizer_parameters = rep_layer_test.parameters())

        #-------------------------------------------------------------#

        # For each epoch
        for epoch in range(1, n_epochs+1):

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

            # Make the gradients of the representation layer for the
            # training samples zero.
            optimizer_rep_train.zero_grad()

            # Make the gradients of the representation layer for the
            # testing samples zero.
            optimizer_rep_test.zero_grad()

            # Initialize the losses to 0 for the current epoch.
            losses_list.append(\
                [epoch, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

            #---------------------------------------------------------#

            # Set a list containing the data to be used for the
            # training loop for training and testing data,
            # respectively.
            train_loop_data = \
                [# Training data
                 (data_loader_train, rep_layer_train,
                  optimizer_rep_train, (1, 2, 3), n_samples_train),
                 # Testing data
                 (data_loader_test, rep_layer_test,
                  optimizer_rep_test, (4, 5, 6), n_samples_test)]

            #---------------------------------------------------------#

            # For each set of data (training and testing)
            for data_loader, rep_layer, optimizer_rep, \
                losses_ixs, n_samples \
                    in train_loop_data:

                #-----------------------------------------------------#

                # For each batch of samples, the mean gene expression
                # in the samples in the batch, and the unique indexes
                # of the samples in the batch
                for samples_exp, samples_mean_exp, samples_ixs \
                    in data_loader:

                    # Move the gene expression of the samples to the
                    # correct device.
                    samples_exp = samples_exp.to(self.device)

                    # Move the mean gene expression of the samples to
                    # the correct device.
                    samples_mean_exp = samples_mean_exp.to(self.device)

                    #-------------------------------------------------#

                    # Make the gradients for the Gaussian mixture model
                    # zero.
                    optimizer_gmm.zero_grad()

                    # Make the gradients for the decoder zero.
                    optimizer_dec.zero_grad()

                    #-------------------------------------------------#

                    # Get the representations for the current samples.
                    z = rep_layer(ixs = samples_ixs)

                    #-------------------------------------------------#

                    # If the chosen output module means that the
                    # r-values are not learned
                    if isinstance(\
                        self.dec.nb,
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
                        #                  'n_comp' *
                        #                  'n_rep_per_comp'
                        #
                        # - 2nd dimension: the dimensionality of the
                        #                  output (= gene) space ->
                        #                  'n_genes'
                        pred_means = self.dec(z = z)

                    # If the chosen output module means that the
                    # r-values are learned
                    elif isinstance(\
                        self.dec.nb,
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
                        #                  'n_comp' *
                        #                  'n_rep_per_comp'
                        #
                        # - 2nd dimension: the dimensionality of the
                        #                  output (= gene) space ->
                        #                  'n_genes'
                        pred_means, pred_log_r_values = self.dec(z = z)

                    #-------------------------------------------------#

                    # Get the Gaussian mixture model's loss.
                    gmm_loss = self.gmm(x = z).sum()

                    #-------------------------------------------------#

                    # If the chosen output module means that the
                    # r-values are not learned
                    if isinstance(\
                        self.dec.nb,
                        (outputmodules.OutputModuleNBFeatureDispersion,
                         outputmodules.OutputModulePoisson)):

                        # Set the options to compute the reconstruction
                        # loss.
                        recon_loss_options = \
                            {"obs_counts" : samples_exp,
                             "pred_means" : pred_means,
                             "scaling_factors" : samples_mean_exp}

                    # If the chosen output module means that the
                    # r-values are learned
                    elif isinstance(\
                        self.dec.nb,
                        outputmodules.OutputModuleNBFullDispersion):

                        # Set the options to compute the reconstruction
                        # loss.
                        recon_loss_options = \
                            {"obs_counts" : samples_exp,
                             "pred_means" : pred_means,
                             "pred_log_r_values" : pred_log_r_values,
                             "scaling_factors" : samples_mean_exp}

                    # Get the reconstruction loss.
                    recon_loss = \
                        self.dec.nb.loss(**recon_loss_options).sum()

                    #-------------------------------------------------#

                    # Get the overall loss.
                    loss = gmm_loss.clone() + recon_loss.clone()

                    #-------------------------------------------------#

                    # Mark the CPU start time of the backward step.
                    time_start_bw_cpu = time.process_time()

                    # Mark the wall clock start time of the backward
                    # step.
                    time_start_bw_wall = time.time()

                    # Backpropagate the loss.
                    loss.backward()

                    # Mark the end CPU time of the backward step.
                    time_end_bw_cpu = time.process_time()

                    # Mark the wall clock end time of the backward
                    # step.
                    time_end_bw_wall = time.time()

                    # Update the total CPU time used by the backward
                    # step.
                    time_tot_bw_cpu += \
                        time_end_bw_cpu - time_start_bw_cpu

                    # Update the total wall clock time used by the
                    # backward step.
                    time_tot_bw_wall += \
                        time_end_bw_wall - time_start_bw_wall

                    #-------------------------------------------------#

                    # Take a step with the optimizer for the Gaussian
                    # mixture model.
                    optimizer_gmm.step()

                    # Take a step with the optimizer for the decoder.
                    optimizer_dec.step()

                    #-------------------------------------------------#

                    # Get the loss for the Gaussian mixture model for
                    # the current epoch.
                    gmm_loss_epoch = \
                        self._normalize_loss(\
                            loss = gmm_loss.item(),
                            loss_type = "gmm",
                            loss_norm_method = \
                                loss_norm_methods["gmm"],
                            loss_norm_options = \
                                {"n_samples" : n_samples})

                    # Get the reconstruction loss for the current
                    # epoch.
                    recon_loss_epoch = \
                        self._normalize_loss(\
                            loss = recon_loss.item(),
                            loss_type = "recon",
                            loss_norm_method = \
                                loss_norm_methods["recon"],
                            loss_norm_options = \
                                {"n_samples" : n_samples})

                    # Get the overall loss for the current epoch.
                    loss_epoch = \
                        self._normalize_loss(\
                            loss = loss.item(),
                            loss_type = "total",
                            loss_norm_method = \
                                loss_norm_methods["total"],
                            loss_norm_options = \
                                {"n_samples" : n_samples})

                    #-------------------------------------------------#

                    # Update the list of losses.
                    losses_list[-1][losses_ixs[0]] += gmm_loss_epoch
                    losses_list[-1][losses_ixs[1]] += recon_loss_epoch
                    losses_list[-1][losses_ixs[2]] += loss_epoch

                #-----------------------------------------------------#

                # Take a step with the optimizer for the
                # representations.
                optimizer_rep.step()

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
            time_train.append(\
                (epoch,
                 time_tot_epoch_cpu, time_tot_bw_cpu,
                 time_tot_epoch_wall, time_tot_bw_wall))

            # Inform the user about the loss at the current epoch
            # and the CPU time/wall clock time elapsed.
            infostr = \
                f"Epoch {epoch}: loss train " \
                f"{losses_list[-1][3]:.3f}, loss test " \
                f"{losses_list[-1][6]:.3f}, epoch total CPU time " \
                f"{time_tot_epoch_cpu:.3f} s, backward step(s) " \
                f"total CPU time {time_tot_bw_cpu:.3f} s, epoch " \
                f"total wall clock time {time_tot_epoch_wall:.3f} " \
                "s, backward step(s) total wall clock time " \
                f"{time_tot_bw_wall:.3f} s."
            logger.info(infostr)

        #-------------------------------------------------------------#

        # Create the final data frames.
        df_rep_train, df_rep_test, df_loss, df_time = \
            self._get_final_dataframes_train(\
                rep_train = rep_layer_train(),
                rep_test = rep_layer_test(),
                losses_list = losses_list,
                time_train = time_train,
                samples_names_train = samples_names_train,
                samples_names_test = samples_names_test,
                df_other_data_train = df_other_data_train,
                df_other_data_test = df_other_data_test)

        #-------------------------------------------------------------#

        # Save the GMM's parameters.
        torch.save(self.gmm.state_dict(),
                   util.uniquify_file_path(gmm_pth_file))

        # Inform the user that the parameters were saved.
        infostr = \
            "The trained Gaussian mixture model's parameters were " \
            f"successfully saved in '{gmm_pth_file}'."
        logger.info(infostr)

        #-------------------------------------------------------------#

        # Save the decoder's parameters.
        torch.save(self.dec.state_dict(),
                   util.uniquify_file_path(dec_pth_file))

        # Inform the user that the parameters were saved.
        infostr = \
            "The trained decoder's parameters were successfully " \
            f"saved in '{dec_pth_file}'."
        logger.info(infostr)

        #-------------------------------------------------------------#

        # Return the data frames.
        return df_rep_train, df_rep_test, df_loss, df_time
