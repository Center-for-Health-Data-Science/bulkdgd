#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-

#    model.py
#
#    This module contains the class implementing the full DGD model
#    (:class:`core.model.DGDModel`).
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
    "This module contains the class implementing the full DGD model " \
    "(:class:`core.model.DGDModel`)."


#######################################################################


# Import from the standard library.
import logging as log
import os
import platform
import re
import time
# Import from third-party packages.
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
# Import from 'bulkDGD'.
from . import (
    dataclasses,
    decoder,
    latent,
    priors,
    )


#######################################################################


# Get the module's logger.
logger = log.getLogger(__name__)


#######################################################################


def uniquify_file_path(file_path):
    """If ``file_path`` exists, number it uniquely.

    Parameters
    ----------
    file_path : ``str``
        The file path.

    Returns
    -------
    unique_file_path : ``str``
        A unique file path generated from the original file path.
    """
    
    # Get the file's name and extension.
    file_name, file_ext = os.path.splitext(file_path)

    # Set the counter to 1.
    counter = 1

    # If the file already exists
    while os.path.exists(file_path):

        # Set the path to the new unique file.
        file_path = file_name + "_" + str(counter) + file_ext

        # Update the counter.
        counter += 1

    # Return the new path.
    return file_path


#######################################################################


class DGDModel(nn.Module):

    """
    Class implementing the full DGD model.
    """


    ######################## PUBLIC ATTRIBUTES ########################


    # Set the supported optimizers to find the representations.
    OPTIMIZERS = ["adam"]


    ######################### INITIALIZATION ##########################


    def __init__(self,
                 dim,
                 n_comp,
                 cm_type,
                 means_prior_name,
                 means_prior_options,
                 weights_prior_name,
                 weights_prior_options,
                 log_var_prior_name,
                 log_var_prior_options,
                 n_units_hidden_layers,
                 r_init,
                 activation_output,
                 genes_txt_file = None,
                 gmm_pth_file = None,
                 dec_pth_file = None):
        """Initialize an instance of the class.

        The model is initialized on the CPU. To move the model to
        another device, modify the ``device`` property.

        Parameters
        ----------
        dim : ``int``
            The dimensionality of the latent space, the Gaussian
            mixture model, and the first layer of the decoder..

        n_comp : ``int``
            The number of components in the Gaussian mixture model.

        cm_type : ``str``, {``"fixed"``, ``"isotropic"``, \
            ``"diagonal"``}
            The covariance matrix type for the Gaussian mixture model.
        
        means_prior_name : ``str``, {``"softball"``}
            The name of the prior distribution over the means of the
            components of the Gaussian mixture model.

            Currently, only the ``"softball"`` distribution is
            supported.

        means_prior_options : ``dict``
            A dictionary of options for setting up the
            prior distribution over the means of the components of
            the Gaussian mixture model.

            For the ``"softball"`` distribution, the following options
            must be provided:

            - ``"radius"`` (int): the radius of the multi-
              dimensional soft ball.

            - ``"sharpness"`` (int): the sharpness of the ball's
              soft boundary.

        weights_prior_name : ``str``, {``"dirichlet"``}
            The name of the prior distribution over the weights of
            the components of the Gaussian mixture model.

            Currently, only the ``"dirichlet"`` distribution is
            supported.

        weights_prior_oprions : ``dict``
            A dictionary of options for setting up the prior
            distribution over the weights of the components of the
            Gaussian mixture model.

            For the ``"dirichlet"`` distribution, the following options
            must be provided:

            - ``"alpha"`` (float): the alpha of the Dirichlet
              distribution.

        log_var_prior_name : ``str``, {``"gaussian"``}
            The name of the prior distribution over the log-variance
            of the Gaussian mixture model.

            Currently, only the ``"gaussian"`` prior is supported.

        log_var_prior_options : ``dict``
            A dictionary of options for setting up the prior
            distribution over the log-variance of the Gaussian mixture
            model.

            For the ``"gaussian"`` distirbution, the following options
            must be provided:

            - ``"mean"`` (float): the mean of the Gaussian
              distribution.

            - ``"stddev"`` (float): the standard deviation of the
              Gaussian distribution.

        n_units_hidden_layers : ``list``
            The number of units in each hidden layer of the decoder.

            The length of the list determines the number of hidden
            layers.

        r_init : ``int``
            The initial value for 'r' for the negative binomial
            distributions in the decoder's 
            :class:`core.decoder.NBLayer`.

        activation_output : ``str``, {``"sigmoid"``, ``"softplus"``}
            The name of the activation function for the decoder's
            output layer.

        genes_txt_file : ``str``
            A .txt file containing the Ensembl IDs of the genes
            included in the model.

            Training data will be checked to ensure counts are
            reported for all genes.

            The number of output units in the decoder is initialized
            from the number of genes found in this file.

        gmm_pth_file : ``str``, optional
            A .pth file with the GMM's trained parameters
            (means, weights, and log-variance of the components).

            Please ensure that the parameters match the Gaussian
            mixture model's structure.

            Omit it if the model needs training.

        dec_pth_file : ``str``, optional
            A .pth file containing the decoder's trained parameters
            (weights and biases).

            Please ensure that the parameters match the decoder's
            architecture.

            Omit it if the model needs training.
        """

        # Run the superclass' initialization.
        super(DGDModel, self).__init__()

        #-------------------------------------------------------------#

        # Get the genes included in the model.
        genes = \
            self.__class__._load_genes_list(\
                genes_list_file = genes_txt_file)

        #-------------------------------------------------------------#

        # Initialize the Gaussian mixture model.
        self._gmm = \
            latent.GaussianMixtureModel(\
                 dim = dim,
                 n_comp = n_comp,
                 cm_type = cm_type,
                 means_prior_name = means_prior_name,
                 means_prior_options = means_prior_options,
                 weights_prior_name = weights_prior_name,
                 weights_prior_options = weights_prior_options,
                 log_var_prior_name = log_var_prior_name,
                 log_var_prior_options = log_var_prior_options)

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

        # Get the decoder.
        self._dec = \
            decoder.Decoder(\
                n_units_input_layer = dim,
                n_units_hidden_layers = n_units_hidden_layers,
                n_units_output_layer = len(genes),
                r_init = r_init,
                activation_output = activation_output)

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

        # Get the r-values associated with the negative binomials
        # modeling the different genes.
        r_values = \
            torch.exp(self._dec.nb.log_r).squeeze().detach()

        # Associate the r-values with the genes.
        self._r_values = \
            pd.Series(r_values,
                      index = genes)

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

        pth_file : ``str``
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
        genes_list_file : ``str``
            The plain text file containing the genes of interest.

        Returns
        -------
        list_genes : ``list``
            The list of genes.
        """

        # Return the list of genes from the file (exclude blank
        # and comment lines).
        return \
            [l.rstrip("\n") for l in open(genes_list_file, "r") \
             if (not l.startswith("#") and not re.match(r"^\s*$", l))]


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
    def r_values(self):
        """The r-values of the negative binomials modeling the
        counts for the genes included in the model.
        """

        return self._r_values


    @r_values.setter
    def r_values(self,
                 value):
        """Raise an exception if the user tries to modify the value of
        ``r_values`` after initialization.
        """
        
        errstr = \
            "The value of 'r_values' is set at initialization and " \
            "cannot be changed. If you want to change the r-values " \
            "of the negative binomials modeling the counts for the " \
            "genes included in the model, initialize a new instance " \
            f"of '{self.__class__.__name__}'."
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
        optimizer_config : ``dict``
            The configuration for the optimizer.

        optimizer_parameters : ``torch.nn.Parameter``
            The parameters that will be optimized.

        Returns
        -------
        optimizer : ``torch.optim.Optimizer``
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
        dataset : ``bulkDGD.core.dataclasses.GeneExpressionDataset``
            The dataset from which the data loader should be created.

        data_loader_config : ``dict``
            The configuration for the data loader.

        Returns
        -------
        data_loader : ``torch.utils.data.DataLoader``
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
                      epochs,
                      opt_num):
        """Optimize the representation(s) found for each sample.

        Parameters
        ----------
        data_loader : ``torch.utils.data.DataLoader``
            The data loader.

        rep_layer : ``bulkDGD.core.latent.RepresentationLayer``
            The representation layer containing the initial
            representations.

        optimizer : ``torch.optim.Optimizer``
            The optimizer.

        n_comp : ``int``
            The number of components of the Gaussian mixture model
            for which at least one representation was drawn per
            sample.

        n_rep_per_comp : ``int``
            The number of new representations taken per sample
            per component of the Gaussian mixture model.

        epochs : ``int``
            The number of epochs to run the optimization for.

        opt_num : ``int``
            The number of the optimization round (especially useful
            if multiple rounds are run).

        Returns
        -------
        rep_opt : ``torch.Tensor``
            A tensor containing the optimized representations.

            This is a 2D tensor where:

            - The first dimension has a length equal to the number
              of samples.

            - The second dimension has a length equal to the
              dimensionality of the latent space where the
              representations live.

        dec_out_opt : ``torch.Tensor``
            A tensor containing the decoder's outputs for the
            optimized representations.

            This is a 2D tensor where:

            - The first dimension has a length equal to the number
              of samples.

            - The second dimension has a length equal to the
              dimensionality of the decoder output, which is also
              the dimensionality of the gene space.

        time_opt : ``list``
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
        n_genes = self.dec.main[-1].out_features

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

                # Get the outputs in gene space corresponding to the
                # representations found in latent space using the
                # decoder.
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
                # - 2nd dimension: the dimensionality of the output
                #                  (= gene) space ->
                #                  'n_genes'
                dec_out = self.dec(z = z)

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

                # Reshape the decoder's output to match the shape
                # required to compute the loss.
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
                pred_means = dec_out.view(n_samples_in_batch,
                                          n_rep_per_comp,
                                          n_comp,
                                          n_genes)

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
                recon_loss = \
                    self.dec.nb.loss(obs_counts = obs_counts,
                                     scaling_factors = scaling_factors,
                                     pred_means = pred_means)

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

                # Get the average loss for the current epoch.
                rep_avg_loss_epoch = \
                    total_loss.item() / \
                        (n_samples * n_genes * \
                         n_comp * n_rep_per_comp)

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

                # Get the decoder's outputs for the optimized
                # representations.
                dec_out_final = self.dec(z = rep_final)

                # Return the representations, the decoder's outputs,
                # and the time data.
                return rep_final, dec_out_final, time_opt


    def _select_best_rep(self,
                         data_loader,
                         rep_layer,
                         n_rep_per_comp):
        """Select the best representation per sample.

        Parameters
        ----------
        data_loader : ``torch.utils.data.DataLoader``
            The data loader.

        rep_layer : ``bulkDGD.core.latent.RepresentationLayer``
            The representation layer containing the representations
            found for the samples.

        n_rep_per_comp : ``int``
            The number of new representations that were taken per
            sample per component of the Gaussian mixture model.

        Returns
        -------
        rep_opt : ``torch.Tensor``
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
        n_genes = self.dec.main[-1].out_features

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

            # Get the outputs in gene space corresponding to the
            # representations found in latent space using the
            # decoder for the samples in the current batch.
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
            # - 2nd dimension: the dimensionality of the output
            #                  (= gene) space ->
            #                  'n_genes'
            dec_out = self.dec(z)

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

            # Reshape the decoded output to match the shape required to
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
            # - 3rd dimension: the number of components in the Gaussian
            #                  mixture model -> 'n_comp'
            #
            # - 4th dimension: the dimensionality of the output
            #                  (= gene) space -> 'n_genes'      
            pred_means = dec_out.view(n_samples_in_batch,
                                      n_rep_per_comp,
                                      n_comp,
                                      n_genes)
     
            # Get the reconstruction loss (rescale based on the mean
            # expression of the genes in the samples in the batch).
            #
            # The output is a 4D tensor with:
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
            #
            # - 4th dimension: the dimensionality of the output
            #                  (= gene) space ->
            #                  'n_genes'
            recon_loss = \
                self.dec.nb.loss(obs_counts = obs_counts,
                                 scaling_factors = scaling_factors,
                                 pred_means = pred_means)

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
        dataset : ``bulkDGD.core.dataclasses.GeneExpressionDataset``
            The dataset from which the data loader should be created.

        config : ``dict``
            A dictionary with the following keys:

            * ``"n_rep_per_comp", whose associated value should be
              the he number of new representations to be taken per
              component per sample.
            
            * ``"epochs"``, whose associated value should be the
              number of epochs the optimizations is run for.
            
            * ``"opt1"``, whose associated value should be a dictionary
              with the following keys:

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
        rep_opt : ``torch.Tensor``
            A tensor containing the optimized representations.

            This is a 2D tensor where:

            * The first dimension has a length equal to the number
              of samples.

            * The second dimension has a length equal to the
              dimensionality of the latent space, where the
              representations live.

        dec_out_opt : ``torch.Tensor``
            A tensor containing the decoder outputs corresponding
            to the optimized representations.

            This is a 2D tensor where:

            * The first dimension has a length equal to the number
              of samples.

            * The second dimension has a length equal to the
              dimensionality of the decoder output, which is also
              the dimensionality of the gene space.

        time_opt : ``list``
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

        # Get the optimized representations.
        rep, dec_out, time = \
            self._optimize_rep(\
                data_loader = data_loader,
                rep_layer = rep_layer_best,
                optimizer = optimizer,
                n_comp = 1,
                n_rep_per_comp = 1,
                epochs = epochs,
                opt_num = 1)

        #-------------------------------------------------------------#

        # Make the gradients zero.
        optimizer.zero_grad()

        #-------------------------------------------------------------#

        # Return the representations, the decoder's outputs, and the
        # time of the optimization.
        return rep, dec_out, time


    def _get_representations_two_opt(self,
                                     dataset,
                                     config):
        """Get the best representations for a set of samples by
        initializing ``n_rep_per_comp`` representations per each
        component of the Gaussian mixture model per sample, optimizing
        these representations, selecting the best representation for
        for each sample, and optimizing these representations furher.

        Parameters
        ----------
        dataset : ``bulkDGD.core.dataclasses.GeneExpressionDataset``
            The dataset from which the data loader should be created.
    
        config : ``dict``
            A dictionary with the following keys:

            * ``"n_rep_per_comp", whose associated value should be
              the he number of new representations to be taken per
              component per sample.
            
            * ``"epochs"``, whose associated value should be the
              number of epochs the optimizations is run for.
            
            * ``"opt"``, whose associated value should be a dictionary
              with the following keys:

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
        rep_opt : ``torch.Tensor``
            A tensor containing the optimized representations.

            This is a 2D tensor where:

            * The first dimension has a length equal to the number of
              samples.

            * The second dimension has a length equal to the
              dimensionality of the latent space, where the
              representations live.

        dec_out_opt : ``torch.Tensor``
            A tensor containing the decoder outputs corresponding
            to the optimized representations.

            This is a 2D tensor where:

            * The first dimension has a length equal to the number
              of samples.

            * The second dimension has a length equal to the
              dimensionality of the decoder output, which is also
              the dimensionality of the gene space.

        time_opt : ``list``
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

        # Get the optimized representations, the corresponding decoder
        # outputs, and the time information about the optimization's
        # epochs.
        rep_1, dec_out_1, time_1 = \
            self._optimize_rep(\
                data_loader = data_loader,
                rep_layer = rep_layer_init,
                optimizer = optimizer_1,
                n_comp = self.gmm.n_comp,
                n_rep_per_comp = n_rep_per_comp,
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

        # Get the optimized representations, the corresponding decoder
        # outputs, and the time information about the optimization's
        # epochs.
        rep_2, dec_out_2, time_2 = \
            self._optimize_rep(\
                data_loader = data_loader,
                rep_layer = rep_layer_best,
                optimizer = optimizer_2,
                n_comp = 1,
                n_rep_per_comp = n_rep_per_comp,
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

        # Return the representations, the decoder's outputs, and the
        # time information for both rounds.
        return rep_2, dec_out_2, time


    def _get_time_dataframe(self,
                            time_list):
        """Get the data frame containing the information about the
        computing time.

        Parameters
        ----------
        time_list : ``list``
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
                                  dec_out,
                                  time_opt,
                                  samples_names,
                                  genes_names):
        """Get the final data frames containing the representations,
        the decoder outputs corresponding to the representations,
        and the time needed for the optimizations.

        Parameters
        ----------
        rep : ``torch.Tensor``
            A tensor containing the optimized representations.

            This is a 2D tensor where:

            * The first dimension has a length equal to the number
              of samples.

            * The second dimension has a length equal to the
              dimensionality of the latent space, where the
              representations live.

        dec_out_opt : ``torch.Tensor``
            A tensor containing the decoder outputs corresponding
            to the optimized representations.

            This is a 2D tensor where:

            * The first dimension has a length equal to the number
              of samples.

            * The second dimension has a length equal to the
              dimensionality of the decoder output, which is also
              the dimensionality of the gene space.

        time_opt : ``list``
            A list of tuples storing, for each epoch, information
            about the CPU and wall clock time used by the entire
            epoch and by the backpropagation step run within the
            epoch.

        samples_names : ``list``
            A list containing the samples' names.

        genes_names : ``list``
            A list containing the genes' names.
        
        Returns
        -------
        df_rep : ``pandas.DataFrame``
            A data frame containing the representations.

        df_dec_out : ``pandas.DataFrame``
            A data frame containing the decoder outputs corresponding
            to the representations.

        df_time_opt : ``pandas.DataFrame``
            A data frame containing data about the optimization time.
        """

        # Convert the tensor containing the decoder outputs into a
        # list.
        dec_out_list = dec_out.detach().cpu().numpy().tolist()

        # Get a data frame containing the decoder outputs for all
        # samples.
        df_dec_out = pd.DataFrame(dec_out_list)

        # Set the names of the rows of the data frame to be the
        # names/IDs/indexes of the samples.
        df_dec_out.index = samples_names

        # Set the names of the columns of the data frame to be the
        # names of the genes.
        df_dec_out.columns = genes_names

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
        return df_rep, df_dec_out, df_time


    def _get_final_dataframes_train(self,
                                    rep_train,
                                    rep_test,
                                    losses_list,
                                    time_train,
                                    samples_names_train,
                                    samples_names_test):
        """Get the final data frames containing the losses calculated
        during training and the time needed to train the model.

        Parameters
        ----------
        rep_train : ``torch.Tensor``
            A tensor containing the optimized representations for the
            training samples.

            This is a 2D tensor where:

            * The first dimension has a length equal to the number
              of samples.

            * The second dimension has a length equal to the
              dimensionality of the latent space, where the
              representations live.

        rep_test : ``torch.Tensor``
            A tensor containing the optimized representations for the
            testing samples.

            This is a 2D tensor where:

            * The first dimension has a length equal to the number
              of samples.

            * The second dimension has a length equal to the
              dimensionality of the latent space, where the
              representations live.

        losses_list : ``list``
            A list containing the losses calculated during each
            training epoch.
        
        time_train : ``list``
            A list of tuples storing, for each epoch, information
            about the CPU and wall clock time used by the entire
            epoch and by the backpropagation step run within the
            epoch.

        samples_names_train : ``list``
            A list containing the training samples' names.

        samples_names_test : ``list``
            A list containing the testing samples' names.

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


    ######################### PUBLIC METHODS #########################


    def get_representations(self,
                            df_samples,
                            config_rep):
        """Find the best representations for a set of samples.

        Parameters
        ----------
        df_samples : ``pandas.DataFrame``
            A data frame containing the samples.

        config_rep : ``dict``
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

        df_dec_out : ``pandas.DataFrame``
            A data frame containing the decoder outputs
            corresponding to the representations found.

            Here, each row contains the decoder output for a given
            representation, and the columns contain either the
            values of the decoder outputs or additional information
            about the input samples found in the input
            data frame. Columns containing additional
            information, if present in the input data frame, will
            appear last in the data frame.

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
            
        # Get the representations, the corresponding decoder
        # outputs, and the time data.
        rep, dec_out, time_opt = \
            opt_method(dataset = dataset,
                       config = config_rep)

        #-------------------------------------------------------------#

        # Generate the final data frames.
        df_rep, df_dec_out, df_time = \
            self._get_final_dataframes_rep(\
                rep = rep,
                dec_out = dec_out,
                time_opt = time_opt,
                samples_names = samples_names,
                genes_names = genes_names)

        #-------------------------------------------------------------#

        # Add the extra data found in the input data frame to the
        # representations' data frame.
        df_rep = pd.concat([df_rep, df_other_data],
                           axis = 1)

        # Add the extra data found in the input data frame to the
        # decoder outputs' data frame.
        df_dec_out = pd.concat([df_dec_out, df_other_data],
                               axis = 1)

        #-------------------------------------------------------------#

        # Return the data frames.
        return df_rep, df_dec_out, df_time


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
        # for which the maximum proability density was found per
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
        df_rep_data, df_other_data = \
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
                    f"Multiple representations have the highest " \
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


    def rescale_decoder_outputs(self,
                                df):
        """Rescale the decoder outputs, which correspond to the
        means of the negative binomials modeling the expression
        of the genes included in the model.

        Parameters
        ----------
        df : ``pandas.DataFrame``
            A data frame containing the decoder outputs.

            Here, each row contains the decoder output for a given
            representation/sample, and the columns contain either the
            values of the decoder outputs or additional information.

            The columns containing the decoder outputs must be
            named after the corresponding genes' Ensembl IDs.
        
        Returns
        -------
        df_rescaled : ``pandas.DataFrame``
            A data frame containing the rescaled decoder outputs.

            It contains the same columns of the input data frame,
            in the same order they appear in the input data frame.

            However, the values in the columns containing the
            decoder outputs are rescaled.
        """

        # Get the r-values.
        r_values = self.r_values

        #-------------------------------------------------------------#

        # Get the names of the columns containing gene expression
        # data from the original data frame.
        genes_columns = \
            [col for col in df.columns if col.startswith("ENSG")]

        # Create a data frame with only those columns containing gene
        # expression data.
        df_dec_data = df.loc[:,genes_columns]

        # Rescale the decoder outputs.
        df_dec_data = df_dec_data * self.r_values

        #-------------------------------------------------------------#

        # Get the names of the other columns.
        other_columns = \
            [col for col in df.columns if col not in genes_columns]

        # Create a data frame with only those columns containing
        # additional information.
        df_other_data = df.loc[:,other_columns]

        #-------------------------------------------------------------#

        # Make a new data frame with the rescaled outputs.
        df_final = pd.concat([df_dec_data, df_other_data])

        # Re-order the columns in the original order.
        df_final = df_final[df.columns.tolist()]

        #-------------------------------------------------------------#

        # Return the new data frame
        return df_final


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

        config_train : ``dict``
            A dictionary of options for the training.

        gmm_pth_file : ``str``, ``"gmm.pth"``
            The .pth file where to save the GMM's trained parameters
            (means of the components, weights of the components,
            and log-variance of the components).

        dec_pth_file : ``str``, ``"gmm.pth"``
            The .pth file where to save the decoder's trained
            parameters (weights and biases).

        rep_pth_file : ``str``, ``"rep.pth"``
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

        # Get the number of epochs.
        n_epochs = config_train["epochs"]

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

                    # Get the number of samples in the current batch.
                    n_samples_in_batch = len(samples_ixs)

                    #-------------------------------------------------#

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

                    # Get the decoder's outputs for the representations.
                    dec_out = self.dec(z = z)

                    #-------------------------------------------------#

                    # Get the Gaussian mixture model's loss.
                    gmm_loss = self.gmm(x = z).sum()

                    # Get the reconstruction loss.
                    recon_loss = \
                        self.dec.nb.loss(\
                            obs_counts = samples_exp,
                            pred_means = dec_out,
                            scaling_factors = samples_mean_exp).sum()

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
                        gmm_loss.item() / \
                        (n_samples * self.gmm.dim * self.gmm.n_comp)

                    # Get the reconstruction loss for the current
                    # epoch.
                    recon_loss_epoch = \
                        recon_loss.item() / \
                        (n_samples * n_genes)

                    # Get the overall loss for the current epoch.
                    loss_epoch = \
                        loss.item() / \
                        (n_samples * n_genes)

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
                samples_names_test = samples_names_test)

        #-------------------------------------------------------------#

        # Save the GMM's parameters.
        torch.save(self.gmm.state_dict(),
                   uniquify_file_path(gmm_pth_file))

        # Inform the user that the parameters were saved.
        infostr = \
            "The trained Gaussian mixture model's parameters were " \
            f"successfully saved in '{gmm_pth_file}'."
        logger.info(infostr)

        #-------------------------------------------------------------#

        # Save the decoder's parameters.
        torch.save(self.dec.state_dict(),
                   uniquify_file_path(dec_pth_file))

        # Inform the user that the parameters were saved.
        infostr = \
            "The trained decoder's parameters were successfully " \
            f"saved in '{dec_pth_file}'."
        logger.info(infostr)

        #-------------------------------------------------------------#

        # Return the data frames.
        return df_rep_train, df_rep_test, df_loss, df_time

