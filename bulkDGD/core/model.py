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
import platform
import re
import time
# Import from third-party packages.
import pandas as pd
import torch
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


class DGDModel:

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
                 gmm_pth_file,
                 dec_pth_file,
                 genes_txt_file):
        """Initialize an instance of the class.

        Parameters
        ----------
        dim : ``int``
            The dimensionality of the latent space (and, therefore,
            the dimensionality of the Gaussian mixture model and
            of the first layer of the decoder).

        n_comp : ``int``
            The number of components of the Gaussian mixture model.

        cm_type : ``str``, {``"fixed"``, ``"isotropic"``, \
            ``"diagonal"``}
            The type of covariance matrix used by the Gaussian mixture
            model.
        
        means_prior_name : ``str``, {``"softball"``}
            The name of the prior distribution over the means of the
            components of the Gaussian mixture model.

            So far, only the ``"softball"`` prior is supported.

        means_prior_options : ``dict``
            A dictionary of options to set up the prior distribution
            over the means of the components of the Gaussian mixture
            model.

            If the prior is the ``"softball"`` distribution, the
            following options must be passed:

            * ``"radius"``, namely the radius of the multi-
              dimensional soft ball (``int``).

            * ``"sharpness"``, namely the sharpness of the ball's
              soft boundary (``int``).

        weights_prior_name : ``str``, {``"dirichlet"``}
            The name of the prior distribution over the weights of
            the components of the Gaussian mixture model.

            So far, only the ``"dirichlet"`` prior is supported.

        weights_prior_oprions : ``dict``
            A dictionary of options to set up the prior distribution
            over the weights of the components of the Gaussian
            mixture model.

            If the prior is the ``"dirichlet"`` distribution, the
            following options must be set:

            * ``"alpha"``, namely the alpha of the Dirichlet
              distribution (``float``).

        log_var_prior_name : ``str``, {``"gaussian"``}
            The name of the prior distribution over the log-variance
            of the Gaussian mixture model.

            So far, only the ``"gaussian"`` prior is supported.

        log_var_prior_options : ``dict``
            A dictionary of options to set up the prior distribution
            over the log-variance of the Gaussian mixture model.

            If the prior is the ``"gaussian"`` distirbution, the
            follow options must be set:

            * ``"mean"``, namely the mean of the Gaussian distribution.

            * ``"stddev"``, namely the standard deviation of the
              Gaussian distribution.

        n_units_hidden_layers : ``list``
            The number of units in each of the hidden layers. As manu
            hidden layers as the number of items in the list will be
            created.

        r_init : ``int``
            The initial value for 'r', representing the "number of
            failures" after which the "trials" stop in the negative
            binomial distributions of the ``NBLayer``.

        activation_output : ``str``, {``"sigmoid"``, ``"softplus"``}
            The name of the activation function to be used in the
            decoder's output layer.

        gmm_pth_file : ``str``, optional
            A .pth file containing the GMM's trained parameters
            (means of the components, weights of the components,
            and log-variance of the components).

        dec_pth_file : ``str``, optional
            A .pth file containing the decoder's trained parameters
            (weights and biases).

        genes_txt_file : ``str``, optional
            A .txt file containing the Ensembl IDs of the genes
            included in the model.
        """

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
        r_values = torch.exp(self._dec.nb.log_r).squeeze().detach()

        # Associate the r-values with the genes.
        self._r_values = \
            pd.Series(r_values,
                      index = genes)


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
            "Gaussian mixture model associated with the DGD model, " \
            "initialize a new instance of " \
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
            "cannot be changed. If you want to change the decoder " \
            "associated with the DGD model, initialize a new " \
            f"instance of '{self.__class__.__name__}'."
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
    

    ######################### PRIVATE METHODS #########################


    def _get_optimizer(self,
                       optimizer_name,
                       optimizer_options,
                       optimizer_parameters):
        """Get the optimizer.

        Parameters
        ----------
        optimizer_name : ``str``
            The name of the optimizer.

        optimizer_options : ``dict``
            A dictionary of options to set up the optimizer.

        optimizer_parameters : ``torch.nn.Parameter``
            The parameters that will be optimized by the optimizer.

        Returns
        -------
        optimizer : ``torch.optim.Optimizer``
            The optimizer.
        """

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
                                 **optimizer_options)

        #-------------------------------------------------------------#

        # Return the optimizer.
        return optimizer


    def _optimize_multiple_rep_per_sample(self,
                                          rep_layer,
                                          optimizer,
                                          n_samples,
                                          n_rep_per_comp,
                                          data_exp,
                                          mean_exp,
                                          epochs,
                                          opt_num):
        """Optimize multiple representations for each sample
        simultaneosly.

        Parameters
        ----------
        rep_layer : ``bulkDGD.core.latent.RepresentationLayer``
            The representation layer containing the initial
            representations.

        optimizer : ``torch.optim.Optimizer``
            The optimizer.

        n_samples : ``int``
            The number of samples.

        n_rep_per_comp : ``int``
            The number of new representations to be taken per sample
            per component of the Gaussian mixture model.

        data_exp : ``torch.Tensor``
            The gene expression (counts) for all samples in tensor
            form.

            This is a 2D tensor where:
            
            * The first dimension has a length equal to the number of
              samples.

            * The second dimension has a length equal to the number of
              genes.

        mean_exp : ``torch.Tensor``
            The mean gene expression (counts) for each sample in tensor
            form.

            This is a 1D tensor whose length is equal to the number of
            samples.

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

            * The first dimension has a length equal to the number
              of samples.

            * The second dimension has a length equal to the
              dimensionality of the latent space where the
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

        # Get the number of components of the Gaussian mixture model.
        n_comp = self.gmm.n_comp

        # Get the dimensionality of the latent space.
        dim = self.gmm.dim

        # Get the number of genes (= the dimensionality of the decoder
        # output).
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

            # Make the gradients zero.
            optimizer.zero_grad()

            # Get the representations' values from the representation
            # layer.
            # 
            # The representations are stored in a 2D tensor with:
            #
            # - 1st dimension: the number of samples in the current
            #                  batch times the number of components
            #                  in the Gaussian mixture model times the
            #                  number of representations taken per
            #                  component per sample ->
            #                  'n_samples' * 
            #                  'n_comp' *
            #                  'n_rep_per_comp'
            #
            # - 2nd dimension: the dimensionality of the Gaussian
            #                  mixture model ->
            #                  'dim'
            z_all = rep_layer.z

            #---------------------------------------------------------#

            # Reshape the tensor containing the representations.
            #
            # The output is a 4D tensor with:
            #
            # - 1st dimension: the number of samples -> 'n_samples'
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
                              dim)

            # Reshape the tensor again.
            #
            # The output is a 2D tensor with:
            #
            # - 1st dimension: the number of samples times the number
            #                  of components in the Gaussian mixture
            #                  model times the number of
            #                  representations taken per
            #                  component per sample ->
            #                  'n_samples' * 
            #                  'n_rep_per_comp' *
            #                  'n_comp'
            #
            # - 2nd dimension: the dimensionality of the Gaussian
            #                  mixture model ->
            #                  'dim'
            z = z_4d.view(n_samples * \
                            n_rep_per_comp * \
                            n_comp,
                          dim)

            #---------------------------------------------------------#

            # Get the outputs in gene space corresponding to the
            # representations found in latent space using the decoder.
            #
            # The output is a 2D tensor with:
            #
            # - 1st dimension: the number of samples times the number
            #                  of components in the Gaussian mixture
            #                  model times the number of
            #                  representations taken per
            #                  component per sample ->
            #                  'n_samples' * 
            #                  'n_rep_per_comp' *
            #                  'n_comp'
            #
            # - 2nd dimension: the dimensionality of the output
            #                  (= gene) space ->
            #                  'n_genes'
            dec_out = self.dec(z)

            #---------------------------------------------------------#

            # Get the observed gene expression and "expand" the
            # resulting tensor to match the shape required to compute
            # the reconstruction loss.
            #
            # The output is a 4D tensor with:
            #
            # - 1st dimension: the number of samples -> 'n_samples'
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
                data_exp.unsqueeze(1).unsqueeze(1).expand(\
                    -1,
                    n_rep_per_comp,
                    n_comp,
                    -1)

            # Get the scaling factors for the mean of each negative
            # binomial modelling the expression of a gene and reshape
            # it so that it matches the shape required to compute the
            # reconstruction loss.
            #
            # The output is a 4D tensor with:
            #
            # - 1st dimension: the number of samples -> 'n_samples'
            #
            # - 2nd dimension: 1
            #
            # - 3rd dimension: 1
            #
            # - 4th dimension: 1
            scaling_factors = \
                decoder.reshape_scaling_factors(mean_exp,
                                                4)

            # Reshape the decoded output to match the shape required
            # to compute the loss.
            #
            # The output is a 4D tensor with:   
            #
            # - 1st dimension: the number of samples -> 'n_samples'
            #
            # - 2nd dimension: the number of representations taken
            #                  per component per sample ->
            #                  'n_rep_per_comp'
            #
            # -3rd dimension: the number of components in the Gaussian
            #                 mixture model -> 'n_comp'
            #
            # - 4th dimension: the dimensionality of the output
            #                  (= gene) space -> 'n_genes'      
            pred_means = dec_out.view(n_samples,
                                      n_rep_per_comp,
                                      n_comp,
                                      n_genes)

            # Get the reconstruction loss.
            #
            # The output is a 4D tensor with:
            #
            # - 1st dimension: the number of samples -> 'n_samples'
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

            # Get the total reconstruction loss by summing all values
            # in the 'recon_loss' tensor.
            #
            # The output is a tensor containing a single value.
            recon_loss_sum = recon_loss.sum().clone()

            #---------------------------------------------------------#

            # Get the GMM error.
            #
            # 'gmm(z)' computes the negative log density of the
            # probability of the representations 'z' being drawn from
            # the Gaussian mixture model.
            #
            # The output is a 1D tensor with:
            #
            # - 1st dimension: the number of samples times the number
            #                  of components in the Gaussian mixture
            #                  model times the number of
            #                  representations taken per
            #                  component per sample ->
            #                  'n_samples' * 
            #                  'n_rep_per_comp' *
            #                  'n_comp'
            gmm_loss = self.gmm(z)

            # Get the total GMM loss by summing over all values in
            # the 'gmm_loss' tensor.
            # 
            # The output is a tensor containing a single value.
            gmm_loss_sum = gmm_loss.sum().clone()

            #---------------------------------------------------------#

            # Get the total loss.
            #
            # The output is a tensor containing a single value.
            total_loss = recon_loss_sum + gmm_loss_sum

            #---------------------------------------------------------#

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
            time_tot_bw_cpu = time_end_bw_cpu - time_start_bw_cpu

            # Get the total wall clock time used by the backward step.
            time_tot_bw_wall = time_end_bw_wall - time_start_bw_wall

            #---------------------------------------------------------#

            # Get the average loss for the current epoch.
            rep_avg_loss_epoch = \
                total_loss.item() / (n_samples * n_genes * \
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
                rep_final = rep_layer.z

                # Get the decoder outputs.
                dec_out_final = self.dec(rep_final)

                # Return the representations, the decoder outputs,
                # and the time data.
                return rep_final, dec_out_final, time_opt


    def _select_best_rep(self,
                         rep_layer,
                         n_samples,
                         n_rep_per_comp,
                         data_exp,
                         mean_exp):
        """Select the best representation per sample.

        Parameters
        ----------
        rep_layer : ``bulkDGD.core.latent.RepresentationLayer``
            The representation layer containing the initial
            representations.

        n_samples : ``int``
            The number of samples.

        n_rep_per_comp : ``int``
            The number of new representations to be taken
            per component per sample.

        data_exp : ``torch.Tensor``
            The gene expression (counts) for all samples in tensor
            form.

            This is a 2D tensor where:
            
            * The first dimension has a length equal to the number
              of samples.

            * The second dimension has a length equal to the
              number of genes.

        mean_exp : ``torch.Tensor``
            The mean gene expression (counts) for each sample in tensor
            form.

            This is a 1D tensor whose length is equal to the number of
            samples.

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
        """

        # Get the number of components of the Gaussian mixture model.
        n_comp = self.gmm.n_comp

        # Get the dimensionality of the latent space.
        dim = self.gmm.dim

        # Get the number of genes (= dimensionality of the decoder
        # output).
        n_genes = self.dec.main[-1].out_features

        # Get the representations' values from the representation
        # layer.
        # 
        # The representations are stored in a 2D tensor with:
        #
        # - 1st dimension: the number of samples times the number
        #                  of components in the Gaussian mixture
        #                  model times the number of
        #                  representations taken per
        #                  component per sample ->
        #                  'n_samples' * 
        #                  'n_rep_per_comp' *
        #                  'n_comp'
        #
        # - 2nd dimension: the dimensionality of the Gaussian
        #                  mixture model ->
        #                  'dim'
        z_all = rep_layer.z

        # Reshape the tensor containing the representations.
        #
        # The output is a 4D tensor with:
        #
        # - 1st dimension: the number of samples -> 'n_samples'
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
                       dim)

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
        #                  'n_rep_per_comp' *
        #                  'n_comp'
        #
        # - 2nd dimension: the dimensionality of the Gaussian mixture
        #                  model ->
        #                  'dim'
        z = z_4d.view(n_samples * \
                        n_rep_per_comp * \
                        n_comp,
                      dim)

        #-------------------------------------------------------------#

        # Get the outputs in gene space corresponding to the
        # representations found in latent space using the
        # decoder.
        #
        # The output is a 2D tensor with:
        #
        # - 1st dimension: the number of samples times the number
        #                  of components in the Gaussian mixture
        #                  model times the number of
        #                  representations taken per
        #                  component per sample ->
        #                  'n_samples' * 
        #                  'n_rep_per_comp' *
        #                  'n_comp'
        #
        # - 2nd dimension: the dimensionality of the output
        #                  (= gene) space ->
        #                  'n_genes'
        dec_out = self.dec(z)

        #-------------------------------------------------------------#

        # Get the observed counts for the expression of each gene in
        # each sample, and "expand" the resulting tensor to match the
        # shape required to compute the reconstruction loss.
        #
        # The output is a 4D tensor with:
        #
        # - 1st dimension: the number of samples -> 'n_samples'
        #
        # - 2nd dimension: the number of representations taken
        #                  per component per sample ->
        #                  'n_rep_per_comp'
        #
        # - 3rd dimension: the number of components in the Gaussian
        #                 mixture model -> 'n_comp'
        #
        # - 4th dimension: the dimensionality of the output
        #                  (= gene) space -> 'n_genes'
        obs_counts = \
            data_exp.unsqueeze(1).unsqueeze(1).expand(\
                -1,
                n_rep_per_comp,
                n_comp,
                -1)

        # Get the scaling factors for the mean of each negative
        # binomial used to model the expression of a gene and reshape
        # it so that it matches the shape required to compute the
        # reconstruction loss.
        #
        # The output is a 4D tensor with:
        #
        # - 1st dimension: the number of samples -> 'n_samples'
        #
        # - 2nd dimension: 1
        #
        # - 3rd dimension: 1
        #
        # - 4th dimension: 1
        scaling_factors = \
            decoder.reshape_scaling_factors(mean_exp,
                                            4)

        # Reshape the decoded output to match the shape required to
        # compute the reconstruction loss.
        #
        # The output is a 4D tensor with:   
        #
        # - 1st dimension: the number of samples -> 'n_samples'
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
        pred_means = dec_out.view(n_samples,
                                  n_rep_per_comp,
                                  n_comp,
                                  n_genes)
     
        # Get the reconstruction loss (rescale based on the mean
        # expression of the genes in the samples in the batch).
        #
        # The output is a 4D tensor with:
        #
        # - 1st dimension: the number of samples -> 'n_samples'
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
        # This means that the loss is not per-gene anymore, but it is
        # summed over all genes.
        #
        # The output is a 3D tensor with:
        #
        # - 1st dimension: the number of samples -> 'n_samples'
        #
        # - 2nd dimension: the number of representations taken per
        #                  component per sample ->
        #                  'n_rep_per_comp'
        #
        # - 3rd dimension: the number of components in the Gaussian
        #                  mixture model ->
        #                  'n_comp'
        recon_loss_sum = recon_loss.sum(-1).clone()

        # Reshape the reconstruction loss so that it can be summed to
        # the GMM loss (calculated below).
        #
        # This means that all loss values in the 'recon_loss_sum' are
        # now listed in a flat tensor.
        #
        # The output is, therefore, a 1D tensor with:
        #
        # - 1st dimension: the number of samples times the number
        #                  of components in the Gaussian mixture
        #                  model times the number of
        #                  representations taken per
        #                  component per sample ->
        #                  'n_samples' * 
        #                  'n_rep_per_comp' *
        #                  'n_comp'
        recon_loss_sum_reshaped = \
            recon_loss_sum.view(n_samples * \
                                n_rep_per_comp * \
                                n_comp)

        #-------------------------------------------------------------#

        # Get the GMM error. 
        #
        # 'gmm(z)' computes the negative log density of the probability
        # of the representations 'z' being drawn from the Gaussian
        # mixture model.
        #
        # The shape of the loss is consistent with the shape of the
        # reconstruction loss in 'recon_loss_sum_shaped'.
        #
        # The output is, therefore, a 1D tensor with:
        #
        # - 1st dimension: the number of samples times the number
        #                  of components in the Gaussian mixture
        #                  model times the number of
        #                  representations taken per
        #                  component per sample ->
        #                  'n_samples' * 
        #                  'n_rep_per_comp' *
        #                  'n_comp'
        gmm_loss = self.gmm(z).clone()

        #-------------------------------------------------------------#

        # Get the total loss.
        #
        # The loss has has many components as the total number of
        # representations computed for the current batch
        # ('n_rep_per_comp' * 'n_comp' representations for each sample
        # in the batch).
        #
        # The output is a 1D tensor with:
        #
        # - 1st dimension: the number of samples times the number
        #                  of components in the Gaussian mixture
        #                  model times the number of
        #                  representations taken per
        #                  component per sample ->
        #                  'n_samples' * 
        #                  'n_rep_per_comp' *
        #                  'n_comp'
        total_loss = recon_loss_sum_reshaped + gmm_loss

        #-------------------------------------------------------------#

        # Reshape the tensor containing the total loss.
        #
        # The output is a 2D tensor with:
        #
        # - 1st dimension: the number of samples  -> 'n_samples'
        #
        # - 2nd dimension: the number of representations taken
        #                  per component of the Gaussian mixture
        #                  model per sample times the number of
        #                  components -> 
        #                  'n_rep_per_comp' * 'n_comp'
        total_loss_reshaped = \
            total_loss.view(n_samples,
                            n_rep_per_comp * n_comp)

        #-------------------------------------------------------------#

        # Get the best representation for each sample in the
        # current batch.
        #
        # The output is a 1D tensor with:
        #
        # - 1st dimension: the number of samples -> 'n_samples'
        best_rep_per_sample = torch.argmin(total_loss_reshaped,
                                           dim = 1).squeeze(-1)

        #-------------------------------------------------------------#

        # Get the best representations for the samples in the batch
        # from the 'n_rep_per_comp' * 'n_comp' representations taken
        # for each sample.
        #
        # The output is a 2D tensor with:
        #
        # - 1st dimension: the number of samples -> 'n_samples'
        #
        # - 2nd dimension: the dimensionality of the Gaussian mixture
        #                  model ->
        #                  'dim'
        rep = z.view(n_samples,
                     n_rep_per_comp * n_comp,
                     dim)[range(n_samples), best_rep_per_sample]

        #-------------------------------------------------------------#

        # Return the representations.
        return rep


    def _optimize_one_rep_per_sample(self,
                                     rep_layer,
                                     optimizer,
                                     n_samples,
                                     data_exp,
                                     mean_exp,
                                     epochs,
                                     opt_num):
        """Optimize one representation per sample.

        Parameters
        ----------
        rep_layer : ``bulkDGD.core.latent.RepresentationLayer``
            The representation layer containing the initial
            representations.

        optimizer : ``torch.optim.Optimizer``
            The optimizer.

        n_samples : ``int``
            The number of samples.

        data_exp : ``torch.Tensor``
            The gene expression (counts) for all samples in tensor
            form.

            This is a 2D tensor where:
            
            * The first dimension has a length equal to the number
              of samples.

            * The second dimension has a length equal to the
              number of genes.

        mean_exp : ``torch.Tensor``
            The mean gene expression (counts) for each sample in tensor
            form.

            This is a 1D tensor whose length is equal to the
            number of samples.

        epochs : ``int``
            The number of epochs to run the optimization for.

        opt_num : ``int``
            The number of the optimization round (useful especially if
            multiple rounds are run).

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

        # Get the number of components of the Gaussian mixture model.
        n_comp = self.gmm.n_comp

        # Get the dimensionality of the latent space.
        dim = self.gmm.dim

        # Get the number of genes (dimensionality of the decoder
        # output).
        n_genes = self.dec.main[-1].out_features

        # Create a list to store the CPU/wall clock time used in each
        # epoch of the optimizations.
        time_opt = []

        #-------------------------------------------------------------#

        # Inform the user that the optimization is starting.
        infostr = f"Starting optimization number {opt_num}..."
        logger.info(infostr)
        
        # For each epoch
        for epoch in range(1, epochs+1):

            # Mark the CPU start time of the epoch.
            time_start_epoch_cpu = time.process_time()

            # Mark the wall clock start time of the epoch.
            time_start_epoch_wall = time.time()

            # Make the gradients zero.
            optimizer.zero_grad()

            # Find the best representations corresponding to the
            # samples in the batch.
            #
            # The output is a 2D tensor with:
            #
            # - 1st dimension: the number of samples -> 'n_samples'
            #
            # - 2nd dimension: the dimensionality of the Gaussian
            #                  mixture model ->
            #                  'dim'
            z = rep_layer.z

            # Get the output in gene space corresponding to the
            # representation found in latent space through the
            # decoder.
            #
            # The output is a 2D tensor with:
            #
            # - 1st dimension: the number of samples -> 'n_samples'
            #
            # - 2nd dimension: the dimensionality of the output
            #                  (= gene) space ->
            #                  'n_genes'
            dec_out = self.dec(z)

            #---------------------------------------------------------#

            # Get the overall reconstruction loss.
            # 
            # The output is a tensor containing a single value.
            recon_loss = \
                self.dec.nb.loss(obs_counts = data_exp,
                                 scaling_factors = mean_exp,
                                 pred_means = dec_out).sum().clone()

            #---------------------------------------------------------#

            # Get the GMM loss.
            #
            # 'gmm(z)' computes the negative log-probability density
            # of 'z' being drawn from the Gaussian mixture model.
            #
            # The output is a tensor containing a single value.
            gmm_loss = self.gmm(z).sum().clone()

            #---------------------------------------------------------#

            # Compute the total loss.
            #
            # The output is a tensor containing a single value.
            total_loss = recon_loss + gmm_loss

            #---------------------------------------------------------#

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
            time_tot_bw_cpu = time_end_bw_cpu - time_start_bw_cpu

            # Get the total wall clock time used by the backward step.
            time_tot_bw_wall = time_end_bw_wall - time_start_bw_wall

            #---------------------------------------------------------#

            # Get the average loss for the current epoch.
            rep_avg_loss_epoch = \
                total_loss.item() / (n_samples * n_genes)

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
                rep_final = rep_layer.z

                # Get the decoder outputs.
                dec_out_final = self.dec(rep_final)

                # Return the representations, the decoder outputs,
                # and the time data.
                return rep_final, dec_out_final, time_opt


    def _get_representations_one_opt(self,
                                     n_samples,
                                     n_rep_per_comp,
                                     data_exp,
                                     mean_exp,
                                     config):
        """Get the representations for a set of samples by
        initializing ``n_rep_per_comp`` representations per each
        component of the Gaussian mixture model per sample, selecting
        the best representation for each sample, and optimizing these
        representations.

        Parameters
        ----------
        n_samples : ``int``
            The number of samples.

        n_rep_per_comp : ``int``
            The number of new representations to be taken for
            component per sample.

        data_exp : ``torch.Tensor``
            The gene expression (counts) for all samples in tensor
            form.

            This is a 2D tensor where:
            
            * The first dimension has a length equal to the number
              of samples.

            * The second dimension has a length equal to the
              number of genes.

        mean_exp : ``torch.Tensor``
            The mean gene expression (counts) for each sample in tensor
            form.

            This is a 1D tensor whose length is equal to the number of
            samples.
    
        config : ``dict``
            A dictionary with the following keys:

            * ``"epochs"``, whose associated value should be the
              number of epochs the optimizations is run for.

            * ``"type"``, whose associated value should be the name
              of the optimizer to be used. The names of the available
              optimizers are stored into the ``OPTIMIZERS`` class
              attribute.

            * ``"options`", whose associated value should be a
              dictionary whose key-value pairs correspond to the
              options needed to set up the optimizer. The options
              available depend on the chosen optimizer.

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
            latent.RepresentationLayer(values = rep_init)

        #-------------------------------------------------------------#

        # Select the best representation for each sample among those
        # initialized (we initialized at least one per sample per
        # component of the Gaussian mixture model).
        rep_best = \
            self._select_best_rep(\
                rep_layer = rep_layer_init,
                n_samples = n_samples,
                n_rep_per_comp = n_rep_per_comp,
                data_exp = data_exp,
                mean_exp = mean_exp)

        # Create a representation layer containing the best
        # representations found.
        rep_layer_best = \
            latent.RepresentationLayer(values = rep_best)

        #-------------------------------------------------------------#

        # Get the optimizer for the optimization.
        optimizer = \
            self._get_optimizer(\
                optimizer_name = config["optimizer_name"],
                optimizer_options = config["optimizer_options"],
                optimizer_parameters = rep_layer_init.parameters())

        #-------------------------------------------------------------#

        # Get the optimized representations.
        rep_opt, dec_out_opt, time_opt = \
            self._optimize_one_rep_per_sample(\
                rep_layer = rep_layer_best,
                optimizer = optimizer,
                n_samples = n_samples,
                data_exp = data_exp,
                mean_exp = mean_exp,
                epochs = config["epochs"],
                opt_num = 1)

        #-------------------------------------------------------------#

        # Make the gradients zero.
        optimizer.zero_grad()

        #-------------------------------------------------------------#

        # Return the representations, the decoder outputs, and the
        # time of the optimization.
        return rep_opt, dec_out_opt, time_opt


    def _get_representations_two_opt(self,
                                     n_samples,
                                     n_rep_per_comp,
                                     data_exp,
                                     mean_exp,
                                     config):
        """Get the representations for a set of samples by
        initializing ``n_rep_per_comp`` representations per each
        component of the Gaussian mixture model per sample, optimizing
        these representations, selecting the best representation for
        for each sample, and optimizing these representations furher.

        Parameters
        ----------
        n_samples : ``int``
            The number of samples.

        n_rep_per_comp : ``int``
            The number of new representations to be taken per component
            per sample.

        data_exp : ``torch.Tensor``
            The gene expression (counts) for all samples in tensor form.

            This is a 2D tensor where:
            
            * The first dimension has a length equal to the number
              of samples.

            * The second dimension has a length equal to the numner of
              genes.

        mean_exp : ``torch.Tensor``
            The mean gene expression (counts) for each sample in tensor
            form.

            This is a 1D tensor whose length is equal to the number of
            samples.
    
        config : ``dict``
            A dictionary with one key (``"optimization"``) associated
            with a dictionary containing the following keys:

            * ``"epochs"``, whose associated value should be the
              number of epochs the optimizations is run for.

            * ``"type"``, whose associated value should be the name
              of the optimizer to be used. The names of the available
              optimizers are stored into the ``OPTIMIZERS`` class
              attribute.

            * ``"options`", whose associated value should be a
              dictionary whose key-value pairs correspond to the
              options needed to set up the optimizer. The options
              available depend on the chosen optimizer.

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
            latent.RepresentationLayer(values = rep_init)

        #-------------------------------------------------------------#

        # Get the optimizer for the first optimization.
        optimizer_opt1 = \
            self._get_optimizer(\
                optimizer_name = \
                    config["opt1"]["optimizer_name"],
                optimizer_options = \
                    config["opt2"]["optimizer_options"],
                optimizer_parameters = rep_layer_init.parameters())

        #-------------------------------------------------------------#

        # Get the optimized representations, the corresponding decoder
        # outputs, and the time information about the optimization's
        # epochs.
        rep_opt1, dec_out_opt1, time_opt1 = \
            self._optimize_multiple_rep_per_sample(\
                rep_layer = rep_layer_init,
                optimizer = optimizer_opt1,
                n_samples = n_samples,
                n_rep_per_comp = n_rep_per_comp,
                data_exp = data_exp,
                mean_exp = mean_exp,
                epochs = config["opt1"]["epochs"],
                opt_num = 1)

        #-------------------------------------------------------------#

        # Create the representation layer.
        rep_layer_opt1 = \
            latent.RepresentationLayer(values = rep_opt1)

        #-------------------------------------------------------------#

        # Make the gradients zero.
        optimizer_opt1.zero_grad()

        #-------------------------------------------------------------#

        # Select the best representation for each sample among those
        # initialized (at least one per sample per component of the
        # Gaussian mixture model).
        rep_best = \
            self._select_best_rep(\
                rep_layer = rep_layer_opt1,
                n_samples = n_samples,
                n_rep_per_comp = n_rep_per_comp,
                data_exp = data_exp,
                mean_exp = mean_exp)

        # Create a representation layer containing the best
        # representations found.
        rep_layer_best = \
            latent.RepresentationLayer(values = rep_best)

        #-------------------------------------------------------------#

        # Get the optimizer for the second optimization.
        optimizer_opt2 = \
            self._get_optimizer(\
                optimizer_name = \
                    config["opt2"]["optimizer_name"],
                optimizer_options = \
                    config["opt2"]["optimizer_options"],
                optimizer_parameters = rep_layer_best.parameters())

        #-------------------------------------------------------------#

        # Get the optimized representations, the corresponding decoder
        # outputs, and the time information about the optimization's
        # epochs.
        rep_opt2, dec_out_opt2, time_opt2 = \
            self._optimize_one_rep_per_sample(\
                rep_layer = rep_layer_best,
                optimizer = optimizer_opt2, 
                n_samples = n_samples,
                data_exp = data_exp,
                mean_exp = mean_exp,
                epochs = config["opt2"]["epochs"],
                opt_num = 2)

        #-------------------------------------------------------------#

        # Make the gradients zero.
        optimizer_opt2.zero_grad()

        #-------------------------------------------------------------#

        # Concatenate the two lists storing the time data for both
        # optimizations.
        time_opt = time_opt1 + time_opt2

        #-------------------------------------------------------------#

        # Return the representations, the decoder outputs, and the
        # time information for both optimizations' epochs.
        return rep_opt2, dec_out_opt2, time_opt


    def _get_final_dataframes(self,
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

        # Convert the tensor containing the representations into a
        # list.
        rep_list = rep.detach().numpy().tolist()

        # Convert the tensor containing the decoder outputs into a
        # list.
        dec_out_list = dec_out.detach().numpy().tolist()

        #-------------------------------------------------------------#

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

        # Crate a data frame for the CPU/wall clock time.
        df_time = pd.DataFrame(time_opt)

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

        # Name and sort the columns.
        df_time.columns = \
            ["platform", "processor", "num_threads",
             "opt_round", "epoch",
             "time_tot_epoch_cpu", "time_tot_bw_cpu",
             "time_tot_epoch_wall", "time_tot_bw_wall"]

        #-------------------------------------------------------------#

        # Return the data frames.
        return df_rep, df_dec_out, df_time


    ######################### PUBLIC METHODS #########################


    def get_representations(self,
                            df_samples,
                            method,
                            config_opt, 
                            n_rep_per_comp = 1):
        """Find the best representations for a set of samples.

        Parameters
        ----------
        df_samples : ``pandas.DataFrame``
            A data frame containing the samples.

        method : ``str``, {``"one_opt"``, ``"two_opt"``}
            The method to find the representations.

            * ``"one_opt"`` means initializing ``n_rep_per_comp``
              representations per component of the Gaussian mixture
              model per sample, picking the best representation
              for each sample, and optimizing these representations.

              This method, therefore, includes only one optimization
              step.

            * ``"two_opt"`` means initializing ``n_rep_per_comp``
              representations per component of the Gaussian mixture
              model per sample, optimizing them, finding the best
              representation for each sample, and optimizing these
              representations.

              This method, therefore, includes two optimization
              steps.

              This is the method used in the work of Prada, Schuster,
              Liang and coworkers introducing the DGD model to
              perform differential gene expression analysis
              :footcite:t:`prada2023n`.

        config_opt : ``dict``
            A dictionary of options for the optimization(s). It varies
            according to the selected ``method``.

            The supported options for all available methods can be
            found :doc:`here <../rep_config_options>`.

        n_rep_per_comp : ``int``, ``1``
            The number of initial representations to be taken
            per component per sample.

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
            step, and the columns contain data about the platoform
            where the calculation was run, the number of CPU threads
            used by the computation, and the CPU and wall clock
            time used by the entire epoch and by the backpropagation
            step run inside it.

        .. footbibliography
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

        # Get the gene expression for all samples in tensor form.
        #
        # This is a 2D tensor with:
        #
        # - 1st dimension: the number of samples -> 'n_samples'
        #
        # - 2nd dimension: the number of genes -> 'n_genes'
        data_exp = dataset.data_exp

        # Get the mean gene expression for each sample in tensor form.
        #
        # This is a 1D tensor with:
        #
        # - 1st dimension: the number of samples -> 'n_samples'
        mean_exp = dataset.mean_exp

        # Get the names/IDs/indexes of the samples from the data
        # frame's rows' names.
        samples_names = df_expr_data.index

        # Get the names of the genes from the expression data frame's
        # columns' names.
        genes_names = df_expr_data.columns

        #-------------------------------------------------------------#

        # If the user selected the one-optimization method
        if method == "one_opt":

            # Get the representations, the corresponding decoder
            # outputs, and the time data.
            rep, dec_out, time_opt = \
                self._get_representations_one_opt(\
                    n_samples = n_samples,
                    n_rep_per_comp = n_rep_per_comp,
                    data_exp = data_exp,
                    mean_exp = mean_exp,
                    config = config_opt)

        # If the user selected the two-optimizations methods
        elif method == "two_opt":
            
            # Get the representations, the corresponding decoder
            # outputs, and the time data.
            rep, dec_out, time_opt = \
                self._get_representations_two_opt(\
                    n_samples = n_samples,
                    n_rep_per_comp = n_rep_per_comp,
                    data_exp = data_exp,
                    mean_exp = mean_exp,
                    config = config_opt)

        #-------------------------------------------------------------#

        # Generate the final data frames.
        df_rep, df_dec_out, df_time = \
            self._get_final_dataframes(rep = rep,
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
        df_prob_rep = pd.DataFrame(probs_values.detach().numpy())

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
