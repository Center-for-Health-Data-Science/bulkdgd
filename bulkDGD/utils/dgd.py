#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-

#    dgd.py
#
#    Utility functions to use the DGD model.
#
#    Copyright (C) 2023 Valentina Sora 
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


# Description of the module
__doc__ = "Utility functions to use the DGD model."
# Package name
pkg_name = "bulkDGD"


# Standard library
import logging as log
from pkg_resources import resource_filename, Requirement
# Third-party packages
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
# bulkDGD
from bulkDGD.core import (
    dataclasses,
    decoder,
    GMM,
    priors,
    )
from . import misc


# Get the module's logger
logger = log.getLogger(__name__)


#------------------------- Private constants -------------------------#


# Name of the column containing the tissue label in the input
# data frame containing the samples
_TISSUE_COL = "tissue"

# Name of the column containing the maximum probability density
# found, per sample
_MAX_PROB_COL = "max_prob_density"

# Name of the column containing the component for which the
# maximum probability density was found, per each sample
_MAX_PROB_COMP_COL = "max_prob_density_comp"

# Name of the column containing the unique index of the sample
# having the maximum probability for a component
_SAMPLE_IDX_COL = "sample_idx"


#------------------------- Public constants --------------------------#


# Default directory containing the configuration files
CONFIG_REP_DIR = \
    resource_filename(Requirement(pkg_name),
                      f"configs/representations")

# Default PyTorch file containing the trained decoder
try:

    DEC_FILE = \
        resource_filename(Requirement(pkg_name),
                          f"data/model/dec.pth")

# If no file is found in the directory
except KeyError:

    # Set it to None
    DEC_FILE = None

# Default PyTorch file containing the trained Gaussian mixture model
GMM_FILE = \
    resource_filename(Requirement(pkg_name),
                      f"data/model/gmm.pth")

# Default PyTorch file containing the trained representation layer
REP_FILE = \
    resource_filename(Requirement(pkg_name),
                      f"data/model/rep.pth")


# File containing the Ensembl IDs of the genes on which the DGD
# model has been trained
DGD_GENES_FILE = \
    resource_filename(Requirement(pkg_name),
                      f"data/model/training_genes.txt")

# Available optimizers
AVAILABLE_OPTIMIZERS = ["adam"]


#------------------------ Preprocess samples -------------------------#


def preprocess_samples(samples_df):
    """Preprocess new samples to use them with the DGD model.

    Parameters
    ----------
    samples_df : ``pandas.DataFrame``
        A data frame containing the samples to be preprocessed.

    Returns
    -------
    ``tuple``
        A tuple containing:

        * The data frame with the preprocessed samples.
        * The list of genes found in the input data frame but not
          belonging to the gene set used to train the DGD model.
        * The list of genes present in the gene set used to train
          the DGD model but not found in the input data frame.
    """

    # Load the list of genes used to train the DGD model
    genes_list_dgd = misc.get_list(list_file = DGD_GENES_FILE)

    # Remove the Ensembl gene IDs' versions from the names of
    # the genes to obtain all genes for which expression data
    # are available in the data frame. It works also if the
    # Ensembl IDs are specified without their version
    genes_list_df = list(zip(*samples_df.columns.str.split(".")))[0]

    # Change the column names so that only the Ensembl IDs are
    # kept (and not their versions)
    samples_df.columns = genes_list_df

    # Select only the genes used to train the DGD model
    # from the samples' data frame to obrain the data frame
    # containing the preprocessed samples. Sort the columns
    # (= genes) in the order expected by the DGD model, and,
    # if no data were found for some genes, add a default
    # count of 0
    preproc_df = samples_df.reindex(genes_list_dgd,
                                    axis = 1,
                                    fill_value = 0)

    # Create a list containing the genes present in the original
    # data frame but not in the list of genes on which the DGD
    # model was trained on. Use lists instead of sets (which
    # would be faster) to preserve the order
    genes_excluded = \
        [gene for gene in genes_list_df if gene not in genes_list_dgd]

    # Create a list containing the genes present in the list of genes
    # used to train the DGD model but not in the original data frame.
    # Use lists instead of sets (which would be faster) to preserve
    # the order
    genes_missing = \
        [gene for gene in genes_list_dgd if gene not in genes_list_df]

    # Return the data frame with the preprocessed samples and the
    # two lists
    return preproc_df, genes_excluded, genes_missing


#--------------------- Load/Initialize the model ---------------------#


def load_samples_data(csv_file,
                      config,
                      keep_samples_names = True):
    """Load the data frame containing the gene expression data for
    the samples, and return data in a PyTorch-compatible format.

    Parameters
    ----------
    csv_file : ``str``
        A CSV file containing the samples' data. Rows must represent
        the different samples, and columns must represent the
        expression of each gene.

    config : ``dict``
        A dictionary containing the options for loading the data.

    keep_samples_names : ``bool``, default: ``True``
        Whether to keep the name assigned to the samples in
        the input dataframe, if any were found.

    Returns
    -------
    ``tuple``
        A ``tuple`` containing:

        * A ``torch.utils.data.DataLoader`` object with the dataset.
        * A ``list`` representing the unique indexes of the samples.
        * An ``int`` representing the number of samples in the
          dataset.
        * An ``int`` representing the number of genes in the dataset.
        * A ``pandas.DataFrame`` with the labels of the tissues the
          samples belong to.
    """

    # If we need to keep the samples' names
    if keep_samples_names:
        
        # Load the data frame assuming the samples' names
        # are the dataframe's rows' names (= index)
        df = pd.read_csv(csv_file,
                         sep = ",",
                         index_col = 0)

        # The samples' indexes will be the samples' names
        indexes = df.index.tolist()

    # Otherwise
    else:

        # Load the data frame assuming there is no index
        df = pd.read_csv(csv_file,
                         sep = ",",
                         index_col = False)

        # Create the unique samples' indexes
        indexes = list(range(len(df)))

    # Initialize the data frame containing the tissue labels
    # for the samples to an empty data frame
    tissues = pd.DataFrame() 

    # If a column of the data frame contains tissue
    # information
    if _TISSUE_COL in df.columns.tolist():

        # Save the tissue information into a separate
        # data frame
        tissues = df.iloc[:,-1] 

        # Add the indexes to the tissue labels as well
        tissues.index = indexes

        # Remove the column from the original data frame
        df = df.iloc[:,:-1]

        # Inform the user that you found tissue information
        infostr = \
            f"'{_TISSUE_COL}' column found in the input " \
            f"CSV file '{csv_file}'. The column is assumed to " \
            f"contain the labels of the tissues the samples " \
            f"belong to."
        logger.info(infostr)

    # Get the number of samples and the number of genes
    # in the dataset
    n_samples, n_genes = df.shape

    # Create the dataset
    dataset = dataclasses.GeneExpressionDataset(df = df)

    # Load the dataset and the data dimensionality and
    # return the data of interest
    data = \
        (DataLoader(dataset, **config),
         indexes,
         n_samples,
         n_genes,
         tissues)

    # Inform the user about the number of samples and number of
    # genes found in the dataset
    logger.info(f"{n_samples} samples were found in the dataset.")
    logger.info(f"{n_genes} genes were found in the dataset.")

    # If tissue information was found
    if not tissues.empty:

        # Inform the user about the number of unique tissues
        # found
        unique_tissues = tissues.unique().tolist()
        logger.info(\
            f"{len(unique_tissues)} tissues were found in the " \
            f"dataset ({', '.join(unique_tissues)}).")

    # Return the data
    return data


def load_pth_file(pth_file,
                  model_type,
                  model):
    """Load a PyTorch file containing a model's state.

    Parameters
    ----------
    pth_file : ``str``
        A PyTorch file containing the state of the model.

    model_type : ``str``, {``"decoder"``, ``"gmm"``, ``"rep"``}
        The type of model whose state is contained inside
        the PyTorch file. ``"decoder"`` indicates the decoder,
        ``"gmm"`` indicates the Gaussian mixture model,
        and ``"rep"`` indicates the representation layer.
    
    model : ``torch.nn.Module``
        The model whose state is to be loaded.

    Returns
    -------
    ``torch.nn.Module``
        The model after the state has been loaded.
    """

    # If the file was not passed
    if pth_file is None or pth_file == "":

        # If the file contains the trained decoder
        if model_type == "decoder":

            # Set the file to the default one
            pth_file = DEC_FILE

        # If the file contains the trained Gaussian mixture
        # model
        elif model_type == "gmm":

            # Set the file to the default one
            pth_file = GMM_FILE

        # If the file contains the trained representation
        # layer
        elif model_type == "rep":

            # Set the file to the default one
            pth_file = REP_FILE

        # Warn the user that the default file will
        # be used
        warnstr = \
            f"No PTH file was passed for '{model_type}' " \
            f"in the configuration file. The default file " \
            f"'{pth_file}' will be used instead."
        logger.warning(warnstr)

    # Load the file
    model.load_state_dict(torch.load(pth_file))

    # Return the model
    return model


def get_gmm(dim,
            config):
    """Initialize the Gaussian mixture model, load its state,
    and return it.

    Parameters
    ----------
    dim : ``int``
        The simensionality of the space for the Gaussian
        mixture model.

    config : ``dict``
        A dictionary containing the options for initializing
        the Gaussian mixture model and loading its state.

    Returns
    -------
    ``DDGPerturbations.core.GMM.GaussianMixtureModel``
        The Gaussian mixture model.
    """

    # Set the prior for the means of the Gaussian components
    # of the model

    # Get the type of prior
    prior_type = config["mean_prior"]["type"]

    # Get the configuration for the prior
    config_prior = config["mean_prior"]["options"]

    # Get the PyTorch file
    pth_file = config["pth_file"]

    # If the prior is a softball prior
    if prior_type == "softball":

        # Initialize the prior
        mean_prior = priors.softball(dim = dim,
                                     **config_prior)

    # Try to initialize the GMM
    try:

        gmm = GMM.GaussianMixtureModel(dim = dim,
                                       mean_prior = mean_prior,
                                       **config["options"])

    # If something went wrong
    except Exception as e:

        # Raise an error
        errstr = \
            f"It was not possible to initialize the Gaussian " \
            f"mixture model. Error: {e}"
        raise Exception(errstr)

    # Inform the user that the GMM was successfully initialized
    infostr = \
        "The Gaussian mixture model was successfully initialized."
    logger.info(infostr)

    # Try to load the model's state
    try:

        gmm = load_pth_file(pth_file = pth_file,
                            model_type = "gmm",
                            model = gmm)

    # If something went wrong
    except Exception as e:

        # Raise an error
        errstr = \
            f"It was not possible to load the Gaussian " \
            f"mixture model's state from '{pth_file}'. " \
            f"Error: {e}"
        raise Exception(errstr)

    # Inform the user that the GMM's state was successfully
    # loaded
    infostr = \
        f"The Gaussian mixture model's state was successfully " \
        f"loaded from '{pth_file}'."
    logger.info(infostr)

    # Return the model
    return gmm


def get_decoder(dim,
                config):
    """Initialize the decoder, load its state, and return it.

    Parameters
    ----------
    dim : ``int``
        The dimensionality of the input space for the decoder.

    config : ``dict``
        A dictionary containing the options for initializing
        the decoder and loading its state.

    Returns
    -------
    ``DDGPerturbations.core.decoder.Decoder``
        The trained decoder.
    """

    # Get the PyTorch file
    pth_file = config["pth_file"]

    # Try to initialize the decoder
    try:
        
        dec = decoder.Decoder(n_neurons_latent = dim,
                              **config["options"])

    # If something went wrong
    except Exception as e:

        # Raise an error
        errstr = \
            f"It was not possible to initialize the " \
            f"decoder. Error: {e}"
        raise Exception(errstr)

    # Inform the user that the decoder was successfully
    # initialized
    infostr = \
        "The decoder was successfully initialized."
    logger.info(infostr)

    # Try to load the model's state
    try:

        dec = load_pth_file(pth_file = pth_file,
                            model_type = "decoder",
                            model = dec)

    # If something went wrong
    except Exception as e:

        # Raise an error
        errstr = \
            f"It was not possible to load the decoder's " \
            f"state from '{pth_file}'. Error: {e}"
        raise Exception(errstr)

    # Inform the user that the decoder's state was successfully
    # loaded
    infostr = \
        f"The decoder's state was successfully " \
        f"loaded from '{pth_file}'."
    logger.info(infostr)

    # Return the model
    return dec


def get_rep_layer(dim,
                  config):
    """Initialize the representation layer, load its state,
    and return it.

    Parameters
    ----------
    dim : ``int``
        The dimensionality of the representation layer.

    config : ``dict``
        A dictionary containing the options for initializing
        the representation layer and loading its state.

    Returns
    -------
    ``DDGPerturbations.core.decoder.RepresentationLayer``
        The representation layer.
    """

    # Get the PyTorch file
    pth_file = config["pth_file"]

    # Get the number of training samples
    n_samples = config["options"]["n_samples"]

    # Get the values
    values = torch.zeros(size = (n_samples, dim))

    # Try to initialize the representation layer
    try:

        rep_layer = decoder.RepresentationLayer(values = values)
    
    # If something went wront
    except Exception as e:

        # Raise an error
        errstr = \
            f"It was not possible to initialize the " \
            f"representation layer. Error: {e}"
        raise Exception(errstr)

    # Inform the user that the representation layer was
    # successfully initialized
    infostr = \
        "The representation layer was successfully " \
        "initialized."
    logger.info(infostr)   
    
    # Try to load the representation layer's state
    try:

        rep_layer = load_pth_file(pth_file = pth_file,
                                  model_type = "rep",
                                  model = rep_layer)

    # If something went wrong
    except Exception as e:

        # Raise an error
        errstr = \
            f"It was not possible to load the representation " \
            f"layers's state from '{pth_file}'. Error: {e}"
        raise Exception(errstr)

    # Inform the user that the representation layer's state
    # was successfully loaded
    infostr = \
        f"The representation layers's state was successfully " \
        f"loaded from '{pth_file}'."
    logger.info(infostr)

    # Return the representation layer
    return rep_layer


#-------------------------- Representations --------------------------#


def get_representations(dataset,
                        indexes,
                        gmm,
                        dec,
                        n_samples,
                        n_genes,
                        n_reps_per_mix_comp,
                        dim,
                        config_opt1,
                        config_opt2):
    """Initialize and optimize the representations.

    Parameters
    ----------
    dataset : ``torch.utils.data.DataLoader``
        The dataset containing the gene expression data for the
        samples.

    indexes : ``list``
        A list of unique indexes representing the samples.

    gmm : ``DGDPerturbations.core.GMM.GaussianMixtureModel``
        The trained Gaussian mixture model.

    dec : ``DGDPerturbations.core.decoder.Decoder``
        The trained decoder.

    n_samples : ``int``
        The number of samples in the dataset.

    n_genes : ``int``
        The number of genes in the dataset.

    n_reps_per_mix_comp : ``int``
        The number of new representations to be taken
        per component per sample.

    dim : ``int``
        The dimensionality of the Gaussian mixture model.
    
    config_opt1 : ``dict``
        A dictionary of options for the initial optimization.

    config_opt2 : ``dict``
        A dictionary of options for the second optimization.

    Returns
    -------
    ``tuple``
        A tuple containing a ``pandas.DataFrame`` with the loss,
        and a ``np.ndarray`` containing the best representation
        for each sample.
    """


    #---------------------- First optimization -----------------------#


    # Get the initial values for the representations by sampling
    # from the Gaussian mixture model. The representations will be
    # the sampled from the means of the mixture components (since
    # 'sampling_method' is set to '"mean"'. The output is a 2D
    # tensor with:
    #
    # - 1st dimension: the total number of samples times the number of
    #                  components in the Gaussian mixture model times
    #                  the number of representations taken per
    #                  component per sample ->
    #                  'n_samples' *
    #                  'n_mix_comp' * 
    #                  'n_reps_per_mix_comp'
    #
    # - 2nd dimension: the dimensionality of the Gaussian mixture
    #                  model ->
    #                  'dim'
    rep_init_values = \
        gmm.sample_new_points(n_points = n_samples, 
                              sampling_method = "mean",
                              n_reps_per_mix_comp = 2)

    # Initialize the representation layer with 'dim'
    # dimensions and 'n_samples' samples that have values
    # 'rep_init_values'
    rep_samples_layer = \
        decoder.RepresentationLayer(values = rep_init_values)


    #---------------------- Check the optimizer ----------------------#


    # Get the optimizer type
    optimizer_type = config_opt1["type"]

    # If there is an Adam optimizer
    if optimizer_type == "adam":

        # Get the Adam optimizer
        rep_samples_optimizer = \
            torch.optim.Adam(rep_samples_layer.parameters(),
                             **config_opt1["options"])

    # Otherwise
    else:

        # Inform the user that no other optimizer is
        # available so far and raise an error
        errstr = \
            f"The optimizer '{optimizer_type}' is not " \
            f"supported so far. The supported optimisers are: " \
            f"{', '.join(AVAILABLE_OPTIMIZERS)}."
        raise ValueError(errstr)

    # Get the number of components in the Gaussian mixture
    n_mix_comp = gmm.n_mix_comp

    # Initialize an empty list to store the average loss for each
    # epoch.
    # The lenght of the list is equal to the number of epochs
    rep_avg_loss = [0] * config_opt1["epochs"]


    #-------------------- Start the optimization ---------------------#


    # Inform the user that the optimization is starting
    infostr = "Starting the first optimization..."
    logger.info(infostr)

    # For each epoch
    for epoch in range(config_opt1["epochs"]):

        # Make the gradients zero
        rep_samples_optimizer.zero_grad()

        # For each batch:
        # 'expr' : gene expression for all samples in the batch
        # 'mean_expr' : mean gene expression for all samples
        #               in the batch
        # 'sample_ixs' : the indexes of the samples in the batch
        for expr, mean_expr, sample_ixs in dataset:


            #------------ Initialize the representations -------------#


            # Get the representations' values from the representation
            # layer. The representations are stored in a 2D tensor with:
            #
            # - 1st dimension: the number of samples in the current
            #                  batch times the number of components
            #                  in the Gaussian mixture model times the
            #                  number of representations taken per
            #                  component per sample ->
            #                  'n_samples_in_batch' * 
            #                  'n_mix_comp' *
            #                  'n_reps_per_mix_comp'
            #
            # - 2nd dimension: the dimensionality of the Gaussian
            #                  mixture model ->
            #                  'dim'
            z_raw = rep_samples_layer.z

            # Reshape the tensor containing the representations. The
            # output is a 4D tensor with:
            #
            # - 1st dimension: the number of samples in the current
            #                  batch ->
            #                  'n_samples_in_batch'
            # 
            # - 2nd dimension: the number of representations taken
            #                  per component per sample ->
            #                  'n_reps_per_mix_comp'
            #
            # - 3rd dimension: the number of components in the
            #                  Gaussian mixture model ->
            #                  'n_mix_comp'
            #
            # - 4th dimension: the dimensionality of the Gaussian
            #                  mixture model ->
            #                  'dim'
            z_reshaped = gmm.reshape_targets(\
                            z_raw,
                            y_type = "predicted")[sample_ixs]

            # Reshape it again. The output is a 2D tensor with:
            #
            # - 1st dimension: the number of samples in the current
            #                  batch times the number of components
            #                  in the Gaussian mixture model times the
            #                  number of representations taken per
            #                  component per sample ->
            #                  'n_samples_in_batch' * 
            #                  'n_mix_comp' *
            #                  'n_reps_per_mix_comp'
            #
            # - 2nd dimension: the dimensionality of the Gaussian
            #                  mixture model ->
            #                  'dim'
            z = gmm.reshape_targets(z_reshaped,
                                    y_type = "reverse")


            #-------------- Decode the representations ---------------#


            # Get the outputs in gene space corresponding to the
            # representations found in latent space through the
            # decoder. The output is a 2D tensor with:
            #
            # - 1st dimension: the number of samples in the current
            #                  batch times the number of components
            #                  in the Gaussian mixture model times the
            #                  number of representations taken per
            #                  component per sample ->
            #                  'n_samples_in_batch' * 
            #                  'n_mix_comp' *
            #                  'n_reps_per_mix_comp'
            #
            # - 2nd dimension: the dimensionality of the output
            #                  (= gene) space ->
            #                  'n_genes'
            dec_out = dec(z)


            #------------ Compute the reconstruction loss ------------#


            # Get the reconstruction  loss (rescale based on the mean
            # expression of the genes in the samples in the batch).
            # The output is a 4D tensor with:
            #
            # - 1st dimension: the number of samples in the current
            #                  batch ->
            #                  'n_samples_in_batch'
            # 
            # - 2nd dimension: the number of representations taken
            #                  per component per sample ->
            #                  'n_reps_per_mix_comp'
            #
            # - 3rd dimension: the number of components in the
            #                  Gaussian mixture model ->
            #                  'n_mix_comp'
            #
            # - 4th dimension: the dimensionality of the output
            #                  (= gene) space ->
            #                  'n_genes'
            recon_loss = \
                dec.nb.loss(\
                    gmm.reshape_targets(expr,
                                        y_type = "true"),
                    decoder.reshape_scaling_factor(mean_expr, 
                                                   4),
                    gmm.reshape_targets(dec_out,
                                        y_type = "predicted"))

            # Get the total reconstruction loss by summing all
            # values in the 'recon_loss' tensor. The output is
            # a tensor containing a single value.
            recon_loss_sum = recon_loss.sum().clone()


            #----------------- Compute the GMM loss ------------------#


            # Get the GMM error. 'gmm(z)' computes the negative log
            # density of the probability of the representations 'z'
            # being drawn from the mixture model. The output is
            # a 1D tensor with:
            #
            # - 1st dimension: the number of samples in the current
            #                  batch times the number of components
            #                  in the Gaussian mixture model times the
            #                  number of representations taken per
            #                  component per sample ->
            #                  'n_samples_in_batch' * 
            #                  'n_mix_comp' *
            #                  'n_reps_per_mix_comp'
            gmm_loss = gmm(z)

            # Get the total GMM loss by summing over all values in
            # the 'gmm_loss' tensor. The output is a tensor containing
            # a single value.
            gmm_loss_sum = gmm_loss.sum().clone()


            #---------------- Compute the total loss -----------------#


            # Get the total loss - 'loss' is a tensor containing
            # a single number 
            total_loss = recon_loss_sum + gmm_loss_sum


            #--------------- Propagate the total loss ----------------#


            # Propagate the loss backward
            total_loss.backward()


            #---------------- Update the average loss ----------------#


            # Update the average loss for the current epoch
            rep_avg_loss[epoch] += \
                total_loss.item() / (n_samples * n_genes * \
                                     n_mix_comp * n_reps_per_mix_comp)


        #------------------------ Take a step ------------------------#


        # Take an optimization step
        rep_samples_optimizer.step()

        # Inform the user about the loss at the current epoch
        infostr = f"Epoch {epoch}: loss {rep_avg_loss[epoch]}."
        logger.info(infostr)


    #----------------- Find the best representations -----------------#


    # Make the gradients zero
    rep_samples_optimizer.zero_grad()
    
    # Create an empty tensor to store the new values for the
    # representations. The output is a 2D tensor with:
    #
    # - 1st dimension: the total number of samples ->
    #                  'n_samples'
    #
    # - 2nd dimension: the dimensionality of the Gaussian mixture
    #                  model ->
    #                  'dim'
    rep_new_values = torch.empty((n_samples, dim))
    
    # For each batch:
    # 'expr' : gene expression for all samples in the batch
    # 'mean_expr' : mean gene expression for all samples
    #               in the batch
    # 'sample_ixs' : the indexes of the samples in the batch
    for expr, mean_expr, sample_ixs in dataset:

        # 'z_raw' is a 2D tensor with:
        #
        # - 1st dimension: the number of samples in the current
        #                  batch times the number of components
        #                  in the Gaussian mixture model times the
        #                  number of representations taken per
        #                  component per sample ->
        #                  'n_samples_in_batch' * 
        #                  'n_mix_comp' *
        #                  'n_reps_per_mix_comp'
        #
        # - 2nd dimension: the dimensionality of the Gaussian
        #                  mixture model ->
        #                  'dim'

        # Reshape the tensor containing the representations. The
        # output is a 4D tensor with:
        #
        # - 1st dimension: the number of samples in the current
        #                  batch ->
        #                  'n_samples_in_batch'
        # 
        # - 2nd dimension: the number of representations taken per
        #                  component per sample ->
        #                  'n_reps_per_mix_comp'
        #
        # - 3rd dimension: the number of components in the Gaussian
        #                  mixture model ->
        #                  'n_mix_comp'
        #
        # - 4th dimension: the dimensionality of the Gaussian
        #                  mixture model ->
        #                  'dim'
        z_reshaped = gmm.reshape_targets(\
                        z_raw,
                        y_type = "predicted")[sample_ixs]


        # Reshape it again. The output is a 2D tensor with:
        #
        # - 1st dimension: the number of samples in the current
        #                  batch times the number of components
        #                  in the Gaussian mixture model times the
        #                  number of representations taken per
        #                  component per sample ->
        #                  'n_samples_in_batch' * 
        #                  'n_mix_comp' *
        #                  'n_reps_per_mix_comp'
        #
        # - 2nd dimension: the dimensionality of the Gaussian mixture
        #                  model ->
        #                  'dim'
        z = gmm.reshape_targets(z_reshaped,
                                y_type = "reverse")


        #---------------- Decode the representations -----------------#


        # Get the outputs in gene space corresponding to the
        # representations found in latent space through the
        # decoder. The output is a 2D tensor with:
        #
        # - 1st dimension: the number of samples in the current
        #                  batch times the number of components
        #                  in the Gaussian mixture model times the
        #                  number of representations taken per
        #                  component per sample ->
        #                  'n_samples_in_batch' * 
        #                  'n_mix_comp' *
        #                  'n_reps_per_mix_comp'
        #
        # - 2nd dimension: the dimensionality of the output
        #                  (= gene) space ->
        #                  'n_genes'
        dec_out = dec(z)


        #---------- Compute the overall reconstruction loss ----------#


        # Get the reconstruction  loss (rescale based on the mean
        # expression of the genes in the samples in the batch).
        # The output is a 4D tensor with:
        #
        # - 1st dimension: the number of samples in the current
        #                  batch ->
        #                  'n_samples_in_batch'
        # 
        # - 2nd dimension: the number of representations taken per
        #                  component per sample ->
        #                  'n_reps_per_mix_comp'
        #
        # - 3rd dimension: the number of components in the Gaussian
        #                  mixture model ->
        #                  'n_mix_comp'
        #
        # - 4th dimension: the dimensionality of the output
        #                  (= gene) space ->
        #                  'n_genes'
        recon_loss = \
            dec.nb.loss(\
                gmm.reshape_targets(expr,
                                    y_type = "true"),
                decoder.reshape_scaling_factor(mean_expr, 
                                               4),
                gmm.reshape_targets(dec_out,
                                    y_type = "predicted"))

        # Get the total reconstruction loss by summing over the
        # last dimension of the 'recon_loss' tensor. This means
        # that the loss is not per-gene anymore, but summed over
        # all genes. The output is a 3D tensor with:
        #
        # - 1st dimension: the number of samples in the current
        #                  batch ->
        #                  'n_samples_in_batch'
        #
        # - 2nd dimension: the number of representations taken per
        #                  component per sample ->
        #                  'n_reps_per_mix_comp'
        #
        # - 3rd dimension: the number of components in the Gaussian
        #                  mixture model ->
        #                  'n_mix_comp'
        recon_loss_sum = recon_loss.sum(-1).clone()

        # Reshape the reconstruction loss so that it can be
        # summed to the GMM loss (calculated below). This means that
        # all loss values in the 'recon_loss_sum' are now listed in
        # a flat tensor. The output is, therefore, a 1D tensor with:
        #
        # - 1st dimension: the number of samples in the current
        #                  batch times the number of components
        #                  in the Gaussian mixture model times the
        #                  number of representations taken per
        #                  component per sample ->
        #                  'n_samples_in_batch' * 
        #                  'n_mix_comp' *
        #                  'n_reps_per_mix_comp'
        recon_loss_sum_reshaped = \
            gmm.reshape_targets(recon_loss_sum,
                                y_type = "reverse")


        #--------------- Compute the overall GMM loss ----------------#


        # Get the GMM error. 'gmm(z)' computes the negative log
        # density of the probability of the representations 'z'
        # being drawn from the mixture model. The shape of the loss
        # is consistent with the shape of the reconstruction loss in
        # 'recon_loss_sum_shaped'. The output is, therefore, a 
        # a 1D tensor with:
        #
        # - 1st dimension: the number of samples in the current
        #                  batch times the number of components
        #                  in the Gaussian mixture model times the
        #                  number of representations taken per
        #                  component per sample ->
        #                  'n_samples_in_batch' * 
        #                  'n_mix_comp' *
        #                  'n_reps_per_mix_comp'
        gmm_loss = gmm(z).clone()


        #-------------- Compute the overall total loss ---------------#


        # Get the total loss. The loss has has many components as the
        # total number of representations computed for the current
        # batch ('n_reps_per_mix_comp' * 'n_mix_comp' representations
        # for each sample in the batch). The output is a 1D tensor
        # with:
        #
        # - 1st dimension: the number of samples in the current
        #                  batch times the number of components
        #                  in the Gaussian mixture model times the
        #                  number of representations taken per
        #                  component per sample ->
        #                  'n_samples_in_batch' * 
        #                  'n_mix_comp' *
        #                  'n_reps_per_mix_comp'
        total_loss = recon_loss_sum_reshaped + gmm_loss

        # Get the best representations for the samples in the batch
        # from the 'n_reps_per_mix_comp' * 'n_mix_comp' representations
        # taken for each sample.
        # The output is a 2D tensor with:
        #
        # - 1st dimension: the number of samples in the current
        #                  bath ->
        #                  'n_samples_in_batch'
        #
        # - 2nd dimension: the dimensionality of the Gaussian mixture
        #                  model ->
        #                  'dim'
        rep_new_values[sample_ixs] = \
            gmm.choose_best_representations(z, total_loss)


    #---------------------- Second optimization ----------------------#


    # Set a representation layer for the best representations found
    # for all samples
    best_rep_layer = \
        decoder.RepresentationLayer(values = rep_new_values)


    #---------------------- Check the optimizer ----------------------#


    # Get the optimizer type
    optimizer_type = config_opt2["type"]

    # If there is an Adam optimizer
    if optimizer_type == "adam":

        # Get the Adam optimizer
        best_rep_optimizer = \
            torch.optim.Adam(best_rep_layer.parameters(),
                             **config_opt2["options"])

    # Otherwise
    else:

        # Inform the user that no other optimizer is
        # available so far and raise an error
        errstr = \
            f"The optimizer '{optimizer_type}' is not " \
            f"supported so far. The supported optimisers are: " \
            f"{', '.join(AVAILABLE_OPTIMIZERS)}."
        raise ValueError(errstr)

    # Initialize an empty list to store the best averages
    # for each epoch - the length of the list is equal
    # to the number of epochs
    rep_avg_loss = [0] * config_opt2["epochs"]


    #-------------------- Start the optimization ---------------------#


    # Inform the user that the optimization is starting
    infostr = "Starting the second optimization..."
    logger.info(infostr)
    
    # For each epoch
    for epoch in range(config_opt2["epochs"]):

        # Make the gradients zero
        best_rep_optimizer.zero_grad()

        # Initialize the loss for all samples in the batch
        # to 0.
        sample_avg_loss = np.zeros(n_samples)
        
        # For each batch:
        # 'expr' : gene expression for all samples in the batch
        # 'mean_expr' : mean gene expression for all samples
        #               in the batch
        # 'sample_ixs' : the indexes of the samples in the batch
        for expr, mean_expr, sample_ixs in dataset:
            
            # Find the best representations corresponding to the
            # samples in the batch. The output is a D tensor with:
            #
            # - 1st dimension: the number of samples in the current
            #                  batch ->
            #                  'n_samples_in_batch'
            #
            # - 2nd dimension: the dimensionality of the Gaussian
            #                  mixture model ->
            #                  'dim'
            z = best_rep_layer(sample_ixs)

            # Get the output in gene space corresponding to the
            # representation found in latent space through the
            # decoder. The output is a 2D tensor with:
            #
            # - 1st dimension: the number of samples in the current
            #                  batch ->
            #                  'n_samples_in_batch'
            #
            # - 2nd dimension: the dimensionality of the output
            #                  (= gene) space ->
            #                  'n_genes'
            dec_out = dec(z)


            #------ Compute the per-sample reconstruction loss -------#


            # Get the sample reconstruction loss summing over the
            # last dimension of the 'dec_out' tensor. Thie means that
            # one loss value per sample in the current batch is 
            # returned, meaning that the loss is summed over the
            # genes. The output is, therefore, a 1D tensor with:
            #
            # - 1st dimension: the number of samples in the current
            #                  batch ->
            #                  'n_samples_in_batch'
            recon_loss_sample = \
                dec.nb.loss(expr, mean_expr, dec_out).sum(-1).clone()


            #------------ Compute the per-sample GMM loss ------------#


            # Get the GMM loss - gmm(z)' computes the
            # negative log density of the probability of 'z'
            # being drawn from the mixture model. One loss value
            # is reported per sample in the current batch. The output
            # is, therefore, a 1D tensor with:
            #
            # - 1st dimension: the number of samples in the current
            #                  batch ->
            #                  'n_samples_in_batch'    
            gmm_loss_sample = gmm(z).clone()


            #----------- Compute the per-sample total loss -----------#


            # Get the total per-sample loss. The output is a 1D tensor
            # with:
            #
            # - 1st dimension: the number of samples in the current
            #                  batch ->
            #                  'n_samples_in_batch'    
            total_loss_sample = recon_loss_sample + gmm_loss_sample

            # Save the loss for the current sample, divided
            # by the number of genes
            sample_avg_loss[sample_ixs] = \
                total_loss_sample.clone().detach().cpu().numpy() / \
                n_genes


            #-------- Compute the overall reconstruction loss --------#


            # Get the overall recon loss. The output is a tensor
            # containing a single value
            recon_loss = \
                dec.nb.loss(expr, mean_expr, dec_out).sum().clone()

            #------------- Compute the overall GMM loss --------------#


            # Get the GMM loss. 'gmm(z)' computes the negative log
            # density of the probability of 'z' being drawn from the
            # mixture model. The output is a tensor containing a
            # single value
            gmm_loss = gmm(z).sum().clone()


            #------------ Compute the overall total loss -------------#


            # Compute the total loss. The output is a tensor containing
            # a single value
            total_loss = recon_loss + gmm_loss


            #--------------- Propagate the total loss ----------------#


            # Propagate the loss backward
            total_loss.backward()


            #---------------------- Take a step ----------------------#


            # Update the best average for the current epoch
            rep_avg_loss[epoch] += \
                total_loss.item() / \
                (n_samples * n_genes * n_reps_per_mix_comp)
        
        # Take an optimization step
        best_rep_optimizer.step()

        # Inform the user about the loss at the current epoch
        infostr = f"Epoch {epoch}: loss {rep_avg_loss[epoch]}."
        logger.info(infostr)

        # If we reached the last epoch
        if epoch == config_opt2["epochs"]-1:


            #-------------------- Loss data frame --------------------#


            # Create a data frame to store the loss
            df_loss = pd.DataFrame(sample_avg_loss)


            #-------------- Decoder outputs data frame ---------------#


            # Representation layer for the best representations
            best_rep_final = \
                decoder.RepresentationLayer(values = best_rep_layer.z)

            # Create a list to store all samples' indexes
            all_sample_ixs = []

            # Create a list to store the decoder outputs for all
            # samples/representations
            all_dec_out = []

            # For each batch:
            # 'expr' : gene expression for all samples in the batch
            # 'mean_expr' : mean gene expression for all samples
            #               in the batch
            # 'sample_ixs' : the indexes of the samples in the batch
            for expr, mean_expr, sample_ixs in dataset:
                
                # Add the indexes of the samples in the batch
                # to the list                
                all_sample_ixs.extend(\
                    sample_ixs.detach().numpy().tolist())

                # Add the decoder outputs for the samples in the
                # batch to the list
                all_dec_out.extend(\
                    dec(best_rep_final(sample_ixs)).detach(\
                            ).numpy().tolist())

            # Get a data frame containing the decoder outputs
            # for all samples, associating to each of them
            # the unique index of the sample it comes from
            df_dec_out = pd.DataFrame(all_dec_out)
            df_dec_out.index = indexes

            # Re-index also the loss data frame
            df_loss.index = indexes


            #--------------- Representations dataframe ---------------#


            # Create a data frame for the representations,
            # associating to each of them the unique index
            # of the sample it comes from
            df_rep = pd.DataFrame(best_rep_final.z.detach().numpy())
            df_rep.index = indexes

            # Return the data frames
            return (df_loss, df_rep, df_dec_out)


#----------------------- Probability densities -----------------------#


def get_probability_density(gmm,
                            df_rep,
                            df_tissues):
    """Given a trained Gaussian mixture model and a set of
    representations, get the probability density of each
    component for each representation and the representation(s)
    having the maximum probability density for each component.

    Parameters
    ----------
    gmm : ``DGDPerturbations.src.GMM.GaussianMixtureModel``
        The trained Gaussian mixture model.

    df_rep : ``pandas.DataFrame``
        A data frame containing the representations.

    df_tissues : ``pandas.DataFrame``
        A data frame containing the labels of the tissues the
        samples from which the representations come from
        belong to.

    Returns
    -------
    ``tuple``
        A tuple containing:

        * A ``pandas.DataFrame`` containing the probability
          densities for each representation, together with an
          indication of what the maximum probability density found is
          and for which component is was found, and the label of the
          tissue the original sample belongs to.
        * A ``pandas.DataFrame`` containing, for each component,
          the representation(s) having the maximum probability
          density for the component, together with the probability
          density for that(those) representation(s) and the label(s)
          of the tissue(s) the original sample(s) belong(s) to.
    """
    
    # Get the probability densities of the representations
    # for each component
    probs_values = gmm.sample_probs(x = torch.Tensor(df_rep.values))

    # Convert the result into a data frame
    df_prob_rep = pd.DataFrame(probs_values.detach().numpy())

    # Add a column storing the highest probability density
    # per representation
    df_prob_rep[_MAX_PROB_COL] = df_prob_rep.max(axis = 1)

    # Add a column storing which component has the highest
    # probability density per representation
    df_prob_rep[_MAX_PROB_COMP_COL] = df_prob_rep.idxmax(axis = 1)
    
    # Match the representations with the tissue labels
    df_prob_rep[_TISSUE_COL] = df_tissues

    # Initialize an empty list to store the rows containing
    # the representations/samples that have the highest
    # probability density for each component 
    rows_with_max = []

    # For each component for which at least one representation
    # had maximum probability density
    for comp in df_prob_rep[_MAX_PROB_COMP_COL].unique():

        # Get only those rows corresponding to the current
        # component under consideration
        sub_df = \
            df_prob_rep.loc[df_prob_rep[_MAX_PROB_COMP_COL] == comp]

        # Get the sample with maximum probability for the
        # component (using max() instead of idxmax() because
        # it does not preserve numbers in scientific notation,
        # possibly because it returns a Series with a different
        # data type)
        max_for_comp = \
            sub_df.loc[sub_df[_MAX_PROB_COL] == \
                       sub_df[_MAX_PROB_COL].max()].copy()

        # If more than one representation/sample has the
        # highest probability density
        if len(max_for_comp) > 1:

            # Warn the user so that they are aware of it
            warnstr = \
                f"Multiple representations have the highest " \
                f"probability density for component {comp} " \
                f"({sub_df[_MAX_PROB_COL].max()}): " \
                f"{', '.join(max_for_comp.index.tolist())}."
            logger.warning(warnstr)
        
        # Add a column storing the representation/sample
        # unique index
        max_for_comp[_SAMPLE_IDX_COL] = max_for_comp.index

        # The new index will be the component number
        max_for_comp = max_for_comp.set_index(_MAX_PROB_COMP_COL)
        
        # Append the data frame to the list of data frames
        rows_with_max.append(max_for_comp)

    # Concatenate the data frames
    df_prob_comp = pd.concat(rows_with_max, axis = 0)

    # Return the two data frames
    return df_prob_rep, df_prob_comp