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
from scipy.stats import nbinom
from sklearn.decomposition import PCA
from statsmodels.stats.multitest import multipletests
from torch.utils.data import DataLoader
import torch
# bulkDGD
from bulkDGD.core import (
    dataclasses,
    decoder,
    latent,
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


# The directory containing the configuration files specifying the
# DGD model's parameters and the files containing the trained
# model
CONFIG_MODEL_DIR = \
    resource_filename(Requirement(pkg_name),
                      "configs/model")

# The directory containing the configuration files specifying the
# options for data loading and optimization when finding the
# best representations
CONFIG_REP_DIR = \
    resource_filename(Requirement(pkg_name),
                      "configs/representations")

# Default PyTorch file containing the trained Gaussian mixture model
GMM_FILE = \
    resource_filename(Requirement(pkg_name),
                      "data/model/gmm.pth")

# Default PyTorch file containing the trained decoder
DEC_FILE = \
    resource_filename(Requirement(pkg_name),
                      "data/model/dec.pth")

# File containing the Ensembl IDs of the genes on which the DGD
# model has been trained
DGD_GENES_FILE = \
    resource_filename(Requirement(pkg_name),
                      "data/model/training_genes.txt")

# Available optimizers
AVAILABLE_OPTIMIZERS = ["adam"]

# List of available priors for the means of the Gaussian mixture
# model
AVAILABLE_MEAN_PRIORS = ["softball"]


#------------------------ Preprocess samples -------------------------#


def preprocess_samples(df_samples):
    """Preprocess new samples.

    Parameters
    ----------
    df_samples : ``pandas.DataFrame``
        A data frame containing the samples to be preprocessed.

    Returns
    -------
    df_preproc : ``pandas.DataFrame``
        The data frame with the preprocessed samples.

    genes_excluded : ``list``
        The list of genes found in the input data frame but not
        belonging to the gene set used to train the DGD model.

        These genes are dropped from the ``df_preproc``
        data frame.

    genes_missing : ``list``
        The list of genes present in the gene set used to train
        the DGD model but not found in the input data frame.

        These genes are added with a count of 0 for all samples
        in the ``df_preproc`` data frame.
    """


    #------------------------- Other columns -------------------------#


    # Get the names of the columns containing gene expression data
    # from the original data frame
    genes_columns_old = \
        [col for col in df_samples.columns if col.startswith("ENSG")]

    # Get the Ensembl IDs without the version
    genes_columns = [col.split(".")[0] for col in genes_columns_old]

    # Rename the columns containing gene expression data
    df_samples = \
        df_samples.rename(\
            mapper = dict(zip(genes_columns_old, genes_columns)),
            axis = 1)

    # Get the names of the other columns
    other_columns = \
        [col for col in df_samples.columns if col not in genes_columns]

    # If additional columns were found
    if other_columns:

        # Inform the user of the other columns found
        infostr = \
            f"{len(other_columns)} column(s) containing additional " \
            "information (not gene expression data) were found in " \
            f"the input data frame: {', '.join(other_columns)}."
        logger.info(infostr)


    #------------------------ Check the genes ------------------------#

    # Inform the user that we are looking for duplicate genes
    infostr = "Now looking for duplicated genes..."
    logger.info(infostr)

    # If there are duplicate genes
    if len(genes_columns) > len(set(genes_columns)):
        
        # Get the duplicate genes
        genes_series = pd.Series(genes_columns)
        duplicated_genes = \
            genes_series[genes_series.duplicated()].tolist()

        # Raise a warning informing the user of the duplicated
        # genes
        warnstr = \
            "Duplicated genes were found in the input data " \
            "frame. They will be averaged. The genes are: " \
            f"{', '.join(duplicated_genes)}."
        logger.warning(warnstr)
    else: 
        # Inform the user that no duplicated genes were found
        infostr = "No duplicated genes were found."
        logger.info(infostr)

    # average genes with same name
    df_samples = df_samples.groupby(df_samples.columns, axis=1).mean()


    #----------------------- Duplicate samples -----------------------#


    # Inform the user that we are about to perform a check on
    # duplicated samples
    infostr = "Now looking for duplicated samples..."
    logger.info(infostr)

    # Get duplicate samples, if any
    duplicated_samples = df_samples.duplicated(keep = "first")

    # If duplicate samples were found
    if duplicated_samples.any():

        # Warn the user that the duplicates will be removed
        warnstr = \
            f"{duplicated_samples.values.sum()} duplicated samples " \
            "were found in the data frame at indexes. These " \
            "duplicates will be removed from the data frame, and " \
            "only the first instance of the duplicate row will be " \
            "kept."
        logger.warning(warnstr)

        # Remove the duplicate samples
        df_samples = df_samples[~duplicated_samples]

    # Otherwise
    else:

        # Inform the user that no duplicate samples were found
        infostr = "No duplicated samples were found."
        logger.info(infostr)


    #------------------------ Missing values -------------------------#


    # Inform the user that we are about to perform a check on
    # missing values
    infostr = \
        "Now looking for missing values in the columns containing " \
        "gene expression data..."
    logger.info(infostr)

    # Get samples containing missing values in the columns containing
    # gene counts
    na_samples = df_samples[genes_columns].isnull().any(axis = 1)

    # If there are samples containing missing values
    if na_samples.any():

        # Warn the user of the samples containing missing values 
        warnstr = \
            f"{na_samples.values.sum()} samples with missing values " \
            "in the columns containing gene expression data were " \
            "found. They will be removed from the data frame of " \
            "preprocessed samples."
        logger.warning(warnstr)

        # Remove the samples with missing values
        df_samples = df_samples[~na_samples]

    # Otherwise
    else:

        # Inform the user that no NA values were found in the
        # columns containing gene expression data
        infostr = \
            "No missing values were found in the columns containing " \
            "gene expression data."
        logger.info(infostr)


    #------------------------- Final columns -------------------------#


    # Load the list of genes used to train the DGD model
    genes_list_dgd = misc.get_list(list_file = DGD_GENES_FILE)

    # Warn the user that the genes' columns were rearranged
    infostr = \
        "In the data frame containing the pre-processed samples, " \
        "the columns containing gene expression data will be " \
        "ordered according to the list of genes used to train " \
        f"the DGD model (which can be found in '{DGD_GENES_FILE}')."
    logger.info(infostr)

    # Warn the user that the other columns were rearranged
    infostr = \
        "In the data frame containing the pre-processed samples, " \
        "the columns found in the input data frame which did not " \
        "contain gene expression data, if any were present, " \
        "will appended as the last columns of the data frame, " \
        "and appear in the same order as they did in the input data " \
        "frame."
    logger.info(infostr)

    # Select only the genes used to train the DGD model
    # from the samples' data frame to obrain the data frame
    # containing the preprocessed samples. Sort the columns
    # (= genes) in the order expected by the DGD model, and,
    # if no data were found for some genes, add a default
    # count of 0
    df_preproc = df_samples.reindex(genes_list_dgd + other_columns,
                                    axis = 1,
                                    fill_value = 0)


    #-------------------- Excluded/missing genes ---------------------#


    # Create a list containing the genes present in the original
    # data frame but not in the list of genes on which the DGD
    # model was trained on. Use lists instead of sets (which
    # would be faster) to preserve the order
    genes_excluded = \
        [gene for gene in genes_columns if gene not in genes_list_dgd]

    # If some genes to be excluded were found
    if genes_excluded:

        # Warn the user
        warnstr = \
            f"{len(genes_excluded)} genes found in the input " \
            "samples are not part of the set of genes used to " \
            "train the DGD model. They will be removed from the " \
            "data frame of preprocessed samples."
        logger.warning(warnstr)

    # Otherwise
    else:

        # Inform the user that no genes to be excluded were found
        infostr = \
            "All genes found in the input samples are part of the " \
            "set of genes used to train the DGD model."
        logger.info(infostr)

    # If too few genes (<10%) are left after removing the genes which
    # were not used during training
    if (len(genes_columns) - len(genes_excluded)) / len(genes_columns) < 0.1:

        # Raise an error
        errstr = \
            f"Too many genes in the input data frame were not used " \
            f"during training. Only {len(genes_columns) - len(genes_excluded)} " \
            f"genes out of {len(genes_columns)} were used during " \
            f"training. Please check that the input data frame " \
            f"contains the correct genes."
        raise ValueError(errstr)

    # Create a list containing the genes present in the list of genes
    # used to train the DGD model but not in the original data frame.
    # Use lists instead of sets (which would be faster) to preserve
    # the order
    genes_missing = \
        [gene for gene in genes_list_dgd if gene not in genes_columns]

    # If genes with missing counts were found
    if genes_missing:

        # Warn the user
        warnstr = \
            f"{len(genes_missing)} genes in the set of genes used " \
            "to train the DGD model were not found in the input " \
            "samples. A default count of 0 will be assigned to " \
            "them in all preprocessed samples."
        logger.warning(warnstr)

    # Otherwise
    else:

        # Inform the user that no genes with missing counts were
        # found
        infostr = \
            "All genes used to train the DGD model were found " \
            "in the input samples."
        logger.info(infostr)


    # Return the data frame with the preprocessed samples and the
    # two lists
    return df_preproc, genes_excluded, genes_missing


#----------------------------- Load data -----------------------------#


def load_samples(csv_file,
                 sep = ",",
                 keep_samples_names = True):
    """Load the data frame containing the gene expression data for
    the samples.

    Parameters
    ----------
    csv_file : `str``
        A CSV file containing the samples' data.

        The rows of the data frame should represent the samples,
        while the columns should represent the genes and any
        additional information about the samples.

    sep : ``str``, ``","``
        The column separator in the input CSV file.

    keep_samples_names : ``bool``, default: ``True``load_samples
        Whether to keep the names/IDs/indexes assigned to the
        samples in the input data frame, if any were found.

        If ``True``, the names/IDs/indexes are assumed to be in
        the first column of the input data frame.

    Returns
    -------
    df_expr_data : ``pandas.DataFrame``
        A data frame containing the gene expression data.

        Here, the rows represent the samples and the columns
        represent the genes.

    df_other_data : ``pandas.DataFrame``
        A data frame containing the additional information
        about the samples.

        Here, the rows represent the samples and the columns
        represent any additional information about the samples found
        in the input data frame.
    """


    #---------------------- Load the data frame ----------------------#


    # If we need to keep the original samples' names
    if keep_samples_names:
        
        # Load the data frame assuming the samples' names
        # are the data frame's rows' names (= index)
        df = pd.read_csv(csv_file,
                         sep = sep,
                         index_col = 0,
                         header = 0)

    # Otherwise
    else:

        # Load the data frame assuming there is no index
        df = pd.read_csv(csv_file,
                         sep = sep,
                         index_col = False,
                         header = 0)

        # Inform the user that numeric indexes will be used
        infostr = \
            "Since 'keep_samples_names = False', the samples " \
            "will be identifies using integer indexes starting " \
            "from 0."
        logger.info(infostr)

        # Set the indexes to numeric indexes
        df.index = range(len(df))


    #--------------------- Split the data frame ----------------------#


    # Get the names of the columns containing gene expression data
    # from the original data frame
    genes_columns = \
        [col for col in df.columns if col.startswith("ENSG")]

    # Get the names of the other columns
    other_columns = \
        [col for col in df.columns if col not in genes_columns]

    # If additional columns were found
    if other_columns:

        # Inform the user of the other columns found
        infostr = \
            f"{len(other_columns)} column(s) containing additional " \
            "information (not gene expression data) were found in " \
            f"the input data frame: {', '.join(other_columns)}."
        logger.info(infostr)

    # Create a data frame with only those columns containing gene
    # expression data    
    df_expr_data = df.loc[:,genes_columns]

    # Create a data frame with only those columns containing
    # additional information    
    df_other_data = df.loc[:,other_columns]


    #-------------------- Return the data frames ---------------------#


    # Inform the user about the number of samples found in the dataset
    logger.info(\
        f"{len(df_expr_data)} sample(s) were found in the dataset.")

    # Return the data
    return df_expr_data, df_other_data


def load_representations(csv_file,
                         sep = ","):
    """Load the representations from a CSV file.

    Parameters
    ----------
    csv_file : ``str``
        A CSV file containing a data frame with the representations.

        Each row should contain a representation. The columns should
        contain the representation's values along the latent
        space's dimensions and additional informations about
        the representations, if present (for instance, the loss
        associated with each representation).

    sep : ``str``, ``","``
        The column separator in the input CSV file.

    Returns
    -------
    df_rep_data : ``pandas.DataFrame``
        A data frame containing the representations'
        values along the latent space's dimensions.

        Here, each row contains a representation and the columns
        contain the representations' values along the latent
        space's dimensions.

    df_other_data : ``pandas.DataFrame``
        A data frame containing the additional information about
        the representations found in the input data frame.

        Here, each row contains a representation and the columns
        contain additional information about the representations
        provided in the input data frame.
    """


    #---------------------- Load the data frame ----------------------#


    # Load the data frame with the representations
    df = pd.read_csv(csv_file,
                     sep = sep,
                     index_col = 0,
                     header = 0)


    #--------------------- Split the data frame ----------------------#


    # Get the names of the columns containing the values of
    # the representations along the latent space's dimensions
    latent_dims_columns = \
        [col for col in df.columns if col.startswith("latent_dim_")]

    # Get the names of the other columns
    other_columns = \
        [col for col in df.columns if col not in latent_dims_columns]

    # If additional columns were found
    if other_columns:

        # Inform the user of the other columns found
        infostr = \
            f"{len(other_columns)} column(s) containing additional " \
            "information (not values of the representations along " \
            "the latent space's dimensions) were found in " \
            f"the input data frame: {', '.join(other_columns)}."
        logger.info(infostr)

    # Return a data frame with the representations' values and
    # another with the extra information
    return df[latent_dims_columns], df[other_columns]


def load_decoder_outputs(csv_file,
                         sep = ","):
    """Load the decoder outputs from a CSV file.

    Parameters
    ----------
    csv_file : ``str``
        A CSV file containing a data frame with the decoder outputs.

        Each row should represent the decoder output for a given
        representation, while each columns should contain either the
        values of the decoder output or additional information
        about the decoder outputs.

    sep : ``str``, ``","``
        The column separator in the input CSV file.

    Returns
    -------
    df_dec_data : ``pandas.DataFrame``
        A data frame containing the decoder outputs.

        Here, each row contains the decoder output for a given
        representation. and the columns contain the values
        of the decoder output.

    df_other_data : ``pandas.DataFrame``
        A data frame containing additional information about
        the decoder outputs found in the input data frame (for
        instance, the tissues from which the original samples
        came from).

        Here, each row contains the decoder output for a given
        representations and the columns contain additional
        information about the decoder outputs provided in the
        input data frame.
    """


    #---------------------- Load the data frame ----------------------#


    # Load the data frame with the decoder outputs
    df = pd.read_csv(csv_file,
                     sep = sep,
                     index_col = 0,
                     header = 0)


    #--------------------- Split the data frame ----------------------#


    # Get the names of the columns containing the decoder outputs
    # for the genes
    dec_out_columns = \
        [col for col in df.columns if col.startswith("ENSG")]

    # Get the names of the other columns
    other_columns = \
        [col for col in df.columns if col not in dec_out_columns]

    # If additional columns were found
    if other_columns:

        # Inform the user of the other columns found
        infostr = \
            f"{len(other_columns)} column(s) containing additional " \
            "information (not the decoder outputs for each " \
            "gene) were found in the input data frame: " \
            f"{', '.join(other_columns)}."
        logger.info(infostr)

    # Return a data frame with the decoder outputs and
    # another with the extra information
    return df[dec_out_columns], df[other_columns]


def save_samples(df,
                 csv_file,
                 sep = ","):
    """Save the samples to a CSV file.

    Parameters
    ----------
    df : ``pandas.DataFrame``
        A data frame containing the samples.

    csv_file : ``str``
        The output CSV file.

    sep : ``str``, ``","``
        The column separator in the output CSV file.
    """

    # Save the representations
    df.to_csv(csv_file,
              sep = sep,
              index = True,
              header = True)


def save_representations(df,
                         csv_file,
                         sep = ","):
    """Save the representations to a CSV file.

    Parameters
    ----------
    df : ``pandas.DataFrame``
        A data frame containing the representations.

    csv_file : ``str``
        The output CSV file.

    sep : ``str``, ``","``
        The column separator in the output CSV file.
    """

    # Save the representations
    df.to_csv(csv_file,
              sep = sep,
              index = True,
              header = True)


def save_decoder_outputs(df,
                         csv_file,
                         sep = ","):
    """Save the decoder outputs to a CSV file.

    Parameters
    ----------
    df : ``pandas.DataFrame``
        A data frame containing the decoder outputs.

    csv_file : ``str``
        The output CSV file.

    sep : ``str``, ``","``
        The column separator in the output CSV file.
    """

    # Save the decoder outputs
    df.to_csv(csv_file,
              sep = sep,
              index = True,
              header = True)


#----------------------- Initialize the model ------------------------#


def get_gmm(config):
    """Initialize the Gaussian mixture model, load its trained
    parameters, and return it.

    Parameters
    ----------
    config : ``dict``
        A dictionary containing the options for initializing
        the Gaussian mixture model and loading its trained
        parameters.

    Returns
    -------
    gmm : ``bulkDGD.core.latent.GaussianMixtureModel``
        The trained Gaussian mixture model.
    """


    #----------------- Get the prior over the means ------------------#


    # Get the type of prior
    prior_type = config["mean_prior"]["type"]

    # Get the configuration for the prior
    config_prior = config["mean_prior"]["options"]

    # Get the PyTorch file
    pth_file = config["pth_file"]

    # If the prior is a softball prior
    if prior_type == "softball":

        # Initialize the prior
        mean_prior = \
            priors.SoftballPrior(dim = config["options"]["dim"],
                                 **config_prior)

    # Otherwise
    else:

        # Raise an error
        errstr = \
            f"Invalid prior type '{prior_type}'. Available " \
            f"types are: {', '.join(AVAILABLE_OPTIMIZERS)}."
        raise ValueError(errstr)


    #---------------------- Initialize the GMM -----------------------#


    # Try to initialize the GMM
    try:

        gmm = latent.GaussianMixtureModel(mean_prior = mean_prior,
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


    #------------------ Load the GMM's paramenters -------------------#


    # Try to load the model's parameters
    try:

        gmm.load_state_dict(torch.load(pth_file))

    # If something went wrong
    except Exception as e:

        # Raise an error
        errstr = \
            f"It was not possible to load the Gaussian " \
            f"mixture model's parameters from '{pth_file}'. " \
            f"Error: {e}"
        raise Exception(errstr)

    # Inform the user that the GMM's parameters was successfully
    # loaded
    infostr = \
        f"The Gaussian mixture model's parameters was  " \
        f"successfully loaded from '{pth_file}'."
    logger.info(infostr)

    # Return the model
    return gmm


def get_decoder(config):
    """Initialize the decoder, load its trained parameters,
    and return it.

    Parameters
    ----------
    config : ``dict``
        A dictionary containing the options for initializing
        the decoder and loading its trained parameters.

    Returns
    -------
    dec : ``bulkDGD.core.decoder.Decoder``
        The trained decoder.
    """


    #-------------------- Initialize the decoder ---------------------#


    # Get the PyTorch file
    pth_file = config["pth_file"]

    # Try to initialize the decoder
    try:
        
        dec = decoder.Decoder(**config["options"])

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


    #----------------- Load the decoder's parameters -----------------#


    # Try to load the model's parameters
    try:

        dec.load_state_dict(torch.load(pth_file))

    # If something went wrong
    except Exception as e:

        # Raise an error
        errstr = \
            f"It was not possible to load the decoder's " \
            f"parameters from '{pth_file}'. Error: {e}"
        raise Exception(errstr)

    # Inform the user that the decoder's parameters were successfully
    # loaded
    infostr = \
        f"The decoder's parameters were successfully " \
        f"loaded from '{pth_file}'."
    logger.info(infostr)

    # Return the model
    return dec


def get_model(config_gmm,
              config_dec):
    """Get the trained DGD model (trained Gaussian mixture model
    and the trained decoder).

    Parameters
    ----------
    config_gmm : ``dict``
        A dictionary containing the options for initializing
        the Gaussian mixture model and loading its state.

    config_dec : ``dict``
        A dictionary containing the options for initializing
        the decoder and loading its state.

    Returns
    -------
    gmm : ``bulkDGD.core.latent.GaussianMixtureModel``
        The Gaussian mixture model.

    dec : ``bulkDGD.core.decoder.Decoder``
        The trained decoder.
    """


    #---------------- Get the Gaussian mixture model -----------------#


    # Get the trained Gaussian mixture model
    gmm = get_gmm(config = config_gmm)


    #------------------------ Get the decoder ------------------------#


    # Get the trained decoder
    dec = get_decoder(config = config_dec)

    # Return the GMM and the decoder
    return gmm, dec


#-------------------------- Representations --------------------------#


def get_representations(df,
                        gmm,
                        dec,
                        n_rep_per_comp,
                        config_data,
                        config_opt1,
                        config_opt2):
    """Initialize and optimize the representations.

    Parameters
    ----------
    df : ``pandas.DataFrame``
        A data frame containing the samples.

    gmm : ``bulkDGD.core.latent.GaussianMixtureModel``
        The trained Gaussian mixture model.

    dec : ``bulkDGD.core.decoder.Decoder``
        The trained decoder.

    n_rep_per_comp : ``int``
        The number of new representations to be taken
        per component per sample.

    config_data : ``dict``
        A dictionary of options to load the data.
    
    config_opt1 : ``dict``
        A dictionary of options for the initial optimization.

    config_opt2 : ``dict``
        A dictionary of options for the second optimization.

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
    """
    from datetime import datetime

    # Get the columns containing gene expression data
    genes_columns = \
        [col for col in df.columns if col.startswith("ENSG")]

    # Get the other columns
    other_columns = \
        [col for col in df.columns if col not in genes_columns]

    # Keep only the columns containing gene expression data
    df_expr_data = df[genes_columns]

    # Get a data frame with only the columns containing additional
    # data
    df_other_data = df[other_columns]

    # Create the dataset
    dataset = dataclasses.GeneExpressionDataset(df = df_expr_data)

    # Create the data loader
    dataloader = DataLoader(dataset, **config_data)

    # Get the number of samples and genes in the dataset from
    # the input data frame's shape
    n_samples, n_genes = df_expr_data.shape

    # Get the samples' names in the original order
    ordered_samples_names = df_expr_data.index

    # Create a dictionary mapping each sample's unique numeric
    # index in the DataLoader to its name
    ixs2names = dict(zip(range(len(ordered_samples_names)),
                         ordered_samples_names))

    # Get the dimensionality of the latent space
    dim = gmm.dim


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
    #                  'n_comp' * 
    #                  'n_rep_per_comp'
    #
    # - 2nd dimension: the dimensionality of the Gaussian mixture
    #                  model ->
    #                  'dim'
    rep_init_values = \
        gmm.sample_new_points(n_points = n_samples, 
                              sampling_method = "mean",
                              n_samples_per_comp = n_rep_per_comp)

    # Initialize the representation layer with 'dim'
    # dimensions and 'n_samples' samples that have values
    # 'rep_init_values'
    rep_samples_layer = \
        latent.RepresentationLayer(values = rep_init_values)


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
    n_comp = gmm.n_comp

    # Initialize an empty list to store the average loss for each
    # epoch.
    # The lenght of the list is equal to the number of epochs
    rep_avg_loss = [0] * config_opt1["epochs"]


    #-------------------- Start the optimization ---------------------#


    # Inform the user that the optimization is starting
    infostr = "Starting the first optimization..."
    logger.info(infostr)
    print("First opt: " + datetime.now().strftime("%H:%M:%S"))

    # For each epoch
    for epoch in range(1, config_opt1["epochs"]+1):

        # Make the gradients zero
        rep_samples_optimizer.zero_grad()

        # Get the representations' values from the representation
        # layer. The representations are stored in a 2D tensor with:
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
        z_all = rep_samples_layer.z

        # For each batch:
        # 'expr' : the gene expression for all samples in the batch
        # 'mean_expr' : the mean gene expression for all samples
        #               in the batch
        # 'sample_ixs' : the numeric indexes of the samples in
        #                the batch
        for expr, mean_expr, sample_ixs in dataloader:

            # Get the number of samples in the batch
            n_samples_in_batch = len(sample_ixs)

            #------------ Initialize the representations -------------#


            # Reshape the tensor containing the representations. The
            # output is a 4D tensor with:
            #
            # - 1st dimension: the number of samples in the current
            #                  batch ->
            #                  'n_samples_in_batch'
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
                              dim)[sample_ixs]

            # Reshape it again. The output is a 2D tensor with:
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
            # - 2nd dimension: the dimensionality of the Gaussian
            #                  mixture model ->
            #                  'dim'
            z = z_4d.view(n_samples_in_batch * \
                            n_rep_per_comp * \
                            n_comp,
                          dim)

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
            #                  'n_comp' *
            #                  'n_rep_per_comp'
            #
            # - 2nd dimension: the dimensionality of the output
            #                  (= gene) space ->
            #                  'n_genes'
            dec_out = dec(z)


            #------------ Compute the reconstruction loss ------------#


            # Get the observed gene counts and "expand" it to match
            # the shape required to compute the loss. The output is
            # a 4D tensor with:
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
            obs_count = \
                expr.unsqueeze(1).unsqueeze(1).expand(\
                    -1,
                    n_rep_per_comp,
                    n_comp,
                    -1)

            # Get the scaling factors for the mean of each negative
            # binomial and reshape it so that it matches the shape
            # required to compute the loss. The output is a 4D
            # tensor with:
            #
            # - 1st dimension: the number of samples in the current
            #                  batch -> 'n_samples_in_batch'
            #
            # - 2nd dimension: 1
            #
            # - 3rd dimension: 1
            #
            # - 4th dimension: 1
            mean_scaling_factor = \
                decoder.reshape_scaling_factor(mean_expr,
                                               4)

            # Reshape the decoded output to match the shape required
            # to compute the loss. The output is a 4D tensor with:   
            #
            # - 1st dimension: the number of samples in the current
            #                  batch -> 'n_samples_in_batch'
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
            pred_mean = dec_out.view(n_samples_in_batch,
                                     n_rep_per_comp,
                                     n_comp,
                                     n_genes)

            # Get the reconstruction loss. The output is a 4D tensor
            # with:
            #
            # - 1st dimension: the number of samples in the current
            #                  batch ->
            #                  'n_samples_in_batch'
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
                dec.nb.loss(obs_count = obs_count,
                            scaling_factor = mean_scaling_factor,
                            pred_mean = pred_mean)

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
            #                  'n_comp' *
            #                  'n_rep_per_comp'
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
            rep_avg_loss[epoch-1] += \
                total_loss.item() / (n_samples * n_genes * \
                                     n_comp * n_rep_per_comp)


        #------------------------ Take a step ------------------------#


        # Take an optimization step
        rep_samples_optimizer.step()

        # Inform the user about the loss at the current epoch
        infostr = f"Epoch {epoch}: loss {rep_avg_loss[epoch-1]}."
        logger.info(infostr)
        print(infostr)

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

    # Get the representations' values from the representation
    # layer. The representations are stored in a 2D tensor with:
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
    z_all_after_optim1 = rep_samples_layer.z
    
    # For each batch:
    # 'expr' : the gene expression for all samples in the batch
    # 'mean_expr' : the mean gene expression for all samples
    #               in the batch
    # 'sample_ixs' : the numeric indexes of the samples
    #                in the batch
    for expr, mean_expr, sample_ixs in dataloader:

        # Re-initialize the number of samples in the current batch
        # since we are looping again over the dataset
        n_samples_in_batch = len(sample_ixs)

        # 'z_all_after_optim1' is a 2D tensor with:
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

        # Reshape the tensor containing the representations. The
        # output is a 4D tensor with:
        #
        # - 1st dimension: the number of samples in the current
        #                  batch ->
        #                  'n_samples_in_batch'
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
            z_all_after_optim1.view(n_samples,
                                    n_rep_per_comp,
                                    n_comp,
                                    dim)[sample_ixs]

        # Reshape it again. The output is a 2D tensor with:
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
        z = z_4d.view(n_samples_in_batch * \
                        n_rep_per_comp * \
                        n_comp,
                      dim)

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
        #                  'n_comp' *
        #                  'n_rep_per_comp'
        #
        # - 2nd dimension: the dimensionality of the output
        #                  (= gene) space ->
        #                  'n_genes'
        dec_out = dec(z)


        #---------- Compute the overall reconstruction loss ----------#


        # Get the observed counts for the expression of each gene in
        # each sample, and "expand" it to match the shape required
        # to compute the loss. The output is a 4D tensor with:
        #
        # - 1st dimension: the number of samples in the current
        #                  batch -> 'n_samples_in_batch'
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
        obs_count = \
            expr.unsqueeze(1).unsqueeze(1).expand(\
                -1,
                n_rep_per_comp,
                n_comp,
                -1)

        # Get the scaling factors for the mean of each negative
        # binomial and reshape it so that it matches the shape
        # required to compute the loss. The output is a 4D
        # tensor with:
        #
        # - 1st dimension: the number of samples in the current
        #                  batch -> 'n_samples_in_batch'
        #
        # - 2nd dimension: 1
        #
        # - 3rd dimension: 1
        #
        # - 4th dimension: 1
        mean_scaling_factor = \
            decoder.reshape_scaling_factor(mean_expr,
                                           4)

        # Reshape the decoded output to match the shape required
        # to compute the loss. The output is a 4D tensor with:   
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
        pred_mean = dec_out.view(n_samples_in_batch,
                                 n_rep_per_comp,
                                 n_comp,
                                 n_genes)
 
        # Get the reconstruction loss (rescale based on the mean
        # expression of the genes in the samples in the batch).
        # The output is a 4D tensor with:
        #
        # - 1st dimension: the number of samples in the current
        #                  batch ->
        #                  'n_samples_in_batch'
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
        recon_loss = dec.nb.loss(obs_count = obs_count,
                                 scaling_factor = mean_scaling_factor,
                                 pred_mean = pred_mean)

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
        #                  'n_rep_per_comp'
        #
        # - 3rd dimension: the number of components in the Gaussian
        #                  mixture model ->
        #                  'n_comp'
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
        #                  'n_comp' *
        #                  'n_rep_per_comp'
        recon_loss_sum_reshaped = \
            recon_loss_sum.view(n_samples_in_batch * \
                                n_rep_per_comp * \
                                n_comp)

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
        #                  'n_comp' *
        #                  'n_rep_per_comp'
        gmm_loss = gmm(z).clone()


        #-------------- Compute the overall total loss ---------------#


        # Get the total loss. The loss has has many components as the
        # total number of representations computed for the current
        # batch ('n_rep_per_comp' * 'n_comp'
        # representations for each sample in the batch).
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


        # Reshape the tensor containing the total loss. The output
        # is a 2D tensor with.
        #
        # - 1st dimension: the number of samples in the current
        #                  batch -> 'n_samples_in_batch'
        #
        # - 2nd dimension: the number of representations taken
        #                  per component of the Gaussian mixture
        #                  model per sample times the number of
        #                  components -> 
        #                  'n_rep_per_comp' * 'n_comp'
        total_loss_reshaped = \
            total_loss.view(n_samples_in_batch,
                            n_rep_per_comp * n_comp)

        # Get the best representation for each sample in the
        # current batch. The output is a 1D tensor with:
        #
        # - 1st dimension: the number of samples in the current
        #                  batch -> 'n_samples_in_batch'
        best_rep_per_sample = torch.argmin(total_loss_reshaped,
                                           dim = 1).squeeze(-1)

        # Get the best representations for the samples in the batch
        # from the 'n_rep_per_comp' * 'n_comp'
        # representations taken for each sample.
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
            z.view(n_samples_in_batch,
                   n_rep_per_comp * n_comp,
                   dim)[range(n_samples_in_batch), best_rep_per_sample]


    #---------------------- Second optimization ----------------------#


    # Set a representation layer for the best representations found
    # for all samples
    best_rep_layer = \
        latent.RepresentationLayer(values = rep_new_values)


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
    print("Second opt: " + datetime.now().strftime("%H:%M:%S"))
    
    # For each epoch
    for epoch in range(1, config_opt2["epochs"]+1):

        # Make the gradients zero
        best_rep_optimizer.zero_grad()

        # Initialize the loss for all samples in the batch
        # to 0.
        sample_avg_loss = np.zeros(n_samples)
        
        # For each batch:
        # 'expr' : the gene expression for all samples in the batch
        # 'mean_expr' : the mean gene expression for all samples
        #               in the batch
        # 'sample_ixs' : the numeric indexes of the samples
        #                in the batch
        for expr, mean_expr, sample_ixs in dataloader:

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
                dec.nb.loss(obs_count = expr,
                            scaling_factor = mean_expr,
                            pred_mean = dec_out).sum(-1).clone()


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
                dec.nb.loss(obs_count = expr,
                            scaling_factor = mean_expr,
                            pred_mean = dec_out).sum().clone()

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
            rep_avg_loss[epoch-1] += \
                total_loss.item() / \
                (n_samples * n_genes * n_rep_per_comp)
        
        # Take an optimization step
        best_rep_optimizer.step()

        # Inform the user about the loss at the current epoch
        infostr = f"Epoch {epoch}: loss {rep_avg_loss[epoch-1]}."
        logger.info(infostr)
        print(infostr)

        # If we reached the last epoch
        if epoch == config_opt2["epochs"]:

            # Representation layer for the best representations
            best_rep_final = \
                latent.RepresentationLayer(values = best_rep_layer.z)

            # Create a list to store the samples' names (possibly
            # shuffled since it may be that the data were shuffled
            # when loaded with the DataLoader)
            all_sample_names = []

            # Create a list to store the representations
            all_rep = []

            # Create a list to store the decoder outputs for all
            # samples/representations
            all_dec_out = []

            # For each batch:
            # 'expr' : the gene expression for all samples in the batch
            # 'mean_expr' : the mean gene expression for all samples
            #               in the batch
            # 'sample_ixs' : the numeric indexes of the samples in the
            #                batch
            for expr, mean_expr, sample_ixs in dataloader:
                
                # Add the names of the samples in the batch
                # to the list                
                all_sample_names.extend(\
                    [ixs2names[ix] for ix \
                     in sample_ixs.numpy().tolist()])

                # Get the representations for the samples in the
                # current batch
                rep_batch = best_rep_final(sample_ixs)

                # Add the representations for the current samples
                # to the list
                all_rep.extend(\
                    rep_batch.detach().numpy().tolist())

                # Add the decoder outputs for the samples in the
                # batch to the list
                all_dec_out.extend(\
                    dec(rep_batch).detach().numpy().tolist())


            #-------------- Decoder outputs data frame ---------------#


            # Get a data frame containing the decoder outputs
            # for all samples, associating to each of them
            # the unique names/IDs/indexes of the sample it comes from
            df_dec_out = pd.DataFrame(all_dec_out)
            df_dec_out.index = all_sample_names

            # Re-order the samples' names according to the original
            # samples' order
            df_dec_out = df_dec_out.reindex(ordered_samples_names)

            # Set the names of the columns of the data frame to be the
            # names of the genes
            df_dec_out.columns = df_expr_data.columns

            # Add the extra data found in the input data frame to the
            # decoder outputs' data frame
            df_dec_out[df_other_data.columns] = df_other_data # pd.concat([df_dec_out, df_other_data])


            #-------------- Representations data frame ---------------#


            # Create a data frame for the representations,
            # associating to each of them the unique index
            # of the sample it comes from
            df_rep = pd.DataFrame(all_rep)
            df_rep.index = all_sample_names

            # Re-order the samples also in the representations'
            # data frame
            df_rep = df_rep.reindex(ordered_samples_names)

            # Name the columns of the data frame as the dimensions
            # of the latent space
            df_rep.columns = \
                [f"latent_dim_{i}" for i \
                 in range(1, df_rep.shape[1]+1)]

            # Create a series to store the loss
            series_loss = pd.Series(sample_avg_loss)

            # Re-index also the loss data frame
            series_loss.index = all_sample_names

            # Re-order the samples also in the loss' series
            series_loss = series_loss.reindex(ordered_samples_names)

            # Add the 'loss' column to the data frame (it will
            # be the current last column)
            df_rep["loss"] = series_loss

            # Add the extra data found in the input data frame to the
            # representations' data frame
            df_rep[df_other_data.columns] = df_other_data # pd.concat([df_rep, df_other_data])


            #---------------- Return the data frames -----------------#


            # Return the data frames
            return df_rep, df_dec_out


#---------------------------- Statistics -----------------------------#


def get_r_values(dec):
    """Get the r-values of the negative binomials modeling the
    expression of the genes included in the DGD model from the
    trained decoder.

    Parameters
    ----------
    dec : ``bulkDGD.core.decoder.Decoder``
        The trained decoder.

    Returns
    -------
    r_values : ``torch.Tensor``
        The r-values of the negative binomials modeling the
        expression of the genes (one per gene).

        This is a 1D tensor whose size equals the dimensionality
        of the gene space.
    """

    return torch.exp(dec.nb.log_r).squeeze().detach()


def get_p_values(obs_counts_sample,
                 pred_means_sample,
                 r_values,
                 resolution = 1):
    """Calculate the p-value associated to the predicted mean
    of each negative binomial.

    Parameters
    ----------
    obs_counts_sample : ``torch.Tensor``
        The observed gene counts in a single sample.

        This is a 1D tensor whose length is equal to the
        dimensionality of the gene space.

    pred_mean_sample : ``torch.Tensor``
        The (rescaled) predicted mean of the negative binomial
        modeling the gene counts for a single sample.

        This is a 1D tensor whose length is equal to the
        dimensionality of the gene space.

    r_values : ``torch.Tensor``
        A tensor containing one r-value for each negative binomial
        (= one r-value for each gene).

        This is a 1D tensor whose length is equal to the
        dimensionality of the gene space.

    resolution : ``int``, ``1``
        How accurate the calculation of the p-values should be.

        The ``resolution`` corresponds to the coarseness of the sum
        over the probability mass function of each negative binomial
        to compute the corresponding p-value.

        The lower the ``resolution``, the more accurate the
        calculation of the p-values.

        If ``1`` (the default), the calculation will be exact
        (but it will be more computationally expensive).

    Returns
    -------
    p_values : ``numpy.ndarray``
        A 1D array containing one p-value per gene.
    
    ks : ``numpy.ndarray``
        A 2D array containing the count values at 
        which the probability mass function was evaluated
        to compute the p-values. The array has as many
        rows as the number of genes and as many columns as
        the number of count values.
    
    pmfs : ``numpy.ndarray``
        A 2D array containing the value of the
        probability mass function for each count value
        at which it was evaluated. The array has as many
        rows as the number of genes and as many columns as
        the number of count values.
    """

    # Get the mean gene counts for the sample. The output is
    # a single value
    obs_counts_mean_sum = \
        torch.mean(obs_counts_sample).unsqueeze(-1)

    # Get the rescaled predicted means of the negative binomials
    # (one for each gene). This is a 1D tensor with:
    #
    # - 1st dimension: the dimensionality of the output (= gene)
    #                  space
    pred_means_sample = \
        decoder.NBLayer.rescale(pred_means_sample,
                                obs_counts_mean_sum)

    # Create an empty list to store the p-valued computed per gene
    # in the current sample, the value of the probability mass
    # function, and the 'k'
    results_sample = []

    # For each gene's (rescaled) predicted mean counts, observed
    # counts, and r-value
    for pred_mean_gene_i, obs_count_gene_i, r_value_i \
        in zip(pred_means_sample, obs_counts_sample, r_values):


        #----------------------- Calculate 'p' -----------------------#


        # Calculate the probability of "success" from the r-value
        # (number of successes till the experiment is stopped) and
        # the mean of the negative binomial. This is a single value,
        # and is calculated from the mean 'm' as:
        #
        # m = r(1-p) / p
        # mp = r - rp
        # mp + rp = r
        # p(m+r) = r
        # p = r / (m + r)
        p_i = pred_mean_gene_i.item() / \
              (pred_mean_gene_i.item() + r_value_i.item())


        #-------------- Get the tail value for the sum ---------------#

        
        # Get the count value at which the value of the percent
        # point function (the inverse of the cumulative mass
        # function) is 0.99999. This corresponds to the value in
        # the probability mass function beyond which lies
        # 0.00001 of the mass. This is a single value.
        #
        # Since SciPy's negative binomial function is implemented
        # as function of the number of failures, their 'p' is
        # equivalent to our '1-p' and their 'n' is our 'r'
        tail = nbinom.ppf(q = 0.99999,
                          n = r_value_i.item(),
                          p = 1 - p_i)


        #---------------- Get the probability masses -----------------#


        # If no resolution was passed
        if resolution is None:
            
            # We are going to sum with steps of lenght 1.
            # This is a 1D tensor with length is equal to 'tail',
            # since we are taking steps of size 1 starting
            # from 0 and ending in 'tail'
            k = torch.arange(\
                    start = 0,
                    end = tail,
                    step = 1)
        
        # Otherwise
        else:
            
            # We are going to integrate with steps of length
            # 'resolution'. This is a 1D tensor whose length
            # is euqal to the number of 'resolution'-sized
            # steps between 0 and 'tail'
            k = torch.linspace(\
                    start = 0,
                    end = int(tail),
                    steps = int(resolution)).round().double()

        # Integrate to find the value of the probability mass
        # function for each count value in the 'k' tensor.
        # The output is a 1D tensor whose length is equal to
        # the length of 'k'
        pmf = \
            decoder.NBLayer.log_prob_mass(\
                k = k,
                m = pred_mean_gene_i,
                r = r_value_i).to(torch.float64)


        #---------------------- Get the p-value ----------------------#


        # Find the value of the probability mass function for the
        # actual value of the count for gene 'i', 'obs_count_gene_i'.
        # This is a single value
        prob_obs_count_gene_i = \
            decoder.NBLayer.log_prob_mass(\
                k = obs_count_gene_i,
                m = pred_mean_gene_i,
                r = r_value_i).to(torch.float64)

        # Find the probability that a point falls lower than the
        # observed count (= sum over all values of 'k' lower than
        # the value of the probability mass function at the actual
        # count value. Exponentiate it since for now we dealt with
        # log-probability masses, and we want the actual probability.
        # The output is a single value
        lower_probs = \
            pmf[pmf <= prob_obs_count_gene_i].exp().sum()

        # Get the total mass of the "discretized" probability mass
        # function we computed above
        norm_const = pmf.exp().sum()
        
        # Calculate the p-value as the ratio between the probability
        # mass associated to the event where a point falls lower
        # than the observed count and the total probability mass
        p_val = lower_probs / norm_const

        # Save the p-value found for the current gene, the value of
        # the probability mass function, and the k
        results_sample.append((p_val.item(),
                               k.detach().numpy(),
                               pmf.detach().numpy()))

    # Create three lists containing all p-values, all PMFs, and
    # all 'k' values
    p_values, ks, pmfs = list(zip(*results_sample))
    
    # Return an array of p-values and two arrays containing the PMFs
    # and the 'k' values
    return np.array(p_values), np.stack(ks), np.stack(pmfs)


def get_q_values(p_values,
                 alpha = 0.05,
                 method = "fdr_bh"):
    """Get the q-values associated to a set of p-values. The q-values
    are the p-values adjusted for the false discovery rate.

    Parameters
    ----------
    p_values : ``list`` or ``tuple``
        The p-values.

    alpha : ``float``, ``0.05``
        The family-wise error rate for the calculation of the
        q-values.

    method : ``str``, ``fdr_bh``
        The method used to adjust the p-values. The available
        methods are listed in the documentation for
        ``statsmodels.stats.multitest.multipletests``.

    Returns
    -------
    q_values : ``numpy.ndarray``
        A 1D array containing the q-values (adjusted p-values).

    rejected : ``numpy.ndarray``
        A 1D array containing booleans defining whether a p-value
        in the input data frame was rejected (``True``) or
        not (``False``).
    """

    # Adjust the p-values
    rejected, q_values, _, _ = multipletests(pvals = p_values,
                                             alpha = alpha,
                                             method = method)

    # Return the q-values and the rejected p-values
    return q_values, rejected


def get_log2_fold_changes(obs_counts_sample,
                          pred_means_sample):
    """Get the log2-fold change of the gene expression.

    Parameters
    ----------
    obs_counts_sample : ``torch.Tensor``
        The observed gene counts in a single sample.

        This is a 1D tensor whose length is equal to the
        dimensionality of the gene space.

    pred_mean_sample : ``torch.Tensor``
        The (rescaled) predicted mean of the negative binomial
        modeling the gene counts for a single sample.

        This is a 1D tensor whose length is equal to the
        dimensionality of the gene space.

    Returns
    -------
    ``torch.Tensor``
        The log2-fold change associated to each gene in the
        given sample.

        This is a 1D tensor whose length is equal to the
        dimensionality of the gene space.
    """
    
    # Return the log-fold change for each gene by dividing the
    # predicted mean count by the observed count. A small value
    # (1e-6) is added to ensure we do not divide by zero.
    return torch.log2(\
            (pred_means_sample + 1e-6) / (obs_counts_sample + 1e-6))


def perform_dea(obs_counts_sample,
                pred_means_sample,
                sample_name = None,
                statistics = \
                    ["p_values", "q_values", "log2_fold_changes"],
                genes_names = None,
                r_values = None,
                resolution = 1,
                alpha = 0.05,
                method = "fdr_bh"):
    """Perform differential expression analysis (DEA).

    Parameters
    ----------
    obs_counts_sample : ``torch.Tensor``
        The observed gene counts in a single sample.

        This is a 1D tensor whose length is equal to the
        dimensionality of the gene space.

    pred_mean_sample : ``torch.Tensor``
        The (rescaled) predicted mean of the negative binomial
        modeling the gene counts for a single sample.

        This is a 1D tensor whose length is equal to the
        dimensionality of the gene space.

    statistics : ``list``,
                 {``["p_values", "q_values", "log2_fold_changes"]``}
        The statistics to be computed. By default, all of them
        will be computed (``"p_values"``, ``"q_values"``,
        ``"log2_fold_changes"``).

    genes_names : ``list``, optional
        The names of the genes on which DEA is performed. If provided,
        the genes will be the names of the rows of the output data
        frame. If not, the rows will be indexed starting from 0.

    r_values : ``torch.Tensor``
        A tensor containing one r-value for each negative binomial
        (= one r-value for each gene).

        This is a 1D tensor whose length is equal to the
        dimensionality of the gene space.

    resolution : ``int``, ``1``
        How accurate the calculation of the p-values should be.

        The ``resolution`` corresponds to the coarseness of the sum
        over the probability mass function of each negative binomial
        to compute the corresponding p-value.

        The lower the ``resolution``, the more accurate the
        calculation of the p-values.

        If ``1`` (the default), the calculation will be exact
        (but it will be more computationally expensive).

    alpha : ``float``, ``0.05``
        The family-wise error rate for the calculation of the
        q-values.

    method : ``str``, ``fdr_bh``
        The method used to adjust the p-values. The available
        methods are listed in the documentation for
        ``statsmodels.stats.multitest.multipletests``.

    Returns
    -------
    ``pandas.DataFrame``
        A data frame whose rows represent the genes on which
        DEA was performed, and whose columns contain the statistics
        computed (p-values, q_values, log2-fold changes). If not
        all statistics were computed, the columns corresponding
        to the missing ones will be empty.
    """
    
    # Initialize all the statistics to None
    p_values = None
    q_values = None
    log2_fold_changes = None

    # Set the column names that will be used in the final data frame
    columns_names = ["p_value", "q_value", "log2_fold_change"]


    #--------------------------- p-values ----------------------------#


    # If the user requested the calculation of p-values
    if "p_values" in statistics:

        # If no r-values were passed
        if r_values is None:

            # Raise an error
            errstr = \
                f"'r-values' are needed to compute the " \
                f"p-values."
            raise RuntimeError(errstr)

        # Calculate the p-values
        p_values, ks, pmfs = \
            get_p_values(obs_counts_sample = obs_counts_sample,
                         pred_means_sample = pred_means_sample,
                         r_values = r_values,
                         resolution = resolution)


    #--------------------------- q-values ----------------------------#


    # If the user requested the calculation of q-values
    if "q_values" in statistics:

        # If no p-values were calculated
        if p_values is None:

            # Raise an error
            errstr = \
                f"The calculation of p-values is needed to " \
                f"compute the q-values. This can be done " \
                f"by adding 'p_values' to 'stats'."
            raise RuntimeError(errstr)

        # Calculate the q-values
        q_values, rejected = \
            get_q_values(p_values = p_values)


    #----------------------- log2-fold changes -----------------------#


    # If the user requested the calculation of fold changes
    if "log2_fold_changes" in statistics:

        # Calculate the fold changes
        log2_fold_changes = \
            get_log2_fold_changes(\
                obs_counts_sample = obs_counts_sample,
                pred_means_sample = pred_means_sample)

    # Get the results for the statistics that were computed
    stats_results = \
        [pd.Series(stat) if stat is not None
         else pd.Series()
         for stat in (p_values, q_values, log2_fold_changes)]


    #----------------------- Output data frame -----------------------#


    # Create a data frame from the statistics computed
    df_stats = pd.concat(stats_results,
                         axis = 1)

    # Set the columns' names to the ones defined above
    df_stats.columns = columns_names

    # If the genes' names were passed
    if genes_names is not None:
        
        # The names of the rows of the data frame will be the gene
        # names
        df_stats.index = genes_names

    # Return the data frame
    return df_stats, sample_name


#-------------------------------- PCA --------------------------------#


def perform_2d_pca(df_rep,
                   pc_columns = ["PC1", "PC2"],
                   groups = None,
                   groups_column = "group"):
    """Perform a 2D principal component analysis (PCA) on a set
    of representations.

    Parameters
    ----------
    df_rep : ``pandas.DataFrame``
        Data frame containing the representations. The rows of the
        data frame should represent the samples, while the columns
        should represent the dimensions of the space where the
        representations live.

    pc_columns : ``list``, ``["PC1", "PC2"]``
        A list with the names of the two columns that will contain
        the values of the first two principal components.

    Returns
    -------
    ``pandas.DataFrame``
        A data frame containing the results of the PCA. The rows
        will contain the representations, while the columns
        will  contain the values of each representation's
        projection along the principal components.
    """

    # Get the representations' values
    rep_values = df_rep.values

    # Get the representations' names/Ds
    rep_names = df_rep.index.tolist()

    # Set up the PCA
    pca = PCA(n_components = 2)

    # Fit the model
    pca.fit(rep_values)
    
    # Fit the model and apply the dimensionality reduction
    projected = pca.fit_transform(rep_values)

    # Create a data frame containing the projected points
    df_projected = pd.DataFrame(projected,
                                columns = pc_columns,
                                index = rep_names)

    # Return the data frame
    return df_projected


#----------------------- Probability densities -----------------------#


def get_probability_density(gmm,
                            df_rep):
    """Given a trained Gaussian mixture model and a set of
    representations, get the probability density of each
    component for each representation and the representation(s)
    having the maximum probability density for each component.

    Parameters
    ----------
    gmm : ``bulkDGD.core.latent.GaussianMixtureModel``
        The trained Gaussian mixture model.

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