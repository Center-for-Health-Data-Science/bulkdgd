#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-

#    defaults.py
#
#    General default values.
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
__doc__ = "General default values."


#######################################################################


# Import from the standard library.
import os


#######################################################################


# Set the default directories for the configuration files.
CONFIG_DIRS = \
    {# Set the directory containing the configuration files specifying
     # the DGD model's parameters and, possibly, the files containing
     # the parameters of the trained model.
     "model" : \
        os.path.join(os.path.dirname(__file__),
                     "configs/model"),

     # Set the directory containing the configuration files specifying
     # the options for the optimization round(s) when finding the best
     # representations for a set of samples.
     "representations" : \
        os.path.join(os.path.dirname(__file__),
                     "configs/representations"),

    # Set the directory containing the configuration files specifying
    # the options to generate plots.
    "plotting" : \
        os.path.join(os.path.dirname(__file__),
                     "configs/plotting"),

    # Set the directory containing the configuration files specifying
    # the options for training the model.
    "training" : \
        os.path.join(os.path.dirname(__file__),
                     "configs/training"),

    # Set the directory containing the configuration files specifying
    # the options to create a new list of genes for the bulkDGD model.
    "genes" : \
        os.path.join(os.path.dirname(__file__),
                     "configs/genes"),

    # Set the directory containing the configuration files specifying
    # the options to perform dimensionality reduction analyses.
    "dimensionality_reduction" : \
        os.path.join(os.path.dirname(__file__),
                     "configs/dimensionality_reduction")}

#---------------------------------------------------------------------#

# Set the default configuration files for performing dimensionality
# reduction analyses.
CONFIG_FILES_DIM_RED = \
    {# Set the default configuration file for performing a PCA.
     "pca" : \
        os.path.join(os.path.dirname(__file__),
                     "configs/dimensionality_reduction/pca.yaml"),
     
     # Set the default configuration file for performing a KPCA.
     "kpca" : \
        os.path.join(os.path.dirname(__file__),
                     "configs/dimensionality_reduction/kpca.yaml"),
     
     # Set the default configuration file for performing a MDS.
     "mds" : \
        os.path.join(os.path.dirname(__file__),
                     "configs/dimensionality_reduction/mds.yaml"),
     
     # Set the default configuration file for performing a t-SNE.
     "tsne" : \
        os.path.join(os.path.dirname(__file__),
                     "configs/dimensionality_reduction/tsne.yaml")}

#---------------------------------------------------------------------#

# Set the default configuration files for generating different types of
# plots.
CONFIG_FILES_PLOT = \
    {# Set the default configuration file for plotting the results of
     # a PCA.
     "pca" : \
        os.path.join(os.path.dirname(__file__),
                     "configs/plotting/dim_red.yaml"),
     
     # Set the default configuration file for plotting the results of
     # a KPCA.
     "kpca" : \
        os.path.join(os.path.dirname(__file__),
                     "configs/plotting/dim_red.yaml"),
     
     # Set the default configuration file for plotting the results of
     # a MDS.
     "mds" : \
        os.path.join(os.path.dirname(__file__),
                     "configs/plotting/dim_red.yaml"),
     
     # Set the default configuration file for plotting the results of
     # a PCA.
     "tsne" : \
        os.path.join(os.path.dirname(__file__),
                     "configs/plotting/dim_red.yaml"),}


#---------------------------------------------------------------------#

# Set the default files used for setting up the model.
DATA_FILES_MODEL = \
    {# Set the default PyTorch file containing the parameters of the
     # trained Gaussian mixture model.
     "gmm" : \
        os.path.join(os.path.dirname(__file__),
                     "data/model/gmm/gmm.pth"),

    # Set the default PyTorch file containing the parameters of the
    # trained decoder.
    "dec" : \
        os.path.join(os.path.dirname(__file__),
                     "data/model/dec/dec.pth"),

    # Set the default file containing the Ensembl IDs of the genes
    # included in the DGD model.
    "genes" : \
        os.path.join(os.path.dirname(__file__),
                     "data/model/genes/genes.txt")}


#######################################################################


# Set the template against which to check the model's configuration.
CONFIG_MODEL_TEMPLATE = \
    {# Set the dimensionality of the input.
     "input_dim" : int,

     # Set the options for the Gaussian mixture model.
     "gmm_options" : None,
     "gmm_pth_file" : str,

     # Set the options for the decoder.
     "dec_options" : None,
     "dec_pth_file" : str,
    
     # Set the file containing the genes included in the model.
     "genes_txt_file" : str}

# Set the templates against which to check the configuration for
# the different types of optimization schemes that can be used
# when finding the representations for new samples.
CONFIG_REP_TEMPLATE = \
    {# Set the template for the 'one_opt' scheme.
     "one_opt" : \
        {# Set the number of representations to get per component
         # of the Gaussian mixture model per sample.
         "n_rep_per_comp" : int,

         # Set the options to configure the data loader.
         "data_loader" : None,

         # Set the optimization scheme.
         "scheme" : str,

         # Set how to output the losses.
         "loss" :  \
            {# Set how to output the GMM loss.
             "gmm" : \
                {# Set how to normalize the loss.
                 "norm_method" : str},
             # Set how to output the reconstruction loss.
             "recon" : \
                {# Set how to normalize the loss.
                 "norm_method" : str},
             # Set how to output the total loss.
             "total" : \
                {# Set how to normalize the loss.
                 "norm_method" : str}},

         # Set the options for the optimization.
         "opt" : \
            {# Set the number of epochs.
             "epochs" : int,
            # Set the optimizer
             "optimizer" : \
                {"name" : str,
                 "options" : None}}},

    # Set the template for the 'two_opt' scheme.
    "two_opt" : \
        {# Set the number of representations to get per component
         # of the Gaussian mixture model per sample.
         "n_rep_per_comp" : int,

         # Set the options to configure the data loader.
         "data_loader" : None,

         # Set the optimization scheme.
         "scheme" : str,

         # Set how to output the losses.
         "loss" :  \
            {# Set how to output the GMM loss.
             "gmm" : \
                {# Set how to normalize the loss.
                 "norm_method" : str},
             # Set how to output the reconstruction loss.
             "recon" : \
                {# Set how to normalize the loss.
                 "norm_method" : str},
             # Set how to output the total loss.
             "total" : \
                {# Set how to normalize the loss.
                 "norm_method" : str}},

         # Set the options for the first optimization.
         "opt1" : \
            {# Set the number of epochs.
             "epochs" : int,
            # Set the optimizer
             "optimizer" : \
                {"name" : str,
                 "options" : None}},
         
         # Set the options for the second optimization.
         "opt2" : \
            {# Set the number of epochs.
             "epochs" : int,
            # Set the optimizer
             "optimizer" : \
                {"name" : str,
                 "options" : None}}}}


# Set the template against which to check the training configuration.
CONFIG_TRAIN_TEMPLATE = \
    {# Set the number of epochs.
     "epochs" : int,

     # Set the options to configure the data loader.
     "data_loader" : None,

     # Set how to output the losses.
     "loss" :  \
        {# Set how to output the GMM loss.
         "gmm" : \
            {# Set how to normalize the loss.
             "norm_method" : str},
         # Set how to output the reconstruction loss.
         "recon" : \
            {# Set how to normalize the loss.
             "norm_method" : str},
         # Set how to output the total loss.
         "total" : \
            {# Set how to normalize the loss.
             "norm_method" : str}},

     # Set the options to train the Gaussian mixture model.
     "gmm" : \
        {"optimizer" : \
            {"name" : str,
             "options" : None}},
     
     # Set the options to train the decoder.
     "dec" : \
        {"optimizer" : \
            {"name" : str,
             "options" : None}},

    # Set the options to optimize the representations.
    "rep" : \
        {"optimizer" : \
            {"name" : str,
             "options" : None}},}
