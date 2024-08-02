#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-

#    defaults.py
#
#    Default directories/files to load and save files.
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
__doc__ = "Default directories/files to load and save files."


#######################################################################


# Import from the standard library.
import os


#######################################################################


# Set the directory containing the configuration files specifying the
# DGD model's parameters and, possibly, the files containing the
# parameters of the trained model.
CONFIG_MODEL_DIR = \
    os.path.join(os.path.dirname(__file__),
                 "configs/model")

#---------------------------------------------------------------------#

# Set the directory containing the configuration files specifying the
# options for the optimization round(s) when finding the best
# representations for a set of samples.
CONFIG_REP_DIR = \
    os.path.join(os.path.dirname(__file__),
                 "configs/representations")

#---------------------------------------------------------------------#

# Set the directory containing the configuration files specifying the
# options to generate plots.
CONFIG_PLOT_DIR = \
    os.path.join(os.path.dirname(__file__),
                 "configs/plot")

#---------------------------------------------------------------------#

# Set the directory containing the configuration files specifying the
# options for training the model.
CONFIG_TRAIN_DIR = \
    os.path.join(os.path.dirname(__file__),
                 "configs/training")

#---------------------------------------------------------------------#

# Set the directory containing the configuration files specifying the
# options to create a new list of genes for the bulkDGD model.
CONFIG_GENES_DIR = \
    os.path.join(os.path.dirname(__file__),
                 "configs/genes")

#---------------------------------------------------------------------#

# Set the default configuration file for plotting the results of the
# PCA.
CONFIG_PLOT_PCA = \
    os.path.join(os.path.dirname(__file__),
                 "configs/plot/pca_scatter.yaml")

#---------------------------------------------------------------------#

# Set the default PyTorch file containing the parameters of the trained
# Gaussian mixture model.
GMM_FILE = \
    os.path.join(os.path.dirname(__file__),
                 "data/gmm.pth")

#---------------------------------------------------------------------#

# Set the default PyTorch file containing the parameters of the trained
# decoder.
DEC_FILE = \
    os.path.join(os.path.dirname(__file__),
                 "data/dec.pth")

#---------------------------------------------------------------------#

# Set the default file containing the Ensembl IDs of the genes included
# in the DGD model.
GENES_FILE = \
    os.path.join(os.path.dirname(__file__),
                 "data/genes.txt")
