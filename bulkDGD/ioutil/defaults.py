#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-

#    defaults.py
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
__doc__ = "Default directories/files for I/O operations."


# Standard library
import os


#---------------------------------------------------------------------#

# The directory containing the configuration files specifying the
# DGD model's parameters and the files containing the trained
# model
CONFIG_MODEL_DIR = \
    os.path.join(os.path.dirname(__file__),
                 "configs/model")

#---------------------------------------------------------------------#

# The directory containing the configuration files specifying the
# options for data loading and optimization when finding the
# best representations
CONFIG_REP_DIR = \
    os.path.join(os.path.dirname(__file__),
                 "configs/representations")

#---------------------------------------------------------------------#

# The default configuration file for plotting the results of the PCA
CONFIG_PLOT_PCA = \
    os.path.join(os.path.dirname(__file__),
                 "configs/plot/pca_scatter.yaml")

#---------------------------------------------------------------------#

# Default PyTorch file containing the trained Gaussian mixture model
GMM_FILE = \
    os.path.join(os.path.dirname(__file__),
                 "data/gmm.pth")

#---------------------------------------------------------------------#

# Default PyTorch file containing the trained decoder
DEC_FILE = \
    os.path.join(os.path.dirname(__file__),
                 "data/dec.pth")

#---------------------------------------------------------------------#

# File containing the Ensembl IDs of the genes included in the DGD
# model
GENES_FILE = \
    os.path.join(os.path.dirname(__file__),
                 "data/genes.txt")
