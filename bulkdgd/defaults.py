#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-

#    defaults.py
#
#    General default values.
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
__doc__ = "General default values."


#######################################################################


# Import from the standard library.
import os


#######################################################################


# Set the default directories for the configuration files.
CONFIG_DIRS = {

    # Set the directory containing the configuration files for the
    # model (and, possibly, the trained model's parameters).
    "model" : \
        os.path.join(os.path.dirname(__file__),
                     "configs/model"),

    #-----------------------------------------------------------------#

    # Set the directory containing the configuration files for
    # finding the representations.
    "representations" : \
        os.path.join(os.path.dirname(__file__),
                     "configs/representations"),

    #-----------------------------------------------------------------#

    # Set the directory containing the configuration files specifying
    # the options to generate plots.
    "plotting" : \
        os.path.join(os.path.dirname(__file__),
                     "configs/plotting"),

    #-----------------------------------------------------------------#

    # Set the directory containing the configuration files specifying
    # the options for training the model.
    "training" : \
        os.path.join(os.path.dirname(__file__),
                     "configs/training"),

    #-----------------------------------------------------------------#

    # Set the directory containing the configuration files specifying
    # the options to create a new list of genes for the BulkDGD model.
    "genes" : \
        os.path.join(os.path.dirname(__file__),
                     "configs/genes"),

    #-----------------------------------------------------------------#

    # Set the directory containing the configuration files specifying
    # the options to perform dimensionality reduction analyses.
    "dimensionality_reduction" : \
        os.path.join(os.path.dirname(__file__),
                     "configs/dimensionality_reduction"),

    #-----------------------------------------------------------------#

    }


#######################################################################


# Set the default configuration files for performing dimensionality
# reduction analyses.
CONFIG_FILES_DIM_RED = {

    # Set the default configuration file for performing a PCA.
    "pca" : \
        os.path.join(os.path.dirname(__file__),
                     "configs/dimensionality_reduction/pca.yaml"),

    #-----------------------------------------------------------------#

    # Set the default configuration file for performing a KPCA.
    "kpca" : \
        os.path.join(os.path.dirname(__file__),
                     "configs/dimensionality_reduction/kpca.yaml"),

    #-----------------------------------------------------------------#

    # Set the default configuration file for performing a MDS.
    "mds" : \
        os.path.join(os.path.dirname(__file__),
                     "configs/dimensionality_reduction/mds.yaml"),

    #-----------------------------------------------------------------#

    # Set the default configuration file for performing a t-SNE.
    "tsne" : \
        os.path.join(os.path.dirname(__file__),
                     "configs/dimensionality_reduction/tsne.yaml"),

    #-----------------------------------------------------------------#

    # Set the default configuration file for performing a UMAP.
    "umap" : \
        os.path.join(os.path.dirname(__file__),
                     "configs/dimensionality_reduction/umap.yaml"),

    #-----------------------------------------------------------------#

    }


#######################################################################


# Set the default configuration files for generating different types of
# plots.
CONFIG_FILES_PLOT = {

    # Set the default configuration file for plotting the results of
    # a PCA.
    "pca" : \
        os.path.join(os.path.dirname(__file__),
                     "configs/plotting/scatterplot.yaml"),

    #-----------------------------------------------------------------#

    # Set the default configuration file for plotting the results of
    # a KPCA.
    "kpca" : \
        os.path.join(os.path.dirname(__file__),
                     "configs/plotting/scatterplot.yaml"),

    #-----------------------------------------------------------------#

    # Set the default configuration file for plotting the results of
    # a MDS.
    "mds" : \
        os.path.join(os.path.dirname(__file__),
                     "configs/plotting/scatterplot.yaml"),

    #-----------------------------------------------------------------#

    # Set the default configuration file for plotting the results of
    # a t-SNE.
    "tsne" : \
        os.path.join(os.path.dirname(__file__),
                     "configs/plotting/scatterplot.yaml"),

    #-----------------------------------------------------------------#

    # Set the default configuration file for plotting the results of
    # a UMAP.
    "umap" : \
        os.path.join(os.path.dirname(__file__),
                     "configs/plotting/scatterplot.yaml"),

    #-----------------------------------------------------------------#

    # Set the default configuration file for plotting a scatterplot.
    "scatterplot" : \
        os.path.join(os.path.dirname(__file__),
                     "configs/plotting/scatterplot.yaml"),

    #-----------------------------------------------------------------#

    # Set the default configuration file for plotting a histogram.
    "histogram" : \
        os.path.join(os.path.dirname(__file__),
                     "configs/plotting/histogram.yaml"),

    #-----------------------------------------------------------------#

    # Set the default configuration file for plotting a bi-histogram.
    "histogram_bihist" : \
        os.path.join(os.path.dirname(__file__),
                     "configs/plotting/histogram_bihist.yaml"),

    #-----------------------------------------------------------------#

    # Set the default configuration file for plotting two overlapping
    # histograms.
    "histogram_overlap" : \
        os.path.join(os.path.dirname(__file__),
                     "configs/plotting/histogram_overlap.yaml"),

    #-----------------------------------------------------------------#

    # Set the default configuration file for plotting a box plot.
    "boxplot" : \
        os.path.join(os.path.dirname(__file__),
                     "configs/plotting/boxplot.yaml"),

    #-----------------------------------------------------------------#

    # Set the default configuration file for plotting a violin plot.
    "violinplot" : \
        os.path.join(os.path.dirname(__file__),
                     "configs/plotting/violinplot.yaml"),

    #-----------------------------------------------------------------#

    # Set the default configuration file for plotting a line plot.
    "lineplot" : \
        os.path.join(os.path.dirname(__file__),
                     "configs/plotting/lineplot.yaml"),

    #-----------------------------------------------------------------#

    # Set the default configuration file for plotting an enrichment
    # scores plot.
    "enrichplot" : \
        os.path.join(os.path.dirname(__file__),
                     "configs/plotting/enrichplot.yaml"),
    }


#######################################################################


# Set the seeds the ensemble's members were trained with, in the order
# the ensemble reports them (the members differ only in the seed).
ENSEMBLE_SEEDS = ("seed37", "seed41", "seed43", "seed47", "seed53",
                  "seed59", "seed61", "seed67", "seed71", "seed73",
                  "seed79", "seed83", "seed89", "seed97", "seed101")


#---------------------------------------------------------------------#


# Set the member loaded by default (the one reported in the paper).
BASE_SEED = "seed37"


#######################################################################


def model_dir(seed: str = BASE_SEED) -> str:
    """Get the directory holding a member's fitted parameters.

    Parameters
    ----------
    seed : :class:`str`, ``BASE_SEED``
        The member's seed.

    Returns
    -------
    model_dir : :class:`str`
        The directory.
    """

    # Return the directory.
    return os.path.join(os.path.dirname(__file__),
                        "data",
                        "model",
                        seed)


def model_config_dir(seed: str = BASE_SEED) -> str:
    """Get the directory holding a member's configuration.

    Parameters
    ----------
    seed : :class:`str`, ``BASE_SEED``
        The member's seed.

    Returns
    -------
    model_config_dir : :class:`str`
        The directory.
    """

    # Return the directory.
    return os.path.join(os.path.dirname(__file__),
                        "configs",
                        "model",
                        seed)


def data_files_model(seed: str = BASE_SEED) -> dict:
    """Get the files needed to set up a member of the ensemble.

    Parameters
    ----------
    seed : :class:`str`, ``BASE_SEED``
        The member's seed.

    Returns
    -------
    data_files : :class:`dict`
        The files. 'dec.pth' is downloaded on first use (see
        'DECODER_PTH_URL').
    """

    # Return the files.
    return {

        # Set the trained Gaussian mixture model's parameters.
        "gmm" : os.path.join(model_dir(seed), "gmm.pth"),

        # Set the trained decoder's parameters (downloaded).
        "dec" : os.path.join(model_dir(seed), "dec.pth"),

        # Set the genes, shared by all members.
        "genes" : os.path.join(os.path.dirname(__file__),
                               "data/model/genes/genes.txt"),

        # Set the model's configuration.
        "config" : os.path.join(model_config_dir(seed), "model.yaml"),

        # Set the seeds the model was trained with.
        "seeds" : os.path.join(model_config_dir(seed), "seeds.yaml"),
        }


#######################################################################


# Set the default files used for setting up the model (the base
# member).
DATA_FILES_MODEL = data_files_model(BASE_SEED)


#######################################################################


# Set the URL template for downloading a trained decoder's parameters.
# '{version}' is the installed 'bulkdgd' version, '{seed}' the member.
DECODER_PTH_URL = \
    "https://github.com/Center-for-Health-Data-Science/bulkdgd/" \
    "releases/download/v{version}/dec_{seed}.pth"
