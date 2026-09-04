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
    
    # Set the directory containing the configuration files specifying
    # the DGD model's parameters and, possibly, the files containing
    # the parameters of the trained model.
    "model" : \
        os.path.join(os.path.dirname(__file__),
                     "configs/model"),
    
    #-----------------------------------------------------------------#

    # Set the directory containing the configuration files specifying
    # the options for the optimization round(s) when finding the best
    # representations for a set of samples.
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


# The seeds the shipped ensemble was trained with, in the order the
# ensemble reports them.
#
# The members differ in nothing but the seed: the architecture, the
# gene universe and the train/test split are identical across all
# fifteen, so a per-seed directory holds only what the seed changed -
# the fitted parameters, and the record of the seed itself.
ENSEMBLE_SEEDS = ("seed37", "seed41", "seed43", "seed47", "seed53",
                  "seed59", "seed61", "seed67", "seed71", "seed73",
                  "seed79", "seed83", "seed89", "seed97", "seed101")


#-----------------------------------------------------------------#


# The member a bare 'BulkDGD()' loads.
#
# Nothing distinguishes it from the other fourteen except that the
# paper reports it, so results quoted for "the model" can be
# reproduced without knowing which seed produced them.
BASE_SEED = "seed37"


#######################################################################


def model_dir(seed: str = BASE_SEED) -> str:
    """The directory holding one member's fitted parameters."""

    return os.path.join(os.path.dirname(__file__), "data", "model", seed)


def model_config_dir(seed: str = BASE_SEED) -> str:
    """The directory holding one member's configuration."""

    return os.path.join(os.path.dirname(__file__), "configs", "model",
                        seed)


def data_files_model(seed: str = BASE_SEED) -> dict:
    """The files needed to set one member of the ensemble up.

    'dec.pth' is named here whether or not it exists: it is fetched
    from the release on first use (see 'DECODER_PTH_URL'), and this is
    where it lands.
    """

    return {

        # The trained Gaussian mixture's parameters. Small enough to
        # ship with the package.
        "gmm" : os.path.join(model_dir(seed), "gmm.pth"),

        # The trained decoder's parameters. Not shipped; downloaded.
        "dec" : os.path.join(model_dir(seed), "dec.pth"),

        # The gene universe, which every member shares.
        "genes" : os.path.join(os.path.dirname(__file__),
                               "data/model/genes/genes.txt"),

        # The model's own architecture, and the seeds it was trained
        # with.
        "config" : os.path.join(model_config_dir(seed), "model.yaml"),

        "seeds" : os.path.join(model_config_dir(seed), "seeds.yaml"),
        }


#######################################################################


# Set the default files used for setting up the model.
#
# Kept as a mapping for the callers that read it directly; it resolves
# to the base member, which is what those callers meant when there was
# only one model to mean.
DATA_FILES_MODEL = data_files_model(BASE_SEED)


#######################################################################


# Set the URL template from which a trained decoder's parameters (too
# large to be distributed with the package itself) can be downloaded on
# demand. '{version}' is filled in with the installed 'bulkdgd'
# version, so a given release always downloads the exact decoders it
# was tested with, and '{seed}' selects the member.
#
# ONE ASSET PER MEMBER, AND PER RELEASE. Each decoder is 1.79 GiB in
# float64, close enough to the 2 GiB limit on a single release asset
# that a wider decoder or a later dtype change would breach it. Tying
# the name to both the version and the seed means a future release can
# change the shape of what it ships without any older install trying
# to read it.
DECODER_PTH_URL = \
    "https://github.com/Center-for-Health-Data-Science/bulkdgd/" \
    "releases/download/v{version}/dec_{seed}.pth"
