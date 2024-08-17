Command-line interface
======================

.. toctree::
   :maxdepth: 1
   :hidden:

   bulkdgd_get_recount3
   bulkdgd_get_genes
   bulkdgd_preprocess_samples
   bulkdgd_find_representations
   bulkdgd_find_probability_density
   bulkdgd_dea
   bulkdgd_reduction
   bulkdgd_train


bulkDGD is structured as an importable Python package.

However, a command-line interface is provided for some of the most common tasks bulkDGD is used for.

This interface consists of a series of executables installed together with the package:

* :doc:`bulkdgd get recount3 <bulkdgd_get_recount3>` allows the seamless retrieval of RNA-seq data and their associated metadata from the Recount3 platform.

* :doc:`bulkdgd get genes <bulkdgd_get_genes>` allows the creation of custom lists of genes to use with the bulkDGD model.

* :doc:`bulkdgd preprocess samples <bulkdgd_preprocess_samples>` allows the preprocessing of samples' data before using them with the bulkDGD model.

* :doc:`bulkdgd find representations <bulkdgd_find_representations>` allows finding the best representations in the latent space defined by the bulkDGD model for a set of new samples.

* :doc:`bulkdgd dea <bulkdgd_dea>` allows performing differential gene expression analysis between a set of samples and their 'normal' counterparts found by the bulkDGD model.

* :doc:`bulkdgd reduction <bulkdgd_reduction>` allows performing dimensionality reduction analyses and plotting the results.

* :doc:`bulkdgd find probability_density <bulkdgd_find_probability_density>` allows finding, for a given a set of representations, the probability density of each representation for each component of the Gaussian mixture model that defines the bulkDGD model's latent space.

* :doc:`bulkdgd train <bulkdgd_train>` allows training the bulkDGD model.
