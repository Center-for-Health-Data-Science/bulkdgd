Command-line interface
======================

.. toctree::
   :maxdepth: 1
   :hidden:

   bulkdgd_get_recount3
   bulkdgd_get_genes
   bulkdgd_preprocess_samples
   bulkdgd_find_representations
   bulkdgd_find_probdens
   bulkdgd_dea
   bulkdgd_reduction
   bulkdgd_train


bulkdgd is structured as an importable Python package.

However, a command-line interface is provided for some of the most common tasks bulkdgd is used for.

This interface consists of a series of executables installed together with the package:

* :doc:`bulkdgd_get_recount3 <bulkdgd_get_recount3>` allows the seamless retrieval of RNA-seq data and their associated metadata from the Recount3 platform.

* :doc:`bulkdgd_get_genes <bulkdgd_get_genes>` allows the creation of custom lists of genes to use with the bulkdgd model.

* :doc:`bulkdgd_preprocess_samples <bulkdgd_preprocess_samples>` allows the preprocessing of samples' data before using them with the bulkdgd model.

* :doc:`bulkdgd_find_representations <bulkdgd_find_representations>` allows finding the best representations in the latent space defined by the bulkdgd model for a set of new samples.

* :doc:`bulkdgd_dea <bulkdgd_dea>` allows performing differential gene expression analysis between a set of samples and their 'normal' counterparts found by the bulkdgd model.

* :doc:`bulkdgd_reduction_pca / bulkdgd_reduction_kpca / bulkdgd_reduction_mds / bulkdgd_reduction_tsne / bulkdgd_reduction_umap <bulkdgd_reduction>` allows performing dimensionality reduction analyses and plotting the results.

* :doc:`bulkdgd_find_probdens <bulkdgd_find_probdens>` allows finding, for a given set of representations, the probability density of each representation for each component of the Gaussian mixture model that defines the bulkdgd model's latent space.

* :doc:`bulkdgd_train <bulkdgd_train>` allows training the model.
