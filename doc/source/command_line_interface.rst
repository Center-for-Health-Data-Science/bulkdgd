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


bulkdgd is an importable Python package.

It also provides a command-line interface for some of the most common tasks bulkdgd is used for.

The interface consists of executables installed together with the package:

* :doc:`bulkdgd_get_recount3 <bulkdgd_get_recount3>` builds a dataset of RNA-seq read counts and harmonized metadata from studies hosted on the Recount3 platform.

* :doc:`bulkdgd_get_genes <bulkdgd_get_genes>` creates custom lists of genes to use with the bulkdgd model.

* :doc:`bulkdgd_preprocess_samples <bulkdgd_preprocess_samples>` preprocesses samples' data before using them with the bulkdgd model.

* :doc:`bulkdgd_find_representations <bulkdgd_find_representations>` finds the best representations in the latent space defined by the bulkdgd model for a set of new samples.

* :doc:`bulkdgd_dea <bulkdgd_dea>` performs differential gene expression analysis between a set of samples and their 'normal' counterparts found by the bulkdgd model.

* :doc:`bulkdgd_reduction_pca / bulkdgd_reduction_kpca / bulkdgd_reduction_mds / bulkdgd_reduction_tsne / bulkdgd_reduction_umap <bulkdgd_reduction>` perform dimensionality reduction analyses and plot the results.

* :doc:`bulkdgd_find_probdens <bulkdgd_find_probdens>` finds, for a given set of representations, the probability density of each representation for each component of the Gaussian mixture model that defines the bulkdgd model's latent space.

* :doc:`bulkdgd_train <bulkdgd_train>` trains the model.
