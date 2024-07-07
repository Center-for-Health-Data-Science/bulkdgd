Command-line interface
======================

.. toctree::
   :maxdepth: 1
   :hidden:

   dgd_get_recount3_data
   dgd_preprocess_samples
   dgd_get_representations
   dgd_perform_dea
   dgd_perform_pca
   dgd_get_probability_density
   dgd_train


bulkDGD is structured as an importable Python package.

However, a command-line interface is provided for some of the most common tasks bulkDGD is used for.

This interface consists of a series of executables installed together with the package:

* :doc:`dgd_get_recount3_data <dgd_get_recount3_data>` allows the seamless retrieval of RNA-seq data and their associated metadata from the Recount3 platform.

* :doc:`dgd_preprocess_samples <dgd_preprocess_samples>` allows the preprocessing of samples' data before using them with the DGD model.

* :doc:`dgd_get_representations <dgd_get_representations>` allows finding the best representations in the latent space defined by the DGD model for a set of new samples.

* :doc:`dgd_perform_dea <dgd_perform_dea>` allows performing differential gene expression analysis between a set of samples and their 'normal' counterparts found by the DGD model.

* :doc:`dgd_perform_pca <dgd_perform_pca>` allows performing and plotting the results of a 2D principal component analysis on a set of representations.

* :doc:`dgd_get_probability_density <dgd_get_probability_density>` allows finding, for a given a set of representations, the probability density of each representation for each component of the Gaussian mixture model that defines the DGD model's latent space.

* :doc:`dgd_train <dgd_train>` allows training the DGD model.
