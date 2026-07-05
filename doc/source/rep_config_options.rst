.. _rep_config_options:

Configuration for the optimization scheme
=========================================

So far, two optimization schemes to find the representations have been implemented:

* ``one_opt``, which consists of only one round of optimization for the best representations found for the samples in latent space. The ``one_opt`` scheme is implemented in the YAML file ``bulkdgd/configs/representations/one_opt.yaml``.

* ``two_opt``, which consists of two consecutive rounds of optimizations. Indeed, multiple candidate representations per sample are found, optimized, and the best one for each sample is picked from the pool. Then, a second round of optimization (similar to the one run under the ``one_opt`` scheme) is performed on these selected representations. The ``two_opt`` scheme is implemented in the YAML file ``bulkdgd/configs/representations/two_opt.yaml``.

The options to customize these schemes can be passed as a nested dictionary or are specified in a YAML configuration file.

The function that loads the configuration file is :func:`bulkdgd.ioutil.load_config_rep`.

The options that can be specified are described below.

* ``"scheme_type"`` is the optimization scheme to use. This can be:

   * ``"one_opt"`` for the optimization scheme with only one round of optimization.

   * ``"two_opt"`` for the optimization scheme with two rounds of optimization.

* ``"latent_type"`` is the type of latent space used in the model. This can be:

   * ``"lgmm"`` for the legacy Gaussian Mixture Model implementation.

   * ``"tgmm"`` for the TorchGMM implementation.

* ``"n_rep_per_comp"`` is the number of representations to initialize per component per sample. This is a positive integer and defaults to ``1``.

* ``"data_loader_options"`` is a dictionary of options to initialize the data loader used to load the data for the optimization. It can contain the following options:

   * ``"batch_size"`` is the batch size to use for the data loader. This is a positive integer and defaults to ``128``.

   * ``"shuffle"`` is a boolean that indicates whether to shuffle the data at each epoch. It defaults to ``False``.

* ``"reporting_options"`` is a dictionary of options for reporting during the optimization. It can contain the following options:

   * ``"loss"`` is a dictionary of options to report the loss. It can contain the following options:

      * ``"reduction_type"`` is the reduction method to use for the loss function. This can be:

         * ``"sum"``, which computes the sum of the loss over the batch. This is the default value if not specified.

         * ``"mean"``, which computes the mean of the loss over the batch.
   
      * ``"latent"`` is a dictionary of options for the latent space loss. It can contain the following options:
      
         * ``"norm_type"`` is the method to use to normalize the loss when reporting it. This can be:

            * ``"none"``, which does not normalize the loss. This is the default value if not specified.

            * ``"n_samples"``, which normalizes the loss by the number of samples in the batch.

            * ``"n_samples * latent_dim"``, which normalizes the loss by the number of samples times the latent dimension.

         * ``"lambda"`` is the weight to use for the latent space loss. This is a non-negative float that defaults to ``1.0``.
      
      * ``"decoder"`` is a dictionary of options for the reconstruction loss. It can contain the following options:
      
         * ``"norm_type"`` is the method to use to normalize the loss when reporting it. This can be:

            * ``"none"``, which does not normalize the loss. This is the default value if not specified.

            * ``"n_samples"``, which normalizes the loss by the number of samples in the batch.

            * ``"n_samples * n_genes"``, which normalizes the loss by the number of samples times the number of genes.
      
      * ``"total"`` is a dictionary of options for the total loss. It can contain the following options:
      
         * ``"norm_type"`` is the method to use to normalize the loss when reporting it. This can be:

            * ``"none"``, which does not normalize the loss. This is the default value if not specified.

            * ``"n_samples"``, which normalizes the loss by the number of samples in the batch.

            * ``"n_samples * n_genes"``, which normalizes the loss by the number of samples times the number of genes.

* ``"scheme_options"`` is a dictionary of options specific to the optimization scheme. The options vary depending on the scheme type and the latent space type.

   * For the ``one_opt`` scheme with the legacy GMM (``"lgmm"``):

      * ``"loss_reduction_type"`` is the reduction method to use for the loss function. This can be:

         * ``"sum"``, which computes the sum of the loss over the batch. This is the default value if not specified.

         * ``"mean"``, which computes the mean of the loss over the batch.

      * ``"optimization"`` is a dictionary of options for the optimization. It can contain the following options:

         * ``"epochs"`` is the number of epochs to run the optimization for. This is a positive integer and defaults to ``50``.

         * ``"optimizer_type"`` is the type of optimizer to use. This can be:

            * ``"adam"``, which uses the Adam optimizer.

            * ``"adamw"``, which uses the AdamW optimizer. This is the default.

         * ``"optimizer_options"`` is a dictionary of options for the optimizer. It can contain the following options:

            * ``"lr"`` is the learning rate. This is a positive float that defaults to ``0.01``.

            * ``"weight_decay"`` is the weight decay. This is a non-negative float that defaults to ``0.0``.

            * ``"betas"`` is a list of two floats that specify the beta parameters for the optimizer. The defaults are ``[0.9, 0.999]``.

   * For the ``one_opt`` scheme with TorchGMM (``"tgmm"``):

      * ``"loss_reduction_type"`` is the reduction method to use for the loss function. This can be:

         * ``"sum"``, which computes the sum of the loss over the batch. This is the default value if not specified.

         * ``"mean"``, which computes the mean of the loss over the batch.

      * ``"latent_loss_calculation"`` is a dictionary of options for the latent space loss calculation. It can contain:

         * ``"lambda"`` is the weight to use for the latent space loss. This is a non-negative float that defaults to ``1.0``.

      * ``"optimization"`` is a dictionary of options for the optimization. It can contain the following options:

         * ``"epochs"`` is the number of epochs to run the optimization for. This is a positive integer and defaults to ``50``.

         * ``"optimizer_type"`` is the type of optimizer to use. This can be:

            * ``"adam"``, which uses the Adam optimizer.

            * ``"adamw"``, which uses the AdamW optimizer. This is the default.

         * ``"optimizer_options"`` is a dictionary of options for the optimizer. It can contain the following options:

            * ``"lr"`` is the learning rate. This is a positive float that defaults to ``0.01``.

            * ``"weight_decay"`` is the weight decay. This is a non-negative float that defaults to ``0.0``.

            * ``"betas"`` is a list of two floats that specify the beta parameters for the optimizer. The defaults are ``[0.9, 0.999]``.

   * For the ``two_opt`` scheme with the legacy GMM (``"lgmm"``):

      * ``"loss_reduction_type"`` is the reduction method to use for the loss function. This can be:

         * ``"sum"``, which computes the sum of the loss over the batch. This is the default value if not specified.

         * ``"mean"``, which computes the mean of the loss over the batch.

      * ``"optimization_1"`` is a dictionary of options for the first optimization round. It can contain the following options:

         * ``"epochs"`` is the number of epochs to run the first optimization for. This is a positive integer and defaults to ``10``.

         * ``"optimizer_type"`` is the type of optimizer to use. This can be:

            * ``"adam"``, which uses the Adam optimizer.

            * ``"adamw"``, which uses the AdamW optimizer. This is the default.

         * ``"optimizer_options"`` is a dictionary of options for the optimizer. It can contain the following options:

            * ``"lr"`` is the learning rate. This is a positive float that defaults to ``0.01``.

            * ``"weight_decay"`` is the weight decay. This is a non-negative float that defaults to ``0.0``.

            * ``"betas"`` is a list of two floats that specify the beta parameters for the optimizer. The defaults are ``[0.9, 0.999]``.

      * ``"optimization_2"`` is a dictionary of options for the second optimization round. It has the same structure as ``"optimization_1"`` but with ``"epochs"`` defaulting to ``50``.

   * For the ``two_opt`` scheme with TorchGMM (``"tgmm"``):

      * ``"loss_reduction_type"`` is the reduction method to use for the loss function. This can be:

         * ``"sum"``, which computes the sum of the loss over the batch. This is the default value if not specified.

         * ``"mean"``, which computes the mean of the loss over the batch.

      * ``"latent_loss_calculation"`` is a dictionary of options for the latent space loss calculation. It can contain:

         * ``"lambda"`` is the weight to use for the latent space loss. This is a non-negative float that defaults to ``1.0``.

      * ``"optimization_1"`` is a dictionary of options for the first optimization round. It can contain the following options:

         * ``"epochs"`` is the number of epochs to run the first optimization for. This is a positive integer and defaults to ``10``.

         * ``"optimizer_type"`` is the type of optimizer to use. This can be:

            * ``"adam"``, which uses the Adam optimizer.

            * ``"adamw"``, which uses the AdamW optimizer. This is the default.

         * ``"optimizer_options"`` is a dictionary of options for the optimizer. It can contain the following options:

            * ``"lr"`` is the learning rate. This is a positive float that defaults to ``0.01``.

            * ``"weight_decay"`` is the weight decay. This is a non-negative float that defaults to ``0.0``.

            * ``"betas"`` is a list of two floats that specify the beta parameters for the optimizer. The defaults are ``[0.9, 0.999]``.

      * ``"optimization_2"`` is a dictionary of options for the second optimization round. It has the same structure as ``"optimization_1"`` but with ``"epochs"`` defaulting to ``50``.