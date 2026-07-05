.. _train_config_options:

Configuration for training the bulkdgd model
============================================

To train an instance of :class:`core.model.BulkDGD`, we need to set a number of options.

These options can be passed as a nested dictionary or are specified in a YAML configuration file.

The function that loads the configuration file is :func:`bulkdgd.ioutil.load_config_train`.

The options that can be specified are described below.

* ``"n_epochs"`` is the number of epochs to train the model for. This is a positive integer and defaults to ``200``.

* ``"loss_reduction_type"`` is the reduction method to use for the loss function. This can be:

   * ``"sum"``, which computes the sum of the loss over the batch. This is the default value if not specified.

   * ``"mean"``, which computes the mean of the loss over the batch.

* ``"data_loader_options"`` is a dictionary of options to initialize the data loaders used to load the data for training and testing. It can contain the following options:
   
   * ``"train"`` is a dictionary of options to initialize the data loader used to load the training data. It can contain the following options:
      
      * ``"batch_size"`` is the batch size to use for the training data loader. This is a positive integer and defaults to ``64``.
      
      * ``"shuffle"`` is a boolean that specifies whether to shuffle the training data at each epoch. It defaults to ``True``.
   
   * ``"test"`` is a dictionary of options to initialize the data loader used to load the test data. It can contain the following options:
      
      * ``"batch_size"`` is the batch size to use for the test data loader. This is a positive integer and defaults to ``64``.
      
      * ``"shuffle"`` is a boolean that specifies whether to shuffle the test data at each epoch. It defaults to ``False``.

* ``"reporting_options"`` is a dictionary of options for reporting. It can contain the following options:

   * ``"loss"`` is a dictionary of options to report the loss. It can contain the following options:
   
      * ``"latent"`` is a dictionary of options for the latent space loss. It can contain the following options:
         
         * ``"norm_type"`` is the method to use to normalize the loss when reporting it. This can be:

            * ``"none"``, which does not normalize the loss. This is the default value if not specified.

            * ``"n_samples"``, which normalizes the loss by the number of samples in the batch.

            * ``"n_samples * latent_dim"``, which normalizes the loss by the number of samples times the latent dimension.
         
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
   
   * ``"metrics"`` is a dictionary of options to report the metrics. It can contain the following options:

      * ``"latent"`` is a list of metrics to compute for the latent space during training. The available metrics are listed below. The default value is ``["silhouette_score", "adjusted_rand_index_score"]``.

         Unsupervised metrics (do not require ground-truth labels):

         * ``"bic"``, which reports the Bayesian Information Criterion (BIC) of the latent space model. Lower is better.

         * ``"silhouette_score"``, which reports the silhouette score of the representations with respect to the cluster assignments given by the latent space model. Higher is better.

         * ``"davies_bouldin_score"``, which reports the Davies-Bouldin score of the representations with respect to the cluster assignments given by the latent space model. Lower is better.

         * ``"calinski_harabasz_score"``, which reports the Calinski-Harabasz score of the representations with respect to the cluster assignments given by the latent space model. Higher is better.

         Supervised metrics (require ground-truth labels):

         * ``"adjusted_rand_index_score"``, which reports the adjusted Rand index of the cluster assignments given by the latent space model with respect to the true labels. Higher is better.

         * ``"normalized_mutual_info_score"``, which reports the normalized mutual information of the cluster assignments given by the latent space model with respect to the true labels. Higher is better.
   
   * ``"optional_outputs"`` is a dictionary of options to specify which optional outputs to save during training. It can contain the following options:

      * ``"representations_epoch"`` is a dictionary of options to specify how to report the representations at the end of each epoch. It can contain the following options:

         * ``"enabled"`` is a boolean that specifies whether to save the representations at the end of each epoch. This defaults to ``False``.

         * ``"stride"`` is the stride (in epochs) at which to save the representations. This is a positive integer that defaults to ``1``.

         * ``"dir"`` is the directory where to save the representations. If not specified, it defaults to ``None``.
      
      * ``"latent_probs_epoch"`` is a dictionary of options to specify how to report the latent probabilities at the end of each epoch. It can contain the following options:

         * ``"enabled"`` is a boolean that specifies whether to save the latent probabilities at the end of each epoch. This defaults to ``False``.

         * ``"stride"`` is the stride (in epochs) at which to save the latent probabilities. This is a positive integer that defaults to ``1``.

         * ``"dir"`` is the directory where to save the latent probabilities. If not specified, it defaults to ``None``.
      
      * ``"latent_means_epoch"`` is a dictionary of options to specify how to report the latent means at the end of each epoch. It can contain the following options:

         * ``"enabled"`` is a boolean that specifies whether to save the latent means at the end of each epoch. This defaults to ``False``.

         * ``"stride"`` is the stride (in epochs) at which to save the latent means. This is a positive integer that defaults to ``1``.

         * ``"dir"`` is the directory where to save the latent means. If not specified, it defaults to ``None``.
      
      * ``"genes_saliency_maps_epoch"`` is a dictionary of options to specify how to report the genes' saliency maps at the end of each epoch. It can contain the following options:

         * ``"enabled"`` is a boolean that specifies whether to save the genes' saliency maps at the end of each epoch. This defaults to ``False``.

         * ``"stride"`` is the stride (in epochs) at which to save the genes' saliency maps. This is a positive integer that defaults to ``1``.

         * ``"dir"`` is the directory where to save the genes' saliency maps. If not specified, it defaults to ``None``.
      
      * ``"pathways_saliency_maps_epoch"`` is a dictionary of options to specify how to report the pathways' saliency maps at the end of each epoch. It can contain the following options:

         * ``"enabled"`` is a boolean that specifies whether to save the pathways' saliency maps at the end of each epoch. This defaults to ``False``.

         * ``"stride"`` is the stride (in epochs) at which to save the pathways' saliency maps. This is a positive integer that defaults to ``1``.

         * ``"dir"`` is the directory where to save the pathways' saliency maps. If not specified, it defaults to ``None``.

* ``"latent_type"`` is the type of latent space to use. This can be:

   * ``"lgmm"`` for the legacy Gaussian Mixture Model implementation.

   * ``"tgmm"`` for the TorchGMM implementation.

* ``"latent_training_options"`` is a dictionary of options to train the latent space. The options vary depending on the latent space implementation specified in ``"latent_type"``.

   * If the latent space implementation is the legacy Gaussian Mixture Model (``"lgmm"``):

      * ``"optimizer_type"`` is the type of optimizer to use for training the GMM. This can be:

         * ``"adam"``, which uses the Adam optimizer.

         * ``"adamw"``, which uses the AdamW optimizer. This is the default value if not specified.
        
      * ``"optimizer_options"`` is a dictionary of options for the optimizer. The options depend on the optimizer type. It can contain the following options:

         * ``"lr"`` is the learning rate. This is a positive float that defaults to ``0.01``.

         * ``"weight_decay"`` is the weight decay. This is a non-negative float that defaults to ``0.0``.

         * ``"betas"`` is a list of two floats that specify the beta parameters for the Adam or AdamW optimizer. The defaults are ``[0.9, 0.999]``.

      * ``"lr_scheduler_type"`` is the type of learning rate scheduler to use. This can be:

         * ``None``, which does not use a learning rate scheduler. This is the default value if not specified.

         * ``"one_cycle"``, which uses the OneCycleLR scheduler.
        
      * ``"lr_scheduler_options"`` is a dictionary of options for the learning rate scheduler. For the ``"one_cycle"`` scheduler, the options are:

         * ``"max_lr"`` is the maximum learning rate to use. This is a positive float that defaults to ``0.01``.

         * ``"pct_start"`` is the percentage of the total number of epochs to use for the increasing phase of the learning rate schedule. This is a float between 0 and 1 that defaults to ``0.25``.

         * ``"anneal_strategy"`` is the annealing strategy to use. This can be:

            * ``"cos"``, which uses a cosine annealing strategy. This is the default value if not specified.

            * ``"linear"``, which uses a linear annealing strategy.
           
         * ``"cycle_momentum"`` is a boolean that specifies whether to cycle the momentum during training. This defaults to ``True``.

         * ``"base_momentum"`` is the base momentum. This is a float between 0 and 1 that defaults to ``0.85``.

         * ``"max_momentum"`` is the maximum momentum. This is a float between 0 and 1 that defaults to ``0.9``.

         * ``"div_factor"`` is the factor by which to divide the maximum learning rate to get the initial learning rate. This is a positive float that defaults to ``25.0``.

         * ``"final_div_factor"`` is the factor by which to divide the initial learning rate to get the minimum learning rate at the end of training. This is a positive float that defaults to ``1000.0``.

         * ``"three_phase"`` is a boolean that specifies whether to use a three-phase learning rate schedule. This defaults to ``False``.
      
      * ``"components_removal_type"`` is the type of components removal to use during training. This can be:

         * ``None``, which disables component removal. This is the default value if not specified.

         * ``"weight_threshold"``, which removes components whose weight falls below a given threshold.
      
      * ``"components_removal_options"`` is a dictionary of options for the components removal. For the ``"weight_threshold"`` type, the options are:

         * ``"threshold"`` is the threshold below which a component's weight triggers its removal. This is a non-negative float that defaults to ``1e-8``.

   * If the latent space implementation is the TorchGMM one (``"tgmm"``):

      * ``"loss_calculation"`` is a dictionary of options to specify how to calculate the latent space loss. It can contain the following options:
         
         * ``"lambda"`` is the weight to use for the latent space loss in the total loss. This is a non-negative float that defaults to ``1.0``.
      
      * ``"model_selection_type"`` is the type of model selection to use for selecting the best model during training. This can be:

         * ``None``, which disables model selection. This is the default value if not specified.

         * ``"metric"``, which selects the best model based on a metric.
      
      * ``"model_selection_options"`` specifies the metric to use for model selection when ``"model_selection_type"`` is ``"metric"``. The available metrics are:

         * ``"bic"``, which uses the Bayesian Information Criterion (BIC). Lower is better. This is the default value if not specified.

         * ``"silhouette_score"``, which uses the silhouette score. Higher is better.

         * ``"calinski_harabasz_score"``, which uses the Calinski-Harabasz score. Higher is better.

         * ``"davies_bouldin_score"``, which uses the Davies-Bouldin score. Lower is better.
      
      * ``"fitting"`` is a dictionary of options to specify how to fit the GMM during training. It can contain the following options:

         * ``"first_epoch"`` is the epoch at which to start fitting the GMM. This is a non-negative integer that defaults to ``25``.

         * ``"refit_final"`` is a boolean that specifies whether to refit the GMM at the end of training. This defaults to ``True``.

         * ``"refit_interval"`` is the interval (in epochs) at which to refit the GMM during training. This is a non-negative integer that defaults to ``0`` (no periodic refitting).

         * ``"max_iter_first_epoch"`` is the maximum number of EM iterations to use for the first epoch of GMM fitting. This is a positive integer that defaults to ``1000``.

         * ``"max_iter_full_refit"`` is the maximum number of EM iterations to use for full-refit epochs. This is a positive integer that defaults to ``100``.

         * ``"max_iter_warm_refit"`` is the maximum number of EM iterations to use for warm-refit epochs. This is a positive integer that defaults to ``100``.

         * ``"max_iter_final_refit"`` is the maximum number of EM iterations to use for the final refit at the end of training. This is a positive integer that defaults to ``1000``.

      * ``"components_removal_type"`` is the type of components removal to use during training. This can be:

         * ``None``, which disables component removal. This is the default value if not specified.

         * ``"weight_threshold"``, which removes components whose weight falls below a given threshold.
      
      * ``"components_removal_options"`` is a dictionary of options for the components removal. For the ``"weight_threshold"`` type, the options are:

         * ``"threshold"`` is the threshold below which a component's weight triggers its removal. This is a non-negative float that defaults to ``1e-8``.
        
* ``"decoder_training_options"`` is a dictionary of options to train the decoder. It can contain the following options:

   * ``"optimizer_type"`` is the type of optimizer to use for training the decoder. This can be:

      * ``"adam"``, which uses the Adam optimizer.

      * ``"adamw"``, which uses the AdamW optimizer. This is the default value if not specified.
      
   * ``"optimizer_options"`` is a dictionary of options for the optimizer. The options depend on the optimizer type. It can contain the following options:

      * ``"lr"`` is the learning rate. This is a positive float that defaults to ``0.001``.

      * ``"weight_decay"`` is the weight decay. This is a non-negative float that defaults to ``0.0``.

      * ``"betas"`` is a list of two floats that specify the beta parameters for the Adam or AdamW optimizer. The defaults are ``[0.9, 0.999]``.

   * ``"lr_scheduler_type"`` is the type of learning rate scheduler to use. This can be:

      * ``None``, which does not use a learning rate scheduler. This is the default value if not specified.

      * ``"one_cycle"``, which uses the OneCycleLR scheduler.
    
   * ``"lr_scheduler_options"`` is a dictionary of options for the learning rate scheduler. For the ``"one_cycle"`` scheduler, the options are:

      * ``"max_lr"`` is the maximum learning rate to use. This is a positive float that defaults to ``0.01``.

      * ``"pct_start"`` is the percentage of the total number of epochs to use for the increasing phase of the learning rate schedule. This is a float between 0 and 1 that defaults to ``0.25``.

      * ``"anneal_strategy"`` is the annealing strategy to use. This can be:

         * ``"cos"``, which uses a cosine annealing strategy. This is the default value if not specified.

         * ``"linear"``, which uses a linear annealing strategy.
         
      * ``"cycle_momentum"`` is a boolean that specifies whether to cycle the momentum during training. This defaults to ``True``.

      * ``"base_momentum"`` is the base momentum. This is a float between 0 and 1 that defaults to ``0.85``.

      * ``"max_momentum"`` is the maximum momentum. This is a float between 0 and 1 that defaults to ``0.9``.

      * ``"div_factor"`` is the factor by which to divide the maximum learning rate to get the initial learning rate. This is a positive float that defaults to ``25.0``.

      * ``"final_div_factor"`` is the factor by which to divide the initial learning rate to get the minimum learning rate at the end of training. This is a positive float that defaults to ``1000.0``.

      * ``"three_phase"`` is a boolean that specifies whether to use a three-phase learning rate schedule. This defaults to ``False``.

* ``"representations_training_options"`` is a dictionary of options to train the representations. It can contain the following options:

   * ``"train_noise_type"`` is the type of noise to add to the representations during training. This can be:

      * ``None`` or ``"none"``, which does not add any noise to the representations during training. This is the default value if not specified.

      * ``"gaussian"``, which adds Gaussian noise to the representations.
   
   * ``"train_noise_options"`` is a dictionary of options to specify the noise to add to the representations during training. For the ``"gaussian"`` type, the options are:

      * ``"scale"`` is the base noise scale. This is a non-negative float that defaults to ``0.0`` (no noise).

      * ``"start"`` is the starting noise multiplier (at epoch 1). This is a non-negative float that defaults to ``1.0``.

      * ``"end"`` is the ending noise multiplier (at the final epoch). This is a non-negative float that defaults to ``0.01``.

      * ``"within_radius_prob"`` is the probability of the noise being within a given radius. This is a float between 0 and 1 that defaults to ``0.95``.

      * ``"gain"`` is the gain factor for the noise. This is a non-negative float that defaults to ``1.0``.

   * ``"optimizer_type"`` is the type of optimizer to use for training the representations. This can be:

      * ``"adam"``, which uses the Adam optimizer.

      * ``"adamw"``, which uses the AdamW optimizer. This is the default value if not specified.
      
   * ``"optimizer_options"`` is a dictionary of options for the optimizer. The options depend on the optimizer type. It can contain the following options:

      * ``"lr"`` is the learning rate. This is a positive float that defaults to ``0.001``.

      * ``"weight_decay"`` is the weight decay. This is a non-negative float that defaults to ``0.0``.

      * ``"betas"`` is a list of two floats that specify the beta parameters for the Adam or AdamW optimizer. The defaults are ``[0.9, 0.999]``.

   * ``"lr_scheduler_type"`` is the type of learning rate scheduler to use. This can be:

      * ``None``, which does not use a learning rate scheduler. This is the default value if not specified.

      * ``"one_cycle"``, which uses the OneCycleLR scheduler.

   * ``"lr_scheduler_options"`` is a dictionary of options for the learning rate scheduler. For the ``"one_cycle"`` scheduler, the options are:

      * ``"max_lr"`` is the maximum learning rate to use. This is a positive float that defaults to ``0.01``.

      * ``"pct_start"`` is the percentage of the total number of epochs to use for the increasing phase of the learning rate schedule. This is a float between 0 and 1 that defaults to ``0.25``.

      * ``"anneal_strategy"`` is the annealing strategy to use. This can be:

         * ``"cos"``, which uses a cosine annealing strategy. This is the default value if not specified.

         * ``"linear"``, which uses a linear annealing strategy.
         
      * ``"cycle_momentum"`` is a boolean that specifies whether to cycle the momentum during training. This defaults to ``True``.

      * ``"base_momentum"`` is the base momentum. This is a float between 0 and 1 that defaults to ``0.85``.

      * ``"max_momentum"`` is the maximum momentum. This is a float between 0 and 1 that defaults to ``0.9``.

      * ``"div_factor"`` is the factor by which to divide the maximum learning rate to get the initial learning rate. This is a positive float that defaults to ``25.0``.

      * ``"final_div_factor"`` is the factor by which to divide the initial learning rate to get the minimum learning rate at the end of training. This is a positive float that defaults to ``1000.0``.

      * ``"three_phase"`` is a boolean that specifies whether to use a three-phase learning rate schedule. This defaults to ``False``.

* ``"early_stopping_type"`` is the type of early stopping criteria to use during training. This can be:

   * ``None``, which disables early stopping. This is the default value if not specified.

   * ``"loss"``, which uses the total loss for early stopping. Training will be stopped if the total loss does not improve for a number of epochs specified by the ``"patience"`` option.

* ``"early_stopping_options"`` is a dictionary of options for the early stopping criteria. It can contain the following options:

   * ``"patience"`` is the number of epochs with no improvement after which training will be stopped if early stopping is enabled. This is a positive integer that defaults to ``10``.