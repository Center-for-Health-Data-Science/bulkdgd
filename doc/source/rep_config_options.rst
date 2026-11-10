.. _rep_config_options:

Configuration for the optimization scheme
=========================================

Two optimization schemes to find the representations are implemented:

* ``two_opt``, which consists of two consecutive rounds of optimization. In the first, multiple candidate representations per sample are optimized, and the best one for each sample is picked from the pool. In the second, the selected representations are optimized further. The ``two_opt`` scheme is implemented in the YAML file ``bulkdgd/configs/representations/two_opt.yaml``.

* ``two_opt_multiseed``, which runs ``two_opt`` once per initialization seed and keeps every seed's answer instead of one. It is implemented in the YAML file ``bulkdgd/configs/representations/two_opt_multiseed.yaml``.

.. _rep_multiseed:

The multi-seed scheme
---------------------

The seed decides **where the search starts, and nothing else**: it places the candidate representations drawn from the components of the Gaussian mixture model. Two seeds can arrive at different local optima, and the genes called from them differ.

Each seed is run in sequence, at the tensor shape a standalone run uses, so each seed's result is bitwise identical to a standalone run with that seed.

.. note::

   ``two_opt_multiseed`` requires ``latent_type: tgmm``, since the legacy Gaussian mixture model places its candidates without a seed.

   It also requires ``loss_reduction_type: sum``.

   The seeds must be **distinct**. Repeated seeds raise an error.

Outputs
-------

``two_opt_multiseed`` produces one representation per sample **per seed**, together with the matching predicted means and r-values, and one table that ``two_opt`` does not produce: the loss of each sample's selected representation at the end of each optimization round.

After the run, these are available on the model as ``multiseed_results``, a dictionary with the keys ``"seeds"``, ``"representations"``, ``"pred_means"``, ``"pred_r_values"`` and ``"losses"``. The first four map a seed to its result; ``"losses"`` is a :class:`pandas.DataFrame` with one row per sample, indexed by sample name, and two columns per seed:

.. code-block:: text

   sample                        loss_opt1_seed7  loss_opt2_seed7  loss_opt1_seed13  loss_opt2_seed13
   GTEX-1117F-0226-SM-5GZZ7.1          100322.87         97806.77         105977.95         100584.39
   GTEX-1117F-1326-SM-5EGHH.1           97275.88         92556.16          91213.52          90363.78

``loss_opt1_seed<N>`` is the total loss of the candidate that won seed *N*'s selection, as of the last epoch of the first optimization. ``loss_opt2_seed<N>`` is the same representation's loss after the second optimization.

The method returns the **first** seed's representations, predicted means and r-values.

.. note::

   These results are kept in memory, in ``multiseed_results``. The package does not write them to disk.

The options to customize these schemes can be passed as a nested dictionary or specified in a YAML configuration file.

The function that loads the configuration file is :func:`bulkdgd.ioutil.load_config_rep`.

The options that can be specified are described below.

* ``"scheme_type"`` is the optimization scheme to use. This can be:

   * ``"two_opt"`` for the optimization scheme with two rounds of optimization. This is the default value if not specified.

   * ``"two_opt_multiseed"`` for the same scheme run once per initialization seed. See :ref:`rep_multiseed`. It takes every option ``two_opt`` takes, and reads its seeds from ``scheme_options.initialization.seeds``, a list of distinct integers.

* ``"latent_type"`` is the type of latent space used in the model. This can be:

   * ``"lgmm"`` for the legacy Gaussian Mixture Model implementation.

   * ``"tgmm"`` for the TorchGMM implementation. This is the default value if not specified.

* ``"n_rep_per_comp"`` is the number of representations to initialize per component per sample. This is a positive integer and defaults to ``1``.

* ``"use_batch_size_padding"`` is a boolean that specifies whether to pad the samples to a whole number of batches with copies of the first sample, so that every batch has the same shape. The padding samples are dropped from the outputs. It defaults to ``True``.

* ``"data_loader_options"`` is a dictionary of options to initialize the data loader used to load the data for the optimization. It can contain the following options:

   * ``"batch_size"`` is the batch size to use for the data loader. This is a positive integer and defaults to ``16``.

   * ``"shuffle"`` is a boolean that indicates whether to shuffle the data at each epoch. It defaults to ``False``.

* ``"reporting_options"`` is a dictionary of options for reporting during the optimization. It can contain the following options:

   * ``"loss"`` is a dictionary of options to report the loss. It can contain the following options:

      * ``"reduction_type"`` is the reduction method to use for the loss function. This can be:

         * ``"sum"``, which computes the sum of the loss over the batch. This is the default value if not specified.

         * ``"mean"``, which computes the mean of the loss over the batch.
   
      * ``"latent"`` is a dictionary of options for the latent space loss. It can contain the following options:
      
         * ``"norm_type"`` is the method to use to normalize the loss when reporting it. This can be:

            * ``"none"``, which does not normalize the loss.

            * ``"n_samples"``, which normalizes the loss by the number of samples in the batch. This is the default value if not specified.

            * ``"n_samples * latent_dim"``, which normalizes the loss by the number of samples times the latent dimension.
      
      * ``"decoder"`` is a dictionary of options for the reconstruction loss. It can contain the following options:
      
         * ``"norm_type"`` is the method to use to normalize the loss when reporting it. This can be:

            * ``"none"``, which does not normalize the loss.

            * ``"n_samples"``, which normalizes the loss by the number of samples in the batch. This is the default value if not specified.

            * ``"n_samples * n_genes"``, which normalizes the loss by the number of samples times the number of genes.
      
      * ``"total"`` is a dictionary of options for the total loss. It can contain the following options:
      
         * ``"norm_type"`` is the method to use to normalize the loss when reporting it. This can be:

            * ``"none"``, which does not normalize the loss.

            * ``"n_samples"``, which normalizes the loss by the number of samples in the batch. This is the default value if not specified.

            * ``"n_samples * n_genes"``, which normalizes the loss by the number of samples times the number of genes.

* ``"scheme_options"`` is a dictionary of options specific to the optimization scheme. The options vary depending on the scheme type and the latent space type.

   * ``"initialization"`` controls how candidate representations are
     drawn for a TorchGMM latent space. It contains an integer ``"seed"``
     (or a list of integer ``"seeds"`` for ``two_opt_multiseed``) and a
     ``"mode"``:

      * ``"sample_keyed"`` is the default. BulkDGD derives
        an independent stream from the seed and the sample ID, so a
        sample gets the same candidates whatever its position, the
        samples beside it, or its chunk.

      * ``"legacy_positional"`` consumes one stream in input order, so
        the candidate assigned to a sample depends on its position and on
        the size of the input chunk. Configurations that reproduce
        published results must request this mode explicitly.

      * ``"legacy_indexed"`` reconstructs the candidates
        ``"legacy_positional"`` assigned, while allowing the counts table
        to be reordered or subsetted. In addition to the seed, it requires
        ``"index_file"``, ``"original_n_samples"`` and ``"chunk_size"``.
        ``"index_file"`` is a one-column CSV whose row names are sample
        IDs and whose integer values are their zero-based absolute
        positions in the original input::

          sample_id,position
          SRR8261581,0
          SRR8261574,1

        ``"chunk_size"`` is the size of the original outer input chunk,
        not ``data_loader_options.batch_size``. Sample IDs and
        positions must both be unique.

   * For the ``two_opt`` scheme with the legacy GMM (``"lgmm"``):

      * ``"loss_reduction_type"`` is the reduction method to use for the loss function. This can be:

         * ``"sum"``, which computes the sum of the loss over the batch. This is the default value if not specified.

         * ``"mean"``, which computes the mean of the loss over the batch.

      * ``"contamination"`` is the weight of an outlier component mixed into each gene's likelihood, a negative binomial with the same mean and r-value ``"contamination_r"``. ``0.0`` disables it. This is a float in [0, 1) that defaults to ``0.0``.

      * ``"contamination_r"`` is the r-value of the outlier component. This is a positive float that defaults to ``0.05``.

      * ``"warm_start"`` is an optional dictionary that seeds the search with a data-driven starting point instead of leaving every candidate to a draw from the mixture. It can contain:

         * ``"pth_file"`` is the path to a fitted ridge predictor, written by :func:`bulkdgd.core.warmstart.fit_from_model_dir`. The prediction **replaces** one of each sample's candidates (``"n_rep_per_comp"`` times the number of components) rather than being added to them, so the number of candidates is unchanged. It is absent by default, in which case every candidate comes from the mixture.

           The predictor maps a sample's counts to its representation and is fitted on the model's own training representations. It only chooses where the search starts.

      * ``"optimization_1"`` is a dictionary of options for the first optimization round. It can contain the following options:

         * ``"epochs"`` is the number of epochs to run the first optimization for. This is a positive integer and defaults to ``10``.

         * ``"optimizer_type"`` is the type of optimizer to use. This can be:

            * ``"adam"``, which uses the Adam optimizer.

            * ``"adamw"``, which uses the AdamW optimizer. This is the default.

            * ``"lbfgs"``, which uses the L-BFGS optimizer.

         * ``"optimizer_options"`` is a dictionary of options for the optimizer. For ``"adam"`` and ``"adamw"``, it can contain the following options:

            * ``"lr"`` is the learning rate. This is a positive float that defaults to ``0.01``.

            * ``"weight_decay"`` is the weight decay. This is a non-negative float that defaults to ``0.0``.

            * ``"betas"`` is a list of two floats that specify the beta parameters for the optimizer. The defaults are ``[0.9, 0.999]``.

            For ``"lbfgs"``, it can contain the following options:

            * ``"lr"`` is the learning rate. This is a positive float that defaults to ``1.0``.

            * ``"max_iter"`` is the maximum number of L-BFGS iterations per epoch. This is a positive integer that defaults to ``100``.

            * ``"history_size"`` is the number of past updates kept to approximate the curvature. This is a positive integer that defaults to ``20``.

            * ``"line_search_fn"`` is the line search to use. This can be:

               * ``"strong_wolfe"``, which uses the strong Wolfe line search. This is the default.

               * ``None``, which uses a fixed step.

            * ``"tolerance_grad"`` is the gradient below which the optimizer stops iterating. This is a positive float that defaults to ``1e-9``.

            * ``"tolerance_change"`` is the change in the parameters or the loss below which the optimizer stops iterating. This is a positive float that defaults to ``1e-12``.

         * ``"noise_type"`` is the type of noise to inject into the representations while they are being optimized. This can be:

            * ``None`` or ``"none"``, which does not inject any noise. This is the default.

            * ``"gaussian"``, which adds Gaussian noise to the decoder's input only, as ``"train_noise_type"`` ``"gaussian"`` without ``"persist"`` does in :ref:`train_config_options`, with the defaults below.

         * ``"noise_options"`` is a dictionary of options for that noise, used when ``"noise_type"`` is ``"gaussian"``. It can contain:

            * ``"scale"`` is the base scale of the noise. This is a non-negative float that defaults to ``0.0``, which disables the injection.

            * ``"start"`` and ``"end"`` are the multipliers the scale is cosine-annealed between, from this round's first epoch to its last. These are non-negative floats defaulting to ``1.0`` and ``0.01``.

            * ``"within_radius_prob"`` is the probability defining the hypersphere the noise is normalized against. This is a float in [0, 1] that defaults to ``0.95``.

            * ``"gain"`` is a final multiplier on the noise. This is a non-negative float that defaults to ``1.0``.

            Each round is annealed over its own epochs and is configured separately.

      * ``"optimization_2"`` is a dictionary of options for the second optimization round. It has the same structure as ``"optimization_1"`` but with ``"epochs"`` defaulting to ``50``.

   * For the ``two_opt`` scheme with TorchGMM (``"tgmm"``):

      * ``"loss_reduction_type"`` is the reduction method to use for the loss function. This can be:

         * ``"sum"``, which computes the sum of the loss over the batch. This is the default value if not specified.

         * ``"mean"``, which computes the mean of the loss over the batch.

      * ``"contamination"`` is the weight of an outlier component mixed into each gene's likelihood, a negative binomial with the same mean and r-value ``"contamination_r"``. ``0.0`` disables it. This is a float in [0, 1) that defaults to ``0.0``.

      * ``"contamination_r"`` is the r-value of the outlier component. This is a positive float that defaults to ``0.05``.

      * ``"latent_loss_calculation"`` is a dictionary of options for the latent space loss calculation. It can contain:

         * ``"lambda"`` is the weight to use for the latent space loss. This is a non-negative float that defaults to ``1.0``.

      * ``"warm_start"`` is an optional dictionary that seeds the search with a data-driven starting point instead of leaving every candidate to a draw from the mixture. It can contain:

         * ``"pth_file"`` is the path to a fitted ridge predictor, written by :func:`bulkdgd.core.warmstart.fit_from_model_dir`. The prediction **replaces** one of each sample's candidates (``"n_rep_per_comp"`` times the number of components) rather than being added to them, so the number of candidates is unchanged. It is absent by default, in which case every candidate comes from the mixture.

           The predictor maps a sample's counts to its representation and is fitted on the model's own training representations. It only chooses where the search starts.

      * ``"optimization_1"`` is a dictionary of options for the first optimization round. It can contain the following options:

         * ``"epochs"`` is the number of epochs to run the first optimization for. This is a positive integer and defaults to ``300``.

         * ``"optimizer_type"`` is the type of optimizer to use. This can be:

            * ``"adam"``, which uses the Adam optimizer.

            * ``"adamw"``, which uses the AdamW optimizer. This is the default.

            * ``"lbfgs"``, which uses the L-BFGS optimizer.

         * ``"optimizer_options"`` is a dictionary of options for the optimizer. For ``"adam"`` and ``"adamw"``, it can contain the following options:

            * ``"lr"`` is the learning rate. This is a positive float that defaults to ``0.01``.

            * ``"weight_decay"`` is the weight decay. This is a non-negative float that defaults to ``0.0``.

            * ``"betas"`` is a list of two floats that specify the beta parameters for the optimizer. The defaults are ``[0.9, 0.999]``.

            For ``"lbfgs"``, it can contain the following options:

            * ``"lr"`` is the learning rate. This is a positive float that defaults to ``1.0``.

            * ``"max_iter"`` is the maximum number of L-BFGS iterations per epoch. This is a positive integer that defaults to ``100``.

            * ``"history_size"`` is the number of past updates kept to approximate the curvature. This is a positive integer that defaults to ``20``.

            * ``"line_search_fn"`` is the line search to use. This can be:

               * ``"strong_wolfe"``, which uses the strong Wolfe line search. This is the default.

               * ``None``, which uses a fixed step.

            * ``"tolerance_grad"`` is the gradient below which the optimizer stops iterating. This is a positive float that defaults to ``1e-9``.

            * ``"tolerance_change"`` is the change in the parameters or the loss below which the optimizer stops iterating. This is a positive float that defaults to ``1e-12``.

         * ``"noise_type"`` is the type of noise to inject into the representations while they are being optimized. This can be:

            * ``None`` or ``"none"``, which does not inject any noise. This is the default.

            * ``"gaussian"``, which adds Gaussian noise to the decoder's input only, as ``"train_noise_type"`` ``"gaussian"`` without ``"persist"`` does in :ref:`train_config_options`, with the defaults below.

         * ``"noise_options"`` is a dictionary of options for that noise, used when ``"noise_type"`` is ``"gaussian"``. It can contain:

            * ``"scale"`` is the base scale of the noise. This is a non-negative float that defaults to ``0.0``, which disables the injection.

            * ``"start"`` and ``"end"`` are the multipliers the scale is cosine-annealed between, from this round's first epoch to its last. These are non-negative floats defaulting to ``1.0`` and ``0.01``.

            * ``"within_radius_prob"`` is the probability defining the hypersphere the noise is normalized against. This is a float in [0, 1] that defaults to ``0.95``.

            * ``"gain"`` is a final multiplier on the noise. This is a non-negative float that defaults to ``1.0``.

            Each round is annealed over its own epochs and is configured separately.

      * ``"optimization_2"`` is a dictionary of options for the second
        optimization round. It has the same structure as
        ``"optimization_1"`` but with ``"epochs"`` defaulting to
        ``500``.
