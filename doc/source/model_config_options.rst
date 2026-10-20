.. _model_config_options:

Configuration for creating an instance of the BulkDGD model
===========================================================

Creating a new instance of :class:`core.model.BulkDGD` requires a number of options.

These options can be passed as a nested dictionary or specified in a YAML configuration file.

The function that loads the configuration file is :func:`bulkdgd.ioutil.load_config_model`.

.. _model_shipped:

The trained model that ships with the package
---------------------------------------------

None of these options has to be set to obtain the published model. An architecture is described by hand only to train a new model:

.. code-block:: python

   from bulkdgd.core.model import BulkDGD

   # The trained model, as published.
   model = BulkDGD()

   # Another member of the ensemble it belongs to.
   model = BulkDGD(seed = "seed41")

A :class:`bulkdgd.core.model.BulkDGD` built with no architecture reads the architecture, the gene list, the scaling factor and the precision from the configuration shipped for the requested member, and loads that member's fitted parameters. For every member this is 32 latent dimensions, a ``"tgmm"`` mixture of **48** components with a ``"tied_spherical"`` covariance, a ``"nb_full_dispersion"`` decoder with hidden layers of 500 and 8000 units, the curated list of **14,740** genes, the ``"median"`` scaling factor and ``"float64"``.

``"seed"`` selects which member is loaded, and defaults to ``"seed37"``, the base member. It cannot be combined with ``"latent_dim"``, ``"latent_options"`` and ``"decoder_options"``, which describe an architecture to be built from scratch: giving them together with a seed raises an error. These three must be given together, or not at all.

Any other argument passed is kept, so a single value can be overridden without restating the rest: ``BulkDGD(device = "cuda")`` is still the trained model.

The ensemble the member belongs to is loaded the same way:

.. code-block:: python

   from bulkdgd.ensemble.ensemble import BulkDGDEnsemble

   ensemble = BulkDGDEnsemble()

which loads the fifteen members ``seed37``, ``seed41``, ``seed43``, ``seed47``, ``seed53``, ``seed59``, ``seed61``, ``seed67``, ``seed71``, ``seed73``, ``seed79``, ``seed83``, ``seed89``, ``seed97`` and ``seed101``. They differ only in the seed they were trained with: the architecture, the gene universe and the train/test split are the same for all of them. The members are built one at a time, when they are requested, so an ensemble asked for one member loads only that member.

Where the shipped files are
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each member has one directory of fitted parameters and one of configuration, named for its seed:

* ``bulkdgd/data/model/<seed>/gmm.pth`` holds the member's fitted Gaussian mixture.

* ``bulkdgd/configs/model/<seed>/model.yaml`` holds its architecture, and ``bulkdgd/configs/model/<seed>/seeds.yaml`` the seeds it was trained with.

* ``bulkdgd/data/model/genes/genes.txt`` holds the gene list, which every member shares.

The decoders are not shipped. Each is 1.79 GiB in ``float64``, so they are release assets named ``dec_<seed>.pth``, one per seed and release. Each is downloaded once, the first time that member is built, into that member's own directory as ``dec.pth``. See the :doc:`installation instructions <installation>` for the offline case.

.. note::

   A per-member ``model.yaml`` names an architecture and no parameter files, so loading it on its own gives an untrained model of the right shape. The command-line tools build the model from the single configuration file they are given, so that file must also name the parameters: add ``"latent_pth_file"`` to ``"latent_options"`` and ``"decoder_pth_file"`` to ``"decoder_options"``.

   Setting both to ``"default"`` always gives the base member. Every member has the same shape, so a configuration for another member that sets ``"default"`` gets ``seed37``'s mixture and decoder without any error. To load any other member, name its two files explicitly.

Options
-------

The options that can be specified are described below.

* ``"genes_txt_file"`` is the path to the plain text file containing the list of genes used in the model. This file should contain one gene name per line in Ensembl ID format.

* ``"latent_dim"`` is the dimensionality of the latent space, a positive integer.

* ``"latent_type"`` is the type of latent space to use. This is a string that can take one of the following values: ``"lgmm"`` for the legacy Gaussian Mixture Model implementation, and ``"tgmm"`` for the TorchGMM implementation.

* ``"latent_options"`` is a dictionary containing the options for the GMM. The options that can be specified in this dictionary depend on the type of GMM used.

   * If you are loading a trained model, ``"latent_pth_file"`` is the path to the .pth file containing the state dictionary of the latent space. 

   * For the legacy GMM (``"lgmm"``):

      * ``"n_components"`` is the number of Gaussian components in the GMM, a positive integer.

      * ``"covariance_type"`` is the type of covariance to use. This is a string that can take one of the following values:
         
         * ``"fixed"`` for a fixed covariance matrix.
         * ``"isotropic"`` for an isotropic covariance matrix.
         * ``"diagonal"`` for a diagonal covariance matrix. This is the default.
      
      * ``"means_prior_type"`` is the type of prior distribution used for the means of the GMM components. This is a string that can take one of the following values:
         
         * ``"softball"`` for a softball prior. This is the default.
      
      * ``"means_prior_options"`` is a dictionary containing the options for the means prior distribution. The options that can be specified in this dictionary depend on the type of prior distribution used.

         * For the softball prior (``"softball"``):

            * ``"radius"`` is the radius of the sphere on which the means are constrained to lie, a positive float.
            * ``"sharpness"`` is how sharply the means are constrained to lie on the sphere, a positive float.
      
      * ``"weights_prior_type"`` is the type of prior distribution used for the weights of the GMM components. This is a string that can take one of the following values:
         
         * ``"dirichlet"`` for a Dirichlet prior. This is the default.
      
      * ``"weights_prior_options"`` is a dictionary containing the options for the weights prior distribution. The options that can be specified in this dictionary depend on the type of prior distribution used.
         
         * For the Dirichlet prior (``"dirichlet"``):

            * ``"alpha"`` is the concentration parameter of the Dirichlet prior, a positive float.
      
      * ``"log_var_prior_type"`` is the type of prior distribution used for the log-variances of the GMM components. This is a string that can take one of the following values:

         * ``"gaussian"`` for a Gaussian prior. This is the default.
      
      * ``"log_var_prior_options"`` is a dictionary containing the options for the log-variances prior distribution. The options that can be specified in this dictionary depend on the type of prior distribution used.

         * For the Gaussian prior (``"gaussian"``):

            * ``"mean"`` is the mean of the Gaussian prior, a float.
            * ``"stddev"`` is the standard deviation of the Gaussian prior, a positive float.
   
   * For the TorchGMM implementation (``"tgmm"``):

      * ``"n_components"`` is the number of Gaussian components in the GMM, a positive integer. If not specified, the default value is ``35``. The models that ship with the package were trained with ``48``.

      * ``"covariance_type"`` is the type of covariance to use. This is a string that can take one of the following values:

         * ``"full"`` for a full covariance matrix.
         * ``"diag"`` for a diagonal covariance matrix.
         * ``"spherical"`` for a spherical covariance matrix. This is the default.
         * ``"tied_full"`` for a full tied covariance matrix.
         * ``"tied_diag"`` for a diagonal tied covariance matrix.
         * ``"tied_spherical"`` for a spherical tied covariance matrix.

     * ``"init_means"`` is the method used to initialize the means of the GMM components. This is a string that can take one of the following values:

         * ``"kmeans"`` for K-means initialization.
         * ``"kpp"`` for K-means++ initialization.
         * ``"random"`` for random initialization.
         * ``"points"`` for initialization using random points from the dataset.
         * ``"maxdist"`` for initialization using the points with the maximum determinant of the covariance matrix. This is the default value if not specified, and it is what the shipped models were trained with.
    
     * ``"init_weights"`` is the method used to initialize the weights of the GMM components. This is a string that can take one of the following values:
         
         * ``"uniform"`` for uniform initialization. This is the default value if not specified.
         * ``"random"`` for random initialization.
         * ``"kmeans"`` for K-means initialization.
   
     * ``"init_covariances"`` is the method used to initialize the covariances of the GMM components. This is a string that can take one of the following values:
         
         * ``"empirical"`` for empirical covariance initialization. This is the default value if not specified.
         * ``"eye"`` for identity covariance initialization.
         * ``"random"`` for random initialization.
         * ``"global"`` for global covariance initialization.
   
     * ``"tol"`` is the convergence threshold based on relative improvement in log-likelihood. If not specified, the default value is ``1e-4``.

     * ``"reg_covar"`` is the non-negative regularization added to the diagonal of the covariance, which keeps the covariance matrices positive definite. If not specified, the default value is ``1e-6``.

     * ``"n_init"`` is the number of initializations to perform. The default value is 1.
   
     * ``"random_state"`` is the random seed used for initialization, an integer. If not specified, the default value is None, and the seed is determined by PyTorch's internal seed.

       This affects more than the GMM. When it is set, ``tgmm`` calls :func:`torch.manual_seed` with it as it places the mixture's components, which reseeds PyTorch's global generator for everything drawn afterwards: the representations' initialization, the order of the batches, and the noise added during training. Leaving it unset and calling :func:`bulkdgd.reproducibility.set_seeds` once, before the model is built, makes the placement reproducible along with everything else, without this side effect. See :doc:`reproducibility <reproducibility>`.

     * ``"cem"`` is a boolean that specifies whether to use the classification expectation-maximization (CEM) algorithm for fitting the GMM. If not specified, the default value is False, and the standard expectation-maximization (EM) algorithm is used.

* ``"gmm_final"`` is an optional dictionary containing the options for a second Gaussian mixture model, fitted to the representations *after* training and saved to its own file, ``gmm_final.pth``.

  This is not the prior. The prior is the mixture in ``"latent_options"``: training uses it, ``gmm.pth`` holds it, and finding a representation for a new sample uses it. It is not replaced, and ``gmm_final.pth`` is never used in the search for a representation.

  The final mixture is for questions about the *density* of the latent space: a sample's probability density, which component it belongs to, how atypical it is, and which directions a sampler should draw in. During training the mixture's covariance has little effect on the representations, so the prior can use the cheapest covariance that works. For questions about density, the covariance is what matters.

  A model without this section writes no such file.

   * ``"covariance_type"`` is the type of covariance the final mixture should have, from the same set of values as the ``"tgmm"`` covariance types above. There is no default: it must be given.

   * ``"shrinkage"`` is how far each component's own covariance is pulled back towards the one shared by all of them, between 0.0 and 1.0. If not specified, the default value is 0.0.

     It has no effect on the tied covariance types, which are already the shared covariance, and it is **required** for ``"full"``: a per-component full covariance in 32 dimensions has 528 parameters per component, fitted only from that component's samples, and an unshrunk fit returns negative variances.

   * ``"reg_covar"`` is the value added to the diagonal of the covariance. If not specified, the trained mixture's own value is used.

* ``"decoder_options"`` is a dictionary containing the options for the decoder.

   * If you are loading a trained model, ``"decoder_pth_file"`` is the path to the .pth file containing the state dictionary of the decoder.

   * ``"n_units_hidden_layers"`` is a list of positive integers that specifies the number of units in each hidden layer of the decoder. For example, if ``"n_units_hidden_layers"`` is set to ``[128, 64]``, then the decoder will have two hidden layers, with 128 units in the first hidden layer and 64 units in the second hidden layer.

   * ``"activations"`` is a list of strings that specifies the activation function to use in each hidden layer of the decoder. The length of this list should be equal to the length of the ``"n_units_hidden_layers"`` list. Each string in this list can take one of the following values:
         
      * ``"relu"`` for ReLU activation.
      * ``"elu"`` for ELU activation.
      * ``"gelu"`` for GELU activation (smooth everywhere, so the decoder map has no activation boundaries).
      
   * ``"dropout"`` is a float between 0 and 1 that specifies the dropout rate to use in the decoder. This is the fraction of the input units to drop during training. If not specified, the default value is 0, which means that no dropout will be applied.

   * ``"output_module_name"`` is the name of the output module used in the decoder. This is a string that can take one of the following values:
         
      * ``"poisson"`` for the Poisson output module.
      * ``"nb_feature_dispersion"`` for the negative binomial output module with r-values learned per gene.
      * ``"nb_full_dispersion"`` for the negative binomial output module with r-values learned per gene and sample.

  No other names are accepted: a name not in this list raises an error when the configuration is loaded.

  .. note::

     ``"nb_full_dispersion_shrunk"`` and ``"nb_full_dispersion_shrunk_tied"``, and their ``"shrinkage_lambda"`` option, are **not supported**. A configuration that names either of them is rejected when it is loaded.

   * ``"output_module_options"`` is a dictionary containing the options for the output module. The options that can be specified in this dictionary depend on the type of output module used.

      * For the Poisson output module (``"poisson"``):

         * ``"activation"`` is the activation function to use in the output module. This is a string that can take one of the following values:
               
            * ``"sigmoid"`` for sigmoid activation.
            * ``"softplus"`` for softplus activation.
         
      * For the negative binomial output module with r-values learned per gene (``"nb_feature_dispersion"``):

         * ``"activation"`` is the activation function to use in the output module. This is a string that can take one of the following values:
               
            * ``"sigmoid"`` for sigmoid activation.
            * ``"softplus"`` for softplus activation.
            
         * ``"r_init"`` is the initial value of the r-values in the negative binomial output module, a positive integer. If not specified, the default value is ``2``.
         
      * For the negative binomial output module with r-values learned per gene and sample (``"nb_full_dispersion"``):

         * ``"activation"`` is the activation function to use in the output module. This is a string that can take one of the following values:

            * ``"sigmoid"`` for sigmoid activation.
            * ``"softplus"`` for softplus activation.

* ``"scaling_factor"`` is how a sample's scaling factor is computed. The scaling factor is the number the decoder's predicted means are multiplied by to put them on the scale of the sample's own counts. This is a string that can take one of the following values:

   * ``"mean"`` for the mean count over all of the sample's genes. This is the default.
   * ``"median"`` for the median count over all of the sample's genes.

  The mean is sensitive to the few genes that take a large and variable share of a library, such as the mitochondrial genes, whose share reflects how the sample was handled rather than the tissue it came from. The median is much less affected by them.

  The scaling factor belongs to the model rather than to a run of it, so it is set here and not in the training configuration. The decoder is fitted against the scaling factor, and the median of a sample's counts is about a third of its mean, so a model trained with one and used with the other has every predicted mean off by about a factor of three, without any error. :func:`bulkdgd.ioutil.load_config_model` reads the value back from this file whenever the model is loaded, so finding representations uses the same scaling factor as training.

  The median is only safe on a gene list filtered to the genes expressed in the samples. A median is zero as soon as half of a sample's genes are zero, and a scaling factor of zero makes every predicted mean zero and the negative binomial undefined. :class:`bulkdgd.core.dataclasses.GeneExpressionDataset` raises an error instead of accepting a zero.

  :meth:`bulkdgd.core.model.BulkDGD.impute` works with both scaling factors when the unmeasured counts are given as NaN. For a sample with only some genes measured, the scaling factor is estimated from the measured genes. With ``"mean"``, the unmeasured part of the sum over the genes is filled in with the model's own expectation. With ``"median"``, the median of the measured counts is multiplied by the ratio of the model's predicted median over all genes to its predicted median over the measured genes. Passing a panel of measured genes as ``genes_measured`` works only with ``"mean"``, and raises an error on a median-scaled model.

* ``"dtype"`` is the precision the model's parameters are built in. This is a string that can take one of the following values:

   * ``"float32"`` for single precision, which is torch's own default.
   * ``"float64"`` for double precision. This is the default.

  The precision cannot be changed after the model is built. A module's parameters are created in torch's default dtype at the moment the module is constructed, and :meth:`torch.nn.Module.load_state_dict` copies a checkpoint *into* the existing parameters, casting as it goes: a float64 checkpoint read into a model built in float32 gives a float32 decoder, without a warning. The Gaussian mixture behaves differently: ``tgmm`` keeps the tensors it is handed rather than copying into its own, so it stays float64 while the decoder becomes float32, and the first matrix multiplication between the two raises ``mat1 and mat2 must have the same dtype``.

  Like ``"scaling_factor"``, this option belongs to the model and is read back from this file whenever the model is loaded, so a model trained in double precision is loaded in double precision without the caller setting torch's default. Torch's default dtype is restored once the model is built, since it is global.


