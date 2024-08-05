.. _model_config_options:


Configuration for creating an instance of the DGD model
=======================================================

To create a new instance of :class:`core.model.DGDModel`, we need to set the following options:

* ``input_dim``, the dimensionality of the model's input.

* ``gmm_options``, a dictionary of options to set up the Gaussian mixture model.

* ``dec_options``, a dictionary of options to set up the decoder.

* ``genes_txt_file``, the path to a plain text file containing the list of genes that will be included in the model.

If we want to load a model that has already been trained, we should provide also the following options:

* ``gmm_pth_file``, a PyTorch file containing the trained paramenters of the Gaussian mixture model.

* ``dec_pth_file``, a PyTorch file containing the trained parameters of the decoder.

This is how the ``gmm_options`` dictionary should look like:

.. code-block:: python

   {# Set the number of components in the Gaussian mixture model.
    #
    # Type: int.
    n_comp: 45,

    # Set the type of covariance matrix used by the Gaussian mixture
    # model.
    #
    # Type: str.
    # 
    # Options:
    # - 'fixed' for a fixed covariance matrix.
    # - 'isotropic' for an isotropic covariance matrix.
    # - 'diagonal' for a diagonal covariance matrix.
    cm_type: diagonal

    # Set the prior distribution over the means of the components of
    # the Gaussian mixture model.
    #
    # Type: str.
    #
    # Options:
    # - 'softball' for a softball distribution.
    means_prior_name: softball

    # Set the options to set up the prior distribution (they vary
    # depending on the prior defined by 'means_prior_name').
    means_prior_options:

      # Set these options if 'means_prior_name' is 'softball'.

      # Set the radius of the soft ball.
      #
      # Type: int.
      radius: 7

      # Set the sharpness of the soft boundary of the ball.
      #
      # Type: int.
      sharpness: 10

    # Set the prior distribution over the weights of the components of
    # the Gaussian mixture model.
    #
    # Type: str.
    #
    # Options:
    # - 'dirichlet' for a Dirichlet distribution.
    weights_prior_name: dirichlet

    # Set the options to set up the prior (they vary according to the
    # prior defined by 'weights_prior_name').
    weights_prior_options:

      # Set these options if 'weights_prior_name' is 'dirichlet'.

      # Set the alpha of the Dirichlet distribution determining the
      # uniformity of the weights of the components in the Gaussian
      # mixture model.
      #
      # Type: int.
      alpha: 5

    # Set the prior distribution over the log-variances of the
    # components of the Gaussian mixture model.
    #
    # Type: str.
    #
    # Options:
    # - 'gaussian' for a Gaussian distribution.
    log_var_prior_name: gaussian

    # Set the options to set up the prior (they vary according to the
    # prior defined by 'log_var_prior_name').
    log_var_prior_options:

      # Set these options if 'log_var_prior_name' is 'gaussian'.

      # Set the mean of the Gaussian distribution calculated as
      # 2 * log(mean).
      #
      # Type: float.
      mean : 0.1

      # Set the standard deviation of the Gaussian distribution.
      #
      # Type: float.
      stddev: 1.0}

And this is how the ``dec_options`` dictionary should look like:

.. code-block:: python

   {# Set the number of units in the hidden layers.
    #
    # Type: list of int.
    n_units_hidden_layers: [500, 8000]

    # Set the name of the decoder's output module.
    #
    # Type: str.
    #
    # Options:
    # - 'nb_feature_dispersion' for negative binomial distributions
    #   with means learned per gene per sample and r-values learned per
    #   gene.
    # - 'nb_full_dispersion' for negative binomial distributions with
    #   both means and r-values learned per gene per sample.
    output_module_name: nb_feature_dispersion

    # Set the options for the output module.
    output_module_options:

      # Set the name of the activation function in the output module.
      #
      # Type: str.
      #
      # Options:
      # - 'sigmoid' for a sigmoid function.
      # - 'softplus' for a softplus function.
      activation: softplus

      # Set the initial r-value for the negative binomial distributions
      # modeling the genes' counts.
      #
      # Type: int.
      r_init: 2

If we are loading the options from a YAML configuration file similar to those provided in the ``bulkDGD/ioutil/configs/model`` directory, we can set up the model as follows:

.. code-block:: python

   # Import 'ioutil' and the 'core.model' module.
   from bulkDGD import ioutil
   from bulkDGD.core import model

   # Let's assume we load the 'model_untrained.yaml' configuration file.

   # Load the configuration from the configuration file.
   config = ioutil.load_config_model(config_file = "model_untrained")

   # The configuration contains a 'input_dim' section, a 'gmm_options'
   # section, and a 'dec_opt' section.

   # Initialize the model.
   dgd_model = model.DGDModel(**config)
