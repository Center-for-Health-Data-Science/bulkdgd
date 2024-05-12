# `configs/model`

Last updated: 12/05/2024

## `model.yaml`

This is an example of a YAML configuration file containing the configuration for the DGD model and used by `dgd_get_representations`, `dgd_perform_dea`, and `dgd_get_probability_desity` (`-cm`, `--config-file-model` option).

The provided configuration file is compatible with the parameters used in the trained deep generative decoder (not uploaded on GitHub because of its size; it can be found [here](https://drive.google.com/file/d/1SZaoazkvqZ6DBF-adMQ3KRcy4Itxsz77/view?usp=sharing)) and the Gaussian mixture model (`bulkDGD/ioutil/data/gmm.pth`). 

Suppose you want to change the architectures of the model components. In that case, you need to retrain the different components of the model, provide the corresponding PyTorch files, and update the given configuration file accordingly.

The configuration can be loaded using the `bulkDGD.ioutil.load_config_model` function.

The configuration file has the following structure:

```yaml
# Configuration file containing the configuration for the full DGD
# model.


####################### GAUSSIAN MIXTURE MODEL ########################


# Set the PyTorch file containing the parameters of the trained GMM.
#
# Make sure that the file contains a GMM whose parameters fit the
# architecture specified in the 'options' section.
#
# Type: str.
#
# Default: 'default'.
gmm_pth_file: default

#---------------------------------------------------------------------#

# Set the dimensionality of the Gaussian mixture model.
#
# Type: int.
#
# Default: 50.
dim: 50

#---------------------------------------------------------------------#

# Set the number of components in the Gaussian mixture model.
#
# Type: int.
#
# Default: 45.
n_comp: 45

#---------------------------------------------------------------------#

# Set the type of covariance matrix used by the Gaussian mixture model.
#
# Type: str.
# 
# Options:
# - 'fixed' for a fixed covariance matrix.
# - 'isotropic' for an isotropic covariance matrix.
# - 'diagonal' for a diagonal covariance matrix.
#
# Default: 'diagonal'.
cm_type: diagonal

#---------------------------------------------------------------------#

# Set the prior distribution over the means of the components of the
# Gaussian mixture model.
#
# Type: str.
#
# Options:
# - 'softball' for a softball distribution.
#
# Default: 'softball'.
means_prior_name: softball

# Set the options to set up the prior distribution (they vary according
# to the prior defined by 'means_prior_name').
means_prior_options:

  # Set these options if 'means_prior_name' is 'softball'.

  # Set the radius of the soft ball.
  #
  # Type: int.
  #
  # Default: 7.
  radius: 7

  # Set the sharpness of the soft boundary of the ball.
  #
  # Type: int.
  #
  # Default: 10.
  sharpness: 10

#---------------------------------------------------------------------#

# Set the prior distribution over the weights of the components of the
# Gaussian mixture model.
#
# Type: str.
#
# Options:
# - 'dirichlet' for a Dirichlet distribution.
#
# Default: 'dirichlet'.
weights_prior_name: dirichlet

# Set the options to set up the prior (they vary according to the prior
# defined by 'weights_prior_name').
weights_prior_options:

  # Set these options if 'weights_prior_name' is 'dirichlet'.

  # Set the alpha of the Dirichlet distribution determining the
  # uniformity of the weights of the components in the Gaussian mixture
  # model.
  #
  # Type: int.
  #
  # Default: 5.
  alpha: 5

#---------------------------------------------------------------------#

# Set the prior distribution over the log-variances of the components
# of the Gaussian mixture model.
#
# Type: str.
#
# Options:
# - 'gaussian' for a Gaussian distribution.
#
# Default: 'gaussian'.
log_var_prior_name: gaussian

# Set the options to set up the prior (they vary according to the prior
# defined by 'log_var_prior_name').
log_var_prior_options:

  # Set these options if 'log_var_prior_name' is 'gaussian'.

  # Set the mean of the Gaussian distribution calculated as
  # 2 * log(mean).
  #
  # Type: float.
  #
  # Default: 0.1.
  mean : 0.1

  # Set the standard deviation of the Gaussian distribution.
  #
  # Type: float.
  #
  # Default: 1.0.
  stddev: 1.0


############################### DECODER ###############################


# Set the PyTorch file containing the parameters of the trained
# decoder.
#
# Make sure that the file contains a decoder whose parameters
# fit the architecture specified in the 'options' section.
#
# Type: str.
#
# Default: 'default'.
dec_pth_file: default

#---------------------------------------------------------------------#
    
# Set the number of units in the hidden layers.
#
# Type: list of int.
#
# Default: [500, 8000].
n_units_hidden_layers: [500, 8000]

#---------------------------------------------------------------------#

# Set the initial r-value for the negative binomial distributions
# modeling the output layer.
#
# Type: int.
#
# Default: 2.
r_init: 2

#---------------------------------------------------------------------#

# Set the name of the activation function to be used in the output
# layer of the decoder.
#
# Type: str.
#
# Options:
# - 'sigmoid' for a sigmoid function.
# - 'softplus' for a softplus function.
#
# Default: 'softplus'.
activation_output: softplus


################################ GENES ################################


# Set the plain text file containing the list of genes included in the
# DGD model.
#
# Type: str.
#
# Default: 'default'.
genes_txt_file: default

```

