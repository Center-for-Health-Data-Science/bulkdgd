# `configs/model`

Last updated: 07/09/2023

## `model.yaml`

Example of a YAML configuration file used for `dgd_get_representations` and `dgd_get_probability_desity` (`-cm`, `--config-file-model` option).

The provided configuration file is compatible with the parameters used in the trained deep generative decoder (not uploaded on GitHub because of its size, it can be found [here](https://drive.google.com/file/d/1SZaoazkvqZ6DBF-adMQ3KRcy4Itxsz77/view?usp=sharing)) and the Gaussian mixture model (`bulkDGD/data/model/gmm.pth`). 

Suppose you want to change the model components' architectures. In that case, you need to re-train the different components of the model, provide the corresponding PyTorch files, and update the given configuration file accordingly.

The configuration file has the following structure (`int`, `str`, `float`, `bool`, etc., represent the data type expected for each field):

```yaml


#---------------------- Gaussian mixture model -----------------------#


# PyTorch file containing the parameters of the trained GMM.
# Make sure that the file contains a GMM whose parameters
# fit the architecture specified in the 'options' section.
gmm_pth_file: str

# Dimensionality of the Gaussian mixture model
dim: int

# Number of components in the mixture
n_comp: int

# Type of covariance matrix. Choices are 'diagonal',
# 'fixed', and 'isotropic'
cm_type: str

# Name of the prior over the means of the components
# of the Gaussian mixture model
means_prior_name: str

# Options to set up the prior (they vary according to
# the type of prior)
means_prior_options:

  # Radius of the soft ball
  radius: int

  # Sharpness of the soft boundary of the ball
  sharpness: int

# Name of the prior over the weights of the components
# of the Gaussian mixture model
weights_prior_name: str

# Options to set up the prior (they vary according to
# the type of prior)
weights_prior_options:

  # Alpha of the Dirichlet distribution determining
  # the uniformity of the weights of the components
  # in the mixture
  alpha: int

# Name of the prior over the log-variances of the components
# of the Gaussian mixture model
log_var_prior_name: str

# Options to set up the prior (they vary according to
# the type of prior)
log_var_prior_options:

  # Mean of the Gaussian distribution calculated
  # as 2 * log(mean)
  mean : float

  # Standard deviation of the Gaussian distribution
  stddev: float


#------------------------------ Decoder ------------------------------#


# PyTorch file containing the parameters of the trained decoder.
# Make sure that the file contains a decoder whose parameters
# fit the architecture specified in the 'options' section
dec_pth_file: str
    
# Number of units in the hidden layers
n_units_hidden_layers: list of int

# Initial "number of successes" (r) for the negative
# binomial distributions modeling the output layer
r_init: int

# Name of the activation function to be used in the
# output layer
activation_output: str


#------------------------------- Genes -------------------------------#


# Plain text file containing the list of genes included in the model
genes_txt_file: default
```

