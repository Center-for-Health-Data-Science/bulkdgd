# `configs/model`

Last updated: 27/07/2023

## `config_model.yaml`

Example of a YAML configuration file used for `dgd_get_representations` and `dgd_get_probability_desity` (`-cm`, `--config-file-model` option).

The provided configuration file is compatible with the parameters used in the trained deep generative decoder (not uploaded on GitHub because of its size, it can be found [here](https://drive.google.com/file/d/1SZaoazkvqZ6DBF-adMQ3KRcy4Itxsz77/view?usp=sharing)), Gaussian mixture model (`bulkDGD/data/model/gmm.pth`), and representation layer (`bulkDGD/data/model/rep.pth`). 

Suppose you want to change the model components' architectures. In that case, you need to re-train the different components of the model, provide the corresponding PyTorch files, and update the given configuration file accordingly.

The configuration file has the following structure (`int`, `str`, `float`, `bool`, etc., represent the data type expected for each field):

```yaml
# Dimensionality of the latent (= representation) space
dim_latent: int


# Section containing the options for setting up the
# decoder
dec:

  # PyTorch file containing the parameters of the trained decoder.
  # Make sure that the file contains a decoder whose parameters
  # fit the architecture specified in the 'options' section.
  pth_file: str
  
  # Section containing the options to initialize the decoder
  options:
  
    # The number of neurons in the input layer of the decoder
    # does not need to be specified since it corresponds to
    # the dimensionality of the latent space

    # Number of neurons in the first hidden layer
    n_neurons_hidden1: int

    # Number of neurons in the second hidden layer
    n_neurons_hidden2: int

    # Number of neurons in the output layer (= number of
    # genes the decoder has been trained on)
    n_neurons_out: int

    # Initial "number of successes" (r) for the negative
    # binomial distributions modeling the output layer
    r_init: int

    # Type of scaling used in the negative binomial
    # distributions. Choices are: 'library', 'total_count',
    # 'mean', and 'median', with 'library' and 'total_count'
    # determining the selection of the sigmoid activation function,
    # and 'mean' and 'median' meaning a the relu activation
    # function will be used
    scaling_type: str


# Section containing the options for setting up the Gaussian
# mixture model (GMM)
gmm:

  # PyTorch file containing the parameters of the trained GMM.
  # Make sure that the file contains a GMM whose parameters
  # fit the architecture specified in the 'options' section.
  pth_file: str

  # Section containing the options to initialize the Gaussian
  # mixture model (GMM)
  options:

    # Number of components in the mixture
    n_comp: int

    # Type of covariance matrix. Choices are 'diagonal',
    # 'fixed', and 'isotropic'
    cm_type: str

    # Parameters to initialize the Gaussian prior
    # on the log-variants of the components:
    # - mean: 2 * log(logbeta_params[0])
    # - stddev: logbeta_params[1]
    logbeta_params: list of two floats

    # Alpha of the Dirichlet distribution determining
    # the uniformity of the weights of the components
    # in the mixture
    alpha: int

  # Prior on the means of the Gaussians
  mean_prior:

    # Type of prior (e.g., "softball")
    type: str

    # Section containing the options for the prior
    # (they vary according to the type of prior)
    options:

      # For the softball prior, radius of the ball
      radius: int

      # For the softball prior, sharpness of the soft
      # boundary
      sharpness: int


# Section containing the options for setting up the
# representation layer
rep_layer:

  # PyTorch file containing the parameters of the trained
  # representation layer. Make sure that the file contains
  # a representation layer whose parameters fit the
  # architecture specified in the 'options' section.
  pth_file: str

  # Section containing the options to initialize the
  # representation layer
  options:

    # Number of samples the representation layer
    # has been trained on
    n_samples: int
```

