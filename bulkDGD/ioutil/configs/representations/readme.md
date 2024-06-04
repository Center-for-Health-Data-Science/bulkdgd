# `configs/representations`

Last updated: 12/05/2024

## `one_opt.yaml`

This is an example of a YAML configuration file used for `dgd_get_representations` (`-cr`, `--config-file-rep` option) to get the representations using only one round of optimization.

The configuration can be loaded using the `bulkDGD.ioutil.load_config_rep` function.

The configuration file has the following structure:

```yaml
# Configuration file specifying the options to get the representations
# with only one round of optimization.


#######################################################################


# Set how many representations to initialize per sample per component
# of the Gaussian mixture model.
#
# Type: int.
#
# Default: 1.
n_rep_per_comp: 1


#######################################################################


# Set the options for the optimization.
optimization:

  # Set the number of epochs.
  #
  # Type: int.
  #
  # Default: 60.
  epochs: 60

  #-------------------------------------------------------------------#

  # Set the optimizer to be used.
  #
  # Type: str.
  #
  # Options:
  # - 'adam' for the Adam optimizer.
  #
  # Default: 'adam'.
  optimizer_name: adam

  # Set the options for the optimizer (they vary according to the
  # optimizer defined by 'optimizer_name').
  optimizer_options:

    # Set these options if 'optimizer_name' is 'adam'.

    # Set the learning rate.
    #
    # Type: float.
    #
    # Default: 0.01.
    lr: 0.01

    # Set the weight decay.
    #
    # Type: float.
    #
    # Default: 0.0.
    weight_decay: 0.0

    # Set the betas.
    #
    # Type: list of float.
    #
    # Default: [0.5, 0.9].
    betas: [0.5, 0.9]
```

## `two_opt.yaml`

This is an example of a YAML configuration file used for `dgd_get_representations` (`-cr`, `--config-file-rep` option) to get the representations using two rounds of optimization.

The configuration can be loaded using the `bulkDGD.ioutil.load_config_rep` function.

The configuration file has the following structure:

```yaml
# Configuration file specifying the options to get the representations
# with two rounds of optimization.


#######################################################################


# Set how many representations to initialize per sample per component
# of the Gaussian mixture model.
#
# Type: int.
#
# Default: 1.
n_rep_per_comp: 1


#######################################################################


# Set the options for the optimizations.
optimization:

  # Set the options for the first optimization round.
  opt1:

    # Set the number of epochs.
    #
    # Type: int.
    #
    # Default: 10.
    epochs: 10

    # Set the optimizer to be used.
    #
    # Type: str.
    #
    # Options:
    # - 'adam' for the Adam optimizer.
    #
    # Default: 'adam'.
    optimizer_name: adam

    # Set the options for the optimizer (they vary according to the
    # optimizer defined by 'optimizer_name').
    optimizer_options:

      # Set these options if 'optimizer_name' is 'adam'.

      # Set the learning rate.
      #
      # Type: float.
      #
      # Default: 0.01.
      lr: 0.01

      # Set the weight decay.
      #
      # Type: float.
      #
      # Default: 0.0
      weight_decay: 0.0

      # Set the betas.
      #
      # Type: list of float.
      #
      # Default: [0.5, 0.9].
      betas: [0.5, 0.9]

  #-------------------------------------------------------------------#

  # Set the options for the second optimization round.
  opt2:

    # Set the number of epochs.
    #
    # Type: int.
    #
    # Default: 50.
    epochs: 50

    # Set the optimizer to be used.
    #
    # Type: str.
    #
    # Options:
    # - 'adam' for the Adam optimizer.
    #
    # Default: 'adam'.
    optimizer_name: adam

    # Set the options for the optimizer (they vary according to the
    # optimizer defined by 'optimizer_name').
    optimizer_options:

      # Set these options if 'optimizer_name' is 'adam'.

      # Set the learning rate.
      #
      # Type: float.
      #
      # Default: 0.01.
      lr: 0.01

      # Set the weight decay.
      #
      # Type: float.
      #
      # Default: 0.0
      weight_decay: 0.0

      # Set the betas.
      #
      # Type: list of float.
      #
      # Default: [0.5, 0.9].
      betas: [0.5, 0.9]
```
