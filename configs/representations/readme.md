# `configs/representations`

Last updated: 18/08/2023

## `one_opt.yaml`

Example of a YAML configuration file used for `dgd_get_representations` (`-cr`, `--config-file-rep` option) to get the representations using only one round of optimization.

The configuration file has the following structure (`int`, `str`, `float`, `bool`, etc., represent the data type expected for each field):

```yaml
# How many representations to initialize per component of
# the Gaussian mixture model per sample
n_rep_per_comp: int

# Section containing the options for the optimization
optimization:

  # Number of epochs
  epochs: int

  # Name of the optimizer (e.g., "adam")
  optimizer_name: str

  # Section containing the options for the optimizer
  # (they vary according to the type of optimizer)
  optimizer_options:

    # Learning rate
    lr: float

    # Weight decay
    weight_decay: float

    # Betas
    betas: [float, float]
```

## `two_opt.yaml`

Example of a YAML configuration file used for `dgd_get_representations` (`-cr`, `--config-file-rep` option) to get the representations using two rounds of optimization.

The configuration file has the following structure (`int`, `str`, `float`, `bool`, etc., represent the data type expected for each field):

```yaml
# How many representations to initialize per component of
# the Gaussian mixture model per sample
n_rep_per_comp: int

# Section containing the options for the optimization
optimization:

  # First optimization round
  opt1:

    # Number of epochs
    epochs: int

    # Name of the optimizer (e.g., "adam")
    optimizer_name: str

    # Section containing the options for the optimizer
    # (they vary according to the type of optimizer)
    optimizer_options:

      # Learning rate
      lr: float

      # Weight decay
      weight_decay: float

      # Betas
      betas: [float, float]

  # Second optimization round
  opt2:

    # Number of epochs
    epochs: int

    # Name of the optimizer (e.g., "adam")
    optimizer_name: str

    # Section containing the options for the optimizer
    # (they vary according to the type of optimizer)
    optimizer_options:

      # Learning rate
      lr: float

      # Weight decay
      weight_decay: float

      # Betas
      betas: [float, float]
```
