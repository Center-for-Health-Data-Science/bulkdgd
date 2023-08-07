# `configs/representations`

Last updated: 27/07/2023

## `config_rep.yaml`

Example of a YAML configuration file used for `dgd_get_representations` (`-cr`, `--config-file-rep` option).

The configuration file has the following structure (`int`, `str`, `float`, `bool`, etc., represent the data type expected for each field):

```yaml
# Section containing the options for loading the gene expression data
data:
  
  # The data will be loaded in batches of 'batch_size' size
  batch_size: int
  
  # Whether to shuffle the data when loading them
  shuffle: bool


# Section containing the options for the different
# rounds of optimization used when finding the
# best representations for new samples
optimization:

  # Section containing the options for the initial
  # round of optimization
  opt1:

    # Number of epochs
    epochs: int

    # Type of optimizer (e.g., "adam")
    type: str

    # Section containing the options for the optimizer
    # (they vary according to the type of optimizer)
    options:

      # Learning rate
      lr: float

      # Weight decay
      weight_decay: float

      # Betas
      betas: list of two floats

  # Section containing the options for further
  # optimization
  opt2:

    # Number of epochs
    epochs: int

    # Type of optimizer (e.g., "adam")
    type: str

    # Section containing the options for the optimizer
    # (they vary according to the type of optimizer)
    options:

      # Learning rate
      lr: float

      # Weight decay
      weight_decay: float

      # Betas
      betas: list of two floats
```


