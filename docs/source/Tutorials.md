# Tutorials

In this section, you will find examples of how to use the `bulkDGD` package for different tasks.

If you want an overview of the command-line utilities available in `bulkDGD`, see the {doc}`Command-line interface <Command-line interface>` section.

Here, we will showcase the usage of the package's functions in the context of larger analysis scripts.

Specifically, we provide detailed tutorials to:

* Find the best representations in latent space for a new set of samples ([Tutorial 1](#tutorial-1-finding-the-best-representations-for-a-set-of-new-samples)).

* Perform differential expression analysis between a set of "treated" samples (for instance, cancer samples) and their corresponding "untreated" samples ("normal" samples) found using the DGD model ([Tutorial 2](#tutorial-2-differential-expression-analysis)).
* Perform principal component analysis (PCA) on a set of representations and plot the results ([Tutorial 3](#tutorial-3-principal-component-analysis)).

The data needed to reproduce the tutorials can be found in the `bulkDGD/tutorials/data` directory, while the scripts containing the code necessary to run the tutorials and the output files prodiced by the tutorials can be found in each tutorial's directory.

## Tutorial 1 - Finding the best representations for a set of new samples

All the code necessary to run this tutorial can be found in the `tutorial_1.py` script.

### The input samples

In this tutorial, we are going to find the best representations in the latent space defined by the DGD model for the samples in the `samples.csv` file. The file has the following structure:

```
,ENSG00000187634,ENSG00000188976,ENSG00000187961,ENSG00000187583,...,tissue
1627,80736,275265,52208,2088,...,testis
111,44899,176358,65177,2660,...,adipose_visceral_omentum
555,60662,381897,90671,24486,...,breast_mammary_tissue
...
```

As we can see, each row contains the expression data for a specific sample. The first column contains the samples' unique names, IDs, or indexes, while the rest of the columns contain either the expression data for a specific gene (identified by its Ensembl ID) or additional information about the samples. In our case, for example, the last column identifies the tissue from which the sample comes.

Before finding the representations for these samples in latent space, we want to make sure that the genes whose expression data are reported in the CSV file correspond to the genes used to train the DGD model and that these genes are reported in the correct order in the file. Furthermore, we would like to know whether we have duplicate samples, duplicate genes, and genes with missing expression values. We can do all this by using the `preprocess_samples` function in the `bulkDGD.utils.dgd` module.

First, we set the logging so that every message above and including the `INFO` level gets reported to have a better idea of what the program is doing. By default, only messages associated with a `WARNING` level or above get reported.

```python
# Import the logging module
import logging as log

# Set the logging options
log.basicConfig(level = "INFO")
```

Then, we load our CSV file as a data frame.

``` python
# Import Pandas and the 'bulkDGD.utils.dgd' module
import pandas as pd
import bulkDGD.utils.dgd as dgd

# Load the samples into a data frame
df_samples = pd.read_csv(# Name of/path to the file
                         "samples.csv",
                         # Column separator used in the file
                         sep = ",",
                         # Name or numerical index of the column
                         # containing the samples' names/IDs/indexes
                         index_col = 0,
                         # Name or numerical index of the row
                         # containing the columns' names
                         header = 0)
```

Then, we can preprocess the samples.

```python
# Preprocess the samples
df_preproc, genes_excluded, genes_missing = \
    dgd.preprocess_samples(df_samples = df_samples)

INFO:bulkDGD.utils.dgd:1 column(s) containing additional information (not gene expression data) were found in the input data frame: tissue.
INFO:bulkDGD.utils.dgd:Now looking for duplicated samples...
INFO:bulkDGD.utils.dgd:No duplicated samples were found.
INFO:bulkDGD.utils.dgd:Now looking for missing values in the columns containing gene expression data...
INFO:bulkDGD.utils.dgd:No missing values were found in the columns containing gene expression data.
INFO:bulkDGD.utils.dgd:Now looking for duplicated genes...
INFO:bulkDGD.utils.dgd:No duplicated genes were found.
INFO:bulkDGD.utils.dgd:In the data frame containing the pre-processed samples, the columns containing gene expression data will be ordered according to the list of genes used to train the DGD model (which can be found in '/Users/testuser/programs/bulkDGD/data/model/training_genes.txt').
INFO:bulkDGD.utils.dgd:In the data frame containing the pre-processed samples, the columns found in the input data frame which did not contain gene expression data, if any were present, will appended as the last columns of the data frame, and appear in the same order as they did in the input data frame.
INFO:bulkDGD.utils.dgd:All genes found in the input samples are part of the set of genes used to train the DGD model.
INFO:bulkDGD.utils.dgd:All genes used to train the DGD model were found in the input samples.
```

By inspecting the log messages, we can see that the function has looked for duplicated samples, duplicated genes, and missing values in the columns containing gene expression data. If the function finds duplicated samples or genes with missing expression values, it raises a warning but keeps the samples where the duplication or missing values were found. However, the function will throw an error if it finds duplicated genes since the DGD model assumes the input samples report expression data for unique genes. That is, isoforms should be ignored.

Then, the function informs us that the columns containing gene expression data were re-ordered according to the list of genes used to train the DGD model, and that all the columns containing additional information about the samples (in our case, the `tissue` column) will be the last columns of the output data frame.

Finally, the function checks that all genes in the input samples are among those used to train the DGD model, and that all genes used in the DGD model are found in the input samples.

The function returned three objects:

* `df_preproc` is a data frame containing the preprocessed samples.
* `genes_excluded` is a list containing the Ensembl IDs of the genes that were found in the input samples but are not part of the set of genes used to train the DGD model. These genes are not present in `df_preproc`.
* `genes_missing` is a list containing the Ensembl IDs of the genes that are part of the set of genes used to train the DGD model but were not found in the input samples. These genes are added to `df_preproc` with a count of 0 for all samples.

### The configuration files

We also need a configuration file containing the specifications of the DGD model and another configuration file with the options to fine-tune the search for the best representations in latent space.

For this, we will use the `bulkDGD/configs/model/config_model.yaml` and the `bulkDGD/configs/representations/config_rep.yaml` files, respectively.

We assume the files have been copied to the current working directory for the tutorial.

We can load the configurations using the `utils.misc.get_config_model` and the `utils.misc.get_config_rep` functions. Therefore, we load the `misc` module first.

```python
# Import the 'bulkDGD.utils.misc' module
from bulkDGD.utils import misc
```

#### The model's configuration

We can load the model's configuration using the `utils.misc.get_config_model` function.

```python
# Load the model's configuration
config_model = misc.get_config_model("config_model.yaml")
```

Once loaded, each configuration consists of a dictionary of options.

```python
# We can inspect the configuration
config_model
>>> {"gmm": {"pth_file": "/Users/testuser/programs/bulkDGD/data/model/gmm.pth", "options": {'dim': 50, "n_comp": 45, "cm_type": "diagonal", "log_var_params": [0.1, 1], "alpha": 5}, "mean_prior": {"type": "softball", "options": {"radius": 7, "sharpness": 10}}}, "dec": {"pth_file": "/Users/testuser/programs/tests/dgd_tests/dec.pth", "options": {"n_neurons_input": 50, "n_neurons_hidden": 500, "n_neurons_hidden2": 8000, "n_neurons_output": 16883, "r_init": 2}}}
```

Here, we provide a brief description of every option. You can find similar descriptions in the comments inside the configuration file.

`"gmm"` maps to a dictionary containing the options to get the trained Gaussian mixture model:

* `"pth_file"` defines the PyTorch file containing the pre-trained Gaussian mixture model. If a file path is specified, the corresponding file will be used. If `default` is specified in the configuration file, the `bulkDGD/data/model/gmm.pth` file will be used, and the absolute path to this file will appear in the configuration.
* `"options"` maps to the options used to initialize the trained Gaussian mixture model. Therefore, you should leave them unchanged unless you re-train the decoder using a different set of options.
  * `"dim"` indicates the dimensionality of the Gaussian mixture model.
  * `"n_comp"` is the number of components in the Gaussian mixture.
  * `"cm_type"` determines the type of covariance matrix of the Gaussian mixture. It can be `fixed`, `isotropic`, or `diagonal`.
  * `"log_var_params"` stores the parameters to initialize the parameters of the Gaussian prior for the log-variance of the mixture model. The two parameters are the mean and standard deviation of the Gaussian prior.
  * `"alpha"` stores the alpha of the Dirichlet distribution, which determines the uniformity of the weights of the components in the Gaussian mixture.
  * `"mean_prior"` stores a set of options referring to the prior used for the means of the components in the Gaussian mixture:
    * `"type"` is the type of prior that should be used. So far, only the `softball` prior has been implemented.
    * `"options"` is a dictionary storing the options that should be used when initializing the prior. The options can vary depending on the `"type"` of prior. For the `softball` prior, we have two options:
      * `"radius"` is the radius of the multi-dimensional ball.
      * `"sharpness"` is the sharpness of the soft boundary of the ball.
* `"dec"` maps to a sub-dictionary containing decoder-specific options.

  * `"pth_file"` indicates the path to the PyTorch file containing the pre-trained decoder. This file is not available inside the GitHub repository because of size constraints but can be downloaded using [this link](https://drive.google.com/file/d/1SZaoazkvqZ6DBF-adMQ3KRcy4Itxsz77/view?usp=sharing). Once downloaded, you can move this file into the `bulkDGD/data/model` directory and simply refer to it by using `default` in the configuration file.
  * `"options"` maps to a sub-dictionary containing options consistent with those used when training the decoder. Therefore, you should leave them unchanged unless you re-train the decoder using a different set of options.
    * `"n_neurons_input"` defines the number of neurons in the input layer of the decoder. This number must correspond to the one set in the `"dim"` option of the `["gmm"]["options"]` section of the configuration file since the input layer of the decoder must have a number of neurons corresponding to the dimensionality of the latent space.
    * `"n_neurons_hidden1"` defines the number of neurons in the first hidden layer of the decoder.
    * `"n_neurons_hidden2` defines the number of neurons in the second hidden layer of the decoder.
    * `"n_neurons_output"` defines the number of neurons in the output layer of the decoder.
    * `"r_init"` defines the initial r-value of each of the negative binomial distributions modeling the expression of the genes.
    * `"activation_output"` refers to the name of the activation function that should be used in the output layer of the decoder. So far, only the `softplus` function is supported.

#### The configuration to find the best representations

We can load the configuration with the options to fine-tune the search for the best representations using the `utils.misc.get_config_rep` function.

```python
# Load the configuration with the options to configure
# the search for the best representations
config_rep = misc.get_config_rep("config_rep.yaml")
```

This configuration also consists of a dictionary of options.

```python
# We can inspect this configuration, too
config_rep
>>> {"data": {"batch_size": 8, "shuffle": True}, "rep": {"n_rep_per_comp": 1, "opt1": {"epochs": 10, "type": "adam", "options": {"lr": 0.01, "weight_decay": 0, "betas": [0.5, 0.9]}}, "opt2": {"epochs": 50, "type": "adam", "options": {"lr": 0.01, "weight_decay": 0, "betas": [0.5, 0.9]}}}}
```

Here, we provide a brief description of every option. You can find similar descriptions in the comments inside the configuration file.

`"data"` maps to a dictionary containing the options to specify how the samples should be loaded (they are passed to the [`torch.data.utils.DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) constructor):

* `"batch_size"` determines the size of the samples' batches. In fact, samples are loaded in chunks of `"batch_size"` size, and each batch is processed independently.
* `"shuffle"` determines whether the samples should be randomly shuffled at every epoch.

`"rep"` maps to a dictionary containing the options defining the parameters to search for the best representations for the given samples:

* `"n_rep_per_comp"` defines the number of representations to be initialized per component of the Gaussian mixture model per sample.

* `"opt1"` maps to a dictionary with the options for the first optimization:
  * `"epochs"` defines the number of epochs.
  * `"type"` defines the type of optimizer to be used. So far, only the `"adam"` optimizer is supported.
  * `"options"` maps to a dictionary containing the options to set up the optimizer. These options may vary according to the optimizer used. For `"adam"`, we have three options, which are passed to the [`torch.optim.Adam`](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html) constructor:
    * `"lr"` defines the learning rate.
    * `"weight_decay"` defines the weight decay.
    * `"betas"` defines the coefficients used for computing the running averages of the gradient and its square.
* `"opt2"` maps to a dictionary with the options for the second optimization:
  * `"epochs"` defines the number of epochs.
  * `"type"` defines the type of optimizer to be used. So far, only the `"adam"` optimizer is supported.
  * `"options"` maps to a dictionary containing the options to set up the optimizer. These options may vary according to the optimizer used. For `"adam"`, we have three options, which are passed to the [`torch.optim.Adam`](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html) constructor:
    * `"lr"` defines the learning rate.
    * `"weight_decay"` defines the weight decay.
    * `"betas"` defines the coefficients used for computing running averages of gradient and its square.

### The trained DGD model

Then, we need to get the trained DGD model. We do so with the `get_model` function in the `bulkDGD.utils.dgd` module.

```python
# Get the trained DGD model (Gaussian mixture model
# and decoder)
gmm, dec = dgd.get_model(config_gmm = config_model["gmm"],
                         config_dec = config_model["dec"])

INFO:bulkDGD.core.latent:The SoftballPrior prior will be used as prior over the means of the mixture components.
INFO:bulkDGD.utils.dgd:The Gaussian mixture model was successfully initialized.
INFO:bulkDGD.utils.dgd:The Gaussian mixture model's state was successfully loaded from '/Users/testuser/programs/bulkDGD/data/model/gmm.pth'.
INFO:bulkDGD.utils.dgd:The decoder was successfully initialized.
INFO:bulkDGD.utils.dgd:The decoder's state was successfully loaded from '/Users/testuser/programs/bulkDGD/data/model/dec.pth'.
```

As we can see from the log messages, both the Gaussian mixture model and the decoder were first initialized, and then their state (the trained parameters) were loaded.

### The representations

We can now use the `get_representations` function in the `bulkDGD.utils.dgd` module to find the best representations for our samples of interest.

```python
# Get the representations
df_rep, df_dec_out = \
    dgd.get_representations(\
        # The data frame containing the preprocessed samples
        df = df_preproc,
        # The trained Gaussian mixture model
        gmm = gmm,
        # The trained decoder
        dec = dec,
        # How many representations to initialize per component
        # of the Gaussian mixture model per sample
        n_rep_per_comp = config_rep["rep"]["n_rep_per_comp"],
        # The configuration to load the samples
        config_data = config_rep["data"],
        # The configuration for the first optimization
        config_opt1 = config_rep["rep"]["opt1"],
        # The configuration for the second optimization
        config_opt2 = config_rep["rep"]["opt2"])

INFO:bulkDGD.utils.dgd:Starting the first optimization...
INFO:bulkDGD.utils.dgd:Epoch 1: loss 53.51452738405719.
INFO:bulkDGD.utils.dgd:Epoch 2: loss 37.78411296090515.
...
INFO:bulkDGD.utils.dgd:Epoch 10: loss 21.375471316397174.
INFO:bulkDGD.utils.dgd:Starting the second optimization...
INFO:bulkDGD.utils.dgd:Epoch 1: loss 16.999307299157767.
INFO:bulkDGD.utils.dgd:Epoch 2: loss 16.984738363634538.
...
INFO:bulkDGD.utils.dgd:Epoch 50: loss 16.913638140095696.
```

We can see from the log messages how the optimization is proceeding and inspect the per-epoch total loss.

The `get_representations` function returns two objects:

* `df_rep` is a data frame containing the representations. It has the following structure:

  Each row contains the best representation found for each sample, with the first column containing the name/ID/index of the sample as provided in the input data frame. The other columns contain the values of the samples' representation along the latent space's dimensions, the loss associated with each representation (`loss` column), and the additional information found for the samples in the input data frame.

* `df_dec_out` is a data frame containing the outputs of the decoder for the representations found.

  As for `df_rep`, each row represents the best representation found for each sample, with the first column containing the name/ID/index of the sample as provided in the input data frame. The other columns here contain the means of the negative binomials modeling the expression of each gene in each sample according to the DGD model and the additional information found for the samples in the input data frame.

The r-values of the negative binomials can be obtained with the `bulkDGD.utils.dgd.get_r_values` function, which gets them from the trained decoder.

```python
# Get the r-values
r_values = dgd.get_r_values(dec = dec)
```

We can now save the representations and decoder outputs into CSV files using the `save_representations` and `save_decoder_outputs` in the `bulkDGD.utils.dgd` module.

```python
# Save the representations
dgd.save_representations(\
    # The data frame containing the representations
    df = df_rep,
    # The output CSV file
    csv_file = "representations.csv",
    # The field separator in the output CSV file
    sep = ",")

# Save the decoder outputs
dgd.save_decoder_outputs(\
    # The data frame containing the decoder outputs
    df = df_dec_out,
    # The output CSV file
    csv_file = "decoder_outputs.csv",
    # The field separator in the output CSV file
    sep = ",")
```

## Tutorial 2 - Differential expression analysis

[coming soon]

## Tutorial 3 - Principal component analysis

[coming soon]