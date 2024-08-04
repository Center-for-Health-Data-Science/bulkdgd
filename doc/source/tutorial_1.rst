Tutorial 1 - Finding the best representations for a set of new samples
======================================================================

The code and input/output data regarding this tutorial can be found in the ``bulkDGD/tutorials/tutorial_1`` directory.

Step 1 - Preprocess the input samples
-------------------------------------

In this tutorial, we are going to find the best representations in the latent space defined by the DGD model for a set of samples.

We are going to use the samples provided in the ``samples.csv`` file.

The file has the following structure:

.. code-block::

   ,ENSG00000187634,ENSG00000188976,ENSG00000187961,ENSG00000187583,...,tissue
   1627,80736,275265,52208,2088,...,testis
   111,44899,176358,65177,2660,...,adipose_visceral_omentum
   555,60662,381897,90671,24486,...,breast_mammary_tissue
   ...

As we can see, each row contains the expression data for a specific sample. The first column contains the samples' unique names, IDs, or indexes, while the rest of the columns contain either the expression data for a specific gene (identified by its Ensembl ID) or additional information about the samples. In our case, for example, the last column, named ``tissue``, identifies the tissue from which each sample comes.

Before finding the representations for these samples in latent space, we want to make sure that the genes whose expression data are reported in the CSV file correspond to the genes included in the DGD model and that these genes are reported in the correct order in the file. Furthermore, we would like to know whether we have duplicate samples, duplicate genes, and genes with missing expression values. We can do all this using the :func:`ioutil.preprocess_samples` function.

First, we set the logging so that every message above and including the ``INFO`` level gets reported to have a better idea of what the program is doing. By default, only messages associated with a ``WARNING`` level or above get reported.

.. code-block:: python

   # Import the 'logging' module.
   import logging as log

   # Set the logging options.
   log.basicConfig(level = "INFO")

Then, we load our CSV file as a data frame using the :func:`ioutil.load_samples` function.

.. code-block:: python

   # Import Pandas and the 'ioutil' module from 'bulkDGD'.
   import pandas as pd
   import bulkDGD
   from bulkDGD import ioutil

   # Load the samples into a data frame.
   df_samples = \
      ioutil.load_samples(# The CSV file where the samples are stored
                          csv_file = input_csv,
                          # The field separator in the CSV file
                          sep = ",",
                          # Whether to keep the original samples' names/
                          # indexes (if True, they are assumed to be in
                          # the first column of the data frame)
                          keep_samples_names = True,
                          # Whether to split the input data frame into
                          # two data frames, one containing only gene
                          # expression data and the other containing
                          # the extra data about the samples                    
                          split = False)

Then, we can preprocess the samples.

.. code-block:: python

   # Preprocess the samples.
   df_preproc, genes_excluded, genes_missing = \
       ioutil.preprocess_samples(df_samples = df_samples)

The function looks for duplicated samples, duplicated genes, and missing values in the columns containing gene expression data. If the function finds duplicated samples or genes with missing expression values, it raises a warning but keeps the samples where the duplication or missing values were found. However, the function will throw an error if it finds duplicated genes since the DGD model assumes the input samples report expression data for unique genes.

Then, the function re-orders the columns containing gene expression data according to the list of genes included in the DGD model and places all the columns containing additional information about the samples (in our case, the ``tissue`` column) as the last columns of the output data frame.

Finally, the function checks that all genes in the input samples are among those included in the DGD model, and that all genes used in the DGD model are found in the input samples.

The function returns three objects:

* ``df_preproc`` is a data frame containing the preprocessed samples.

* ``genes_excluded`` is a list containing the Ensembl IDs of the genes that were found in the input samples but are not part of the set of genes included in the DGD model. These genes are absent from ``df_preproc``. In our case, no genes were excluded.

* ``genes_missing`` is a list containing the Ensembl IDs of the genes that are part of the set of genes included in the the DGD model but were not found in the input samples. These genes are added to ``df_preproc`` with a count of 0 for all samples. In our case, no genes were missing.

Step 2 - Get the trained DGD model
----------------------------------

In order to set up the DGD model and load its trained parameters, we need a configuration file specifying the options to initialize the model and the path to the files containing the parameters of the trained model.

In this case, we will use the ``bulkDGD/ioutil/configs/model/model.yaml`` file. We can refer to this file using only the file's name (without extension) since it is located in the ``bulkDGD/ioutil/configs/model`` directory.

We can load the configuration using the :func:`ioutil.load_config_model` function.

.. code-block:: python
   
   # Load the configuration.
   config_model = ioutil.get_config_model("model")

Once loaded, the configuration consists of a dictionary of options, which maps to the arguments required by the :class:`core.model.DGDModel` constructor.

You can find more information about the available options to configure the DGD model :doc:`here <model_config_options>`.

Then, we can initialize the trained DGD model.

.. code-block:: python
   
   # Import the 'model' module from 'bulkDGD.core'.
   from bulkDGD.core import model
   
   # Get the trained DGD model (Gaussian mixture model
   # and decoder).
   dgd_model = model.DGDModel(**config_model)

Step 3 - Get the optimization scheme
------------------------------------

Before finding the representations, we need to define the scheme that will be used to optimize the representations in latent space.

The scheme is contained in a YAML configuration file similar to that containing the DGD model's configuration.

In this case, we will use the ``bulkDGD/ioutil/configs/representations/two_opt.yaml`` file. We can refer to this file using only the file's name (without extension) since it is located in the ``bulkDGD/ioutil/configs/representations`` directory.

We can load the configuration using the :func:`ioutil.load_config_rep` function.

Once loaded, the configuration consists of a dictionary of options.

You can find more information about the supported optimization schemes and corresponding options :doc:`here <rep_config_options>`.

.. code-block:: python
   
   # Load the configuration.
   config_rep = ioutil.load_config_rep("two_opt")

Step 4 - Find and optimize the representations
----------------------------------------------

We can now use the :meth:`core.model.DGDModel.get_representations` method to find and optimize the representations for our input samples.

.. code-block:: python
   
   # Get the representations, the predicted scaled means and
   # r-values of the negative binomials modeling the genes' counts for
   # the in silico samples corresponding to the representations found,
   # and the time spent finding the representations.
   df_rep, df_pred_means, df_pred_r_values, df_time_opt = \
       dgd_model.get_representations(\
           # The data frame with the samples
           df_samples = df_samples,
           # The configuration to find the representations                        
           config_rep = config_rep)

The method returns four objects:

* ``df_rep`` is a ``pandas.DataFrame`` containing the optimized representations. In this data frame, each row represents a different representation, and each column represents either the value of the representatione along a dimension of the latent space (in the ``latent_dim_*`` columns) or additional information about the original samples (in our case, the ``tissue`` column).

* ``df_pred_means`` is a ``pandas.DataFrame`` containing the predicted scaled means of the negative binomials used to model the RNA-seq genes' counts in the in silico samples corresponding to the best representations found. In this data frame, each row represents a different sample, and each column represents either the scaled mean of the negative binomial for a specific gene (in the columns named after the genes' Ensembl IDs) or additional information about the original samples (in our case, the ``tissue`` column).

* ``df_pred_r_values`` is a ``pandas.DataFrame`` containing the predicted r-values of the negative binomials used to model the RNA-seq genes' counts in the in silico samples corresponding to the best representations found. In this data frame, each row represents a different sample, and each column represents either the r-value of the negative binomial for a specific gene (in the columns named after the genes' Ensembl IDs) or additional information about the original samples (in our case, the ``tissue`` column).

* ``df_time`` is a ``pandas.DataFrame`` containing information about the CPU and wall clock time used by each optimization epoch and each backpropagation step through the decoder (one per epoch).

Step 5 - Save the outputs
-------------------------

We can save the preprocessed samples, the representations, the predicted scaled means, the predicted r-values, and the information about the optimization time to CSV files using the :func:`ioutil.save_samples`, :func:`ioutil.save_representations`, :func:`ioutil.save_decoder_outputs`, and :func:`ioutil.save_time` functions.

.. code-block:: python
   
   # Save the preprocessed samples.
   ioutil.save_samples(\
       # The data frame containing the samples
       df = df_preproc,
       # The output CSV file
       csv_file = "samples_preprocessed.csv",
       # The field separator in the output CSV file
       sep = ",")

   # Save the representations.
   ioutil.save_representations(\
       # The data frame containing the representations
       df = df_rep,
       # The output CSV file
       csv_file = "representations.csv",
       # The field separator in the output CSV file
       sep = ",")

   # Save the predicted scaled means.
   ioutil.save_decoder_outputs(\
       # The data frame containing the predicted scaled means
       df = df_pred_means,
       # The output CSV file
       csv_file = "pred_means.csv",
       # The field separator in the output CSV file
       sep = ",")

   # Save the predicted r-values.
   ioutil.save_decoder_outputs(\
       # The data frame containing the predicted r-values
       df = df_pred_r_values,
       # The output CSV file
       csv_file = "pred_r_values.csv",
       # The field separator in the output CSV file
       sep = ",")

   # Save the time data.
   ioutil.save_time(\
       # The data frame containing the time data
       df = df_time_opt,
       # The output CSV file
       csv_file = "time_opt.csv",
       # The field separator in the output CSV file
       sep = ",")
