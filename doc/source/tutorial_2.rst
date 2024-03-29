Tutorial 2 - Differential Expression Analysis
=============================================

In this tutorial, we will use the DGD model to perform differential gene expression analysis (DEA) for a set of samples.

The code and input/output data regarding this tutorial can be found in the ``bulkDGD/tutorials/tutorial_2`` directory.

In this tutorial, we will walk you through how to integrate differential expression analysis with bulkDGD in a Python script. 

However, if you need to perform DEA on a large set of samples, we recommend using the :doc:`dgd_perform_dea <dgd_perform_dea>` executable, which parallelizes the calculations.

Step 1 - Get the samples' representations in latent space and corresponding decoder outputs
-------------------------------------------------------------------------------------------

First, we need to find the best representations for these samples in the latent space defined by the DGD model. You can follow the instructions provided in :doc:`Tutorial 1 <tutorial_1>` to find the representations and the corresponding decoder outputs, which represent the scaled means of the negative binomials modeling the expression of the genes included in the model. Specifically, the decoder outputs are what we need to perform differential expression analysis, since we are going to compare them (after re-scaling) to the experimental gene counts found in the original samples.

For this tutorial, we use the first ten preprocessed samples in the ``samples_preprocessed.csv`` file and the corresponding ten pre-calculated decoder outputs in the ``decoder_outputs.csv`` file, both obtained following :doc:`Tutorial 1 <tutorial_1>`.

First, we set the logging so that every message above and including the ``INFO`` level gets reported to have a better idea of what the program is doing. By default, only messages associated with a ``WARNING`` level or above get reported.

.. code-block:: python

   # Import the logging module
   import logging as log

   # Set the logging options
   log.basicConfig(level = "INFO")

We load the samples using the :func:`ioutil.load_samples` function.

.. code-block:: python

   # Import the 'dgd.ioutil' module
   from bulkDGD import ioutil
   
   # Load the preprocessed samples into a data frame
   df_samples = \
       ioutil.load_samples(# The CSV file where the samples are stored
                           csv_file = "samples_preprocessed.csv",
                           # The field separator used in the CSV file
                           sep = ",",
                           # Whether to keep the original samples' names/
                           # indexes (if True, they are assumed to be in
                           # the first column of the data frame 
                           keep_samples_names = True,
                           # Whether to split the input data frame into
                           # two data frames, one containing only gene
                           # expression data and the other containing
                           # additional information about the samples
                           split = False)
 
   # Get only the first ten rows
   df_samples = df_samples.iloc[:10,:]

Then, we load the decoder outputs using the :func:`ioutil.load_decoder_outputs` function.

.. code-block:: python
   
   # Load the decoder outputs into a data frame
   df_dec_out = \
      ioutil.load_decoder_outputs(# The CSV file where the decoder outputs
                                  # are stored
                                  csv_file = "decoder_outputs.csv",
                                  # The field separator used in the CSV
                                  # file
                                  sep = ",",
                                  # Whether to split the input data frame
                                  # into two data frame, one containing
                                  # only the decoder outputs and the other
                                  # containing additional information
                                  # about the original samples
                                  split = False)
   
   # Get only the first ten rows
   df_dec_out = df_dec_out.iloc[:10,:]

Step 2 - Get the trained DGD model
----------------------------------

In order to set up the DGD model and load its trained parameters, we need a configuration file specifying the options to initialize it and the path to the files containing the trained model.

In this case, we will use the ``bulkDGD/ioutil/configs/model/model.yaml`` file. We assume this file was copied to the current working directory.

We can load the configuration using the :func:`ioutil.load_config_model` function.

.. code-block:: python
   
   # Load the configuration
   config_model = ioutil.load_config_model("model.yaml")

Once loaded, the configuration consists of a dictionary of options, which maps to the arguments required by the :class:`core.model.DGDModel` constructor.

Then, we can initialize the trained DGD model.

.. code-block:: python
   
   # Import the 'core.model' module
   from bulkDGD.core import model
   
   # Get the trained DGD model (Gaussian mixture model
   # and decoder)
   dgd_model = model.DGDModel(**config_model)

Step 3 - Perform differential expression analysis
-------------------------------------------------

Since the raw decoder outputs are scaled by the r-valuse of the negative binomial distributions modeling the genes (one r-value per distribution, meaning one r-value per gene), we need to get these r-values. They are stored in the :attr:`core.model.DGDModel.r_values` attribute.

.. code-block:: python

    # Get the r-values
    r_values = dgd_model.r_values

Then, we can perform differential expression analysis for each sample with the :func:`analysis.dea.perform_dea` function, and save the results to CSV files (one per sample).

.. code-block:: python

   # Import the 'dea' module
   from bulkDGD.analysis import dea

   # For each sample
   for sample in df_samples.index:

       # Perform differential expression analysis
       dea_results, _ = \
           dea.perform_dea(# The observed gene counts for the current
                           # sample
                           obs_counts = df_samples.loc[sample,:],
                           # The predicted means - decoder outputs for
                           # the current sample
                           pred_means = df_dec_out.loc[sample,:],
                           # Which statistics should be computed and
                           # included in the results
                           statistics = \
                               ["p_values", "q_values",
                                "log2_fold_changes"],
                           # The r-values of the negative binomials
                           r_values = r_values,
                           # The resolution for the p-values calculation
                           # (the higher, the more accurate the
                           # calculation; set to 'None' for an exact
                           # calculation)
                           resolution = 1e4,
                           # The family-wise error rate for the
                           # calculation of the q-values
                           alpha = 0.05,
                           # The method used to calculate the q-values
                           method = "fdr_bh")

       # Save the results
       dea_results.to_csv(# The CSV file where to save the results
                          # for the current sample
                          f"dea_sample_{sample}.csv",
                          # The field separator to use in the output
                          # CSV file
                          sep = ",",
                          # Whether to keep the rows' names
                          index = True,
                          # Whether to keep the columns' names
                          header = True)