Tutorial 2 - Differential Expression Analysis
=============================================

In this tutorial, we will use the bulkDGD model to perform differential gene expression analysis (DEA) for a set of samples.

The code and input/output data regarding this tutorial can be found in the ``bulkDGD/tutorials/tutorial_2`` directory.

In this tutorial, we will walk you through how to integrate differential expression analysis with bulkDGD in a Python script. 

However, if you need to perform DEA on a large set of samples, we recommend using the :doc:`bulkdgd_dea <bulkdgd_dea>` executable, which parallelizes the calculations.

Step 1 - Get the samples' representations in latent space and corresponding decoder's outputs
---------------------------------------------------------------------------------------------

First, we need to find the best representations for these samples in the latent space defined by the bulkDGD model. You can follow the instructions provided in :doc:`Tutorial 1 <tutorial_1>` to find the representations and the corresponding decoder outputs. These outputs are the predicted scaled means and predicted r-values of the negative binomials modeling the genes' counts in silico samples corresponding to the representations.

For this tutorial, we use the first ten preprocessed samples in the ``samples_preprocessed.csv`` file and the corresponding ten pre-calculated scaled means and r-values in the ``pred_means.csv`` and ``pred_r_values.csv`` files obtained following :doc:`Tutorial 1 <tutorial_1>`.

First, we set the logging so that every message above and including the ``INFO`` level gets reported to have a better idea of what the program is doing. By default, only messages associated with a ``WARNING`` level or above get reported.

.. code-block:: python

   # Import the 'logging' module.
   import logging as log

   # Set the logging options.
   log.basicConfig(level = "INFO")

We load the samples using the :func:`ioutil.load_samples` function.

.. code-block:: python

   # Import the 'ioutil' module from 'bulkDGD'.
   from bulkDGD import ioutil
   
   # Load the preprocessed samples into a data frame.
   df_samples = \
       ioutil.load_samples(# The CSV file where the samples are stored
                           csv_file = "samples_preprocessed.csv",
                           # The field separator used in the CSV file
                           sep = ",",
                           # Whether to keep the original samples' names/
                           # indexes (if True, they are assumed to be in
                           # the first column of the data frame) 
                           keep_samples_names = True,
                           # Whether to split the input data frame into
                           # two data frames, one containing only gene
                           # expression data and the other containing
                           # additional information about the samples
                           split = False)
 
   # Get only the first ten samples.
   df_samples = df_samples.iloc[:10,:]

Then, we load the predicted scaled means using the :func:`ioutil.load_decoder_outputs` function.

.. code-block:: python
   
   # Load the predicted scaled means into a data frame.
   df_pred_means = \
       ioutil.load_decoder_outputs(# The CSV file where the predicted
                                   # scaled means are stored
                                   csv_file = "decoder_outputs.csv",
                                   # The field separator used in the CSV
                                   # file
                                   sep = ",",
                                   # Whether to split the input data frame
                                   # into two data frames, one containing
                                   # only the predicted scaled means and
                                   # the other containing additional
                                   # information about the original samples
                                   split = False)

   # Get only the first ten rows.
   df_dec_out = df_dec_out.iloc[:10,:]

Finally, we load the predicted r-values using the :func:`ioutil.load_decoder_output` function. If we used an instance of the bulkDGD model using Poisson distributions instead of negative binomial distributions to model the predicted genes' counts, we would not have an output file with the r-values and we would not need to load them.

.. code-block:: python

   # Load the predicted r-values into a data frame.
   df_pred_r_values = \
      ioutil.load_decoder_outputs(# The CSV file where the predicted
                                  # r-values are stored
                                  csv_file = "decoder_outputs.csv",
                                  # The field separator used in the CSV
                                  # file
                                  sep = ",",
                                  # Whether to split the input data frame
                                  # into two data frames, one containing
                                  # only the predicted r-values and
                                  # the other containing additional
                                  # information about the original samples
                                  split = False)

   # Get only the first ten rows.
   df_pred_r_values = df_pred_r_values.iloc[:10,:]

Step 3 - Perform differential expression analysis
-------------------------------------------------

We can perform differential expression analysis for each sample with the :func:`analysis.dea.perform_dea` function, and save the results to CSV files (one per sample).

.. code-block:: python

   # Import the 'dea' module from 'bulkDGD.analysis'.
   from bulkDGD.analysis import dea

   # For each sample
   for sample in df_samples.index:

       # Perform differential expression analysis.
       dea_results, _ = \
           dea.perform_dea(# The observed gene counts for the current
                           # sample
                           obs_counts = df_samples.loc[sample,:],
                           # The predicted scaled means for the current
                           # sample
                           pred_means = df_pred_means.loc[sample,:],
                           # The r-values for the current sample
                           r_values = df_pred_r_values.loc[sample,:],
                           # Which statistics should be computed and
                           # included in the results
                           statistics = \
                               ["p_values", "q_values",
                                "log2_fold_changes"],
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

       # Save the results.
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
