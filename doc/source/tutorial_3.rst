Tutorial 3 - Training the DGD model
===================================

The code and input/output data regarding this tutorial can be found in the ``bulkDGD/tutorials/tutorial_3`` directory.

Step 1 - Set the DGD model
--------------------------

In this tutorial, we are going to train the DGD model using a new set of samples.

First, we create a new instance of the DGD model with the desired options.

To do so, we need a configuration file with these options.

In this case, we will use the ``model_untrained.yaml`` file already present in the tutorial's directory.

First, we set the logging so that every message above and including the ``INFO`` level gets reported to have a better idea of what the program is doing. By default, only messages associated with a ``WARNING`` level or above get reported.

.. code-block:: python

   # Import the 'logging' module.
   import logging as log

   # Set the logging options.
   log.basicConfig(level = "INFO")

Then, we can load the configuration for the DGD model using the :func:`ioutil.load_config_model` function.

.. code-block:: python

   # Import the 'ioutil' module from 'bulkDGD'.
   from bulkDGD import ioutil
   
   # Load the configuration.
   config_model = ioutil.get_config_model("model.yaml")

Once loaded, the configuration consists of a dictionary of options, which maps to the arguments required by the :class:`core.model.DGDModel` constructor. One of these options is the ``genes_txt_file``, which maps to the path to a plain text file containing a list of Ensembl IDs representing the genes that should be included in the DGD model. If this option is set to ``"default"``, the list of genes defined in ``bulkDGD/ioutil/data/genes.txt`` is used.

Here, we use the custom list contained in the ``custom_genes.txt`` file (in the ``model_untrained.yaml`` configuration file, ``genes_txt_file`` is set to ``custom_genes.txt``).

We can now initialize the DGD model.

.. code-block:: python
   
   # Import the 'model' module from 'bulkDGD.core'.
   from bulkDGD.core import model
   
   # Get the DGD model (Gaussian mixture model and decoder).
   dgd_model = model.DGDModel(**config_model)

If we have a GPU available, we can move the model there.

.. code-block:: python

   # Import 'torch'.
   import torch 

   # If a CPU with CUDA is available.
   if torch.cuda.is_available():

       # Set the GPU as the device.
       device = torch.device("cuda")

   # Otherwise
   else:

       # Set the CPU as the device.
       device = torch.device("cpu")

   # Move the model to the device.
   dgd_model.device = device


Step 2 - Preprocess the input samples
-------------------------------------

We are going to use the samples provided in the ``samples_train.csv`` (training samples) and ``samples_test.csv`` (test samples) files.

The files have the following structure:

.. code-block::

   ,ENSG00000187634,ENSG00000188976,ENSG00000187961,ENSG00000187583,...,tissue
   1627,80736,275265,52208,2088,...,testis
   111,44899,176358,65177,2660,...,adipose_visceral_omentum
   555,60662,381897,90671,24486,...,breast_mammary_tissue
   ...

As we can see, each row contains the expression data for a specific sample. The first column contains the samples' unique names, IDs, or indexes, while the rest of the columns contain either the expression data for a specific gene (identified by its Ensembl ID) or additional information about the samples. In our case, for example, the last column, named ``tissue``, identifies the tissue from which each sample comes.

Before proceeding with the training, we want to make sure that the genes whose expression data are reported in the CSV files correspond to the genes included in the DGD model and that these genes are reported in the correct order in the files. Furthermore, we would like to know whether we have duplicate samples, duplicate genes, and genes with missing expression values. We can do all this using the :func:`ioutil.preprocess_samples` function.

We load our CSV files as data frames using the :func:`ioutil.load_samples` function.

.. code-block:: python

   # Load the training samples into a data frame.
   df_train_raw = \
      ioutil.load_samples(# The CSV file where the samples are stored
                          csv_file = "samples_train.csv",
                          # The field separator in the CSV file
                          sep = ",",
                          # Whether to keep the original samples' names/
                          # indexes (if True, they are assumed to be in
                          # the first column of the data frame 
                          keep_samples_names = True,
                          # Whether to split the input data frame into
                          # two data frames, one containing only gene
                          # expression data and the other containing
                          # the extra data about the samples                    
                          split = False)

    # Load the test samples into a data frame.
    df_test_raw = \
        ioutil.load_samples(# The CSV file where the samples are stored
                            csv_file = "samples_test.csv",
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

Then, we can preprocess the samples.

.. code-block:: python

   # Preprocess the training samples.
   df_train, genes_excluded_train, genes_missing_train = \
       ioutil.preprocess_samples(df_samples = df_train_raw,
                                 genes_txt_file = "custom_genes.txt")

   # Preprocess the test samples.
   df_test, genes_excluded_test, genes_missing_test = \
       ioutil.preprocess_samples(df_samples = df_test_raw,
                                 genes_txt_file = "custom_genes.txt")

The function looks for duplicated samples, duplicated genes, and missing values in the columns containing gene expression data. If the function finds duplicated samples or genes with missing expression values, it raises a warning but keeps the samples where the duplication or missing values were found. However, the function will throw an error if it finds duplicated genes since the DGD model assumes the input samples report expression data for unique genes.

Then, the function re-orders the columns containing gene expression data according to the list of genes included in the DGD model and places all the columns containing additional information about the samples (in our case, the ``tissue`` column) as the last columns of the output data frame.

Finally, the function checks that all genes in the input samples are among those included in the DGD model, and that all genes used in the DGD model are found in the input samples.

The function returns three objects:

* ``df_train``/``df_test`` is a data frame containing the preprocessed samples.

* ``genes_excluded_train``/``genes_excluded_test`` is a list containing the Ensembl IDs of the genes that were found in the input samples but are not part of the set of genes included in the DGD model. These genes are absent from ``df_train``/``df_test``. In our case, no genes were excluded.

* ``genes_missing_train``/``genes_missing_test`` is a list containing the Ensembl IDs of the genes that are part of the set of genes included in the the DGD model but were not found in the input samples. These genes are added to ``df_train``/``df_test`` with a count of 0 for all samples. In our case, no genes were missing.

Step 3 - Get the training options
---------------------------------

Before training the DGD model, we need to obtain the configuration for the training procedure (which optimizers to use, for how many epochs to train, etc.). Here, we load the configuration from the ``bulkDGD/ioutil/configs/training/training.yaml`` configuration file. However, the configuration can also be stored in a dictionary whose structure is described :doc:`here <train_config_options>`.

.. code-block:: python
   
   # Load the configuration for training the DGD model. Since the
   # file is stored in the 'bulkDGD/ioutil/configs/training'
   # directory, we can simply refer to it by its name (without the
   # .yaml extension).
   config_train = ioutil.load_config_train("training")

Step 4 - Train the DGD model
----------------------------

We can now train the DGD model.

.. code-block:: python
   
   # Train the DGD model
   df_rep_train, df_rep_test, df_loss, df_time = \
        dgd_model.train(df_train = df_train,
                        df_test = df_test,
                        config_train = config_train)

The functions returns four objects:

* ``df_rep_train`` is a ``pandas.DataFrame`` containing the representations found for the training samples in latent space. In this data frame, each row represents a different representation, and each column represents either the value of the representatione along a dimension of the latent space (in the ``latent_dim_*`` columns) or additional information about the original samples (in our case, the ``tissue`` column).

* ``df_rep_test`` is a ``pandas.DataFrame`` containing the representations found for the test samples in latent space. In this data frame, each row represents a different representation, and each column represents either the value of the representatione along a dimension of the latent space (in the ``latent_dim_*`` columns) or additional information about the original samples (in our case, the ``tissue`` column).

* ``df_loss`` is a ``pandas.DataFrame`` containing the losses computed per-epoch during the training procedure.

* ``df_time`` is a ``pandas.DataFrame`` containing information about the CPU and wall clock time used by each training epoch and by the backpropagation steps through the decoder.

Furthermore, the function writes out two files, ``dec.pth`` and ``gmm.pth``, containing the parameters of the trained decoder and Gaussian mixture model, respectively. If these files already exist in the working directory (if, for instance, you have already trained the model multiple times), a numerical suffix will be added to the new files as not to overwrite the old ones. Therefore, you will have ``dec_2.pth`` and ``gmm_2.pth`` in case ``dec.pth``, ``dec_1.pth``, ``gmm.pth``,  and ``gmm_1.pth`` already exist. 

Step 5 - Save the outputs
-------------------------

We can save the preprocessed samples, the representations, the losses, and the information about the training time to CSV files using the :func:`ioutil.save_samples`, :func:`ioutil.save_representations`, :func:`ioutil.save_loss`, and :func:`ioutil.save_time` functions.

.. code-block:: python
   
   # Save the preprocessed training samples.
   ioutil.save_samples(\
       # The data frame containing the samples
       df = df_train,
       # The output CSV file
       csv_file = "samples_preprocessed_train.csv",
       # The field separator in the output CSV file
       sep = ",")

   # Save the preprocessed test samples.
   ioutil.save_samples(\
       # The data frame containing the samples
       df = df_test,
       # The output CSV file
       csv_file = "samples_preprocessed_test.csv",
       # The field separator in the output CSV file
       sep = ",")

   # Save the representations for the training samples.
   ioutil.save_representations(\
       # The data frame containing the representations
       df = df_rep_train,
       # The output CSV file
       csv_file = "representations_train.csv",
       # The field separator in the output CSV file
       sep = ",")

   # Save the representations for the test samples.
   ioutil.save_representations(\
       # The data frame containing the representations
       df = df_rep_train,
       # The output CSV file
       csv_file = "representations_test.csv",
       # The field separator in the output CSV file
       sep = ",")

   # Save the losses.
   ioutil.save_loss(\
       # The data frame containing the losses
       df = df_loss,
       # The output CSV file
       csv_file = "loss.csv",
       # The field separator in the output CSV file
       sep = ",")

   # Save the time data.
   ioutil.save_time(\
       # The data frame containing the time data
       df = df_time,
       # The output CSV file
       csv_file = "train_time.csv",
       # The field separator in the output CSV file
       sep = ",")
