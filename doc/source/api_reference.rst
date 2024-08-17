API reference
=============

.. toctree::
   :maxdepth: 1
   :hidden:

   analysis
   core
   genes
   ioutil
   plotting
   recount3
   model_config_options
   rep_config_options
   train_config_options

bulkDGD consists of several packages:

* :doc:`analysis <analysis>`, containing utilities to analyze the data produced by the bulkDGD model.

* :doc:`core <core>`, containing the core components of the bulkDGD model and the model itself.

* :doc:`genes <genes>`, containing utilities to create customized lists of genes to use with the bulkDGD model.

* :doc:`ioutil <ioutil>`, containing utilities for loading and saving files and pre-process data.

* :doc:`plotting <plotting>`, containing plotting utilities.

* :doc:`recount3 <recount3>`, containing utilities to interact with the Recount3 platform.

Some functions and methods use dictionary-based configurations for several tasks, such as setting up the bulkDGD model, finding the best representations for a new set of samples and training the bulkDGD model. More detailed descriptions of such configurations are available here:

* :doc:`configuration used to set up the bulkDGD model <model_config_options>`.

* :doc:`configuration used to set the optimization scheme <rep_config_options>` when finding representations for a new set of samples.

* :doc:`configuration used to train the bulkDGD model <train_config_options>`.
