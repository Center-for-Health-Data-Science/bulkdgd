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
   reproducibility
   model_config_options
   rep_config_options
   train_config_options

bulkdgd consists of several packages:

* :doc:`analysis <analysis>`, containing utilities to analyze the data produced by the bulkdgd model.

* :doc:`core <core>`, containing the core components of the bulkdgd model and the model itself.

* ``ensemble``, containing ``bulkdgd.ensemble.ensemble.BulkDGDEnsemble``, the ensemble of models differing only in the seed they were trained with, and the tiered consensus drawn from it. A bare ``BulkDGDEnsemble()`` loads the fifteen members that ship with the package; see :ref:`model_shipped`.

* :doc:`genes <genes>`, containing utilities to create customized lists of genes to use with the bulkdgd model.

* :doc:`ioutil <ioutil>`, containing utilities for loading and saving files and preprocessing data.

* :doc:`plotting <plotting>`, containing plotting utilities.

* :doc:`recount3 <recount3>`, containing utilities to interact with the Recount3 platform.

* :doc:`reproducibility <reproducibility>`, containing :func:`bulkdgd.reproducibility.set_seeds`, which seeds the generators a run draws from. It must be called before the model is built, because building the model draws the decoder's weights.

Some functions and methods take dictionary-based configurations, for instance to set up the bulkdgd model, find the best representations for a new set of samples and train the bulkdgd model. These configurations are described in:

* :doc:`configuration used to set up the bulkdgd model <model_config_options>`.

* :doc:`configuration used to set the optimization scheme <rep_config_options>` when finding representations for a new set of samples.

* :doc:`configuration used to train the bulkdgd model <train_config_options>`.
