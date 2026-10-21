``ioutil`` - utilities for I/O operations
=========================================

.. automodule:: bulkdgd.ioutil
  
   .. autofunction:: bulkdgd.ioutil.load_config_model

   .. autofunction:: bulkdgd.ioutil.load_config_rep

   .. autofunction:: bulkdgd.ioutil.load_config_train

   .. autofunction:: bulkdgd.ioutil.load_config_dim_red

   .. autofunction:: bulkdgd.ioutil.load_config_plot

   .. autofunction:: bulkdgd.ioutil.load_config_genes

   .. autofunction:: bulkdgd.ioutil.load_config_recount3

   .. autofunction:: bulkdgd.ioutil.load_decoder_outputs

   .. autofunction:: bulkdgd.ioutil.load_samples

   .. autofunction:: bulkdgd.ioutil.load_representations

   .. autofunction:: bulkdgd.ioutil.save_representations

   .. autofunction:: bulkdgd.ioutil.save_samples

   .. autofunction:: bulkdgd.ioutil.save_decoder_outputs

   .. autofunction:: bulkdgd.ioutil.preprocess_samples

   .. autofunction:: bulkdgd.ioutil.save_table

   .. autofunction:: bulkdgd.ioutil.load_table

How a table is written
----------------------

Every table the package writes follows the same rule as :func:`bulkdgd.ioutil.save_table`: the format follows the **suffix of the path it is given**: ``.parquet`` (or ``.pq``) is written as Parquet, and anything else as delimited text. The format is never changed from the one the path asks for.

Use Parquet when the numbers will be compared between runs. These tables carry ``float64`` activations, predicted means and r-values, and a ``float64`` written as text is rounded to its decimal form: a round trip moves the value by up to about 1e-12.

Most command-line tools use ``.csv`` in their default output names, so they write text unless given an output name ending in ``.parquet``. The exceptions are the ``bulkdgd_reduction_*`` tools, whose default ``-oa`` name ends in ``.parquet``, and ``bulkdgd_dea``, which always writes its results as ``.parquet`` files.
