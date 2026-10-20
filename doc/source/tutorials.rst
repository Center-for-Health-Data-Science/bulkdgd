Tutorials
=========

.. toctree::
   :maxdepth: 1
   :hidden:

   tutorial_notebooks/tutorial_1
   tutorial_notebooks/tutorial_2
   tutorial_notebooks/tutorial_3
   tutorial_notebooks/tutorial_4
   tutorial_notebooks/tutorial_5
   tutorial_notebooks/tutorial_6

This section shows how to use the bulkdgd package for different tasks.

For an overview of the command-line utilities available in bulkdgd, see the :doc:`Command-line interface <command_line_interface>` section.

The tutorials use the package's functions inside larger analysis scripts to:

* Find the best representations in latent space for a new set of samples (:doc:`Tutorial 1 <tutorial_notebooks/tutorial_1>`).

* Perform differential expression analysis between a set of "treated" samples (for instance, cancer samples) and their corresponding "untreated" samples ("normal" samples) found using the bulkdgd model (:doc:`Tutorial 2 <tutorial_notebooks/tutorial_2>`).

* Train the bulkdgd model on a new set of samples (:doc:`Tutorial 3 <tutorial_notebooks/tutorial_3>`).

* Download and prepare a set of samples from the Recount3 platform for use with the bulkdgd model, using a real glioblastoma dataset as an example (:doc:`Tutorial 4 <tutorial_notebooks/tutorial_4>`).

* Use bulkdgd directly from R, via ``reticulate``, to find representations and perform DEA without writing any Python code (:doc:`Tutorial 5 <tutorial_notebooks/tutorial_5>`).

* Run a set of samples through the ensemble of models that ships with the package (representations, DEA, the tiered consensus and gene set enrichment), and decide when the ensemble is needed and when a single model is enough (:doc:`Tutorial 6 <tutorial_notebooks/tutorial_6>`).

The data and Python notebooks needed to reproduce the tutorials are in the ``tutorial_*`` directories inside the ``tutorials`` directory at the root of the repository.
