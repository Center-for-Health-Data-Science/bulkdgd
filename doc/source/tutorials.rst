Tutorials
=========

.. toctree::
   :maxdepth: 1
   :hidden:

   tutorial_notebooks/tutorial_1
   tutorial_notebooks/tutorial_2
   tutorial_notebooks/tutorial_3
   tutorial_notebooks/tutorial_4

In this section, you will find examples of how to use the bulkdgd package for different tasks.

If you want an overview of the command-line utilities available in bulkdgd, see the :doc:`Command-line interface <command_line_interface>` section.

Here, we will showcase the usage of the package's functions in the context of larger analysis scripts.

Specifically, we provide detailed tutorials to:

* Find the best representations in latent space for a new set of samples (:doc:`Tutorial 1 <tutorial_notebooks/tutorial_1>`).

* Perform differential expression analysis between a set of "treated" samples (for instance, cancer samples) and their corresponding "untreated" samples ("normal" samples) found using the bulkdgd model (:doc:`Tutorial 2 <tutorial_notebooks/tutorial_2>`).

* Train the bulkdgd model on a new set of samples (:doc:`Tutorial 3 <tutorial_notebooks/tutorial_3>`).

* Download and prepare a set of samples from the Recount3 platform for use with the bulkdgd model, using a real glioblastoma dataset as an example (:doc:`Tutorial 4 <tutorial_notebooks/tutorial_4>`).

The data and Python notebooks needed to reproduce the tutorials can be found in the different ``tutorial_*`` directories inside the ``tutorials`` directory at the root of the repository.
