Getting started
===============

bulkDGD is a Python package containing the DGD generative model for the gene expression of human tissues from bulk RNA-Seq data described in the work of Schuster and Krogh :cite:p:`schuster2021deep,schuster2021manifold` and Prada and coworkers :cite:p:`prada2023n`.

bulkDGD can be used, for instance, to find differentially expressed genes between normal human samples and diseased samples.

Installation
------------

We provide detailed instructions to install bulkDGD in a Python virtual environment or in a ``conda`` environment in the :doc:`Installation <installation>` section.

Usage
-----

The modules of bulkDGD can be imported and used to build customized scripts and pipelines.

The :doc:`API reference section <api_reference>` provides a detailed descriptions of bulkDGD's subpackages and modules.

However, we also provide a small :doc:`command-line interface <command_line_interface>` to automate some of the most common tasks for which the DGD model can be used for more bio-oriented audiences.

Tutorials
---------

Our :doc:`tutorials <tutorials>` provide detailed, step-by-step explanations of how to perform different tasks using bulkDGD.

Issues, bugs, and questions
---------------------------

You can report to us any bug, issue, or question about the package by `opening an issue <https://docs.github.com/en/issues/tracking-your-work-with-issues/creating-an-issue>`_ in the `issues section <https://github.com/Center-for-Health-Data-Science/bulkDGD/issues>`_ of our GitHub repository.

Citing
------

If you use our software for your research, please cite the following articles:

* Schuster, Viktoria, and Anders Krogh. "A manifold learning perspective on representation learning: Learning decoder and representations without an encoder." *Entropy* 23.11 (2021): 1403.

* Schuster, Viktoria, and Anders Krogh. "The deep generative decoder: Using MAP estimates of representations." *arXiv preprint arXiv:2110.06672* (2021).

* Prada-Luengo, Inigo, et al. "N-of-one differential gene expression without control samples using a deep generative model." *bioRxiv* (2023): 2023-01.

References
----------

.. bibliography::
