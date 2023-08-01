# Getting started

`bulkDGD` is a Python package containing the DGD generative model for the gene expression of human tissues from bulk RNA-Seq data described in the work of Schuster and Krogh [^1][^2] and Prada and coworkers [^3].

## Installation

We provide detailed instructions to install `bulkDGD` in a Python virtual environment or in a `conda` environment in the {doc}`Installation <Installation>` section.

## Usage

The modules of `bulkDGD` can be imported and used to build customized scripts and pipelines.

However, we also provide a small command-line interface to automate some of the most common tasks for which the DGD model can be used for more bio-oriented audiences.

The package consists of two sub-packages, whose content is described in their corresponding sections:

* [Core](./core.rst)
* [Utils](./utils.rst)

Furthermore, the {doc}`Command-line interface <Command-line interface>` section provides a description of the command-line interface.

## Issues, bugs, and questions

You can report to us any bug, issue, or question about the package by [opening an issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/creating-an-issue) in the [issues section](https://github.com/Center-for-Health-Data-Science/BulkDGD/issues) of your GitHub repository.

## Citing

If you use our software for your research, please cite the following articles:

* Schuster, Viktoria, and Anders Krogh. "A manifold learning perspective on representation learning: Learning decoder and representations without an encoder." *Entropy* 23.11 (2021): 1403.

* Schuster, Viktoria, and Anders Krogh. "The deep generative decoder: Using MAP estimates of representations." *arXiv preprint arXiv:2110.06672* (2021).

* Prada-Luengo, Inigo, et al. "N-of-one differential gene expression without control samples using a deep generative model." *bioRxiv* (2023): 2023-01.

## References

[^1]:Schuster, Viktoria, and Anders Krogh. "A manifold learning perspective on representation learning: Learning decoder and representations without an encoder." *Entropy* 23.11 (2021): 1403.
[^2]: Schuster, Viktoria, and Anders Krogh. "The deep generative decoder: Using MAP estimates of representations." *arXiv preprint arXiv:2110.06672* (2021).
[^3]: Prada-Luengo, Inigo, et al. "N-of-one differential gene expression without control samples using a deep generative model." *bioRxiv* (2023): 2023-01.

