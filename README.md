<h1 align="center">
<img src="./branding/bulkdgd_logo.png" width="300">
</h1><br>

<a href='https://bulkdgd.readthedocs.io/en/latest/?badge=latest'>
    <img src='https://readthedocs.org/projects/bulkdgd/badge/?version=latest' alt='Documentation Status' />
</a>


``bulkdgd`` is a Python package that uses the Deep Generative Decoder (DGD) to model the gene expression of healthy human tissues from bulk RNA-Seq data.

The first version of the DGD was developed in 2023 (Schuster and Krogh, 2023), and its first application to bulk RNA-Seq data was presented by Prada-Luengo, Schuster, Liang, and coworkers (Prada-Luengo, Schuster, Liang, et al., 2023).

``bulkdgd`` can also be used from R without writing any Python code. For differential expression analysis with ``bulkdgd`` in R, see the [R tutorial](https://bulkdgd.readthedocs.io/en/latest/tutorial_notebooks/tutorial_5.html).

All [recount3](https://rna.recount.bio/) samples used to train and test the model were processed from raw files with the [Monorail](https://github.com/langmead-lab/monorail-external/) pipeline. We recommend processing your own raw files with the same pipeline before using them with ``bulkdgd``.

## Quick install

```sh
pip install bulkdgd
```

The trained decoder's parameters (``dec.pth``, too large to distribute on PyPI) are downloaded automatically the first time you use the pre-trained model. See the [installation instructions](https://bulkdgd.readthedocs.io/en/latest/installation.html) for details, including how to download them manually on offline machines.

## Version 2.0.1

* The default pre-trained model (both the Gaussian mixture and the decoder's parameters) has been updated, fixing a mismatch between the packaged model configuration and the previously shipped decoder.
* The trained decoder's parameters (``dec.pth``) are no longer distributed with the package or downloaded manually: ``bulkdgd`` downloads them automatically, once, the first time the pre-trained model is used.

## Version 2.0.0 - major release

This **major release** supersedes all previous versions of the package (previously named ``bulkDGD``). Existing users should switch to it. The main changes from the previous version are:

* The package has been renamed from ``bulkDGD`` to ``bulkdgd`` (the import name and CLI commands, already lowercase, are unchanged).
* The default pre-trained model has been retrained on GTEx data with a new Gaussian mixture model implementation (``tgmm``) and a smaller, curated gene list.
* A bug left over from an earlier refactor, which made several CLI commands (`bulkdgd_train`, `bulkdgd_find_probdens`, `bulkdgd_find_representations`) fail immediately on invocation, has been fixed.
* All tutorials have been converted from standalone Python scripts into executed Jupyter notebooks, and a new tutorial covers downloading and preparing samples from [Recount3](https://rna.recount.bio/).
* The documentation has been revised: broken API cross-references, outdated installation instructions and missing CLI usage examples have been fixed, and the documentation builds without warnings.

* **Documentation**: ``bulkdgd``'s documentation can be found [here](https://bulkdgd.readthedocs.io/en/latest/).

* **Bug reports**: please report bugs or problems with ``bulkdgd`` in the [issues](https://github.com/Center-for-Health-Data-Science/bulkdgd/issues) section on GitHub.

**License**

``bulkdgd`` is freely available under the terms of the GNU General Public License (Version 3, 29 June 2007).

**References**

(Schuster and Krogh, 2023) Schuster, Viktoria, and Anders Krogh. "The Deep Generative Decoder: MAP estimation of representations improves modelling of single-cell RNA data." *Bioinformatics* 39.9 (2023): btad497.

(Prada-Luengo, Schuster, Liang, et al., 2023) Prada-Luengo, Iñigo, et al. "N-of-one differential gene expression without control samples using a deep generative model." *Genome Biology* 24.1 (2023): 263.
