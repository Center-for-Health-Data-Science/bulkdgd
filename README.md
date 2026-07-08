<h1 align="center">
<img src="./branding/bulkdgd_logo.png" width="300">
</h1><br>

<a href='https://bulkdgd.readthedocs.io/en/latest/?badge=latest'>
    <img src='https://readthedocs.org/projects/bulkdgd/badge/?version=latest' alt='Documentation Status' />
</a>


``bulkdgd`` is a Python package providing an interface to use the Deep Generative Decoder (DGD) to model the gene expression of healthy human tissues from bulk RNA-Seq data.

The first version of the DGD was developed in 2023 (Schuster and Krogh, 2023) and the first application of the model to bulk RNA-Seq data is presented in the work of Prada-Luengo, Schuster, Liang, and coworkers (Prada-Luego, Schuster, Liang, et al., 2023).

``bulkdgd`` is a Python package, but it can be used directly from R without writing any Python code. For how to perform differential expression analysis using ``bulkdgd`` in R, see the [R tutorial](https://bulkdgd.readthedocs.io/en/latest/tutorial_notebooks/tutorial_5.html).

All the [recount3](https://rna.recount.bio/) samples used to train and test the model were processed from raw files through the [Monorail](https://github.com/langmead-lab/monorail-external/) pipeline. If you have your own raw files, we recommend processing them through the same pipeline before using them with ``bulkdgd``.

## Quick install

```sh
pip install bulkdgd
```

The trained decoder's parameters (``dec.pth``, too large to distribute on PyPI) are downloaded automatically the first time you use the pre-trained model - see the [installation instructions](https://bulkdgd.readthedocs.io/en/latest/installation.html) for details, including a manual-download fallback for offline machines.

## Version 2.0.1

* The default pre-trained model has been updated (both the Gaussian mixture and the decoder's parameters), fixing a mismatch between the packaged model configuration and the previously-shipped decoder.
* The trained decoder's parameters (``dec.pth``) are no longer distributed with the package or manually downloaded - ``bulkdgd`` now downloads them automatically, once, the first time the pre-trained model is used.

## Version 2.0.0 - major release

This is a **major release** superseding all previous versions of the package (previously named ``bulkDGD``). Existing users should switch to it. The biggest changes compared to the previous version:

* The package has been renamed from ``bulkDGD`` to ``bulkdgd`` (the import name and CLI commands are unaffected, since they already used the lowercase form).
* The pre-trained model shipped by default has been retrained on GTEx data using a new Gaussian mixture model implementation (``tgmm``) and a curated, smaller gene list, replacing the previous default model.
* Several CLI commands (`bulkdgd_train`, `bulkdgd_find_probdens`, `bulkdgd_find_representations`) had a leftover bug from an earlier refactor that made them fail immediately on invocation - this has been fixed.
* All tutorials have been converted from standalone Python scripts into executed Jupyter notebooks, and a new tutorial covers downloading and preparing samples from [Recount3](https://rna.recount.bio/).
* The documentation has been substantially overhauled: broken API cross-references, outdated installation instructions, and missing CLI usage examples have all been fixed, and the docs now build cleanly with no warnings.

* **Documentation**: ``bulkdgd``'s documentation can be found [here](https://bulkdgd.readthedocs.io/en/latest/).

* **Bug reports**: please report any bugs or problems you encounter with ``bulkdgd`` in the dedicated [issues](https://github.com/Center-for-Health-Data-Science/bulkdgd/issues) section on GitHub.

**License**

``bulkdgd`` is freely available under the terms of the GNU General Public License (Version 3, 29 June 2007).

**References**

(Schuster and Krogh, 2023) Schuster, Viktoria, and Anders Krogh. "The Deep Generative Decoder: MAP estimation of representations improves modelling of single-cell RNA data." *Bioinformatics* 39.9 (2023): btad497.

(Prada-Luengo, Schuster, Liang, et al., 2023) Prada-Luengo, Iñigo, et al. "N-of-one differential gene expression without control samples using a deep generative model." *Genome Biology* 24.1 (2023): 263.
