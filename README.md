<h1 align="center">
<img src="./branding/bulkdgd_logo.png" width="300">
</h1><br>

<a href='https://bulkdgd.readthedocs.io/en/latest/?badge=latest'>
    <img src='https://readthedocs.org/projects/bulkdgd/badge/?version=latest' alt='Documentation Status' />
</a>


``bulkdgd`` is a Python package providing an interface to use the Deep Generative Decoder (DGD) developed by Schuster and Krogh (Schuster and Krogh, 2023) to model the gene expression of healthy human tissues from bulk RNA-Seq data.

The first application of the model to bulk RNA-Seq data is presented in the work of Prada-Luengo, Schuster, Liang, and coworkers (Prada-Luego, Schuster, Liang, et al., 2023).

## Version 2.0.0 - major release

This is a **major release** superseding all previous ``bulkDGD`` versions. Existing users should switch to it. The biggest changes compared to the previous version:

* The package has been renamed from ``bulkDGD`` to ``bulkdgd`` (the import name and CLI commands are unaffected, since they already used the lowercase form).
* The pre-trained model shipped by default has been retrained on GTEx data using a new Gaussian mixture model implementation (``tgmm``) and a curated, smaller gene list, replacing the previous default model.
* Several CLI commands (`bulkdgd_train`, `bulkdgd_find_probdens`, `bulkdgd_find_representations`) had a leftover bug from an earlier refactor that made them fail immediately on invocation - this has been fixed.
* All tutorials have been converted from standalone Python scripts into executed Jupyter notebooks, and a new tutorial covers downloading and preparing samples from [Recount3](https://rna.recount.bio/).
* The documentation has been substantially overhauled: broken API cross-references, outdated installation instructions, and missing CLI usage examples have all been fixed, and the docs now build cleanly with no warnings.

* **Documentation**: bulksgs's documentation can be found [here](https://bulkdgd.readthedocs.io/en/latest/).

* **Bug reports**: please report any bugs or problems you encounter with bulkDGD in the dedicated [issues](https://github.com/Center-for-Health-Data-Science/bulkDGD/issues) section on GitHub.

**License**

``bulkdgd`` is freely available under the terms of the GNU General Public License (Version 3, 29 June 2007).

**References**

(Schuster and Krogh, 2023) Schuster, Viktoria, and Anders Krogh. "The Deep Generative Decoder: MAP estimation of representations improves modelling of single-cell RNA data." *Bioinformatics* 39.9 (2023): btad497.

(Prada-Luengo, Schuster, Liang, et al., 2023) Prada-Luengo, Iñigo, et al. "N-of-one differential gene expression without control samples using a deep generative model." *Genome Biology* 24.1 (2023): 263.
