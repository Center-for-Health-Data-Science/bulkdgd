# Tutorial 5 - Using bulkdgd from R

`bulkdgd` is a Python package, but it can be driven directly from R through the [`reticulate`](https://rstudio.github.io/reticulate/) package, without writing any Python code yourself. This tutorial shows how to load the pre-trained `bulkdgd` model, find the best representations in latent space for your own samples, and perform differential expression analysis (DEA) against the model's in silico control. This is the same workflow as [Tutorial 1](tutorial_1.ipynb) and [Tutorial 2](tutorial_2.ipynb), called from R instead of Python.

This tutorial is for R users who want to use `bulkdgd` inside an existing R analysis pipeline, in a role similar to DESeq2. Unlike DESeq2, `bulkdgd` needs no separate "control" samples, since the decoder's prediction for each sample acts as its own in silico control.

The full script referenced below is [`tutorial_5.R`](https://github.com/Center-for-Health-Data-Science/bulkDGD/blob/main/tutorials/tutorial_5/tutorial_5.R) in the `tutorials/tutorial_5` directory of the repository, together with the example data set used here.

## Prerequisites

* `bulkdgd` must already be installed in a Python environment (`conda` or `virtualenv`); see the [installation instructions](../installation.rst). `reticulate` does not install `bulkdgd` for you; it only connects R to a Python environment that already has it.
* The `reticulate` R package must be installed (`install.packages("reticulate")`).

## Design: reticulate for I/O, command-line tools for the heavy computation

Finding representations and running DEA both involve heavy PyTorch computation: gradient-based optimisation for the former, large probability-mass-function sums for the latter. The script does *not* run either of these in-process through `reticulate`. Instead, it calls bulkdgd's own command-line tools, [`bulkdgd_find_representations`](../bulkdgd_find_representations.md) and [`bulkdgd_dea`](../bulkdgd_dea.md), via R's `system2()`, and uses `reticulate` in-process only for the lightweight, non-numerical step of matching your samples' genes against the model's.

```{note}
Running PyTorch's autograd through R's embedded Python interpreter can crash the R session with a segmentation fault. This is a known class of `reticulate`/PyTorch interoperability issue, unrelated to `bulkdgd` itself: a minimal `tensor$backward()` call works, but the larger computation graph built while optimising a representation can crash R, while the same call runs without problems as a plain Python process. Running that computation through bulkdgd's command-line tools avoids the issue.
```

## Connect reticulate to the right Python environment

Point `reticulate` at the environment in which you installed `bulkdgd`, then check that the package is importable before going any further:

```r
library(reticulate)

use_condaenv("bulkdgd-env", required = TRUE)
# or, for a virtualenv installation:
# use_virtualenv("~/venvs/bulkdgd-env", required = TRUE)

if (!py_module_available("bulkdgd")) {
  stop("bulkdgd is not importable in the selected Python environment.")
}

ioutil <- import("bulkdgd.ioutil")

# bulkdgd's command-line tools are installed as scripts alongside the
# Python interpreter reticulate is using - resolving them this way
# means the script works even when that environment's 'bin' directory
# isn't on the shell's PATH (usually the case in an RStudio session
# not launched from an activated conda/venv shell).
bulkdgd_bin_dir <- dirname(py_config()$python)
```

The script also defines a `run_bulkdgd_cli()` helper that runs one of these tools via `system2()` and stops with its full output if it fails (see the script for its definition). It prints that output with `cat()` rather than including it in the `stop()` message, because R truncates error messages at about 1000 characters by default (`options("warning.length")`), which can hide the line that explains the failure.

## Prepare your samples

`bulkdgd` expects a data frame with **samples as rows and genes as columns**, columns named after the genes' Ensembl IDs (versioned or unversioned, e.g. `ENSG00000187634` or `ENSG00000187634.13`). Row names are the sample names/IDs.

This is the *transpose* of the layout DESeq2 uses (DESeq2's `counts` matrix has genes as rows and samples as columns). If you start from such a matrix, transpose it:

```r
df_samples <- as.data.frame(t(counts))
```

Then match your genes against the model's gene set (genes the model does not know are dropped, and genes it expects but your data lack are added with a count of 0), and save the result to CSV, since the command-line tools that follow read their input from disk:

```r
preproc_result <- ioutil$preprocess_samples(df_samples = df_samples)
df_preproc     <- preproc_result[[1]]

ioutil$save_samples(df = df_preproc, csv_file = "samples_preprocessed.csv", sep = ",")
```

## Find the representations

`"two_opt"` is bulkdgd's default two-round optimisation scheme, passed as a bare name because bare names are resolved against bulkdgd's own packaged configuration directories.

The model's configuration is passed as a path instead. The pre-trained model shipped with bulkdgd (a Gaussian-mixture latent space and a decoder, trained on GTEx data) has one configuration file per member of the ensemble, in a sub-directory that the bare-name lookup does not reach, and those files name an architecture and no parameter files. Copy `bulkdgd/configs/model/seed37/model.yaml` next to your samples as `model_trained.yaml`, add `latent_pth_file: "default"` under `latent_options` and `decoder_pth_file: "default"` under `decoder_options`, and pass the copy. See the [model](../model_config_options.rst) and [representation-finding](../rep_config_options.rst) configuration options for the details.

```r
run_bulkdgd_cli("bulkdgd_find_representations", c(
  "-is", "samples_preprocessed.csv",
  "-cm", "model_trained.yaml",
  "-cr", "two_opt",
  "-or", "representations.csv",
  "-om", "pred_means.csv",
  "-ov", "pred_r_values.csv",
  "-ot", "opt_time.csv",
  "-d", getwd(),
  "-lc", "-v"
))
```

Finding a representation is a per-sample optimisation in latent space (not a single forward pass), so this step is the slowest: from seconds to a few minutes per sample on CPU, depending on the optimisation settings. Start with a few samples the first time you run it.

## Differential expression analysis

For each sample, `bulkdgd` compares its observed gene counts against the decoder's own prediction for that sample and computes p-values, Benjamini-Hochberg-adjusted q-values, and log2 fold changes, writing one Parquet file per sample (`dea_sample_<name>.parquet`):

```r
run_bulkdgd_cli("bulkdgd_dea", c(
  "-is", "samples_preprocessed.csv",
  "-im", "pred_means.csv",
  "-iv", "pred_r_values.csv",
  "-odp", "dea_sample_",
  "-d", getwd(),
  "-lc", "-v"
))
```

Read a result file back, with pandas through `reticulate`, as an R data frame (columns `p_value`, `q_value`, `log2_fold_change`, plus the observed count and the decoder's predicted mean/r-value) and work with it as you would with a DESeq2 results table:

```r
dea_files <- list.files(pattern = "^dea_sample_.*\\.parquet$")

pd <- import("pandas")
df_dea_first <- py_to_r(pd$read_parquet(dea_files[1]))
df_sig <- df_dea_first[df_dea_first$q_value < 0.05, ]
df_sig <- df_sig[order(-abs(df_sig$log2_fold_change)), ]
```

See [`bulkdgd_dea`](../bulkdgd_dea.md) for the full set of options (exact vs. approximate p-value calculation, alternative multiple-testing correction methods, gene set enrichment analysis, and more).
