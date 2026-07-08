# Tutorial 5 - Using bulkdgd from R

`bulkdgd` is a Python package, but it can be driven directly from R through the [`reticulate`](https://rstudio.github.io/reticulate/) package, without writing any Python code yourself. This tutorial shows how to load the pre-trained `bulkdgd` model, find the best representations in latent space for your own samples, and perform differential expression analysis (DEA) against the model's in silico control - the same workflow as [Tutorial 1](tutorial_1.ipynb) and [Tutorial 2](tutorial_2.ipynb), called from R instead of Python.

This is meant for R users who want to use `bulkdgd` inside an existing R analysis pipeline, in a role similar to how one would use DESeq2 - except that `bulkdgd` needs no separate "control" samples, since the decoder's prediction for each sample acts as its own in silico control.

The full, ready-to-run script referenced below is [`tutorial_5.R`](https://github.com/Center-for-Health-Data-Science/bulkDGD/blob/main/tutorials/tutorial_5/tutorial_5.R) in the `tutorials/tutorial_5` directory of the repository, together with the example data set used here.

## Prerequisites

* `bulkdgd` must already be installed in a Python environment (`conda` or `virtualenv`) - see the [installation instructions](../installation.rst). `reticulate` does not install `bulkdgd` for you; it only connects R to a Python environment that already has it.
* The `reticulate` R package must be installed (`install.packages("reticulate")`).

## Design: reticulate for I/O, command-line tools for the heavy computation

Finding representations and running DEA both involve heavy PyTorch computation - gradient-based optimisation for the former, large probability-mass-function sums for the latter. The script deliberately does *not* run either of these in-process through `reticulate`. Instead, it calls bulkdgd's own command-line tools, [`bulkdgd_find_representations`](../bulkdgd_find_representations.md) and [`bulkdgd_dea`](../bulkdgd_dea.md), via R's `system2()`, and only uses `reticulate` in-process for the lightweight, non-numerical step of matching your samples' genes against the model's.

```{note}
This isn't a style preference - running PyTorch's autograd through R's embedded Python interpreter can segfault the R session outright. This is a known class of `reticulate`/PyTorch interoperability issue (unrelated to `bulkdgd` itself): a minimal `tensor$backward()` call works fine, but the more elaborate computation graph built while optimising a representation reliably crashed R in testing, with no such crash when the exact same call ran as a plain Python process. Delegating that computation to bulkdgd's command-line tools sidesteps the issue entirely and was verified end-to-end (real TCGA samples, a real trained decoder) before writing this tutorial.
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

The script also defines a small `run_bulkdgd_cli()` helper that runs one of these tools via `system2()` and stops with its full output if it fails - see the script for its definition. It prints that output with `cat()` rather than folding it into the `stop()` message, because R truncates error messages at ~1000 characters by default (`options("warning.length")`), which is long enough to hide the one line that actually explains a failure.

## Prepare your samples

`bulkdgd` expects a data frame with **samples as rows and genes as columns**, columns named after the genes' Ensembl IDs (versioned or unversioned, e.g. `ENSG00000187634` or `ENSG00000187634.13`). Row names are the sample names/IDs.

This is the *transpose* of the layout DESeq2 uses (DESeq2's `counts` matrix is genes-as-rows, samples-as-columns) - if that's where you're starting from:

```r
df_samples <- as.data.frame(t(counts))
```

Then match your genes against the model's gene set - genes the model doesn't know about are dropped, and genes it expects but that are missing from your data are added back with a count of 0 - and save the result to CSV, since the command-line tools that follow read their input from disk:

```r
preproc_result <- ioutil$preprocess_samples(df_samples = df_samples)
df_preproc     <- preproc_result[[1]]

ioutil$save_samples(df = df_preproc, csv_file = "samples_preprocessed.csv", sep = ",")
```

## Find the representations

`"model_tgmm_trained"` is the pre-trained model shipped with bulkdgd (Gaussian-mixture latent space + decoder, trained on GTEx data); `"two_opt"` is bulkdgd's default two-round optimisation scheme. Both are bare names resolved against bulkdgd's own packaged configuration directories - pass a full path instead to use your own YAML config file (see the [model](../model_config_options.rst) and [representation-finding](../rep_config_options.rst) configuration options).

```r
run_bulkdgd_cli("bulkdgd_find_representations", c(
  "-is", "samples_preprocessed.csv",
  "-cm", "model_tgmm_trained",
  "-cr", "two_opt",
  "-or", "representations.csv",
  "-om", "pred_means.csv",
  "-ov", "pred_r_values.csv",
  "-ot", "opt_time.csv",
  "-d", getwd(),
  "-lc", "-v"
))
```

Finding a representation is a per-sample optimisation in latent space (not a single forward pass), so this step is the slow part - from seconds to a few minutes per sample on CPU depending on the optimisation settings. Start with a handful of samples the first time you run this.

## Differential expression analysis

For each sample, `bulkdgd` compares its observed gene counts against the decoder's own prediction for that sample and computes p-values, Benjamini-Hochberg-adjusted q-values, and log2 fold changes, writing one CSV file per sample (`dea_sample_<name>.csv`):

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

From here it's plain R - read a result file back in as a genuine R data frame (columns `p_value`, `q_value`, `log2_fold_change`, plus the observed count and the decoder's predicted mean/r-value for reference) and work with it as you would a DESeq2 results table:

```r
dea_files <- list.files(pattern = "^dea_sample_.*\\.csv$")

df_dea_first <- read.csv(dea_files[1], row.names = 1)
df_sig <- df_dea_first[df_dea_first$q_value < 0.05, ]
df_sig <- df_sig[order(-abs(df_sig$log2_fold_change)), ]
```

See [`bulkdgd_dea`](../bulkdgd_dea.md) for the full set of options (exact vs. approximate p-value calculation, alternative multiple-testing correction methods, gene set enrichment analysis, and more).
