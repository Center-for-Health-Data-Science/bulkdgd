#!/usr/bin/env Rscript

# tutorial_5.R
#
# Tutorial 5 - Using bulkdgd from R
#
# This script shows how to call the Python 'bulkdgd' package from R,
# via 'reticulate', to:
#
#   1. Find the best representations in latent space for a set of new
#      bulk RNA-Seq samples, using the pre-trained bulkdgd model.
#   2. Use the decoder's predictions as an in silico "control" for
#      each sample and perform differential expression analysis (DEA)
#      against it - no separate control samples needed.
#
# This is meant to be dropped into an RStudio session (or run with
# 'Rscript tutorial_5.R') and adapted to your own count matrix. It
# mirrors Tutorials 1 and 2 of the Python documentation; read those
# first if you want the full detail on what each step does under the
# hood: https://bulkdgd.readthedocs.io/en/latest/tutorials.html
#
# Design note: finding representations and running DEA both involve
# heavy PyTorch computation (gradient-based optimisation for the
# former; large probability-mass-function sums for the latter). Both
# are therefore delegated to bulkdgd's own command-line tools
# ('bulkdgd_find_representations', 'bulkdgd_dea') via 'system2()',
# rather than called in-process through 'reticulate'. This is not
# just a style choice: running PyTorch's autograd through R's
# embedded Python interpreter can segfault the R session outright (a
# known class of reticulate/PyTorch interoperability issue, unrelated
# to bulkdgd itself) - the same computation is completely reliable as
# a plain external process. Only the lightweight, non-numerical steps
# (loading/matching genes) are called in-process below.
#
# Prerequisites
# --------------
# * bulkdgd must already be installed in a Python environment (conda
#   or virtualenv) - see
#   https://bulkdgd.readthedocs.io/en/latest/installation.html.
#   'reticulate' does NOT install bulkdgd for you.
# * The 'reticulate' R package must be installed
#   (install.packages("reticulate")).


#######################################################################
## 0. Connect reticulate to the Python environment where bulkdgd lives
#######################################################################

library(reticulate)

# Point this at whichever environment bulkdgd was installed into.
# Pick ONE of the two lines below and adjust the name/path, or delete
# both if reticulate already finds the right Python automatically
# (e.g. because RETICULATE_PYTHON is set, or you are inside a
# 'renv'/'virtualenv' project already using it).

use_condaenv("bulkdgd-env", required = TRUE)
# use_virtualenv("~/venvs/bulkdgd-env", required = TRUE)

# Fail fast with a clear message if bulkdgd is not importable in the
# selected environment.
if (!py_module_available("bulkdgd")) {
  stop(
    "The 'bulkdgd' Python package is not importable in the Python ",
    "environment reticulate is using (", py_config()$python, "). ",
    "Install bulkdgd there first - see ",
    "https://bulkdgd.readthedocs.io/en/latest/installation.html - ",
    "or point 'use_condaenv()'/'use_virtualenv()' above at the ",
    "environment where you already installed it."
  )
}

ioutil <- import("bulkdgd.ioutil")

# bulkdgd's command-line tools are installed as scripts alongside the
# Python interpreter reticulate is using; resolving them this way
# means this script works regardless of whether that environment's
# 'bin' directory is on the shell's PATH (it usually isn't, in an
# RStudio session that wasn't launched from an activated conda/venv
# shell).
bulkdgd_bin_dir <- dirname(py_config()$python)

# Runs a bulkdgd command-line tool and stops with its output if it
# fails, instead of silently continuing with missing output files.
# The full output is printed via cat() rather than folded into the
# stop() message, because R truncates error/warning messages at
# ~1000 characters by default (see options("warning.length")) - long
# enough to hide the one line that actually explains the failure.
run_bulkdgd_cli <- function(command_name, args) {
  command_path <- file.path(bulkdgd_bin_dir, command_name)
  output <- system2(command_path, args = args, stdout = TRUE, stderr = TRUE)
  status <- attr(output, "status")
  if (!is.null(status) && status != 0) {
    cat(output, sep = "\n")
    stop(sprintf(
      "'%s' failed (exit status %s); see its output above (and '%s.log').",
      command_name, status, command_name
    ))
  }
  output
}


#######################################################################
## 1. Prepare your samples
#######################################################################

# bulkdgd expects a data frame with SAMPLES AS ROWS and GENES AS
# COLUMNS, columns named after the genes' Ensembl IDs (e.g.
# "ENSG00000187634", with or without a version suffix such as
# ".13" - both are handled). Row names are used as the sample
# names/IDs.
#
# This is the TRANSPOSE of the layout DESeq2 uses (DESeq2's 'counts'
# matrix is genes-as-rows, samples-as-columns). If you are starting
# from a DESeq2-style count matrix, transpose it first:
#
#   counts     <- your genes-by-samples count matrix
#   df_samples <- as.data.frame(t(counts))
#
# For this tutorial, we use the example data set shipped with the
# Python tutorials (10 GTEx-derived samples).
df_samples <- ioutil$load_samples(
  csv_file = "samples.csv",
  sep = ",",
  # The first column of the CSV file holds the samples' names/IDs.
  keep_samples_names = TRUE,
  split = FALSE
)

# Match the samples' genes against the set of genes included in the
# model: genes the model does not know about are dropped, and genes
# the model expects but that are missing from your data are added
# back in with a count of 0 (with a message telling you which genes
# these were, so you can sanity-check that the missing set is small).
preproc_result <- ioutil$preprocess_samples(df_samples = df_samples)
df_preproc     <- preproc_result[[1]]
genes_excluded <- preproc_result[[2]]
genes_missing  <- preproc_result[[3]]

cat(sprintf(
  "%d gene(s) excluded, %d gene(s) missing (set to 0) after matching to the model's gene set.\n",
  length(genes_excluded), length(genes_missing)
))

ioutil$save_samples(
  df = df_preproc,
  csv_file = "samples_preprocessed.csv",
  sep = ","
)


#######################################################################
## 2. Find the representations
#######################################################################

# "model_tgmm_trained" is the pre-trained model shipped with bulkdgd
# (Gaussian-mixture latent space + decoder, trained on GTEx data).
# "two_opt" is bulkdgd's default two-round optimisation scheme used
# to find each sample's representation. Both can be swapped for your
# own YAML config files (pass a full path instead of a bare name) -
# see
# https://bulkdgd.readthedocs.io/en/latest/model_config_options.html
# and .../rep_config_options.html.
#
# For each sample, this optimises a point in the model's latent space
# so that the decoder's output best reconstructs that sample's gene
# expression (a per-sample MAP optimisation, not a single forward
# pass) - expect this to take from seconds to a few minutes per
# sample on CPU, depending on the optimisation settings and your
# hardware. Start with a handful of samples the first time you run
# this.
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


#######################################################################
## 3. Differential expression analysis
#######################################################################

# For each sample, compare its observed gene counts against the
# decoder's prediction for that same sample (its in silico "control")
# - no separate control samples are needed. This computes p-values,
# Benjamini-Hochberg-adjusted q-values, and log2 fold changes for
# every gene, and writes one CSV file per sample
# ("dea_sample_<name>.csv").
run_bulkdgd_cli("bulkdgd_dea", c(
  "-is", "samples_preprocessed.csv",
  "-im", "pred_means.csv",
  "-iv", "pred_r_values.csv",
  "-odp", "dea_sample_",
  "-d", getwd(),
  "-lc", "-v"
))

# From here on, it's plain R: read a result file back in as a regular
# R data frame (gene as row name; columns 'p_value', 'q_value',
# 'log2_fold_change', plus the observed count and the decoder's
# predicted mean/r-value for reference) and work with it as you would
# a DESeq2 results table. For example, the significantly
# differentially expressed genes (q < 0.05) for the first sample,
# strongest fold change first:
dea_files <- list.files(pattern = "^dea_sample_.*\\.csv$")

df_dea_first <- read.csv(dea_files[1], row.names = 1)
df_sig <- df_dea_first[df_dea_first$q_value < 0.05, ]
df_sig <- df_sig[order(-abs(df_sig$log2_fold_change)), ]

cat(sprintf(
  "Sample '%s': %d significant gene(s) out of %d tested.\n",
  dea_files[1], nrow(df_sig), nrow(df_dea_first)
))
head(df_sig)
