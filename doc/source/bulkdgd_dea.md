# `bulkdgd_dea`

This command can be used to perform differential expression analysis (DEA) of genes between a "treated" sample (for instance, a cancer sample) and an "untreated" (control) sample.

Within the context of the bulkdgd model, the DEA is intended between a "treated" experimental sample and a "control" sample, which is the model's decoder's output for the best representation of the "treated" sample in latent space. Therefore, the decoder output for the best representation of the "treated" sample acts as an in silico control sample.

`bulkdgd_dea` expects two to three inputs. First, a CSV file containing a data frame of experimental "treated" samples. The program assumes that each row represents a sample and each column represents either a gene or additional sample metadata. Second, the program expects a CSV file containing a data frame with the means of the distributions modeling the genes' counts in the in silico "control" samples. A third input is needed if the genes' counts were modeled using negative binomial distributions: a CSV file containing the r-values of the negative binomials modeling the genes' counts in the "control" samples. These last two files are obtained by running the [`bulkdgd_find_representations`](bulkdgd_find_representations.md) command on the "treated" samples.

The output of `bulkdgd_dea` is a CSV file for each sample containing the results of the differential expression analysis. Here, the p-values, q-values (adjusted p-values), and log2-fold changes relative to each gene's differential expression are reported. The program can also return one or more (one per sample) CSV file(s) containing the results of the gene set enrichment analysis, if one or more `-gsf`, `--genes-sets-files` are provided.

## Command line

```
bulkdgd_dea [-h] -is INPUT_SAMPLES -im INPUT_MEANS [-iv INPUT_RVALUES] [-odp OUTPUT_DEA_PREFIX] [-ogp OUTPUT_GSEA_PREFIX] [-mg] [-pr P_VALUES_RESOLUTION] [-pt P_VALUES_THRESHOLD] [-qa Q_VALUES_ALPHA] [-qm Q_VALUES_METHOD] [-qt Q_VALUES_THRESHOLD] [-fct LOG2_FOLD_CHANGE_THRESHOLD] [-gsf GENES_SETS_FILES [GENES_SETS_FILES ...]] [-n N_PROC] [-d WORK_DIR] [-lf LOG_FILE] [-lc] [-v] [-vv]
```

## Options

### Help options

| Option         | Description                     |
| -------------- | ------------------------------- |
| `-h`, `--help` | Show the help message and exit. |

### Input files

| Option                   | Description                                                  |
| ------------------------ | ------------------------------------------------------------ |
| `-is`, `--input-samples` | The input CSV file containing a data frame with the gene expression data for the samples. |
| `-im`, `--input-means`   | The input CSV file containing the data frame with the predicted means of the distributions used to model the genes' counts for each in silico control sample. |
| `-iv`, `--input-rvalues` | The input CSV file containing the data frame with the predicted r-values of the negative binomials for each in silico control sample if negative binomial distributions were used to model the genes' counts. |

### Output files

| Option                   | Description                                                  |
| ------------------------ | ------------------------------------------------------------ |
| `-odp`, `--output-dea-prefix` | The prefix of the output CSV file(s) that will contain the results of the differential expression analysis. Since the analysis will be performed for each sample, one file per sample will be created. The files' names will have the form `{output_csv_prefix}{sample_name}.csv`. The default prefix is `dea_`. |
| `-ogp`, `--output-gsea-prefix`| The prefix of the output CSV file(s) that will contain the results of the gene set enrichment analysis. By default, the analysis will be performed for each sample and one per sample will be created. The files' names will have the form `{output_csv_prefix}{sample_name}.csv`. The default prefix is `gsea_`. If the `-mg`, `--merge-gsea` option is passed, this option is interpreted as the name of the output file where the merged results will be written (stripped of any trailing underscores or dots). |
| `-mg`, `--merge-gsea` | Whether the results of the gene set enrichment analysis will be merged into a single file. By default, the results for each sample are written in separate files. |

### DEA options

| Option                         | Description                                                  |
| ------------------------------ | ------------------------------------------------------------ |
| `-pr`, `--p-values-resolution` | The resolution at which to sum over the probability mass function to compute the p-values. The higher the resolution, the more accurate the calculation. The default is `10000.0`. |
| `-pt`, `--p-values-threshold` | The threshold used to select the significant genes based on the p-values. The default is `0.05`. |
| `-qa`, `--q-values-alpha`      | The alpha value used to calculate the q-values (adjusted p-values). The default is `0.05`. |
| `-qm`, `--q-values-method`     | The method used to calculate the q-values (i.e., to adjust the p-values). The default is `"fdr_bh"`. The available methods can be found in the documentation of `statsmodels.stats.multitest.multipletests`, which is used to perform the calculation. |
| `-qt`, `--q-values-threshold` | The threshold used to select the significant genes based on the q-values. The default is `0.05`. |
| `-fct`, `--log2-fold-change-threshold` | The threshold used to select the significant genes based on the log2-fold changes. The default is `2`. |
| `-gsf`, `--genes-sets-files` | A list of plain text files containing the gene sets to be used in the analysis. The files must contain one gene symbol per line. The gene sets will be used to perform the gene set enrichment analysis. |

### Run options

| Option          | Description                                                  |
| --------------- | ------------------------------------------------------------ |
| `-n`, `--n-proc` | The number of processes to start. The default number of processes started is 1. |
| `-dev`, `--device` | The device on which to calculate the p-values. If not provided, the GPU will be used if it is available, and the CPU otherwise. |
| `-pm`, `--p-values-method` | How to calculate the p-values: `batched`, `per-gene`, or `auto`. The default is `auto`. See [Calculating the p-values](#calculating-the-p-values) below. |

### Calculating the p-values

The p-value of a gene depends only on the distribution modelling that
gene's counts, so the genes are independent of each other, and the
calculation parallelizes exactly. There are two ways of exploiting this,
and they **give the same p-values**.

`batched` calculates the p-values for all the genes at once. This is
what allows them to be calculated on a GPU. On a CPU, `torch` spreads
the calculation over the cores by itself, so `batched` should be used
with a single process (`-n 1`) — a single process already uses the whole
machine.

`per-gene` calculates the p-values one gene at a time, and is
parallelized over the samples with the `-n`, `--n-proc` option. This is
how to use a machine with many cores when no GPU is available. Note that
it can use no more processes than there are samples.

The two kinds of parallelism **do not compose**. Several processes, each
running a `batched` calculation, would each try to use every core on the
machine, and would fight over them — which is slower than either kind of
parallelism on its own. `bulkdgd_dea` warns if it is asked to do this.

`auto`, the default, uses `batched` on a GPU and on a CPU with a single
process, and `per-gene` on a CPU with more than one process.

As a rough guide, on one 56-core node with an AMD MI250X GPU, for 16
samples and 14895 genes, at the default resolution:

| Method | Processes | Device | Time |
| ------ | --------- | ------ | ---- |
| `per-gene` | `-n 1` | CPU | 112.1 s |
| `per-gene` | `-n 7` | CPU | 43.8 s |
| `per-gene` | `-n 28` | CPU | 30.0 s |
| `batched` | `-n 1` | CPU | 25.7 s |
| `batched` | `-n 7` | CPU | 72.4 s (the combination to avoid) |
| `batched` | `-n 1` | GPU | 17.5 s |

### Working directory options

| Option             | Description                                                  |
| ------------------ | ------------------------------------------------------------ |
| `-d`, `--work-dir` | The working directory. The default is the current working directory. |

### Logging options

| Option                    | Description                                                  |
| ------------------------- | ------------------------------------------------------------ |
| `-lf`, `--log-file`       | The name of the log file. The file will be written in the working directory. The default file name is `bulkdgd_dea.log`. |
| `-lc`, `--log-console`    | Show log messages also on the console.                       |
| `-v`, `--logging-verbose` | Enable verbose logging (INFO level).                         |
| `-vv`, `--logging-debug`  | Enable maximally verbose logging for debugging purposes (DEBUG level). |

## Example

```
bulkdgd_dea -is samples.csv -im pred_means.csv -iv pred_r_values.csv -gsf glioblastoma_genes.txt -mg
```

This performs DEA for each sample in `samples.csv` against the in silico control samples described by `pred_means.csv`/`pred_r_values.csv` (as produced by [`bulkdgd_find_representations`](bulkdgd_find_representations.md)), additionally scoring the enrichment of the gene set in `glioblastoma_genes.txt` and merging the per-sample enrichment results into a single file.