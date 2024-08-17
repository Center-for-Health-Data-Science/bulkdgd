# `bulkdgd dea`

This command can be used to perform differential expression analysis (DEA) of genes between a "treated" sample (for instance, a cancer sample) against an "untreated" or "control" sample.

Within the context of the bulkDGD model, the DEA is intended between a "treated" experimental sample and a "control" sample, which is the model's decoder's output for the best representation of the "treated" sample in latent space. Therefore, the decoder output for the best representation of the "treated" sample acts as an in silico control sample.

`bulkdgd dea` expects two to three inputs. First, a CSV file containing a data frame set of experimental "treated" samples. The program assumes that each row represents a sample and each column represents a gene or additional information about the samples. Then, the program expects a CSV file containing a data frame with the means of the distributions modeling the genes' counts in the in silico "control" samples. The third input is needed if the genes' counts were modeled using negative binomial distributions and is a CSV file containing a data frame containing the r-values of the negative binomials modeling the genes' counts in the "control" samples. These last two files are obtained by running the [`bulkdgd find representations`](#bulkdgd_find_representations) command on the "treated" samples.

The output of `bulkdgd dea` is a CSV file for each sample containing the results of the differential expression analysis. Here, the p-values, q-values (adjusted p-values), and log2-fold changes relative to each gene's differential expression are reported.

To speed up DEA's performance on a set of samples, `bulkdgd dea` uses the [Dask](https://www.dask.org/) Python package to parallelize the calculations.

## Command line

```
bulkdgd dea [-h] -is INPUT_SAMPLES -im INPUT_MEANS [-iv INPUT_RVALUES] [-op OUTPUT_PREFIX] [-pr P_VALUES_RESOLUTION] [-qa Q_VALUES_ALPHA] [-qm Q_VALUES_METHOD] [-d WORK_DIR] [-n N_PROC] [-lf LOG_FILE] [-lc] [-v] [-vv]
```

## Options

### Help options

| Option         | Description                     |
| -------------- | ------------------------------- |
| `-h`, `--help` | Show the help message and exit. |

### Input files

| Option                   | Description                                                  |
| ------------------------ | ------------------------------------------------------------ |
| `-is`, `--input-samples` | The input CSV file containing a data frame with the gene expression data for the samples |
| `-im`, `--input-means`   | The input CSV file containing the data frame with the predicted means of the distributions used to model the genes' counts for each in silico control sample. |
| `-iv`, `--input-rvalues` | The input CSV file containing the data frame with the predicted r-values of the negative binomials for each in silico control sample if negative binomial distributions were used to model the genes' counts. |

### Output files

| Option                   | Description                                                  |
| ------------------------ | ------------------------------------------------------------ |
| `-op`, `--output-prefix` | The prefix of the output CSV file(s) that will contain the results of the differential expression analysis. Since the analysis will be performed for each sample, one file per sample will be created. The files' names will have the form `{output_csv_prefix}{sample_name}.csv`. The default prefix is `dea_`. |

### DEA options

| Option                         | Description                                                  |
| ------------------------------ | ------------------------------------------------------------ |
| `-pr`, `--p-values-resolution` | The resolution at which to sum over the probability mass function to compute the p-values. The higher the resolution, the more accurate the calculation. The default is `1e4`. |
| `-qa`, `--q-values-alpha`      | The alpha value used to calculate the q-values (adjusted p-values). The default is `0.05`. |
| `-qm`, `--q-values-method`     | The method used to calculate the q-values (i.e., to adjust the p-values). The default is `"fdr_bh"`. The available methods can be found in the documentation of `statsmodels.stats.multitest.multipletests`, which is used to perform the calculation. |

### Run options

| Option          | Description                                                  |
| --------------- | ------------------------------------------------------------ |
| `-n`, `--nproc` | The number of processes to start. The default number of processes started is 1. |

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