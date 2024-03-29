# `dgd_perform_dea`

This executable can be used to perform differential expression analysis (DEA) of genes between a "treated" sample (for instance, a cancer sample) against an "untreated" or "control" sample.

Within the context of the DGD model, the DEA is intended between a "treated" experimental sample and a "control" sample, which is the model's decoder's output for the best representation of the "treated" sample in latent space. Therefore, the decoder output for the best representation of the "treated" sample acts as an *in silico* control sample.

This approach was first presented in the work of Prada-Luengo, Schuster, Liang, and coworkers [^1].

`dgd_perform_dea` expects three inputs. First, a CSV file containing a data frame set of experimental "treated" samples. The program assumes that each row represents a sample and each column represents a gene or additional information about the samples. Then, the program expects a data frame with the *in silico* "control" samples. This data frame is structured as that containing the "treated" samples and can be obtained by running the [`dgd_get_representations`](#dgd_get_representations) executable on the "treated" samples.

The output of `dgd_perform_dea` is a CSV file containing the results of the differential expression analysis. Here, the p-values, q-values (adjusted p-values), and log2-fold changes relative to each gene's differential expression are reported.

To speed up DEA's performance on a set of samples, `dgd_perform_dea` uses the [Dask](https://www.dask.org/) Python package to parallelize the calculations.

## Command line

```
[-h] -is INPUT_CSV_SAMPLES -id INPUT_CSV_DEC [-op OUTPUT_CSV_PREFIX] -cm CONFIG_FILE_MODEL [-pr P_VALUES_RESOLUTION] [-qa Q_VALUES_ALPHA] [-qm Q_VALUES_METHOD] [-d WORK_DIR] [-n N_PROC] [-lf LOG_FILE] [-lc] [-v] [-vv]
```

## Options

| Option                         | Description                                                  |
| ------------------------------ | ------------------------------------------------------------ |
| `-h`, `--help`                 | Show the help message and exit.                              |
| `-is`, `--input-csv-samples`   | The input CSV file containing a data frame with the gene expression data for the samples |
| `-id`, `--input-csv-dec`       | The input CSV file containing the data frame with the decoder's output for each sample's best representation. |
| `-op`, `--output-csv-prefix`   | The prefix of the output CSV file(s) that will contain the results of the differential expression analysis. Since the analysis will be performed for each sample, one file per sample will be created. The files' names will have the form `{output_csv_prefix}{sample_name}.csv`. The default prefix is `dea_`. |
| `-cm`, `--config-file-model`   | The YAML configuration file specifying the DGD model's parameters and files containing the trained model. If it is a name without an extension, it is assumed to be the name of a configuration file in `$INSTALLDIR/bulkDGD/ioutil/configs/model`. |
| `-pr`, `--p-values-resolution` | The resolution at which to sum over the probability mass function to compute the p-values. The higher the resolution, the more accurate the calculation. The default is `1e4`. |
| `-qa`, `--q-values-alpha`      | The alpha value used to calculate the q-values (adjusted p-values). The default is `0.05`. |
| `-qm`, `--q-values-method`     | The method used to calculate the q-values (i.e., to adjust the p-values). The default is `"fdr_bh"`. The available methods can be found in the documentation of `statsmodels.stats.multitest.multipletests`, which is used to perform the calculation. |
| `-d`, `--work-dir`             | The working directory. The default is the current working directory. |
| `-n`, `--nproc`                | The number of processes to start. The default number of processes started is 1. |
| `-lf`, `--log-file`            | The name of the log file. The file will be written in the working directory. The default file name is `dgd_perform_dea.log`. |
| `-lc`, `--log-console`         | Show log messages also on the console.                       |
| `-v`, `--logging-verbose`      | Enable verbose logging (INFO level).                         |
| `-vv`, `--logging-debug`       | Enable maximally verbose logging for debugging purposes (DEBUG level). |