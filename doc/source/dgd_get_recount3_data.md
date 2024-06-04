# `dgd_get_recount3_data`

This executable retrieves RNA-seq data (and associated metadata) from the [Recount3 platform](https://rna.recount.bio/).

So far, the program supports retrieving data for samples from the [GTEx](https://gtexportal.org/home/), [TCGA](https://www.cancer.gov/ccg/research/genome-sequencing/tcga), and [SRA](https://www.ncbi.nlm.nih.gov/sra) projects.

The executable allows samples to be selected for a single tissue (for GTEx data), cancer type (for TCGA), or project code (for SRA) and to filter them according to the associated metadata. The filtering is performed using a query string in the format supported by the [`pandas.DataFrame.query()`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.query.html) method.

Metadata fields on which it is possible to filter the samples differ between GTEx, TCGA, and SRA samples. A list of the available fields is available in `bulkDGD/recount3/data/gtex_metadata_fields.txt` for GTEx samples, `bulkDGD/recount3/data/tcga_metadata_fields.txt` for TCGA samples, and `bulkDGD/recount3/data/sra_metadata_fields.txt` for SRA samples.

The main output of `dgd_get_recount3_data` is a CSV file containing the RNA-seq data retrieved from Recount3 for the samples of interest. The rows represent the samples, while the columns contain the genes, identified by their Ensembl IDs.

The user also has the option to save the original compressed (`.gz`) files containing the RNA-seq data and the metadata associated with the samples. If these files are found in the working directory for a specific project and sample category, they will not be downloaded again.

To speed up data retrieval and processing,  `dgd_get_recount3_data` uses the [Dask](https://www.dask.org/) Python package to parallelize the calculations.

## Command line

```
dgd_get_recount3_data [-h] [-is INPUT_SAMPLES_BATCHES] [-d WORK_DIR] [-n N_PROC] [-sg] [-sm] [-lf LOG_FILE] [-lc] [-v] [-vv]
```

## Options

| Option                          | Description                                                  |
| ------------------------------- | ------------------------------------------------------------ |
| `-h`, `--help`                  | Show the help message and exit.                              |
| `-i`, `--input-samples-batches` | A CSV file to download samples' data in bulk. The file must contain at least two columns: `input_project_name` with the name of the project the samples belong to and `input_samples_category` with the samples' category. A third column, `query_string`, may specify the query string used to filter each batch of samples. |
| `-d`, `--work-dir`              | The working directory. The default is the current working directory. |
| `-n`, `--n-proc`                | The number of processes to start. The default number of processes started is 1. |
| `-sg`, `--save-gene-sums`       | Save the original GZ file containing the RNA-seq data for the samples. For each batch of samples, the corresponding file will be saved in the working directory and named`{input_project_name}_{input_samples_category}_gene_sums.gz`. This file will be written only once if more than one batch refers to the same `input_project_name` and `input_samples_category`. |
| `-sm`, `--save-metadata`        | Save the original GZ file containing the metadata for the samples. For each batch of samples, the corresponding file will be saved in the working directory and named `{input_project_name}_{input_samples_category}_metadata.gz`. This file will be written only once if more than one batch refers to the same `input_project_name` and `input_samples_category` |
| `-lf`, `--log-file`             | The name of the log file. The file will be written in the working directory. The default file name is `dgd_get_recount3_data.log`. |
| `-lc`, `--log-console`          | Show log messages also on the console.                       |
| `-v`, `--logging-verbose`       | Enable verbose logging (INFO level).                         |
| `-vv`, `--logging-debug`        | Enable maximally verbose logging for debugging purposes (DEBUG level). |