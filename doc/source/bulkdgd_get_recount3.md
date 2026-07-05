# `bulkdgd_get_recount3`

This command retrieves RNA-seq data (and associated metadata) from the [Recount3 platform](https://rna.recount.bio/).

So far, the program supports retrieving data for samples from the [GTEx](https://gtexportal.org/home/), [TCGA](https://www.cancer.gov/ccg/research/genome-sequencing/tcga), and [SRA](https://www.ncbi.nlm.nih.gov/sra) projects.

The executable allows samples to be selected for a single tissue (for GTEx data), cancer type (for TCGA), or project code (for SRA).

`bulkdgd_get_recount3` accepts a CSV file as input containing the batches of samples to be downloaded from Recount3. The CSV file is expected to have two columns:

  * `"recount3_project_name"`, containing the name of the project (`"GTEx"`, `"TCGA"`, or `"SRA"`) the samples belong to.
  * `"recount3_samples_category"`, containing the name of the category the samples belong to (it is a tissue type for GTEx data, a cancer type for TCGA data, and a project code for SRA data).

The main output of `bulkdgd_get_recount3` is several CSV files (one per batch of samples) containing the RNA-seq data retrieved from Recount3 for the samples of interest. The rows represent the samples, while the columns contain the genes identified by their Ensembl IDs or the samples' metadata. This file is named `{recount3_project_name}_{recount3_samples_category}.csv`.

The raw RNA-seq data are converted into read counts by dividing the raw counts by the average mapped length of the reads per sample.

The user also has the option to save the original compressed (`.gz`) files containing the raw RNA-seq data, the quality control metrics (including the average mapped length per sample), and the metadata associated with the samples. If these files are found in the working directory for a specific project and sample category, they will not be downloaded again.

## Command line

```
bulkdgd_get_recount3 [-h] [-ib INPUT_SAMPLES_BATCHES] [-d WORK_DIR] [-n N_PROC] [-sg] [-sq] [-sm] [-lf LOG_FILE] [-lc] [-v] [-vv]
```

## Options

### Help options

| Option         | Description                     |
| -------------- | ------------------------------- |
| `-h`, `--help` | Show the help message and exit. |

### Input files

| Option                           | Description                                                  |
| -------------------------------- | ------------------------------------------------------------ |
| `-ib`, `--input-samples-batches` | A CSV file used to download samples' data in bulk. |

### Output options

| Option                    | Description                                                  |
| ------------------------- | ------------------------------------------------------------ |
| `-sg`, `--save-gene-sums` | Save the original GZ file containing the RNA-seq data for the samples. For each batch of samples, the corresponding file will be saved in the working directory and named `{recount3_project_name}_{recount3_samples_category}_gene_sums.gz`. |
| `-sq`, `--save-qc` | Save the original GZ file containing the quality control metrics for the samples. For each batch of samples, the corresponding file will be saved in the working directory and named `{recount3_project_name}_{recount3_samples_category}_qc.gz`. |
| `-sm`, `--save-metadata`  | Save the original GZ file containing the metadata for the samples. For each batch of samples, the corresponding file will be saved in the working directory and named `{recount3_project_name}_{recount3_samples_category}_metadata.gz`. |

### Run options

| Option           | Description                                                  |
| ---------------- | ------------------------------------------------------------ |
| `-n`, `--n-proc` | The number of processes to start. The default number of processes started is 1. |

### Working directory options

| Option             | Description                                                  |
| ------------------ | ------------------------------------------------------------ |
| `-d`, `--work-dir` | The working directory. The default is the current working directory. |

### Logging options

| Option                    | Description                                                  |
| ------------------------- | ------------------------------------------------------------ |
| `-lf`, `--log-file`       | The name of the log file. The file will be written in the working directory. The default file name is `bulkdgd_get_recount3.log`. |
| `-lc`, `--log-console`    | Show log messages also on the console.                       |
| `-v`, `--logging-verbose` | Enable verbose logging (INFO level).                         |
| `-vv`, `--logging-debug`  | Enable maximally verbose logging for debugging purposes (DEBUG level). |

## Example

Given an input file `batches.csv` containing:

```
recount3_project_name,recount3_samples_category
TCGA,GBM
SRA,SRP027383
SRA,SRP072494
SRA,SRP141440
```

```
bulkdgd_get_recount3 -ib batches.csv -d recount3_data
```

This downloads, in parallel, the TCGA glioblastoma multiforme (GBM) samples and the three SRA glioblastoma studies listed in `batches.csv`, writing one CSV file per batch (for instance, `TCGA_GBM.csv` and `SRA_SRP027383.csv`) in the `recount3_data` directory. See :doc:`Tutorial 4 <tutorial_notebooks/tutorial_4>` for a complete, worked example using this exact dataset.