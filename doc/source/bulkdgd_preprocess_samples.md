# `bulkdgd preprocess samples`

This command allows users to preprocess new samples to use with the bulkDGD model.

It expects as input a CSV file containing a data frame with the gene expression data for the new samples.

Each row must represent a sample, while each column must represent a gene identified by its Ensembl ID or additional information about the samples. The first column is expected to contain the samples' unique names, IDs, or indexes.

During preprocessing, Ensembl IDs indicating [pseudoautosomal regions](http://www.ensembl.org/info/genome/genebuild/human_PARS.html) are treated as different genes.

In detail, sample preprocessing consists of the following steps:

1. Removing duplicated samples.
2. Removing samples containing missing values for the expression of some genes.
3. Excluding all data for genes that are not included in the bulkDGD model. A plain text file containing the list of the genes used in the model (`genes.txt`) is available in `bulkDGD/data/model/genes`.
4. Adding a count of 0 for all genes not found in the input samples but part of the set of genes used to train the bulkDGD model.
5. Sorting the genes in the order expected by the bulkDGD model.

The program will exit with an error if it finds duplicated genes.

Preprocessing new samples is a critical step before finding the samples' best representations in latent space using the bulkDGD model (using, for instance, the [`bulkdgd find representations`](#bulkdgd_find_representations) executable).

The main output of `bulkdgd preprocess samples` is a CSV file containing the preprocessed samples.

The Ensembl IDs of the genes found in the input samples but not present in the gene set used to train the bulkDGD model, if any, are written in a separate output file in plain text format, with one gene per line. 

Furthermore, if any gene present in the gene set used to train the bulkDGD model is not found in the genes for which counts are available in the input samples, the gene will be written to the output CSV file containing the preprocessed samples with a count of 0 for all samples. These "missing" genes are also written in a separate plain text file containing one gene per line.

## Parallelization

If the command is parallelized over several input files, each is assumed to be identically named and placed in a different directory. The paths to such directories must be specified using the `-ds`, `--dirs` option (see the full description in the [Parallelization options](#parallelization-options) section below) and must be relative to the specified working directory (`-d`, `--work-dir` option).

The input files must be placed in different directories and referenced by their corresponding options by name (not path).

The output and log files for each run will be written in the directory where the corresponding input/configuration files were placed. If the command is not parallelized, these files will be written in the working directory.

## Command line

```
bulkdgd preprocess samples [-h] -is INPUT_SAMPLES [-os OUTPUT_SAMPLES] [-oe OUTPUT_GENES_EXCLUDED] [-om OUTPUT_GENES_MISSING] [-d WORK_DIR] [-lf LOG_FILE] [-lc] [-v] [-vv]  [-p] [-n N_PROC] [-ds DIRS [DIRS ...]]
```

## Options

### Help options

| Option         | Description                     |
| -------------- | ------------------------------- |
| `-h`, `--help` | Show the help message and exit. |

### Input files

| Option                   | Description                                                  |
| ------------------------ | ------------------------------------------------------------ |
| `-is`, `--input-samples` | The input CSV file containing a data frame with the samples to be preprocessed. |

### Output files

| Option                           | Description                                                  |
| -------------------------------- | ------------------------------------------------------------ |
| `-os`, `--output-samples`        | The name of the output CSV file containing the data frame with the preprocessed samples. The default file name is `samples_preprocessed.csv`. |
| `-oe`, `--output-genes-excluded` | The name of the output plain text file containing the list of genes whose expression data are excluded from the data frame with the preprocessed samples. The default file name is `genes_excluded.txt`. |
| `-om`, `--output-genes-missing`  | The name of the output plain text file containing the list of genes for which no expression data are found in the input data frame. A default count of 0 is assigned to these genes in the output data frame with the preprocessed samples. The default file name is `genes_missing.txt`. |

### Working directory options

| Option             | Description                                                  |
| ------------------ | ------------------------------------------------------------ |
| `-d`, `--work-dir` | The working directory. The default is the current working directory. |

### Logging options

| Option                    | Description                                                  |
| ------------------------- | ------------------------------------------------------------ |
| `-lf`, `--log-file`       | The name of the log file. The default file name is `bulkdgd_preprocess_samples.log`. |
| `-lc`, `--log-console`    | Show log messages also on the console.                       |
| `-v`, `--logging-verbose` | Enable verbose logging (INFO level).                         |
| `-vv`, `--logging-debug`  | Enable maximally verbose logging for debugging purposes (DEBUG level). |

### Parallelization options

| Option                | Description                                                  |
| --------------------- | ------------------------------------------------------------ |
| `-p`, `--parallelize` | Whether to run the command in parallel.                      |
| `-n`, `--n-proc`      | The number of processes to start. The default number of processes started is 1. |
| `-ds`, `--dirs`       | The directories containing the input/configuration files. It can be either a list of names or paths, a pattern that the names or paths match, or a plain text file containing the names of or the paths to the directories. If names are given, the directories are assumed to be inside the working directory. If paths are given, they are assumed to be relative to the working directory. |
