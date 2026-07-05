# `bulkdgd_get_genes`

This command allows you to create customized lists of genes to use with the bulkdgd model.

`bulkdgd_get_genes` takes as input a YAML configuration file containing:

* The `attributes` to retrieve for the genes of interest from the Ensembl database.

* The `filters` to use on the genes retrieved from the Ensembl database (to keep, for instance, only protein-coding genes or genes producing only protein-coding transcripts).

The command produces two output files:

* A CSV file containing the `attributes` for the genes passing all the `filters`.

* A plain text file containing the Ensembl IDs of the genes reported in the CSV file. This file can be directly used when setting up a new instance of the bulkdgd model or preprocessing a set of samples to be used with the model.

## Parallelization

The command can be run in parallel over different inputs in different directories by using the `-ds`, `--dirs` option. The directories may be specified either by name (if they are in the current working directory) or their absolute or relative path.

* If `-ds dir1 path/to/dir2`, the program will be run in parallel in each directory using the input and configuration files in it. The name of the configuration file may be provided using the `-cg`/ `--config-file-genes` option. The output files and the log file for each run will be saved in the corresponding directory and named according to the file names provided in the `-ol`/`--output-list`, `-oa`/`--output-attributes`, and `-lf`/`--log-file` options.

* If `-ds file.txt`, `file.txt` the file is expected to contain a newline-separated list of either names of directories in the working directory or absolute/relative paths to directories. The name of the configuration file may be provided using the `-cg`/ `--config-file-genes` option. The output files and the log file for each run will be saved in the corresponding directory and named according to the file names provided in the `-ol`/`--output-list`, `-oa`/`--output-attributes`, and `-lf`/`--log-file` options. `file.txt` can, for instance, look like this:

    .. code-block::

       dir1
       dir2
       absolute/path/to/dir3
       ..relative/path/to/dir4
       ...

## Command line

```
bulkdgd_get_genes [-h] [-ol OUTPUT_LIST] [-oa OUTPUT_ATTRIBUTES] -cg CONFIG_FILE_GENES [-d WORK_DIR] [-lf LOG_FILE] [-lc] [-v] [-vv] [-p] [-n N_PROC] [-ds DIRS [DIRS ...]]
```

## Options

### Help options

| Option         | Description                     |
| -------------- | ------------------------------- |
| `-h`, `--help` | Show the help message and exit. |

### Output files

| Option                       | Description                                                  |
| ---------------------------- | ------------------------------------------------------------ |
| `-ol`, `--output-list`       | The name of the output plain text file containing the list of genes of interest, identified using their Ensembl IDs. The default file name is `genes_list.txt`. |
| `-oa`, `--output-attributes` | The name of the output CSV file containing the attributes retrieved from the Ensembl database for the genes of interest. The default file name is `genes_attributes.txt`. |

### Configuration files

| Option                       | Description                                                  |
| ---------------------------- | ------------------------------------------------------------ |
| `-cg`, `--config-file-genes` | The YAML configuration file containing the options used to query the Ensembl database for the genes of interest. If it is a name without an extension, it is assumed to be the name of a configuration file in `$INSTALLDIR/bulkdgd/configs/genes`. |

### Working directory options

| Option             | Description                                                  |
| ------------------ | ------------------------------------------------------------ |
| `-d`, `--work-dir` | The working directory. The default is the current working directory. |

### Logging options

| Option                    | Description                                                  |
| ------------------------- | ------------------------------------------------------------ |
| `-lf`, `--log-file`       | The name of the log file. The default file name is `bulkdgd_get_genes.log`. |
| `-lc`, `--log-console`    | Show log messages also on the console.                       |
| `-v`, `--logging-verbose` | Enable verbose logging (INFO level).                         |
| `-vv`, `--logging-debug`  | Enable maximally verbose logging for debugging purposes (DEBUG level). |

### Parallelization options

| Option                | Description                                                  |
| --------------------- | ------------------------------------------------------------ |
| `-p`, `--parallelize` | Whether to run the command in parallel.                      |
| `-n`, `--n-proc`      | The number of processes to start. The default number of processes started is 1. |
| `-ds`, `--dirs`       | The directories containing the input/configuration files. It can be either a list of names or paths, a pattern that the names or paths match, or a plain text file containing the names of or the paths to the directories. If names are given, the directories are assumed to be inside the working directory. If paths are given, they are assumed to be relative to the working directory. |

## Example

```
bulkdgd_get_genes -cg config_genes_list.yaml -ol genes.txt -oa genes_attributes.csv
```

This queries Ensembl/BioMart according to the filters and attributes defined in `config_genes_list.yaml` and writes the resulting list of Ensembl gene IDs to `genes.txt` and their attributes to `genes_attributes.csv`.
