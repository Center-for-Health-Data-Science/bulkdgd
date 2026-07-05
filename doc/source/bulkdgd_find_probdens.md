# `bulkdgd_find_probdens`

For a set of given representations and the Gaussian mixture model (GMM) modelling the representation space in the bulkdgd model, this command outputs the probability density of each representation for each GMM component.

This is useful, for instance, to identify a representative sample for each GMM component, namely the one having the highest probability for that component for all samples under consideration.

`bulkdgd_find_probdens` takes as input a CSV file containing the representations for a set of samples formatted as the one produced by [`bulkdgd_find_representations`](bulkdgd_find_representations.md). This CSV file stores a data frame in which each row represents a sample's representation, and each column represents a dimension of the latent space where the representations live.

## Parallelization

The command can be run in parallel over different inputs in different directories by using the `-ds`, `--dirs` option. The directories may be specified either by name (if they are in the current working directory) or their absolute or relative path.

* If `-ds dir1 path/to/dir2`, the program will be run in parallel in each directory using the input and configuration files in it. The name of the input file may be provided using the `-ir`/`--input-rep` option, while the name of the configuration file may be provided using the `-cm`/ `--config-file-model` option. The output files and the log file for each run will be saved in the corresponding directory and named according to the file names provided in the `-opr`/`--output-prob-rep`, `-opc`/`--output-prob-comp`, and `-lf`/`--log-file` options.

* If `-ds file.txt`, `file.txt` the file is expected to contain a newline-separated list of either names of directories in the working directory or absolute/relative paths to directories. The name of the input file may be provided using the `-ir`/`--input-rep` option, while the name of the configuration file may be provided using the `-cm`/ `--config-file-model` option. The output files and the log file for each run will be saved in the corresponding directory and named according to the file names provided in the `-opr`/`--output-prob-rep`, `-opc`/`--output-prob-comp`, and `-lf`/`--log-file` options. `file.txt` can, for instance, look like this:

    .. code-block::

       dir1
       dir2
       absolute/path/to/dir3
       ..relative/path/to/dir4
       ...

## Command line

```
bulkdgd_find_probdens [-h] -ir INPUT_REP [-or OUTPUT_PROB_REP] [-oc OUTPUT_PROB_COMP] -cm CONFIG_FILE_MODEL [-d WORK_DIR] [-lf LOG_FILE] [-lc] [-v] [-vv] [-p] [-n N_PROC] [-ds DIRS [DIRS ...]]
```

## Options

### Help options

| Option         | Description                     |
| -------------- | ------------------------------- |
| `-h`, `--help` | Show the help message and exit. |

### Input files

| Option               | Description                                                  |
| -------------------- | ------------------------------------------------------------ |
| `-ir`, `--input-rep` | The input CSV file containing the data frame with the representations. |

### Output files

| Option                      | Description                                                  |
| --------------------------- | ------------------------------------------------------------ |
| `-opr`, `--output-prob-rep`  | The name of the output CSV file containing, for each representation, its probability density for each of the Gaussian mixture model's components, the maximum probability density found, and the component the maximum probability density comes from. The default file name is `probability_density_representations.csv`. |
| `-opc`, `--output-prob-comp` | The name of the output CSV file containing, for each component of the Gaussian mixture model, the representation(s) having the maximum probability density with respect to it. The default file name is `probability_density_components.csv`. |

### Configuration files

| Option                       | Description                                                  |
| ---------------------------- | ------------------------------------------------------------ |
| `-cm`, `--config-file-model` | The YAML configuration file specifying the bulkdgd model's parameters and files containing the trained model. If it is a name without an extension, it is assumed to be the name of a configuration file in `$INSTALLDIR/bulkdgd/configs/model`. |

### Working directory options

| Option             | Description                                                  |
| ------------------ | ------------------------------------------------------------ |
| `-d`, `--work-dir` | The working directory. The default is the current working directory. |

### Logging options

| Option                    | Description                                                  |
| ------------------------- | ------------------------------------------------------------ |
| `-lf`, `--log-file`       | The name of the log file. The default file name is `bulkdgd_find_probdens.log`. |
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
bulkdgd_find_probdens -ir representations.csv -cm model_tgmm_trained.yaml
```

This computes, for each representation in `representations.csv` (as produced by [`bulkdgd_find_representations`](bulkdgd_find_representations.md)), the probability density with respect to each component of the Gaussian mixture model defined in `model_tgmm_trained.yaml`.
