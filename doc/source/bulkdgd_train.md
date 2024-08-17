# `bulkdgd train`

This command allows you to train the bulkDGD model.

`bulkdgd train` takes in input two CSV files, each containing a data frame with the expression data (RNA-seq read counts) for the training and test samples, respectively. The program expects the input data frames to display the different samples as rows and the genes as columns. The genes' names are expected to be their Ensembl IDs. Additional columns containing data about the samples (for instance, the tissue they come from) are allowed.

It is recommended that this preprocessing step be performed with [`bulkdgd preprocess samples`](#bulkdgd_preprocess_samples).

`bulkdgd train` also needs a YAML configuration file specifying the bulkDGD model's parameters. An example can be found in `bulkDGD/configs/model` (`model_untrained.yaml`).

`bulkdgd train` also needs a configuration file defining the options used for the training process (such as the optimizers for each model's component).

The executable produces four output files:

* A CSV file containing a data frame with the representations found for the training samples. Here, the rows represent the samples for which the representations were found. In contrast, the columns represent the dimensions of the latent space where the representations live and any extra information about the representations found in the input data frame.
* A CSV file containing a data frame with the representations found for the test samples. Here, the rows represent the samples for which the representations were found. In contrast, the columns represent the dimensions of the latent space where the representations live and any extra information about the representations found in the input data frame.
* A CSV file containing a data frame with the losses computed during training.
* A CSV file containing information about the CPU and wall clock time used by each epoch during training and the backpropagation steps performed within each epoch.

## Parallelization

If the command is parallelized over several input files or configuration files, each is assumed to be identically named and placed in a different directory. The paths to such directories must be specified using the `-ds`, `--dirs` option (see the full description in the [Options](#Options) section below) and must be relative to the specified working directory (`-d`, `--work-dir` option).

You can also run the command with the same input file using different configuration files and vice versa.

In these cases, the files that vary among the runs must be placed in different directories and referenced by their corresponding options by name (not path).

In contrast, the input/configuration files that stay the same among runs may be placed anywhere and referenced by their corresponding options using their absolute path or path relative to the specified working directory.

The output and log files for each run will be written in the directory where the corresponding input/configuration files were placed. If the command is not parallelized, these files will be written in the working directory.

## Command line

```
bulkdgd train [-h] -it INPUT_TRAIN -ie INPUT_TEST [-ort OUTPUT_REP_TRAIN] [-ore OUTPUT_REP_TEST [-ol OUTPUT_LOSS] [-ot OUTPUT_TIME] -cm CONFIG_FILE_MODEL -ct CONFIG_FILE_TRAIN [-dev DEVICE] [-d WORK_DIR] [-lf LOG_FILE] [-lc] [-v] [-vv] [-p] [-n N_PROC] [-ds DIRS [DIRS ...]]
```

## Options

### Help options

| Option         | Description                     |
| -------------- | ------------------------------- |
| `-h`, `--help` | Show the help message and exit. |

### Input files

| Option                 | Description                                                  |
| ---------------------- | ------------------------------------------------------------ |
| `-it`, `--input-train` | The input CSV file containing a data frame with the gene expression data for the training samples. |
| `-ie`, `--input-test`  | The input CSV file containing a data frame with the gene expression data for the test samples. |

### Output files

| Option                       | Description                                                  |
| ---------------------------- | ------------------------------------------------------------ |
| `-ort`, `--output-rep-train` | The name of the output CSV file containing the data frame with the representation of each training sample in latent space. The default file name is `representations_train.csv`. |
| `-ore`, `--output-rep-test`  | The name of the output CSV file containing the data frame with the representation of each test sample in latent space. The default file name is `representations_test.csv`. |
| `-ol`, `--output-loss`       | The name of the output CSV file containing the data frame with the per-epoch loss(es) for training and test samples. The default file name is `loss.csv`. |
| `-ot`, `--output-time`       | The name of the output CSV file containing the data frame with information about the CPU and wall clock time spent for each training epoch and the backpropagation steps through the decoder. The default file name is `train_time.csv`. |

### Configuration files

| Option                       | Description                                                  |
| ---------------------------- | ------------------------------------------------------------ |
| `-cm`, `--config-file-model` | The YAML configuration file specifying the bulkDGD model's parameters. If it is a name without an extension, it is assumed to be the name of a configuration file in `$INSTALLDIR/bulkDGD/configs/model`. |
| `-ct`, `--config-file-train` | The YAML configuration file specifying the options for training the bulkDGD model. If it is a name without an extension, it is assumed to be the name of a configuration file in `$INSTALLDIR/bulkDGD/configs/training`. |

### Run options

| Option             | Description                                                  |
| ------------------ | ------------------------------------------------------------ |
| `-dev`, `--device` | The device to use. If not provided, the GPU will be used if it is available. Available devices are: `"cpu"`, `"cuda"`. |

### Working directory options

| Option             | Description                                                  |
| ------------------ | ------------------------------------------------------------ |
| `-d`, `--work-dir` | The working directory. The default is the current working directory. |

### Logging options

| Option                    | Description                                                  |
| ------------------------- | ------------------------------------------------------------ |
| `-lf`, `--log-file`       | The name of the log file. The default file name is `bulkdgd_train.log`. |
| `-lc`, `--log-console`    | Show log messages also on the console.                       |
| `-v`, `--logging-verbose` | Enable verbose logging (INFO level).                         |
| `-vv`, `--logging-debug`  | Enable maximally verbose logging for debugging purposes (DEBUG level). |

### Parallelization options

| Option                | Description                                                  |
| --------------------- | ------------------------------------------------------------ |
| `-p`, `--parallelize` | Whether to run the command in parallel.                      |
| `-n`, `--n-proc`      | The number of processes to start. The default number of processes started is 1. |
| `-ds`, `--dirs`       | The directories containing the input/configuration files. It can be either a list of names or paths, a pattern that the names or paths match, or a plain text file containing the names of or the paths to the directories. If names are given, the directories are assumed to be inside the working directory. If paths are given, they are assumed to be relative to the working directory. |