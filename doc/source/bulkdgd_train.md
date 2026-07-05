# `bulkdgd_train`

This command allows you to train the bulkdgd model.

`bulkdgd_train` takes as input:

* A CSV files containing the expression data (RNA-seq read counts) for the training and test samples. The program expects the input data frame to display samples as rows and genes as columns. The samples' names/indexes/IDs should be in the first column of the data frame. The genes' names are expected to be their Ensembl IDs. Additional columns containing sample metadata (for instance, tissue of origin) are allowed.

* A plain text file with a newline-separated list of the names/indexes/IDs of the training samples.

* A plain text file with a newline-separated list of the names/indexes/IDs of the test samples.

It is recommended to pre-process the samples with [`bulkdgd_preprocess_samples`](bulkdgd_preprocess_samples.md) before using them.

`bulkdgd_train` also needs two configuration files:

* A YAML configuration file specifying the bulkdgd model's options. An example of this configuration file can be found in `bulkdgd/configs/model`.

* A YAML configuration file defining the options used for the training process. An example of this configuration file can be found in `bulkdgd/configs/training`.

The executable produces six to eight output files. These files are always produced:

* A CSV file containing a data frame with the representations found for the training samples.

* A CSV file containing a data frame with the representations found for the test samples.

* A CSV file containing a data frame with the means of the distributions modeling the genes for the training samples.

* A CSV file containing a data frame with the means of the distributions modeling the genes for the test samples.

* A CSV file containing a data frame with the losses computed during training.

* A CSV file containing information about the CPU and wall clock time used by each epoch during training.

If the genes' are modeled by negative binomial distributions (instead of Poisson distributions) and the r-values of the distributions are learned per gene, one additional file is produced:

* A CSV file containing a data frame with the r-values of the negative binomial distributions modeling the genes.

If the genes' are modeled by negative binomial distributions (instead of Poisson distributions) and the r-values of the distributions are learned per gene and per sample, two additional files are produced:

* A CSV file containing a data frame with the r-values of the negative binomial distributions modeling the genes for the training samples.

* A CSV file containing a data frame with the r-values of the negative binomial distributions modeling the genes for the test samples.

## Parallelization

The command can be run in parallel over different inputs in different directories by using the `-ds`, `--dirs` option. The directories may be specified either by name (if they are in the current working directory) or their absolute or relative path. 

* If `-ds dir1 path/to/dir2`, the program will be run in parallel in each directory using the input and configuration files in it. The names of the input files may be provided using the `-is`/`--input-samples`, `-it`/`--input-train`, and `-ie`/`--input-test` options, and the names of the configuration files may be provided using the `-icm`/`--input-config-file-model` and `-ict`/`--input-config-file-train` options. The output files and the log file for each run will be saved in the corresponding directory and named according to the file names provided in the `-ort`/`--output-rep-train`, `-ore`/`--output-rep-test`, `-opmt`/`--output-pred-means-train`, `-opme`/`--output-pred-means-test`, `-opv`/`--output-pred-rvalues`, `-opvt`/`--output-pred-rvalues-train`, `-opve`/`--output-pred-rvalues-test`, `-ol`/`--output-loss`, `-ot`/`--output-time`, and `-lf`/`--log-file` options.

* If `-ds file.txt`, `file.txt` the file is expected to contain a newline-separated list of either names of directories in the working directory or absolute/relative paths to directories. The same input, configuration, and output options as above apply. `file.txt` can, for instance, look like this:

    .. code-block::

       dir1
       dir2
       absolute/path/to/dir3
       ..relative/path/to/dir4
       ...
       ...

## Command line

```
bulkdgd_train [-h] -is INPUT_SAMPLES -it INPUT_TRAIN -ie INPUT_TEST [-ilt INPUT_LABELS_TRAIN] [-ile INPUT_LABELS_TEST] -icm INPUT_CONFIG_FILE_MODEL -ict INPUT_CONFIG_FILE_TRAIN [-olat OUTPUT_LATENT] [-odec OUTPUT_DECODER] [-ort OUTPUT_REP_TRAIN] [-ore OUTPUT_REP_TEST] [-opmt OUTPUT_PRED_MEANS_TRAIN] [-opme OUTPUT_PRED_MEANS_TEST] [-opv OUTPUT_PRED_RVALUES] [-opvt OUTPUT_PRED_RVALUES_TRAIN] [-opve OUTPUT_PRED_RVALUES_TEST] [-ol OUTPUT_LOSS] [-omrt OUTPUT_METRICS_TRAIN] [-omre OUTPUT_METRICS_TEST] [-ot OUTPUT_TIME] [-dev DEVICE] [-d WORK_DIR] [-lf LOG_FILE] [-lc] [-v] [-vv] [-p] [-n N_PROC] [-ds DIRS [DIRS ...]]
```

## Options

### Help options

| Option         | Description                     |
| -------------- | ------------------------------- |
| `-h`, `--help` | Show the help message and exit. |

### Input files

| Option                 | Description                                                  |
| ---------------------- | ------------------------------------------------------------ |
| `-is`, `--input-samples` | The input CSV file containing a data frame with the gene expression data for the samples. |
| `-it`, `--input-train` | The input plain text file containing a newline-separated list of sample names/indexes/IDs for the training samples. |
| `-ie`, `--input-test`  | The input plain text file containing a newline-separated list of sample names/indexes/IDs for the test samples. |
| `-ilt`, `--input-labels-train` | The input CSV file containing the labels for the training samples. The file should have two columns, one with the names/indexes/IDs of the samples and one with the corresponding labels. The file should not have a header. This argument is needed only if supervised clustering metrics are computed during training. |
| `-ile`, `--input-labels-test` | The input CSV file containing the labels for the test samples. The file should have two columns, one with the names/indexes/IDs of the samples and one with the corresponding labels. The file should not have a header. This argument is needed only if supervised clustering metrics are computed during training. |
| `-icm`, `--input-config-file-model` | The YAML configuration file specifying the model's parameters. If it is a name without an extension, it is assumed to be the name of a configuration file in `$INSTALLDIR/bulkdgd/configs/model`. |
| `-ict`, `--input-config-file-train` | The YAML configuration file specifying the options for training the model. If it is a name without an extension, it is assumed to be the name of a configuration file in `$INSTALLDIR/bulkdgd/configs/training`. |

### Output files

| Option                       | Description                                                  |
| ---------------------------- | ------------------------------------------------------------ |
| `-olat`, `--output-latent` | The output .pth file where the latent space's parameters will be saved after training. By default, the file will be named `latent.pth`. |
| `-odec`, `--output-decoder` | The output .pth file where the decoder's parameters will be saved after training. By default, the file will be named `decoder.pth`. |
| `-ort`, `--output-rep-train` | The output CSV file containing the data frame with the representation of each training sample in latent space. The default file name is `representations_train.csv`. |
| `-ore`, `--output-rep-test`  | The output CSV file containing the data frame with the representation of each test sample in latent space. The default file name is `representations_test.csv`. |
| `-omt`, `--output-pred-means-train` | The output CSV file containing the data frame with the predicted mean of each gene in each training sample. The default file name is `pred_means_train.csv`. |
| `-ome`, `--output-pred-means-test` | The output CSV file containing the data frame with the predicted mean of each gene in each test sample. The default file name is `pred_means_test.csv`. |
| `-opv`, `--output-pred-rvalues` | The output CSV file containing the data frame with the r-value for each gene. The default file name is `pred_r_values.csv`. If the model's output module returns r-values per gene and per sample, the `-opvt`, `--output-pred-rvalues-train` and `-opve`, `--output-pred-rvalues-test` options should be used instead. |
| `-opvt`, `--output-pred-rvalues-train` | The output CSV file containing the data frame with the r-value for each gene in each training sample. The default file name is `pred_r_values_train.csv`. This option will be used only if the model's output module returns r-values per gene and per sample. |
| `-opve`, `--output-pred-rvalues-test` | The output CSV file containing the data frame with the r-value for each gene in each test sample. The default file name is `pred_r_values_test.csv`. This option will be used only if the model's output module returns r-values per gene and per sample. |
| `-ol`, `--output-loss` | The output CSV file containing the data frame with the per-epoch losses for training and test samples. The default file name is `loss.csv`. |
| `-omrt`, `--output-metrics-train` | The output CSV file containing the data frame with the per-epoch values of the latent space clustering metrics computed during training for the training samples. The default file name is `metrics_train.csv`. This file will be written only if at least one clustering metric is computed during training (they are specified in the `-ict`, `--input-config-file-train` configuration file).|
| `-omre`, `--output-metrics-test` | The output CSV file containing the data frame with the per-epoch values of the latent space clustering metrics computed during training for the test samples. The default file name is `metrics_test.csv`. This file will be written only if at least one clustering metric is computed during training (they are specified in the `-ict`, `--input-config-file-train` configuration file).|
| `-ot`, `--output-time` | The output CSV file containing the data frame with information about the CPU and wall clock time spent for each training epoch. The default file name is `train_time.csv`. |

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

## Example

```
bulkdgd_train -is samples_preprocessed.csv -it samples_train.txt -ie samples_test.txt -icm model.yaml -ict training.yaml
```

This trains a new bulkdgd model on the samples listed in `samples_train.txt`/`samples_test.txt` (a subset of the samples found in `samples_preprocessed.csv`), using the model architecture described in `model.yaml` and the training options described in `training.yaml`.