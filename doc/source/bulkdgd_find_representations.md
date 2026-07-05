# `bulkdgd_find_representations`

This command allows you to get representations in the latent space defined by the bulkdgd model for new samples.

`bulkdgd_find_representations` takes as input a CSV file containing a data frame with the expression data (RNA-seq read counts for the different genes) for the samples of interest. The program expects the input data frame to display the samples as rows and the genes as columns. The genes' names are expected to be their Ensembl IDs. Additional columns containing data about the samples (for instance, the tissue they come from) are allowed.

It is recommended that the samples are preprocessed with [`bulkdgd_preprocess_samples`](bulkdgd_preprocess_samples.md) before running `bulkdgd_find_representations`.

`bulkdgd_find_representations` also needs a YAML configuration file specifying the bulkdgd model's parameters, an example of which can be found in `bulkdgd/configs/model` (`model_tgmm_trained.yaml` for the current Gaussian mixture model implementation, or `model_lgmm_trained.yaml` for the legacy one).

The executable also needs a configuration file defining the data loading and optimization options for finding the representations. Several examples are available in `bulkdgd/configs/representations`.

To run the executable, you also need the trained bulkdgd model, which comes in two PyTorch files:

* A file containing the trained parameters of the Gaussian mixture model describing the latent space. If the `latent_pth_file` option is set to `default` in the model's configuration file, the file `bulkdgd/data/model/gmm/gmm.pth` is loaded.

* A file containing the trained decoder for the original model with the decoder's output module configured to `nb_feature_dispersion`. Since the file is too big to be hosted on GitHub, you can find it [here](https://drive.google.com/file/d/1GKMkVmmcEH8glNrQ4092VWYQgq6maYW1/view?usp=sharing). Save it locally and specify its path in the configuration file, or move it to `bulkdgd/data/model/dec` with the name `dec.pth`. In this latter case, setting the `decoder_pth_file` option to `default` in the model's configuration file automatically loads the correct file.

`bulkdgd_find_representations` produces three to four output files:

* A CSV file containing a data frame with the best representations found. Here, the rows represent the samples for which the representations were found. In contrast, the columns represent the dimensions of the latent space where the representations live and any extra information about the representations found in the input data frame.

* A CSV file containing a data frame with the predicted means of the distributions used to model the genes' counts in the in silico samples associated with the best representations found. Here, the rows represent the samples, and the columns represent the genes (identified by their Ensembl IDs) and any extra information about the representations found in the input data frame. If negative binomial distributions were used to model the genes' counts, the predicted means of these distributions are scaled by the distributions' r-values.

* A CSV file containing a data frame with the predicted r-values of the negative binomials used to model the genes' counts in the in silico samples associated with the best representations found. Here, the rows represent the samples, and the columns represent the genes (identified by their Ensembl IDs) and any extra information about the representations found in the input data frame. This file is produced only if negative binomial distributions are used to model the genes' counts.

* A CSV file containing information about the CPU and wall clock time used by each epoch run when optimizing the representations and by each backpropagation step performed within each epoch.

## Parallelization

The command can be run in parallel over different inputs in different directories by using the `-ds`, `--dirs` option. The directories may be specified either by name (if they are in the current working directory) or their absolute or relative path.

* If `-ds dir1 path/to/dir2`, the program will be run in parallel in each directory using the input and configuration files in it. The name of the input file may be provided using the `-is`/`--input-samples` option, while the name of the configuration file may be provided using the `-cm`/ `--config-file-model` and `-cr`/`--config-file-rep` options. The output files and the log file for each run will be saved in the corresponding directory and named according to the file names provided in the `-or`/`--output-rep`, `-om`/`--output-means`, `-ov`/`--output-rvalues`, `-ot`/`--output-time`, and `-lf`/`--log-file` options.

* If `-ds file.txt`, `file.txt` the file is expected to contain a newline-separated list of either names of directories in the working directory or absolute/relative paths to directories. The name of the input file may be provided using the `-is`/`--input-samples` option, while the names of the configuration files may be provided using the `-cm`/ `--config-file-model` and `-cr`/`--config-file-rep` options. The output files and the log file for each run will be saved in the corresponding directory and named according to the file names provided in the `-or`/`--output-rep`, `-om`/`--output-means`, `-ov`/`--output-rvalues`, `-ot`/`--output-time`, and `-lf`/`--log-file` options. `file.txt` can, for instance, look like this:

    .. code-block::

       dir1
       dir2
       absolute/path/to/dir3
       ..relative/path/to/dir4
       ...

## Command line

```
bulkdgd_find_representations [-h] -is INPUT_SAMPLES [-or OUTPUT_REP] [-om OUTPUT_MEANS] [-ov OUTPUT_RVALUES] [-ot OUTPUT_TIME] -cm CONFIG_FILE_MODEL -cr CONFIG_FILE_REP [-d WORK_DIR] [-lf LOG_FILE] [-lc] [-v] [-vv] [-p] [-n N_PROC] [-ds DIRS [DIRS ...]]
```

## Options

### Help options

| Option         | Description                     |
| -------------- | ------------------------------- |
| `-h`, `--help` | Show the help message and exit. |

### Input files

| Option                   | Description                                                  |
| ------------------------ | ------------------------------------------------------------ |
| `-is`, `--input-samples` | The input CSV file containing the data frame with gene expression data for the samples for which a representation in latent space should be found. |

### Output files

| Option                    | Description                                                  |
| ------------------------- | ------------------------------------------------------------ |
| `-or`, `--output-rep`     | The name of the output CSV file containing the data frame with the representation of each input sample in latent space. The default file name is `representations.csv`. |
| `-om`, `--output-means`   | The name of the output CSV file containing the data frame with the predicted scaled means of the negative binomials for the in silico samples obtained from the best representations found. The default file name is `pred_means.csv`. |
| `-ov`, `--output-rvalues` | The name of the output CSV file containing the data frame with the r-values of the negative binomials for the in silico samples obtained from the best representations found. The default file name is `pred_r_values.csv`. The file is produced only if negative binomial distributions are used to model the genes' counts. |
| `-ot`, `--output-time`    | The name of the output CSV file containing the data frame with information about the CPU and wall clock time spent for each optimization epoch and each backpropagation step through the decoder. The default file name is `opt_time.csv`. |

### Configuration files

| Option                       | Description                                                  |
| ---------------------------- | ------------------------------------------------------------ |
| `-cm`, `--config-file-model` | The YAML configuration file specifying the bulkdgd model's parameters and files containing the trained model. If it is a name without an extension, it is assumed to be the name of a configuration file in `$INSTALLDIR/bulkdgd/configs/model`. |
| `-cr`, `--config-file-rep`   | The YAML configuration file specifying the options for the optimization step(s) when finding the best representations.  If it is a name without an extension, it is assumed to be the name of a configuration file in `$INSTALLDIR/bulkdgd/configs/representations`. |

### Working directory options

| Option             | Description                                                  |
| ------------------ | ------------------------------------------------------------ |
| `-d`, `--work-dir` | The working directory. The default is the current working directory. |

### Logging options

| Option                    | Description                                                  |
| ------------------------- | ------------------------------------------------------------ |
| `-lf`, `--log-file`       | The name of the log file. The default file name is `bulkdgd_find_representations.log`. |
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
bulkdgd_find_representations -is samples_preprocessed.csv -cm model_tgmm_trained.yaml -cr two_opt.yaml
```

This finds the best representations in latent space for the samples in `samples_preprocessed.csv` (as produced by [`bulkdgd_preprocess_samples`](bulkdgd_preprocess_samples.md)), using the trained model described in `model_tgmm_trained.yaml` and the two-round optimization scheme described in `two_opt.yaml`.