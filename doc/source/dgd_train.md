# `dgd_train`

This executable allows you to train the DGD model.

`dgd_train` takes in input two CSV files, each containing a data frame with the expression data (RNA-seq read counts) for the training and test samples, respectively. The program expects the input data frames to display the different samples as rows and the genes as columns. The genes' names are expected to be their Ensembl IDs. Additional columns containing data about the samples (for instance, the tissue they come from) are allowed.

It is recommended that this preprocessing step be performed with [`dgd_preprocess_samples`](#dgd_preprocess_samples).

`dgd_train` also needs a YAML configuration file specifying the DGD model's parameters. An example can be found in `bulkDGD/configs/model` (`model_untrained.yaml`).

`dgd_train` also needs a configuration file defining the options used for the training process (such as the optimizers for each model's component).

The executable produces four output files:

* A CSV file containing a data frame with the representations found for the training samples. Here, the rows represent the samples for which the representations were found. In contrast, the columns represent the dimensions of the latent space where the representations live and any extra information about the representations found in the input data frame.
* A CSV file containing a data frame with the representations found for the test samples. Here, the rows represent the samples for which the representations were found. In contrast, the columns represent the dimensions of the latent space where the representations live and any extra information about the representations found in the input data frame.
* A CSV file containing a data frame with the losses computed during training.
* A CSV file containing information about the CPU and wall clock time used by each epoch during training and the backpropagation steps performed within each epoch.

## Command line

```
dgd_train [-h] -it INPUT_CSV_TRAIN -ie INPUT_CSV_TEST [-ort OUTPUT_CSV_REP_TRAIN] [-ore OUTPUT_CSV_REP_TEST [-ol OUTPUT_CSV_LOSS] [-ot OUTPUT_CSV_TIME] -cm CONFIG_FILE_MODEL -ct CONFIG_FILE_TRAIN [-d WORK_DIR] [-dev DEVICE] [-lf LOG_FILE] [-lc] [-v] [-vv]
```

## Options

| Option                           | Description                                                  |
| -------------------------------- | ------------------------------------------------------------ |
| `-h`, `--help`                   | Show the help message and exit.                              |
| `-it`, `--input-csv-train`       | The input CSV file containing a data frame with the gene expression data for the training samples. |
| `-ie`, `--input-csv-test`        | The input CSV file containing a data frame with the gene expression data for the test samples. |
| `-ort`, `--output-csv-rep-train` | The name of the output CSV file containing the data frame with the representation of each training sample in latent space. The file will be written in the working directory. The default file name is `representations_train.csv`. |
| `-ore`, `--output-csv-rep-test`  | The name of the output CSV file containing the data frame with the representation of each test sample in latent space. The file will be written in the working directory. The default file name is `representations_test.csv`. |
| `-ol`, `--output-csv-loss`       | The name of the output CSV file containing the data frame with the per-epoch loss(es) for training and test samples. The file will be written in the working directory. The default file name is `loss.csv`. |
| `-ot`, `--output-csv-time`       | The name of the output CSV file containing the data frame with information about the CPU and wall clock time spent for each training epoch and the backpropagation steps through the decoder. The file will be written in the working directory. The default file name is `train_time.csv`. |
| `-cm`, `--config-file-model`     | The YAML configuration file specifying the DGD model's parameters. If it is a name without an extension, it is assumed to be the name of a configuration file in `$INSTALLDIR/bulkDGD/configs/model`. |
| `-ct`, `--config-file-train`     | The YAML configuration file containing the options for training the DGD model. If it is a name without an extension, it is assumed to be the name of a configuration file in `$INSTALLDIR/bulkDGD/configs/training`. |
| `-d`, `--work-dir`               | The working directory. The default is the current working directory. |
| `-dev`, `--device`               | The device to use. If not provided, the GPU will be used if it is available. Available devices are: `"cpu"`, `"cuda"`. |
| `-lf`, `--log-file`              | The name of the log file. The file will be written in the working directory. The default file name is `dgd_train.log`. |
| `-lc`, `--log-console`           | Show log messages also on the console.                       |
| `-v`, `--logging-verbose`        | Enable verbose logging (INFO level).                         |
| `-vv`, `--logging-debug`         | Enable maximally verbose logging for debugging purposes (DEBUG level). |