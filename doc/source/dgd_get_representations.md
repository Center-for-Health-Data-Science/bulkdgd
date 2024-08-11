# `dgd_get_representations`

This executable allows you to get representations in the latent space defined by the DGD model for new samples.

`dgd_get_representations` takes as input a CSV file containing a data frame with the expression data (RNA-seq read counts for the different genes) for the samples of interest. The program expects the input data frame to display the different samples as rows and the genes as columns. The genes' names are expected to be their Ensembl IDs. Additional columns containing data about the samples (for instance, the tissue they come from) are allowed.

It is recommended that the samples are preprocessed with [`dgd_preprocess_samples`](#dgd_preprocess_samples) before running `dgd_get_representations`.

`dgd_get_representations` also needs a YAML configuration file specifying the DGD model's parameters, an example of which can be found in `bulkDGD/configs/model` (`model.yaml`).

The executable also needs a configuration file defining the data loading and optimization options for finding the representations. Several examples are available in `bulkDGD/configs/representations`.

To run the executable, you also need the trained DGD model, which comes in two PyTorch files:

* A file containing the trained Gaussian mixture model. If the `gmm_pth_file` option is set to `default` in the model's configuration file, the file `bulkDGD/data/model/gmm/gmm.pth` is loaded.
* A file containing the trained decoder for the original model with the decoder's output module configured to `nb_feature_dispersion`. Since the file is too big to be hosted on GitHub, you can find it [here](https://drive.google.com/file/d/1GKMkVmmcEH8glNrQ4092VWYQgq6maYW1/view?usp=sharing). Save it locally and specify its path in the configuration file, or move it to `bulkDGD/data/model/dec` with the name `dec.pth`. In this latter case, setting the `dec_pth_file` option to `default` in the model's configuration file automatically loads the right file.

`dgd_get_representations` produces three to four output files:

* A CSV file containing a data frame with the best representations found. Here, the rows represent the samples for which the representations were found. In contrast, the columns represent the dimensions of the latent space where the representations live and any extra information about the representations found in the input data frame.
* A CSV file containing a data frame with the predicted means of the distributions used to model the genes' counts in the in silico samples associated with the best representations found. Here, the rows represent the samples, and the columns represent the genes (identified by their Ensembl IDs) and any extra information about the representations found in the input data frame. If negative binomial distributions were used to model the genes' counts, the predicted means of these distributions are scaled by the distributions' r-values.
* A CSV file containing a data frame with the predicted r-values of the negative binomials used to model the genes' counts in the in silico samples associated with the best representations found. Here, the rows represent the samples, and the columns represent the genes (identified by their Ensembl IDs) and any extra information about the representations found in the input data frame. This file is produced only if negative binomial distributions are used to model the genes' counts
* A CSV file containing information about the CPU and wall clock time used by each epoch ran when optimizing the representations and by each backpropagation step performed within each epoch.

## Command line

```
dgd_get_representations [-h] -i INPUT_CSV [-or OUTPUT_CSV_REP] [-om OUTPUT_CSV_MEANS] [-ov OUTPUT_CSV_RVALUES] [-ot OUTPUT_CSV_TIME] -cm CONFIG_FILE_MODEL -cr CONFIG_FILE_REP [-d WORK_DIR] [-lf LOG_FILE] [-lc] [-v] [-vv]
```

## Options

| Option                        | Description                                                  |
| ----------------------------- | ------------------------------------------------------------ |
| `-h`, `--help`                | Show the help message and exit.                              |
| `-i`, `--input-csv`           | The input CSV file containing a data frame with gene expression data for the samples for which a representation in latent space should be found. |
| `-or`, `--output-csv-rep`     | The name of the output CSV file containing the data frame with the representation of each input sample in latent space. The file will be saved in the working directory. The default file name is `representations.csv`. |
| `-om`, `--output-csv-means`   | The name of the output CSV file containing the data frame with the predicted scaled means of the negative binomials for the in silico samples obtained from the best representations found. The file will be written in the working directory. The default file name is `pred_means.csv`. |
| `-ov`, `--output-csv-rvalues` | The name of the output CSV file containing the data frame with the r-values of the negative binomials for the in silico samples obtained from the best representations found. The file will be written in the working directory. The default file name is `pred_r_values.csv`. The file is produced only if negative binomial distributions are used to model the genes' counts. |
| `-ot`, `--output-csv-time`    | The name of the output CSV file containing the data frame with information about the CPU and wall clock time spent for each optimization epoch and each backpropagation step through the decoder. The file will be written in the working directory. The default file name is `opt_time.csv`. |
| `-cm`, `--config-file-model`  | The YAML configuration file specifying the DGD model's parameters and files containing the trained model. If it is a name without an extension, it is assumed to be the name of a configuration file in `$INSTALLDIR/bulkDGD/configs/model`. |
| `-cr`, `--config-file-rep`    | The YAML configuration file containing the options for the optimization step(s) when finding the best representations.  If it is a name without an extension, it is assumed to be the name of a configuration file in `$INSTALLDIR/bulkDGD/configs/representations`. |
| `-d`, `--work-dir`            | The working directory. The default is the current working directory. |
| `-lf`, `--log-file`           | The name of the log file. The file will be written in the working directory. The default file name is `dgd_get_representations.log`. |
| `-lc`, `--log-console`        | Show log messages also on the console.                       |
| `-v`, `--logging-verbose`     | Enable verbose logging (INFO level).                         |
| `-vv`, `--logging-debug`      | Enable maximally verbose logging for debugging purposes (DEBUG level). |