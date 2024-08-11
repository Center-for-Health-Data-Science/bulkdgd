# `dgd_get_probability_density`

For a set of given representations and the Gaussian mixture model (GMM) modeling the representation space in the DGD, this executable outputs the probability density of each representation for each GMM component.

This is useful, for instance, to identify a representative sample for each GMM component, namely the one having the highest probability for that component with respect to all samples under consideration.

`dgd_get_probability_density` takes as input a CSV file containing the representations for a set of samples formatted as the one produced by [`dgd_get_representations`](#dgd_get_representations). This CSV file stores a data frame in which each row represents a sample's representation, and each column represents a dimension of the latent space where the representations live.

## Command line

```
dgd_get_probability_density [-h] -i INPUT_CSV [-or OUTPUT_CSV_PROB_REP] [-oc OUTPUT_CSV_PROB_COMP] -cm CONFIG_FILE_MODEL [-d WORK_DIR] [-lf LOG_FILE] [-lc] [-v] [-vv]
```

## Options

| Option                          | Description                                                  |
| ------------------------------- | ------------------------------------------------------------ |
| `-h`, `--help`                  | Show the help message and exit.                              |
| `-i`, `--input-csv`             | The input CSV file containing the data frame with the representations. |
| `-or`, `--output-csv-prob-rep`  | The name of the output CSV file containing, for each representation, its probability density for each of the Gaussian mixture model's components, the maximum probability density found, and the component the maximum probability density comes from. The file will be written in the working directory. The default file name is `probability_density_representations.csv`. |
| `-oc`, `--output-csv-prob-comp` | The name of the output CSV file containing, for each component of the Gaussian mixture model, the representation(s) having the maximum probability density with respect to it. The file will be written in the working directory. The default file name is `probability_density_components.csv`. |
| `-cm`, `--config-file-model`    | The YAML configuration file specifying the DGD model's parameters and files containing the trained model. If it is a name without an extension, it is assumed to be the name of a configuration file in `$INSTALLDIR/bulkDGD/configs/model`. |
| `-d`, `--work-dir`              | The working directory. The default is the current working directory. |
| `-lf`, `--log-file`             | The name of the log file. The file will be written in the working directory. The default file name is `dgd_get_probability_density.log`. |
| `-lc`, `--log-console`          | Show log messages also on the console.                       |
| `-v`, `--logging-verbose`       | Enable verbose logging (INFO level).                         |
| `-vv`, `--logging-debug`        | Enable maximally verbose logging for debugging purposes (DEBUG level). |
