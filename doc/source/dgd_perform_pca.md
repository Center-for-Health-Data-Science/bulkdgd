# `dgd_perform_pca`

This executable performs a two-dimensional principal component analysis on a set of given representations.

`dgd_perform_pca` expects as input a CSV file containing a data frame with the representations. Each row of the data frame should contain data for a different representation, and each column should contain either the values of the representations along a dimension of the latent space or additional information about the representations. The program expects the columns containing the values of the representations to be named `latent_dim_*` where `*` represents an integer from 1 to $$D$$, with $$D$$ being the dimensionality of the latent space where the representations live.

The executable produces two outputs:

* A CSV file containing a data frame with the results of the PCA for the representations.
* A file containing a scatter plot of the results of the PCA.

The executable can also take as input a configuration file specifying the plot's aesthetics and output format. If not provided, the default configuration file `bulkDGD/configs/plot/config_pca_scatter.yaml` is used to generate the plot.

## Command line

```
dgd_perform_pca [-h] -i INPUT_CSV [-oc OUTPUT_CSV_PCA] [-op OUTPUT_PLOT_PCA] [-cp CONFIG_FILE_PLOT] [-gc GROUPS_COLUMN] [-d WORK_DIR] [-lf LOG_FILE] [-lc] [-v] [-vv]
```

## Options

| Option                      | Description                                                  |
| --------------------------- | ------------------------------------------------------------ |
| `-h`, `--help`              | Show the help message and exit.                              |
| `-i`, `--input-csv`         | The input CSV file containing the data frame with the representations. |
| `-oc`, `--output-csv-pca`   | The name of the output CSV file containing the results of the PCA. The file will be written in the working directory. The default file name is `pca.csv`. |
| `-op`, `--output-plot-pca`  | The name of the output file containing the plot displaying the results of the PCA. The file will be written in the working directory. The default file name is `pca.pdf`. |
| `-cp`, `--config-file-plot` | The YAML configuration file specifying the aesthetics of the plot and the plot's output format. If not provided, the default configuration file (`$INSTALLDIR/configs/plot/config_pca_scatter.yaml`) will be used. |
| `-gc`, `--groups-column`    | The name/index of the column in the input data frame containing the groups by which the samples will be colored in the output plot. By default, the program assumes that no such column is present. |
| `-d`, `--work-dir`          | The working directory. The default is the current working directory. |
| `-lf`, `--log-file`         | The name of the log file. The file will be written in the working directory. The default file name is `dgd_perform_pca.log`. |
| `-lc`, `--log-console`      | Show log messages also on the console.                       |
| `-v`, `--logging-verbose`   | Enable verbose logging (INFO level).                         |
| `-vv`, `--logging-debug`    | Enable maximally verbose logging for debugging purposes (DEBUG level). |
