# `configs/plot`

Last updated: 12/05/2023

## `r_values_hist`

This is an example of a configuration file describing the aesthetics of a histogram representing the distribution of a set of r-values corresponding to different negative binomial distributions. The file also contains the options specifying the output file format where the plot will be saved. A comment line above each option describes it.

The configuration in this file can be loaded with the `bulkDGD.ioutil.get_config_plot` function and then passed to the `bulkDGD.plotting.plot_r_values_hist` function, which generates the histogram. 

## `pca_scatter`

This is an example of a configuration file describing the aesthetics of a scatter plot displaying the results of a two-dimensional principal component analysis (PCA). The file also contains the options specifying the output file format where the plot will be saved. A comment line above each option describes it.

The configuration in this file can be loaded with the `bulkDGD.ioutil.get_config_plot` function and then passed to the `bulkDGD.plotting.plot_2d_pca` function, which generates the scatter plot.

This is the default configuration file used by `dgd_perform_pca`.

## `time_line.yaml`

This is an example of a configuration file describing the aesthetics of a line plot displaying the CPU/wall clock time spent in each epoch of each round of optimization performed when finding the best representations for a set of samples. The file also contains the options specifying the output file format where the plot will be saved. A comment line above each option describes it.

The configuration in this file can be loaded with the `bulkDGD.ioutil.get_config_plot` function and then passed to the `bulkDGD.plotting.plot_get_representations_time` function, which generates the line plot.