# `configs/plot`

Last updated: 01/08/2023

## `config_r_values_hist`

This is a configuration file describing the aesthetics of a histogram representing the distribution of a set of r-values corresponding to different negative binomial distributions. The file also contains the options specifying the format of the output file where the plot will be saved. A comment line above each option describes it.

The configuration in this file can be loaded with the `utils.misc.get_config_plot` function, then passed to the `utils.plotting.plot_r_values_hist` function, which generates the histogram. 

## `config_pca_scatter`

This is a configuration file describing the aesthetics of a scatter plot displaying the results of a two-dimensional principal component analysis (PCA). The file also contains the options specifying the format of the output file where the plot will be saved. A comment line above each option describes it.

The configuration in this file can be loaded with the `utils.misc.get_config_plot` function, then passed to the `utils.plotting.plot_2d_pca` function, which generates the scatter plot.

This is the default configuration file used by `dgd_perform_pca`.