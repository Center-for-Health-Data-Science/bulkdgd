# Command-line interface

`bulkDGD` is structured as an importable Python package.

However, a command-line interface is provided for some of the most common tasks `bulkDGD` is used for.

This interface consists of a series of executables installed together with the package.

## Executables

### `dgd_get_recount3_data`

This executable is devoted to retrieving RNA-seq data (and associated metadata) from the [Recount3 platform](https://rna.recount.bio/).

So far, the program supports the retrieval of data for samples belonging to the [GTEx](https://gtexportal.org/home/) and [TCGA](https://www.cancer.gov/ccg/research/genome-sequencing/tcga) projects.

The executable allows selecting samples for a single tissue (for GTEx data) or cancer type (for TCGA) and filtering the samples according to the associated metadata. The filtering is performed using a query string in the format supported by the [`pandas.DataFrame.query()`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.query.html) method, which can be passed from the command line as a string or as a plain text file containing the string. Metadata fields on which it is possible to filter the samples differ between GTEx and TCGA samples. A list of the available fields is available in `bulkDGD/bulkDGD/data/recount3/gtex_metadata_fields.txt` for GTEx samples and in `bulkDGD/bulkDGD/data/recount3/tcga_metadata_fields.txt` for TCGA samples.

The main output of `dgd_get_recount3_data` is a CSV file containing the RNA-seq data retrieved from Recount3 for the samples of interest. The rows represent the samples, while the columns contain the genes, which are identified by their Ensembl IDs.

The user also has the option to save the original compressed (`.gz`) files containing the RNA-seq data and the metadata associated with the samples.

#### Command line

```
dgd_get_recount3_data [-h] -ip {gtex,tcga} [-is INPUT_SAMPLES_CATEGORY] [-o OUTPUT_CSV] [-d WORK_DIR] [--save-gene-sums] [--save-metadata] [--query-string QUERY_STRING] [-v] [-vv]
```

#### Options

| Option                            | Description                                                  |
| --------------------------------- | ------------------------------------------------------------ |
| `-h`, `--help`                    | Show the help message and exit.                              |
| `-ip`, `--input-project-name`     | The name of the Recount3 project for which samples will be retrieved. Available projects are: `"gtex"`, `"tcga"`. |
| `-is`, `--input-samples-category` | The category of samples for which RNA-seq data will be retrieved. For GTEx data, this is the name of the tissue the samples belong to. For TCGA data, this is the type of cancer the samples are associated with. |
| `-o`, `--output-csv`              | The name of the output CSV file containing the data frame with the RNA-seq data for the samples. The rows of the data frame represent samples, while the columns represent genes (identified by their Ensembl IDs). The file will be saved in the working directory. The default file name is `{input_project_name}_{input_samples_category}.csv`. |
| `-d`, `--work-dir`                | The working directory. The default is the current working directory. |
| `--query-string`                  | The string that will be used to filter the samples according to their associated metadata using the `pandas.DataFrame.query()` method. The option also accepts a plain text file containing the string since it can be long for complex queries. |
| `--save-gene-sums`                | Save the original GZ file containing the RNA-seq data for the samples. The file will be saved in the working directory and named `{input_project_name}_{input_samples_category}_gene_sums.gz`. |
| `--save-metadata`                 | Save the original GZ file containing the metadata for the samples. The file will be saved in the working directory and named `{input_project_name}_{input_samples_category}_metadata.gz`. |
| `-v`, `--logging-verbose`         | Verbose logging (INFO level).                                |
| `-vv`, `--logging-debug`          | Maximally verbose logging for debugging purposes (DEBUG level). |

### `dgd_preprocess_samples`

This executable allows users to preprocess new samples to use within the DGD model.

It expects as input a CSV file containing a data frame with the gene expression data for the new samples.

Each row must represent a sample, while each column must represent a gene identified by its Ensembl ID. Therefore, each cell of the data frame represents the expression (as RNA-seq read counts) of a gene in a specific sample.

In detail, sample preprocessing consists of the following steps:

1. Excluding all data for genes that are not included in the DGD model. A plain text file containing the list of the genes used in the model is available in `bulkDGD/bulkDGD/data/model/training_genes.txt`.
2. Sorting the genes in the order expected by the DGD model.

Preprocessing new samples is a critical step before trying to find the samples' best representations in latent space using the DGD model (with [`dgd_get_representations`](###`dgd_get_representations`)).

The main output of `dgd_preprocess_samples` is a CSV file containing the preprocessed samples.

The Ensembl IDs of the genes found in the input samples but not present in the gene set used to train the DGD model, if any, are written in a separate output file in plain text format, with one gene per line. 

Furthermore, if any gene present in the gene set used to train the DGD model is not found in the genes for which counts are available in the input samples, the gene will be written to the output CSV file containing the preprocessed samples with a count of 0 for all samples. These "missing" genes are also written in a separate plain text file containing one gene per line.

#### Command line

```
dgd_preprocess_samples [-h] -i INPUT_CSV [-os OUTPUT_CSV_SAMPLES] [-oe OUTPUT_TXT_GENES_EXCLUDED] [-om OUTPUT_TXT_GENES_MISSING] [-sc SAMPLES_NAMES_COLUMN] [-d WORK_DIR] [-v] [-vv]
```

#### Options

| Option                               | Description                                                  |
| ------------------------------------ | ------------------------------------------------------------ |
| `-h`, `--help`                       | Show the help message and exit.                              |
| `-i`, `--input-csv`                  | The input CSV file containing a data frame with the samples to be preprocessed. The columns must represent the genes (identified by their Ensembl IDs), while the rows must represent the samples. |
| `-os`, `--output-csv-samples`        | The name of the output CSV file containing the data frame with the preprocessed samples. The columns will represent the genes (identified by their Ensembl IDs), while the rows will represent the samples. The file will be written in the working directory. The default file name is `samples_preprocessed.csv`. |
| `-oe`, `--output-txt-genes-excluded` | The name of the output plain text file containing the list of genes whose expression data are excluded from the data frame with the preprocessed samples. The file will be written in the working directory. The default file name is `genes_excluded.txt`. |
| `-om`, `--output-txt-genes-missing`  | The name of the output plain text file containing the genes for which no available expression data are found in the input data frame. A default count of 0 is assigned to these genes in the output data frame with the preprocessed samples. The file will be written in the working directory. The default file name is `genes_missing.txt`. |
| `-sc`, `--samples-names-column`      | The name/index of the column containing the IDs/names of the samples, if any. By default, the program assumes that no such column is present. |
| `-tc`, `--tissues-column`            | The name/index of the column containing the names of the tissues the samples belong to, if any. By default, the program assumes that no such column is present. |
| `-d`, `--work-dir`                   | The working directory. The default is the current working directory. |
| `-v`, `--logging-verbose`            | Verbose logging (INFO level).                                |
| `-vv`, `--logging-debug`             | Maximally verbose logging for debugging purposes (DEBUG level). |

### `dgd_get_representations`

This executable allows you to get representations in the latent space defined by the DGD model for new samples.

`dgd_get_representations` takes as input a CSV file containing a data frame with the expression data (RNA-seq read counts) for the samples of interest. The program expects the input data frame to display the different samples as rows and the genes as columns so that each cell contains the expression data for a gene in a sample. The genes' names are expected to be their Ensembl IDs.

The samples passed to the program must have expression data only for the genes included in the DGD model, and the genes should be sorted according to the order expected from the DGD model. A file containing the sorted list of these genes can be found in the `bulkDGD/bulkDGD/data/model/training_genes.txt` file.

It is recommended to perform this preprocessing step with [`dgd_preprocess_samples`](###`dgd_preprocess_samples`).

`dgd_get_representations` also needs a YAML configuration file specifying the DGD model parameters, an example of which can be found in `bulkDGD/bulkDGD/configs/model`.

The executable also needs a configuration file defining the data loading and optimization options for finding the representations, an example of which is available in `bulkDGD/bulkDGD/configs/representations`.

To run the executable, you also need the trained DGD model, which comes in three PyTorch files:

* A file containing the trained Gaussian mixture model. The file `bulkDGD/bulkDGD/data/model/gmm.pth` is loaded by default unless a different path is specified in the configuration file. 
* A file containing the trained representation layer. The file `bulkDGD/bulkDGD/data/model/rep.pth` is loaded by default unless a different path is specified in the configuration file.
* A file containing the trained deep generative decoder. Since the file is too big to be hosted on GitHub, you can find it [here](https://drive.google.com/file/d/1SZaoazkvqZ6DBF-adMQ3KRcy4Itxsz77/view?usp=sharing). Save it locally and specify its path in the configuration file or in `bulkDGD/bulkDGD/data/model` with the name `dec.pth`. This latter strategy would ensure the automatic loading of the file every time you launch the `dgd_get_representations` executable without having to specify its path in the configuration file. 

`dgd_get_representations` produces various output files:

* A CSV file containing a data frame with the best representations found. Here, the rows represent the samples for which the representations were found, while the columns represent the dimensions of the latent space where the representations live.
* A CSV file containing a data frame with the outputs produced by the decoder when the best representations found for the samples are passed through it. Here, the rows represent the samples, and the columns represent the genes (identified by their Ensembl IDs).
* A CSV file containing a data frame with the per-sample loss associated with the best representation found for each sample. This data frame contains only one column, storing the loss, and has as many rows as the samples.
* A CSV file containing a data frame with the labels of the tissues associated with the samples if any column containing information about the tissues was found in the input data frame. This data frame contains only one column, storing the labels, and has as many rows as the samples. This file is not produced if no tissue information is found in the input data frame.

#### Command line

```bash
dgd_get_representations [-h] [-i INPUT_CSV] [-or OUTPUT_CSV_REP] [-ot OUTPUT_CSV_TISSUES] -c CONFIG_FILE [-d WORK_DIR] [-v] [-vv]
```

#### Options

| Option                        | Description                                                  |
| ----------------------------- | ------------------------------------------------------------ |
| `-h`, `--help`                | Show the help message and exit.                              |
| `-i`, `--input-csv`           | The input CSV file containing a data frame with gene expression data for the samples for which a representation in latent space should be found. The rows must represent the samples. The first column must contain the unique names of the samples, while the other columns must represent the genes (identified by their Ensembl IDs). The genes should be the ones the DGD model has been trained on, whose Ensembl IDs can be found in `bulkDGD/bulkDGD/data/model/training_genes.txt`. |
| `-or`, `--output-csv-rep`     | The name of the output CSV file containing a data frame with the representation of each input sample in latent space. The rows represent the samples. The first column contains the unique names of the samples, while the other columns represent the values of the representations along the latent space's dimensions. The file will be saved in the working directory. The default file name is `representations.csv`. |
| `-od`, `--output-csv-dec`     | The name of the output CSV file containing the data frame with the decoder output for each input sample. The rows represent the samples. The first column contains the unique names of the samples, while the other columns represent the genes (identified by their Ensembl IDs). The file will be written in the working directory. The default tile name is `decoder_outputs.csv`. |
| `-ol`, `--output-csv-loss`    | The name of the output CSV file containing the data frame with the per-sample loss associated with the best representation found for each sample of interest. The rows represent the samples, while the only column contains the loss. The file will be written in the working directory. The default file name is `loss.csv`. |
| `-ot`, `--output-csv-tissues` | The output CSV file containing the data frame with the labels of the tissues the samples belong to. The rows represent the samples, while the only column contains the labels of the tissues. The file will be written in the working directory. The default file name is `tissues.csv`. This file will not be generated unless the input CSV file has a `tissue` column containing the labels. |
| `-cm`, `--config-file-model`  | The YAML configuration file specifying the DGD model parameters and files containing the trained model. If it is a name without extension, it is assumed to be the name of a configuration file in `bulkDGD/bulkDGD/configs/model`. |
| `-cr`, `--config-file-rep`    | The YAML configuration file containing the options for data loading and optimization when finding the best representations.  If it is a name without extension, it is assumed to be the name of a configuration file in `bulkDGD/bulkDGD/configs/representations`. |
| `-d`, `--work-dir`            | The working directory. The default is the current working directory. |
| `-v`, `--logging-verbose`     | Verbose logging (INFO level).                                |
| `-vv`, `--logging-debug`      | Maximally verbose logging for debugging purposes (DEBUG level). |

### `dgd_perform_dea`

This executable can be used to perform differential expression analysis (DEA) of genes between a "treated" sample (for instance, a cancer sample) against an "untreated" or "control" sample.

Within the context of the DGD model, the DEA is intended between a "treated" experimental sample and a "control" sample which is the model's decoder's output for the best representation of the "treated" sample in latent space. The decoder output for the best representation of the "treated" sample, therefore, acts as an *in silico* control sample.

This approach was first presented in the work of Prada-Luengo, Schuster, Liang, and coworkers (ref), and this DEA method was shown to outperform one of the current gold-standard approaches.

`dgd_perform_dea` expects three inputs. First, a CSV file containing a data frame set of experimental "treated" samples. The program assumes that each row represents a sample, each column represents a gene, and each cell of the data frame contains the read counts for a gene in a specific sample. Then, the program expects a data frame with the *in silico* "control" samples, This data frame is structured as the one containing the "treated" samples and can be obtained by running the `dgd_get_representations` executable on the "treated" sample (see the [section](###`dgd_get_representations`) dedicated to this executable for more details).

The output of `dgd_perform_dea` is a CSV file containing the results of the differential expression analysis. Here, the p-values, q-values (adjusted p-values), and log2-fold changes relative to each gene's differential expression are reported.

#### Command line

```
dgd_perform_dea [-h] -is INPUT_CSV_SAMPLES -id INPUT_CSV_DEC [-op OUTPUT_CSV_PREFIX] -c CONFIG_FILE [-pr P_VALUES_RESOLUTION] [-qa Q_VALUES_ALPHA] [-qm Q_VALUES_METHOD] [-d WORK_DIR] [-n N_PROC] [-v] [-vv]
```

#### Options

| Option                         | Description                                                  |
| ------------------------------ | ------------------------------------------------------------ |
| `-h`, `--help`                 | Show the help message and exit.                              |
| `-is`, `--input-csv-samples`   | The input CSV file containing a data frame with the gene expression data for the samples. The rows must represent the samples. The first column must contain the samples' unique names or indexes, while the other columns must represent the genes (Ensembl IDs). |
| `-id`, `--input-csv-dec`       | The input CSV file containing the data frame with the decoder output for each sample's best representation found with 'dgd_get_representations'. The rows must represent the samples. The first column must contain the samples' unique names or indexes, while the other columns must represent the genes (Ensembl IDs). |
| `-op`, `--output-csv-prefix`   | The prefix of the output CSV file(s) that will contain the results of the differential expression analysis. Since the analysis will be performed for each sample, one file per sample will be created. The files' names will have the form `{output_csv_prefix}{sample_name}.csv`. The default prefix is `dea_`. |
| `-c`, `--config-file`          | The YAMl configuration file specifying the DGD model parameters and files containing the trained model. If it is a name without extension, it is assumed to be the name of a configuration file in `$INSTALLDIR/bulkDGD/configs/model`. |
| `-pr`, `--p-values-resolution` | The resolution at which to sum over the probability mass function to compute the p-values. The lower the resolution, the more accurate the calculation. A resolution of 1 corresponds to an exact calculation of the p-values. The default is 1. |
| `-qa`, `--q-values-alpha`      | The alpha value used to calculate the q-values (adjusted p-values). The default is 0.05. |
| `-qm`, `--q-values-method`     | The method used to calculate the q-values (i.e., to adjust the p-values). The default is `"fdr_bh"`. The available methods can be found in the documentation of `statsmodels.stats.multitest.multipletests`, which is used to perform the calculation. |
| `-d`, `--work-dir`             | The working directory. The default is the current working directory. |
| `-n`, `--nproc`                | The number of processes to start (the higher the number, the faster the execution of the program). The default number of processes started is 1. |
| `-v`, `--logging-verbose`      | Verbose logging (INFO level).                                |
| `-vv`, `--logging-debug`       | Maximally verbose logging for debugging purposes (DEBUG level). |

### `dgd_get_probability_density`

For a set of given representations and the Gaussian mixture model (GMM) modeling the representation space in the DGD, this executable outputs the probability density of each representation for each GMM component.

This is useful, for instance, to identify a representative sample for each GMM component, namely the one having the highest probability for that component with respect to all samples under consideration.

`dgd_get_probability_density` takes as input a CSV file containing the representations for a set of samples formatted as the one produced by [`dgd_get_representations`](####`dgd_get_representations`). This CSV file stores a data frame where each row represents a sample's representation, and each column represents a dimension of the latent space where the representations live.

#### Command line

```
dgd_get_probability_density [-h] -ir INPUT_CSV_REP [-it INPUT_CSV_TISSUES] [-or OUTPUT_CSV_PROB_REP] [-oc OUTPUT_CSV_PROB_COMP] -c CONFIG_FILE [-d WORK_DIR] [-v] [-vv]
```

#### Options

| Option                          | Description                                                  |
| ------------------------------- | ------------------------------------------------------------ |
| `-h`, `--help`                  | Show the help message and exit.                              |
| `-ir`, `--input-csv-rep`        | The input CSV file containing the data frame with the samples' representations in latent space. The columns represent the values of the representations along the latent space's dimensions, while the rows represent the samples. |
| `-it`, `--input-csv-tissue`     | The input CSV file containing one column with the labels of the tissues the input samples belong to. This input file is optional. |
| `-or`, `--output-csv-prob-rep`  | The name of the output CSV file containing, for each representation, its probability density for each of the Gaussian mixture model's components, the maximum probability density found, the component the maximum probability density comes from, and the label of the tissue the input sample belongs to. The file will be written in the working directory. The default file name is `probability_density_representations.csv`. |
| `-oc`, `--output-csv-prob-comp` | The name of the output CSV file containing, for each component of the Gaussian mixture model, the representation(s) having the maximum probability density with respect to it. The file will be written in the working directory. The default file name is `probability_density_components.csv`. |
| `-cm`, `--config-file-model`    | The YAML configuration file specifying the DGD model parameters and files containing the trained model. If it is a name without extension, it is assumed to be the name of a configuration file in `bulkDGD/bulkDGD/configs/model`. |
| `-d`, `--work-dir`              | The working directory. The default is the current working directory. |
| `-v`, `--logging-verbose`       | Verbose logging (INFO level).                                |
| `-vv`, `--logging-debug`        | Maximally verbose logging for debugging purposes (DEBUG level).ese |
