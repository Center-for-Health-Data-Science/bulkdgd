# `dgd_get_recount3_data`

This executable is devoted to retrieving RNA-seq data (and associated metadata) from the [Recount3 platform](https://rna.recount.bio/).

So far, the program supports the retrieval of data for samples belonging to the [GTEx](https://gtexportal.org/home/) and [TCGA](https://www.cancer.gov/ccg/research/genome-sequencing/tcga) projects.

The executable allows selecting samples for a single tissue (for GTEx data) or cancer type (for TCGA) and filtering the samples according to the associated metadata. The filtering is performed using a query string in the format supported by the [`pandas.DataFrame.query()`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.query.html) method, which can be passed from the command line as a string or as a plain text file containing the string. Metadata fields on which it is possible to filter the samples differ between GTEx and TCGA samples. A list of the available fields is available in `bulkDGD/data/recount3/gtex_metadata_fields.txt` for GTEx samples and in `bulkDGD/data/recount3/tcga_metadata_fields.txt` for TCGA samples.

The main output of `dgd_get_recount3_data` is a CSV file containing the RNA-seq data retrieved from Recount3 for the samples of interest. The rows represent the samples, while the columns contain the genes, which are identified by their Ensembl IDs.

The user also has the option to save the original compressed (`.gz`) files containing the RNA-seq data and the metadata associated with the samples.

## Command line

```
dgd_get_recount3_data [-h] -ip {gtex,tcga} -is INPUT_SAMPLES_CATEGORY [-o OUTPUT_CSV] [-d WORK_DIR] [-sg] [-sm] [-qs QUERY_STRING] [-lf LOG_FILE] [-lc] [-v] [-vv]
```

## Options

| Option                            | Description                                                  |
| --------------------------------- | ------------------------------------------------------------ |
| `-h`, `--help`                    | Show the help message and exit.                              |
| `-ip`, `--input-project-name`     | The name of the Recount3 project for which samples will be retrieved. The available projects are: `"gtex"`, `"tcga"`. |
| `-is`, `--input-samples-category` | The category of samples for which RNA-seq data will be retrieved. For GTEx data, this is the name of the tissue the samples belong to. For TCGA data, this is the type of cancer the samples are associated with. |
| `-o`, `--output-csv`              | The name of the output CSV file containing the data frame with the RNA-seq data for the samples. The file will be written in the working directory. The default file name is `{input_project_name}_{input_samples_category}.csv`. |
| `-d`, `--work-dir`                | The working directory. The default is the current working directory. |
| `-sg`, `--save-gene-sums`         | Save the original GZ file containing the RNA-seq data for the samples. The file will be saved in the working directory and named `{input_project_name}_{input_samples_category}_gene_sums.gz`. |
| `-sm`, `--save-metadata`          | Save the original GZ file containing the metadata for the samples. The file will be saved in the working directory and named `{input_project_name}_{input_samples_category}_metadata.gz`. |
| `-qs`, `--query-string`           | The string that will be used to filter the samples according to their associated metadata using the `pandas.DataFrame.query()` method. The option also accepts a plain text file containing the string since it can be long for complex queries. |
| `-lf`, `--log-file`               | The name of the log file. The file will be written in the working directory. The default file name is `dgd_get_recount3_data.log`. |
| `-lc`, `--log-console`            | Show log messages also on the console.                       |
| `-v`, `--logging-verbose`         | Enable verbose logging (INFO level).                         |
| `-vv`, `--logging-debug`          | Enable maximally verbose logging for debugging purposes (DEBUG level). |