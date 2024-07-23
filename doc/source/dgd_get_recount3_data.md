# `dgd_get_recount3_data`

This executable retrieves RNA-seq data (and associated metadata) from the [Recount3 platform](https://rna.recount.bio/).

So far, the program supports retrieving data for samples from the [GTEx](https://gtexportal.org/home/), [TCGA](https://www.cancer.gov/ccg/research/genome-sequencing/tcga), and [SRA](https://www.ncbi.nlm.nih.gov/sra) projects.

The executable allows samples to be selected for a single tissue (for GTEx data), cancer type (for TCGA), or project code (for SRA) and to filter them according to the associated metadata. The filtering is performed using a query string in the format supported by the [`pandas.DataFrame.query()`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.query.html) method.

A list of the available metadata fields/columns is available in `bulkDGD/recount3/data/gtex_metadata_fields.txt` for GTEx samples, `bulkDGD/recount3/data/tcga_metadata_fields.txt` for TCGA samples, and `bulkDGD/recount3/data/sra_metadata_fields.txt` for SRA samples. More metadata fields may be available for SRA samples depending on the study they refer to. In this case, you can inspect the study's available metadata fields using the [NCBI SRA Run Selector tool](https://www.ncbi.nlm.nih.gov/Traces/study/). However, remember that the SRA Run Selector does not report Recount3-specific metadata, which can be found in the `sra_metadata_fields.txt` file.

`dgd_get_recount3_data` accepts two types of inputs containing the batches of samples to be downloaded from Recount3:

* A CSV file with a comma-separated data frame. The data frame is expected to have at least two columns:

  * `"recount3_project_name"`, containing the name of the project (`"gtex"`, `"tcga"`, or `"sra"`) the samples belong to.
  * `"recount3_samples_category"`, containing the name of the category the samples belong to (it is a tissue type for GTEx data, a cancer type for TCGA data, and a project code for SRA data).

  These additional three columns may also be present:

  * `"query_string"`, containing the query string that should be used to filter each batch of samples by their metadata. The query string is passed to the [`pandas.DataFrame.query()`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.query.html) method. If no ``"query_string"` column is present, the samples will not be filtered.
  * ``metadata_to_keep``, containing a vertical line (`|`)-separated list of names of metadata columns that will be kept in the final data frames, together with the columns containing gene expression data. `"recount3_project_name"` and `"recount3_samples_category"` are valid column names, and, if passed, the final data frames will also include them (each data frame will, of course, contain only one repeated value for each of these columns, since it contains samples from a single category of a single project). By default, all metadata columns (plus the `"recount3_project_name"` and `"recount3_samples_category"` columns) are kept in the final data frames).
  * ``metadata_to_drop``, containing a vertical line (`|`)-separated list of names of metadata columns that will be dropped from the final data frames. The reserved keyword `_all_` can be used to drop all metadata columns from the final data frame of a specific batch of samples. `"recount3_project_name"` and `"recount3_samples_category"` are valid column names, and, if passed, will result in these columns being dropped.

* A YAML file with the format exemplified below. We recommend using a YAML file over a CSV file when you have several studies for which different filtering conditions should be applied.

  ```yaml
  # SRA studies - it can be omitted if no SRA studies are
  # included.
  sra:
  
    # Conditions applied to all SRA studies.
    all:
  
      # Which metadata to keep in all studies (if found). It is
      # a vertical line (|)-separated list of names of metadata
      # columns that will be kept in the  final data frames,
      # together with the columns containing gene expression
      # data.
      #
      # "recount3_project_name"`` and "recount3_samples_category"
      # are valid column names, and, if passed, the final data
      # frames will also include them (each data frame will, of
      # course, contain only one repeated value for each of these
      # columns, since it contains samples from a single category
      # of a single project).
      #
      # By default, all metadata columns (plus the
      # "recount3_project_name" and `"recount3_samples_category"
      # columns) are kept in the final data frames.
      metadata_to_keep:
  
        # Keep in all studies.
        - source_name
        ...
  
      # Which metadata to drop from all studies (if found). It is
      # a vertical line (|)-separated list of names of metadata
      # columns that will be dropped from the final data frames.
      #
      # The reserved keyword '_all_' can be used to drop all
      # columns from the data frames.
      #
      # "recount3_project_name" and "recount3_samples_category"
      # are valid column names and, if passed, will result in
      # these columns being dropped.
      metadata_to_drop:
  
        # Found in all studies.
        - age
        ...
  
    # Conditions applied to SRA study SRP179061.
    SRP179061:
  
      # The query string that should be used to filter each batch
      # of samples by their metadata. The query string is passed
      # to the 'pandas.DataFrame.query()' method for filtering.
  
      # If no query string  is present, the samples will not
      # be filtered.
      query_string: diagnosis == 'Control'
  
      # Which metadata to keep in this study (if found), It
      # follows the same rules as the 'metadata_to_keep' field
      # in the 'all' section.
      metadata_to_keep:
      - tissue
  
      # Which metadata to drop from this study (if found), It
      # follows the same rules as the 'metadata_to_drop' field
      # in the 'all' section.
      metadata_to_drop:
      - Sex
  
  # GTEx studies - it can be omitted if no GTEx studies are
  # included.
  gtex:
  
    # Same format as for SRA studies - single studies are
    # identified by the tissue type each study refers to.
    ...
  
  # TCGA studies - it can be omitted if no TCGA studies are
  # included.
  tcga:
  
    # Same format as for SRA studies - single studies are
    # identified by the cancer type each study refers to.
    ...
  ```

The main output of `dgd_get_recount3_data` is several CSV files (one per batch of samples) containing the RNA-seq data retrieved from Recount3 for the samples of interest. The rows represent the samples, while the columns contain the genes identified by their Ensembl IDs or the samples' metadata. This file is usually named `{recount3_project_name}_{recount3_samples_category}.csv`. If several different batches of samples are downloaded for the same project and samples' category (for instance, for the same GTEx tissue but using a different query string for filtering the samples), the output file for the first batch will be named `{recount3_project_name}_{recount3_samples_category}.csv`, the output file for the second one will be named `{recount3_project_name}_{recount3_samples_category}_1.csv`, and so forth.

The user also has the option to save the original compressed (`.gz`) files containing the RNA-seq data and the metadata associated with the samples. If these files are found in the working directory for a specific project and sample category, they will not be downloaded again.

To speed up data retrieval and processing,  `dgd_get_recount3_data` uses the [Dask](https://www.dask.org/) Python package to parallelize the calculations.

## Command line

```
dgd_get_recount3_data [-h] [-is INPUT_SAMPLES_BATCHES] [-d WORK_DIR] [-n N_PROC] [-sg] [-sm] [-lf LOG_FILE] [-lc] [-v] [-vv]
```

## Options

| Option                          | Description                                                  |
| ------------------------------- | ------------------------------------------------------------ |
| `-h`, `--help`                  | Show the help message and exit.                              |
| `-i`, `--input-samples-batches` | A CSV file or a YAML file used to download samples' data in bulk. |
| `-d`, `--work-dir`              | The working directory. The default is the current working directory. |
| `-n`, `--n-proc`                | The number of processes to start. The default number of processes started is 1. |
| `-sg`, `--save-gene-sums`       | Save the original GZ file containing the RNA-seq data for the samples. For each batch of samples, the corresponding file will be saved in the working directory and named`{recount3_project_name}_{recount3_samples_category}_gene_sums.gz`. This file will be written only once if more than one batch refers to the same `recount3_project_name` and `recount3_samples_category`. |
| `-sm`, `--save-metadata`        | Save the original GZ file containing the metadata for the samples. For each batch of samples, the corresponding file will be saved in the working directory and named `{recount3_project_name}_{recount3_samples_category}_metadata.gz`. This file will be written only once if more than one batch refers to the same `recount3_project_name` and `recount3_samples_category`. |
| `-lf`, `--log-file`             | The name of the log file. The file will be written in the working directory. The default file name is `dgd_get_recount3_data.log`. |
| `-lc`, `--log-console`          | Show log messages also on the console.                       |
| `-v`, `--logging-verbose`       | Enable verbose logging (INFO level).                         |
| `-vv`, `--logging-debug`        | Enable maximally verbose logging for debugging purposes (DEBUG level). |