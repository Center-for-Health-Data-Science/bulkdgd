# `dgd_get_genes_list`

This executable allows you to create customized lists of genes to use with the DGD model.

`dgd_get_genes_list` takes in input a YAML configuration file containing:

* The `attributes` to retrieve for the genes of interest from the Ensembl database.
* The `filters` to use on the genes retrieved from the Ensembl database (to keep, for instance, only protein-coding genes or genes producing only protein-coding transcripts).

The executable produces two output files:

* A CSV file containing the `attributes` for the genes passing all the `filters`.
* A plain text file containing the Ensembl IDs of the genes reported in the CSV file. This file can be directly used when setting up a new instance of the DGD model or preprocessing a set of samples to be used with the model.

## Command line

```
dgd_get_genes_list [-h] [-ol OUTPUT_TXT_LIST] [-oa OUTPUT_CSV_ATTRIBUTES] -cg CONFIG_FILE_GENES [-d WORK_DIR] [-lf LOG_FILE] [-lc] [-v] [-vv]
```

## Options

| Option                           | Description                                                  |
| -------------------------------- | ------------------------------------------------------------ |
| `-h`, `--help`                   | Show the help message and exit.                              |
| `-ol`, `--output-txt-list`       | The name of the output plain text file containing the list of genes of interest, identified using their Ensembl IDs. The file will be written in the working directory. The default file name is `genes_list.txt`. |
| `-oa`, `--output-csv-attributes` | The name of the output CSV file containing the attributes retrieved from the Ensembl database for the genes of interest. The file will be written in the working directory. The default file name is `genes_attributes.txt`. |
| `-cg`, `--config-file-genes`     | The YAML configuration file containing the options used to query the Ensembl database for the genes of interest. If it is a name without an extension, it is assumed to be the name of a configuration file in `$INSTALLDIR/bulkDGD/configs/genes`. |
| `-d`, `--work-dir`               | The working directory. The default is the current working directory. |
| `-lf`, `--log-file`              | The name of the log file. The file will be written in the working directory. The default file name is `dgd_get_genes_list.log`. |
| `-lc`, `--log-console`           | Show log messages also on the console.                       |
| `-v`, `--logging-verbose`        | Enable verbose logging (INFO level).                         |
| `-vv`, `--logging-debug`         | Enable maximally verbose logging for debugging purposes (DEBUG level). |

