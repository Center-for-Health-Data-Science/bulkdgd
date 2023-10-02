# `dgd_preprocess_samples`

This executable allows users to preprocess new samples to use with the DGD model.

It expects as input a CSV file containing a data frame with the gene expression data for the new samples.

Each row must represent a sample, while each column must represent a gene identified by its Ensembl ID or additional information about the samples. The first column is expected to contain the samples' unique names, IDs, or indexes.

During preprocessing, Ensembl IDs indicating [pseudoautosomal regions](http://www.ensembl.org/info/genome/genebuild/human_PARS.html) are treated as different genes.

In detail, sample preprocessing consists of the following steps:

1. Removing duplicated samples.
2. Removing samples containing missing values for the expression of some genes.
3. Excluding all data for genes that are not included in the DGD model. A plain text file containing the list of the genes used in the model is available in `bulkDGD/ioutil/data/genes.txt`.
4. Adding a count of 0 for all genes not found in the input samples but part of the set of genes used to train the DGD model.
5. Sorting the genes in the order expected by the DGD model.

The program will exit with an error if it finds duplicated genes.

Preprocessing new samples is a critical step before finding the samples' best representations in latent space using the DGD model (using, for instance, the [`dgd_get_representations`](#dgd_get_representations) executable).

The main output of `dgd_preprocess_samples` is a CSV file containing the preprocessed samples.

The Ensembl IDs of the genes found in the input samples but not present in the gene set used to train the DGD model, if any, are written in a separate output file in plain text format, with one gene per line. 

Furthermore, if any gene present in the gene set used to train the DGD model is not found in the genes for which counts are available in the input samples, the gene will be written to the output CSV file containing the preprocessed samples with a count of 0 for all samples. These "missing" genes are also written in a separate plain text file containing one gene per line.

## Command line

```
dgd_preprocess_samples [-h] -i INPUT_CSV [-os OUTPUT_CSV_SAMPLES] [-oe OUTPUT_TXT_GENES_EXCLUDED] [-om OUTPUT_TXT_GENES_MISSING] [-d WORK_DIR] [-lf LOG_FILE] [-lc] [-v] [-vv]
```

## Options

| Option                               | Description                                                  |
| ------------------------------------ | ------------------------------------------------------------ |
| `-h`, `--help`                       | Show the help message and exit.                              |
| `-i`, `--input-csv`                  | The input CSV file containing a data frame with the samples to be preprocessed. |
| `-os`, `--output-csv-samples`        | The name of the output CSV file containing the data frame with the preprocessed samples. The file will be written in the working directory. The default file name is `samples_preprocessed.csv`. |
| `-oe`, `--output-txt-genes-excluded` | The name of the output plain text file containing the list of genes whose expression data are excluded from the data frame with the preprocessed samples. The file will be written in the working directory. The default file name is `genes_excluded.txt`. |
| `-om`, `--output-txt-genes-missing`  | The name of the output plain text file containing the genes for which no available expression data are found in the input data frame. A default count of 0 is assigned to these genes in the output data frame with the preprocessed samples. The file will be written in the working directory. The default file name is `genes_missing.txt`. |
| `-d`, `--work-dir`                   | The working directory. The default is the current working directory. |
| `-lf`, `--log-file`                  | The name of the log file. The file will be written in the working directory. The default file name is `dgd_preprocess_samples.log`. |
| `-lc`, `--log-console`               | Show log messages also on the console.                       |
| `-v`, `--logging-verbose`            | Enable verbose logging (INFO level).                         |
| `-vv`, `--logging-debug`             | Enable maximally verbose logging for debugging purposes (DEBUG level). |
