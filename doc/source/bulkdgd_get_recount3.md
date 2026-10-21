# `bulkdgd_get_recount3`

This command builds a dataset of RNA-seq read counts and harmonized metadata from studies hosted on the [Recount3 platform](https://rna.recount.bio/).

The program supports studies from the [GTEx](https://gtexportal.org/home/), [TCGA](https://www.cancer.gov/ccg/research/genome-sequencing/tcga), and [SRA](https://www.ncbi.nlm.nih.gov/sra) projects. A study is a tissue for GTEx, a cancer type for TCGA, and an SRA project (for instance, `SRP114904`) for SRA.

`bulkdgd_get_recount3` takes a YAML configuration file selecting the studies and samples to take and harmonizing their metadata: which columns are kept and how they are renamed, which values are mapped to new values, and which new columns are added. The options the file can contain are described in the [configuration options](recount3_config_options.rst). Example files are in `bulkdgd/configs/recount3`.

The command produces five output files:

* A CSV file containing the samples' read counts, with the samples as rows and the genes (identified by their Ensembl IDs) as columns. The first column, `external_id`, contains the samples' IDs. The read counts are the raw counts divided by each sample's average mapped read length and rounded.

* A CSV file containing the samples' harmonized metadata, one row per sample, in the same order as the counts.

* Two plain text files listing the columns of the counts (the genes) and of the metadata, one per line.

* A plain text file listing the samples' IDs, one per line, in the counts' order.

The files downloaded from Recount3 (the RNA-seq data, the quality control metrics and the metadata), and, for SRA studies, from ENA, GEO and the SRA Run Selector, are saved in the directories given with `-odc`, `-odq` and `-odm`, and are not downloaded again if found there.

## Command line

```
bulkdgd_get_recount3 [-h] -ic INPUT_CONFIG_FILE -is {GTEx,TCGA,SRA} [-ig {26,29}] [-ec EXISTING_COUNTS] [-oc OUTPUT_COUNTS] [-om OUTPUT_METADATA] [-occ OUTPUT_COUNTS_COLUMNS] [-omc OUTPUT_METADATA_COLUMNS] [-osi OUTPUT_SAMPLES_IDS] [-odc OUTPUT_DIR_COUNTS] [-odq OUTPUT_DIR_QC] [-odm OUTPUT_DIR_METADATA] [-st STUDIES [STUDIES ...]] [-mo] [-se] [-d WORK_DIR] [-lf LOG_FILE] [-lc] [-v] [-vv]
```

## Options

### Help options

| Option         | Description                     |
| -------------- | ------------------------------- |
| `-h`, `--help` | Show the help message and exit. |

### Input files

| Option                           | Description                                                  |
| -------------------------------- | ------------------------------------------------------------ |
| `-ic`, `--input-config-file`     | The YAML configuration file selecting the studies and samples the dataset is built from and harmonizing their metadata. If it is a name without an extension, it is assumed to be the name of a configuration file in `bulkdgd/configs/recount3`. |
| `-is`, `--input-study-name`      | The Recount3 project the studies belong to: `GTEx`, `TCGA`, or `SRA`. |
| `-ig`, `--input-gencode-release` | The GENCODE release the counts were annotated with: `26` or `29`. The default is `29`. |
| `-ec`, `--existing-counts`       | An existing CSV file of counts, written by a previous run, to take the metadata's samples from. Only the studies of the samples it lacks are downloaded, and the samples not in the metadata are dropped. By default, the counts of all the studies are downloaded. |

### Output files

| Option                             | Description                                                  |
| ---------------------------------- | ------------------------------------------------------------ |
| `-oc`, `--output-counts`           | The output CSV file of the read counts. The default file name is `counts.csv`. |
| `-om`, `--output-metadata`         | The output CSV file of the harmonized metadata. The default file name is `metadata.csv`. |
| `-occ`, `--output-counts-columns`  | The output plain text file of the counts' columns. The default file name is `counts_columns.txt`. |
| `-omc`, `--output-metadata-columns`| The output plain text file of the metadata's columns. The default file name is `metadata_columns.txt`. |
| `-osi`, `--output-samples-ids`     | The output plain text file of the samples' IDs. The default file name is `samples_ids.txt`. |
| `-odc`, `--output-dir-counts`      | The directory of the RNA-seq data files downloaded from Recount3 (`{project}_{study}_gene_sums.gz`). The default is the working directory. |
| `-odq`, `--output-dir-qc`          | The directory of the quality control files downloaded from Recount3 (`{project}_{study}_qc.gz`). The default is the working directory. |
| `-odm`, `--output-dir-metadata`    | The directory of the metadata files downloaded from Recount3 (`{project}_{study}_metadata.gz`) and, for SRA studies, from ENA (`{study}_ena_filereport.tsv`), GEO (`{geo_code}_family.soft.gz`) and the SRA Run Selector (`{study}_run_selector.csv`). The default is the working directory. |

The output files are written in the working directory, unless their paths are absolute.

### Run options

| Option                   | Description                                                  |
| ------------------------ | ------------------------------------------------------------ |
| `-st`, `--studies`       | The studies to take from the configuration file. By default, all of them are taken. |
| `-mo`, `--metadata-only` | Write only the metadata (and their columns), without getting the counts. |
| `-se`, `--skip-existing` | Skip the run when every output file it writes already exists. By default, the output files are overwritten. |

### Working directory options

| Option             | Description                                                  |
| ------------------ | ------------------------------------------------------------ |
| `-d`, `--work-dir` | The working directory. The default is the current working directory. |

### Logging options

| Option                    | Description                                                  |
| ------------------------- | ------------------------------------------------------------ |
| `-lf`, `--log-file`       | The name of the log file. The file will be written in the working directory. The default file name is `bulkdgd_get_recount3.log`. |
| `-lc`, `--log-console`    | Show log messages also on the console.                       |
| `-v`, `--log-verbose`     | Enable verbose logging (INFO level).                         |
| `-vv`, `--log-debug`      | Enable maximally verbose logging for debugging purposes (DEBUG level). |

## Example

```
bulkdgd_get_recount3 -ic sra_example -is SRA -d sra_dataset -odc raw/gene_sums -odq raw/qc -odm raw/metadata
```

This builds a dataset from the three SRA studies in `bulkdgd/configs/recount3/sra_example.yaml`, writing `counts.csv`, `metadata.csv`, `counts_columns.txt`, `metadata_columns.txt` and `samples_ids.txt` in the `sra_dataset` directory, and the files downloaded from Recount3, ENA and GEO in `sra_dataset/raw`.

```
bulkdgd_get_recount3 -ic gtex_example -is GTEx -st KIDNEY -mo
```

This writes only the harmonized metadata of the GTEx kidney samples selected in `bulkdgd/configs/recount3/gtex_example.yaml`.

The functions the command uses are also available from Python: `bulkdgd.ioutil.load_config_recount3`, `bulkdgd.recount3.get_dataset_metadata`, `bulkdgd.recount3.get_dataset_counts`, `bulkdgd.recount3.save_dataset_metadata` and `bulkdgd.recount3.save_dataset_counts` (see the [recount3 API](recount3.rst)). Tutorial 4 downloads single studies with the lower-level functions.
