# `data`

Last updated: 22/07/2024

## `gtex_metadata_fileds`

A plain text file containing the fields (= columns) found in the files describing the metadata associated with GTEx samples downloaded from the Recount3 platform.

The file is used by the `dgd_get_recount3_data` executable when filtering the GTEx samples through a query string.

Example:

```
# Fields found in the GTEx metadata files downloaded from the Recount3 platform.

rail_id
run_acc
external_id
study
SUBJID
SEX
AGE
DTHHRDY
```

## `sra_metadata_fields.txt`

A plain text file containing the fields (= columns) found in the files describing the metadata associated with SRA samples downloaded from the Recount3 platform. 

The file is used by the `dgd_get_recount3_data` executable when filtering the SRA samples through a query string.

Example:

```
# Fields found in the SRA metadata files downloaded from the Recount3 platform.

rail_id
external_id
study
sample_acc
experiment_acc
```

## `tcga_metadata_fields.txt`

A plain text file containing the fields (= columns) found in the files describing the metadata associated with TCGA samples downloaded from the Recount3 platform.

This file is used by the `dgd_get_recount3_data` executable when filtering the TCGA samples through a query string.

Example:

```
rail_id
external_id
study
tcga_barcode
gdc_file_id
gdc_cases.project.name
gdc_cases.project.released
gdc_cases.project.state
```
