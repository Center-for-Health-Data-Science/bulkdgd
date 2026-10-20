# `data`

## `gtex_metadata_fields.txt`

A plain text file listing the fields (columns) of the metadata files for the GTEx samples downloaded from the Recount3 platform.

`bulkdgd_get_recount3` can filter the GTEx samples with a query string on these fields.

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

A plain text file listing the fields (columns) of the metadata files for the SRA samples downloaded from the Recount3 platform.

`bulkdgd_get_recount3` can filter the SRA samples with a query string on these fields.

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

A plain text file listing the fields (columns) of the metadata files for the TCGA samples downloaded from the Recount3 platform.

`bulkdgd_get_recount3` can filter the TCGA samples with a query string on these fields.

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
