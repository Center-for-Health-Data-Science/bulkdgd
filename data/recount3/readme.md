# `data/recount3`

Last updated: 14/07/2023

## `gtex_tissues.txt`

A plain text file containing the list of GTEx tissues. `dgd_get_recount3_data` uses it to check whether the user-provided tissue is valid.

Example:

```
# GTEx tissue types - STUDY_NA is not included

# Adipose tissue
ADIPOSE_TISSUE

# Adrenal gland
ADRENAL_GLAND

# Blood
BLOOD

# Blood vessel
BLOOD_VESSEL
```

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

## `tcga_cancer_types.txt`

A plain text file containing the list of TCGA cancer types. `dgd_get_recount3_data` uses it to check whether the user-provided cancer type is valid.

Example:

```
# TCGA cancer types (from https://gdc.cancer.gov/resources-tcga-users/tcga-code-tables/tcga-study-abbreviations, CNTL, FPPP, and MISC excluded)

# Adrenocortical carcinoma
ACC

# Bladder Urothelial Carcinoma
BLCA

# Breast Invasive Carcinoma
BRCA

# Cervical squamous cell carcinoma and endocervical carcinoma
CESC
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
