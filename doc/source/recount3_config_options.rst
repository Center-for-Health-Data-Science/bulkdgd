.. _recount3_config_options:

Configuration for building datasets from Recount3 studies
=========================================================

A dataset built with :doc:`bulkdgd_get_recount3 <bulkdgd_get_recount3>` or :func:`bulkdgd.recount3.datasets.get_dataset_metadata` is described by a YAML configuration file. The file selects the Recount3 studies and samples and harmonizes their metadata: which columns are kept and how they are renamed, which values are mapped to new values, and which columns are added.

The function that loads the configuration file is :func:`bulkdgd.ioutil.load_config_recount3`. It checks the file against the options below and raises an error listing every malformed, missing or unknown option with its path in the file (for instance, ``SRP114904.metadata_to_add.sample_organ_or_tissue: must be a dictionary.``). Options that have no effect (for instance, a mapping's old value written as a YAML boolean or number, which matches no text value) are logged as warnings.

The layout of the file depends on the Recount3 project the studies belong to:

* For GTEx and TCGA, the file holds one set of options, applied to all the studies listed in ``"studies"``.

* For SRA, the file maps each SRA study's code (for instance, ``SRP114904``) to that study's options.

Example files are in ``bulkdgd/configs/recount3``: ``gtex_example.yaml``, ``tcga_example.yaml`` and ``sra_example.yaml``.

How the options are applied
---------------------------

The Recount3 metadata of the studies are downloaded (or read from the directory they were saved in) and processed in this order:

* GTEx: ``"query_string"``, ``"metadata_to_add"``, ``"mapping"``, ``"metadata_to_keep"``.

* TCGA: ``"query_string"``, ``"metadata_to_process"``, ``"metadata_to_merge"``, ``"mapping_to_lower"``, ``"metadata_to_add"``, ``"mapping"``, ``"metadata_to_keep"``.

* SRA, study by study: ``"query_string_geo"``, ``"query_string"``, ``"samples_to_drop"``, ``"metadata_to_add_geo"``, ``"metadata_to_keep_geo"`` (after ``"mapping_geo"``), ``"metadata_to_add"``, ``"mapping"``, ``"mapping_with_prefix"``, ``"mapping_with_suffix"``, ``"serial_mapping"``, ``"metadata_to_keep"``, ``"metadata_study"``, ``"mapping_values"``, ``"metadata_from_study"``.

Queries and mappings therefore refer to the columns' original names, before ``"metadata_to_keep"`` renames them, except for ``"mapping_values"`` and ``"metadata_from_study"``, which refer to the new names.

The queries are passed to :meth:`pandas.DataFrame.query`. Column names are written between backticks (for instance, ``"`SMTSD` == 'Kidney - Cortex'"``); the backticks are required for names that are not valid Python identifiers.

The Recount3 metadata hold text, so the old values in the mappings and the values compared in the queries are strings. A YAML key such as ``1.0`` or ``yes`` is read as a number or a boolean, and must be quoted to match the text ``"1.0"`` or ``"yes"``.

GTEx options
------------

* ``"studies"`` is the list of GTEx tissues to take, as named in Recount3 (for instance, ``"BLADDER"``). It is required and must not be empty.

* ``"query_string"`` is a query selecting the samples to keep. By default, all samples are kept.

* ``"metadata_to_keep"`` maps the columns to keep to their new names. A column already having one of the new names is replaced. If it is given, only these columns and those in ``"metadata_to_add"`` are kept; otherwise, all columns are kept.

* ``"mapping"`` maps columns to dictionaries of old values and new values.

* ``"metadata_to_add"`` maps the columns to add to dictionaries of queries and values: each sample matching a query gets its value. Two keys are special: ``"_all_"`` sets the value for all samples (the other keys are then ignored), and ``"_other_"`` sets it for the samples no query matched. A column already in the metadata is replaced.

TCGA options
------------

TCGA takes ``"studies"`` (the cancer types, for instance ``"ACC"``), ``"query_string"``, ``"metadata_to_keep"``, ``"mapping"`` and ``"metadata_to_add"`` as GTEx does, and:

* ``"metadata_to_process"`` maps columns to the options that split their values at separators into new columns:

   * ``"separators"`` is the list of texts the values are split at, in the order they appear in the values. It is required.

   * ``"columns"`` maps each separator's name to the new column holding the lowercased text between that separator and the next one found, or the end of the value. A separator's name is the separator without a leading digit and with its capitalized words lowercased (``"Necrosis"`` is named ``"necrosis"``, ``"2Nuclear Grade III or IV"`` is named ``"nuclear grade III or IV"``). It is required.

   * ``"keep_columns"`` is whether ``"metadata_to_keep"`` keeps the new columns. It defaults to ``true``.

* ``"metadata_to_merge"`` maps new columns to how they are built from other columns, with exactly one of:

   * ``"columns_to_merge"``, a list of columns merged in order: each column fills the missing values left by the columns before it. An item is either a column's name or a dictionary with a ``"column"`` and a ``"pattern"`` (a regular expression whose first group is taken from the column's values). It can be given with ``"mapping"``, a dictionary of old values and new values applied to the merged values, and ``"serial"``, whether to replace the merged values with their rank among the sorted unique values, from 1 (it defaults to ``false``).

   * ``"columns_to_join"``, a list of columns whose values are joined with ``"separator"`` (it defaults to ``"|"``), skipping missing values. Each item is a dictionary with the ``"column"`` and, optionally, a ``"mapping"`` of its old values (matched regardless of case) to new values, a ``"prefix"`` put before each value as ``"{prefix}: {value}"``, and a ``"phrase_prefix"`` and a ``"phrase_suffix"`` put before and after each value, separated by a space.

  The values ``"na"``, ``"NA"`` and ``""`` are missing values. ``"metadata_to_keep"`` keeps the new columns.

* ``"mapping_to_lower"`` is a list of columns whose values are lowercased.

SRA options
-----------

Each study's options are under its SRA code, and are all optional. ``"query_string"``, ``"metadata_to_keep"``, ``"mapping"`` and ``"metadata_to_add"`` work as for GTEx, except that ``"metadata_to_add"`` sets its values as text (``null`` sets a missing value), and ``"metadata_to_keep"`` skips a column whose new name is in ``"metadata_to_add"``.

Besides the columns added to every dataset, a study's metadata have only the library and ENA columns described below and the columns named in ``"metadata_study"``, ``"metadata_to_keep"``, ``"metadata_to_add"``, ``"metadata_to_keep_geo"``, ``"metadata_to_add_geo"`` and ``"metadata_from_study"``. The library columns of the Recount3 metadata (``library_name``, ``library_strategy``, ``library_source``, ``library_selection`` and ``library_layout``) are kept as ``sample_sequencing_library_name``, ``sample_sequencing_library_strategy`` and so on, with harmonized values (for instance, ``RNA-seq``, ``transcriptomic`` and ``polyA``).

The other options are:

* ``"metadata_study"`` maps columns to the value all of the study's samples get. ``"study_geo_code"``, a string (for instance, ``"GSE116111"``), is also the GEO series the GEO options below read.

* ``"samples_to_drop"`` is a list of runs to drop.

* ``"mapping_with_prefix"`` and ``"mapping_with_suffix"`` map columns to a text put before or after each of their values.

* ``"serial_mapping"`` maps columns to a list of their values: each value is replaced by its position in the list, from 1.

* ``"mapping_values"`` maps columns to dictionaries of old substrings and new substrings, replaced in the columns' values.

* ``"metadata_from_study"`` maps other studies' codes to values taken from them. Each study's ``"values"`` maps columns to dictionaries of values and the runs that get them. The values are set as text, a column not in the metadata is added, and every run must be in the study's metadata. ``"matched_on"``, ``"source"`` and ``"fields"`` are notes, not used to build the dataset.

* ``"_use_as_external_id"`` is a column whose values index the samples, in place of their run IDs, while the options are applied. The samples get their run IDs back before ``"metadata_from_study"``. It applies only to studies with Recount3 metadata.

The GEO options read the samples of the study's GEO series, which are matched with the study's samples by their titles (``sample_title`` in the study's metadata) or, for studies without Recount3 metadata, by their SRA experiments. They are ignored if ``"metadata_study"`` has no ``"study_geo_code"``:

* ``"query_string_geo"`` is a query selecting the series' samples. For studies without Recount3 metadata, the study's samples not in the selection are dropped.

* ``"metadata_to_keep_geo"`` maps the series' columns (``Sample_title``, ``Sample_source_name_ch1``, one column per characteristic, named by its label, ...) to the names they are added with.

* ``"mapping_geo"`` maps the series' columns to dictionaries of old values and new values, applied before ``"metadata_to_keep_geo"`` (and ignored without it).

* ``"metadata_to_add_geo"`` adds columns as ``"metadata_to_add"`` does, with queries on the series' columns.

Studies with no Recount3 metadata are built from their SRA Run Selector table, whose columns the options then refer to.

For every study ENA has, each run's experiment (``sample_identifier_ena_experiment``), BioSample (``sample_identifier_ena_biosample``) and number within its experiment, from 1 (``sample_identifier_run_numeric_id``), are taken from ENA, as are the library strategy, selection and source where the metadata lack them.

Columns added to every dataset
------------------------------

Every dataset has the columns ``study_recount3_project_name`` (``gtex``, ``tcga`` or ``sra``) and ``study_recount3_samples_category`` (the sample's study). Some columns are derived from others, if those are in the dataset:

* ``subject_demographics_sex_or_gender``, from ``subject_demographics_sex`` (GTEx), ``subject_demographics_gender`` (TCGA), or ``subject_demographics_sex`` where present and ``subject_demographics_gender`` otherwise (SRA).

* ``subject_identifier_numeric_id`` (GTEx), numbering the subjects in ``subject_identifier_GTEx_id`` from 1, in sorted order.

* ``subject_demographics_age_bin_10_years`` (SRA), the ten-year bin of ``subject_demographics_age_years`` (``"0-9"``, ``"10-19"``, ..., ``"110-119"``).

* ``subject_age_years`` (TCGA), a copy of ``subject_demographics_age_years`` as text. ``subject_demographics_age_years`` is then made a number.

In SRA datasets, ``sample_collection_is_adjacent_normal`` and ``sample_collection_is_metastatic`` are ``"no"`` where not set, and ``study_comment``, if present, gets a note on the sources of the library, ENA and derived columns. In TCGA datasets, ``sample_collection_is_adjacent_normal`` is ``"no"`` where not set, and the survival times (``subject_status_main_condition_clinical_progression_time_to_death_days`` and ``..._time_to_last_follow_up_days``) are made numbers.
