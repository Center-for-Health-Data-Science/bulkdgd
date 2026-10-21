.. _recount3_config_options:

Configuration for building datasets from Recount3 studies
=========================================================

A dataset built with :doc:`bulkdgd_get_recount3 <bulkdgd_get_recount3>` (or :func:`bulkdgd.recount3.get_dataset_metadata`) is described by a YAML configuration file. It selects the Recount3 studies and samples to take and harmonizes their metadata: which columns are kept and how they are renamed, which values are mapped to new values, and which new columns are added.

The function that loads the configuration file is :func:`bulkdgd.ioutil.load_config_recount3`. It checks the file against the options below and reports every malformed option, with its path in the file (for instance, ``SRP114904.metadata_to_add.sample_organ_or_tissue: must be a dictionary.``). Options not listed below are errors. Options that are valid but have no effect (for instance, a mapping whose old values are YAML booleans or numbers, which match no text value) are reported as warnings.

The layout of the file depends on the Recount3 project the studies belong to:

* For GTEx and TCGA, the file holds one set of options, applied to all the studies listed in ``"studies"``.

* For SRA, the file maps each SRA study's code (for instance, ``SRP114904``) to that study's options.

Example files are in ``bulkdgd/configs/recount3``: ``gtex_example.yaml``, ``tcga_example.yaml`` and ``sra_example.yaml``.

How the options are applied
---------------------------

The Recount3 metadata of the studies are downloaded (or read from the directory they were saved in) and processed in this order:

* GTEx: ``"query_string"``, ``"metadata_to_add"``, ``"mapping"``, ``"metadata_to_keep"``.

* TCGA: ``"query_string"``, ``"metadata_to_process"``, ``"metadata_to_merge"``, ``"mapping_to_lower"``, ``"metadata_to_add"``, ``"mapping"``, ``"metadata_to_keep"``.

* SRA, study by study: the GEO options (``"query_string_geo"``), ``"query_string"``, ``"samples_to_drop"``, ``"metadata_to_add_geo"``, ``"metadata_to_keep_geo"`` (after ``"mapping_geo"``), ``"metadata_to_add"``, ``"mapping"``, ``"mapping_with_prefix"``, ``"mapping_with_suffix"``, ``"serial_mapping"``, ``"metadata_to_keep"``, ``"metadata_study"``, ``"mapping_values"``, then ``"metadata_from_study"``.

Queries and mappings therefore refer to the columns' **original** names and values, except for ``"mapping_values"`` and ``"metadata_from_study"``, which refer to the new names.

The queries are passed to :meth:`pandas.DataFrame.query`. Column names are written between backticks (for instance, ``"`SMTSD` == 'Kidney - Cortex'"``), which names that are not valid Python names need.

The Recount3 metadata hold text, so the old values in the mappings and the values in the queries are strings: a YAML key such as ``1.0`` or ``yes`` (read as a number and a boolean) must be quoted to match the text ``"1.0"`` or ``"yes"``.

GTEx options
------------

* ``"studies"`` is the list of GTEx tissues to take, as named in Recount3 (for instance, ``"BLADDER"``). It is required and must not be empty.

* ``"query_string"`` is a query selecting the samples to keep. It is optional; by default, all samples are kept.

* ``"metadata_to_keep"`` maps the columns to keep to their new names. A column already having one of the new names is replaced. If given, only these columns and those in ``"metadata_to_add"`` are kept; by default, all columns are kept.

* ``"mapping"`` maps columns to dictionaries of old values and new values.

* ``"metadata_to_add"`` maps the columns to add to dictionaries of queries and values: each sample matching a query gets its value. Two keys are special: ``"_all_"`` sets the value for all samples (the other keys are then ignored), and ``"_other_"`` sets it for the samples no earlier query matched. A column already in the metadata is replaced.

TCGA options
------------

TCGA takes ``"studies"`` (the cancer types, for instance ``"ACC"``), ``"query_string"``, ``"metadata_to_keep"``, ``"mapping"`` and ``"metadata_to_add"`` as GTEx does, and:

* ``"metadata_to_process"`` maps columns whose values are lists of findings to the options used to split them into new columns:

   * ``"separators"`` is the list of texts that start each finding, in the order they appear in the values. It is required.

   * ``"columns"`` maps each separator's name to the new column holding the text that follows the separator. A separator's name is the separator without a leading digit and with its capitalized words lowercased (``"Necrosis"`` is named ``"necrosis"``, ``"2Nuclear Grade III or IV"`` is named ``"nuclear grade III or IV"``). It is required.

   * ``"keep_columns"`` is whether the new columns are kept by ``"metadata_to_keep"``. It defaults to ``true``.

* ``"metadata_to_merge"`` maps new columns to how they are built from other columns, in one of two ways:

   * ``"columns_to_merge"`` is a list of columns whose values fill each other's missing values, in order. An item is either a column's name or a dictionary with a ``"column"`` and a ``"pattern"`` (a regular expression whose first group is taken from the column's values). It can be given with ``"mapping"``, a dictionary of old values and new values applied to the merged values, and ``"serial"``, whether to replace the merged values with their rank among the sorted unique values, from 1 (it defaults to ``false``).

   * ``"columns_to_join"`` is a list of columns whose values are joined with ``"separator"`` (it defaults to ``"|"``). Each item is a dictionary with the ``"column"`` and, optionally, a ``"mapping"`` of its old values (matched regardless of case) to new values, a ``"prefix"`` put before each value followed by a colon, a ``"phrase_prefix"`` put before each value and a ``"phrase_suffix"`` put after it.

  The values ``"na"``, ``"NA"`` and ``""`` are missing values. The new columns are kept by ``"metadata_to_keep"``.

* ``"mapping_to_lower"`` is a list of columns whose values are lowercased.

SRA options
-----------

Each study's options are under its SRA code, and are all optional. SRA studies take ``"query_string"``, ``"metadata_to_keep"``, ``"mapping"`` and ``"metadata_to_add"`` as GTEx does, with two differences: in ``"metadata_to_add"``, the values are added as text, and ``null`` sets a missing value; and ``"metadata_to_keep"`` does not rename a column to a name in ``"metadata_to_add"``.

A study's dataset has only the columns named in ``"metadata_study"``, ``"metadata_to_keep"``, ``"metadata_to_add"``, ``"metadata_to_keep_geo"`` and ``"metadata_to_add_geo"``, after the library and ENA columns below. The library columns of the Recount3 metadata (``library_name``, ``library_strategy``, ``library_source``, ``library_selection`` and ``library_layout``) are kept as ``sample_sequencing_library_name`` and so on, with harmonized values (``RNA-seq``, ``transcriptomic``, ``polyA``, ...).

The other options are:

* ``"metadata_study"`` maps study-level columns to the value all of the study's samples get. ``"study_geo_code"``, the study's GEO series (for instance, ``"GSE104730"``), is also the series the GEO options below read.

* ``"samples_to_drop"`` is a list of runs to drop.

* ``"mapping_with_prefix"`` and ``"mapping_with_suffix"`` map columns to a text put before or after each of their values.

* ``"serial_mapping"`` maps columns to a list of their values: each value is replaced by its position in the list, from 1.

* ``"mapping_values"`` maps columns to dictionaries of old substrings and new substrings, replaced in the columns' values.

* ``"metadata_from_study"`` maps other studies' codes to values recovered from them, for runs both studies deposited. Each study's ``"values"`` maps columns to dictionaries of values and the runs that get them. ``"matched_on"``, ``"source"`` and ``"fields"`` are notes, not used to build the dataset.

* ``"_use_as_external_id"`` is a column whose values index the samples, in place of their run IDs, while the options are applied: ``"samples_to_drop"`` names these values, and the runs' ENA details are looked up by them. The samples get their run IDs back before ``"metadata_from_study"``.

The GEO options read the samples of the study's GEO series, which are matched with the study's samples by their titles (or, for studies without Recount3 metadata, by their SRA experiments). They are ignored if ``"metadata_study"`` has no ``"study_geo_code"``:

* ``"query_string_geo"`` is a query selecting the series' samples.

* ``"metadata_to_keep_geo"`` maps the series' columns (``Sample_title``, ``Sample_source_name_ch1``, the samples' characteristics, ...) to the names they are added with.

* ``"mapping_geo"`` maps the series' columns to dictionaries of old values and new values, applied before ``"metadata_to_keep_geo"`` (and ignored without it).

* ``"metadata_to_add_geo"`` adds columns as ``"metadata_to_add"`` does, with queries on the series' columns.

Studies with no Recount3 metadata are built from their SRA Run Selector table, whose columns the options then refer to. Each run's experiment (``sample_identifier_ena_experiment``) and BioSample (``sample_identifier_ena_biosample``), its number within its experiment (``sample_identifier_run_numeric_id``), and its library strategy, selection and source where the metadata lack these columns, are taken from ENA.

Columns added to every dataset
------------------------------

Every dataset has the columns ``study_recount3_project_name`` (``gtex``, ``tcga`` or ``sra``) and ``study_recount3_samples_category`` (the sample's study). Some columns are derived from others, if those are in the dataset:

* ``subject_demographics_sex_or_gender``, from ``subject_demographics_sex`` (GTEx), ``subject_demographics_gender`` (TCGA), or ``subject_demographics_sex`` where present and ``subject_demographics_gender`` otherwise (SRA).

* ``subject_identifier_numeric_id`` (GTEx), numbering the subjects in ``subject_identifier_GTEx_id`` from 1.

* ``subject_demographics_age_bin_10_years`` (SRA), the ten-year bin of ``subject_demographics_age_years`` (``"0-9"``, ``"10-19"``, ...).

* ``subject_age_years`` (TCGA), a copy of ``subject_demographics_age_years`` as text, which is made a number.

In SRA datasets, ``sample_collection_is_adjacent_normal`` and ``sample_collection_is_metastatic`` are ``"no"`` where not set, and a ``study_comment`` gets a note on where the library columns and the derived columns come from. In TCGA datasets, ``sample_collection_is_adjacent_normal`` is ``"no"`` where not set, and the survival times (``subject_status_main_condition_clinical_progression_time_to_death_days`` and ``..._time_to_last_follow_up_days``) are made numbers.
