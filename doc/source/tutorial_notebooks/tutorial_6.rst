Tutorial 6 - Differential expression and gene set enrichment with the ensemble
==============================================================================

The genes a single bulkdgd model calls differentially expressed depend on the seed it was trained with, so one model cannot tell which of those genes are real. The ensemble answers a different question: how many of several identically trained models agree on a gene. It is fifteen models sharing an architecture, a gene universe, a training set and a train/test split, and differing only in the seed they were fitted with.

This tutorial takes a set of samples through the ensemble end to end (representations, differential expression, the tiered consensus drawn from what the members call, and gene set enrichment), and closes with when to use the ensemble and when a single model is enough.

It assumes :doc:`Tutorial 1 <tutorial_1>` and :doc:`Tutorial 2 <tutorial_2>`, which do the first two of those steps with one model.

Loading the ensemble
--------------------

``BulkDGDEnsemble()``, called with no arguments, is the ensemble that ships with the package. The members differ only in their seed, so, given neither configuration, the class builds the shared architecture and the members' paths from the files the package ships.

.. code-block:: python

   # Import from the standard library.
   import logging as log
   import os

   # Import from third-party libraries.
   import pandas as pd

   # Import from 'bulkdgd'.
   from bulkdgd import ioutil
   from bulkdgd.ensemble import BulkDGDEnsemble

   # Set the logging options so that every message of level INFO or
   # above is emitted.
   log.basicConfig(level = "INFO")

   # Get the ensemble that ships with the package. The shared
   # architecture is read once, from the base member's configuration
   # file, and one entry is made per shipped seed.
   ensemble = BulkDGDEnsemble()

.. code-block:: text

   INFO:bulkdgd.ensemble.ensemble:The ensemble was successfully set (15 models, seeds: 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101).

Nothing has been loaded at this point. The ensemble knows which members it has, what they are seeded with, and where their fitted parameters live, and it holds no model:

.. code-block:: python

   # The members' names, which are the seeds they are trained with.
   ensemble.names

   # The seeds themselves, as integers.
   ensemble.seeds

   # How many models the ensemble is made of.
   ensemble.n_models

   # The device the members are placed on when they are built.
   ensemble.device

A member is built only when it is asked for, one at a time, by ``get_model``:

.. code-block:: python

   # Build one member, with its trained parameters loaded. It is an
   # ordinary 'bulkdgd.core.model.BulkDGD', and behaves exactly as the
   # one in Tutorial 1.
   model = ensemble.get_model(name = "seed37")

``BulkDGD()`` with no arguments is the base member, ``seed37``, on its own. Apart from being the default, nothing distinguishes it from the other fourteen members. This tutorial runs the whole ensemble; :doc:`Tutorial 1 <tutorial_1>` runs only ``seed37``.

What the first run downloads
----------------------------

Each member's fitted mixture (``gmm.pth``, about 28 KiB) ships with the package. Its decoder does not: each decoder is about 1.79 GiB in ``float64``, close to the 2 GiB limit on a GitHub release asset, and fifteen of them are too large for PyPI. They are downloaded on demand, one asset per member and per release, from

.. code-block:: text

   https://github.com/Center-for-Health-Data-Science/bulkdgd/releases/download/v{version}/dec_{seed}.pth

and saved in the per-seed directory the package expects, ``bulkdgd/data/model/{seed}/dec.pth``, **inside the installed package**. This has three consequences:

* The full ensemble is about **27 GiB** of one-off downloads. Later runs reuse the files.

* The installation must be writable. A read-only or system-wide installation fails on the first member it tries to download.

* A member is downloaded the first time it is built, so the downloads are spread over the run rather than done up front.

``status`` reports what is already on disk. ``trained`` is ``False`` for a member whose decoder has not been downloaded yet:

.. code-block:: python

   # Report what each member already has: whether its parameters are
   # there, how many samples it has differential expression results
   # for, and whether those results are packed.
   ensemble.status()

.. code-block:: text

            seed  trained  n_samples_dea  dea_packed
   name
   seed37     37     True              0       False
   seed41     41    False              0       False
   seed43     43    False              0       False
   ...

Where the results go
--------------------

Every member has a ``"results_dir"``, and everything run with that member is written under it. The shipped configuration puts the results under the working directory, in ``bulkdgd_ensemble_results/{seed}``. The path is resolved when the ensemble is built, so the working directory *at that moment* decides where the results are written.

To write them elsewhere, pass a configuration to the method you are calling. It can point the ensemble's members at different directories, but cannot change which members there are:

.. code-block:: python

   # Keep the members the ensemble already has, and move their
   # results. 'dict(options, results_dir = ...)' copies each member's
   # entry and replaces one key of it.
   config_ensemble = \
       {name : dict(options,
                    results_dir = os.path.join("/data/brca", name))
        for name, options in ensemble.config_ensemble.items()}

Use the same approach to run a second cohort through the same members without overwriting the first cohort's results: one ensemble, with a different ``config_ensemble`` per cohort. Every method below takes it as the optional ``config_ensemble`` argument.

Preprocess the samples
----------------------

The ensemble takes the samples exactly as a single model does (a data frame with the samples as rows and the model's genes as columns), so the preprocessing is the one in :doc:`Tutorial 1 <tutorial_1>`. Every member shares a gene universe, so it is done once for the whole ensemble:

.. code-block:: python

   # Load the samples into a data frame.
   df_samples = \
       ioutil.load_samples(csv_file = "samples.csv",
                           sep = ",",
                           keep_samples_names = True,
                           split = False)

   # Match the samples' genes against the model's: genes the model
   # does not know about are dropped, and genes it expects that are
   # missing are added back with a count of 0.
   df_preproc, genes_excluded, genes_missing = \
       ioutil.preprocess_samples(df_samples = df_samples)

Find the representations
------------------------

Differential expression needs each member's own prediction for each sample, so the representations come first. This step builds the members, so it is also the step that downloads the decoders and takes the longest: finding a representation is a per-sample optimization in latent space, and the ensemble runs it fifteen times.

.. code-block:: python

   # Load the default configuration for the two-round optimization
   # scheme.
   config_rep = ioutil.load_config_rep(None)

   # Find the representations with every member, one member at a time.
   results_rep = \
       ensemble.find_representations(
           # The data frame with the pre-processed samples
           df_samples = df_preproc,
           # The configuration for the search, which is the one a
           # single 'BulkDGD' takes
           config_rep = config_rep,
           # Whether to skip the members that already have their
           # representations on disk
           resume = True)

Each member writes ``representations.csv``, ``pred_means.csv`` and ``time.csv`` into its results' directory, plus ``pred_r_values.csv`` when the output module has r-values, which the shipped model's does. The predicted means are the in silico control the next step compares the observed counts against.

A member that fails is recorded, and the members after it are still run, so check what came back:

.. code-block:: python

   # What happened to each member: 'done', 'skipped' or 'failed'.
   {name : result["status"] for name, result in results_rep.items()}

   # Why a member failed, if one did.
   results_rep["seed41"]["error"]

Every stage below returns a dictionary of the same shape, keyed by member name, with the keys ``"name"``, ``"seed"``, ``"status"``, ``"outputs"``, ``"error"`` and ``"elapsed"``.

.. note::

   ``resume = True`` skips a member that already has *all* of its outputs. This makes an interrupted run cheap to restart, but it also silently reuses stale results: a cohort re-run with different options into the same directories is skipped, not recomputed. Point the results elsewhere, or pass ``resume = False``.

Differential expression across the ensemble
-------------------------------------------

``dea`` compares each sample's observed counts against each member's predicted means, and computes p-values, q-values and log2-fold changes exactly as :doc:`Tutorial 2 <tutorial_2>` does with one model. It reads ``pred_means.csv`` and ``pred_r_values.csv`` from each member's results' directory, where the previous step wrote them, and **builds no model at all**, so this stage needs neither the decoders nor a GPU. The two calls must therefore be given the same ``config_ensemble``, since it says where those files are:

.. code-block:: python

   # Perform differential expression analysis with every member.
   results_dea = \
       ensemble.dea(
           # The data frame with the same pre-processed samples
           df_samples = df_preproc,
           # The configuration for the analysis
           config_dea = \
               {# Where a member's results are written. A relative
                # path is taken relative to the member's results'
                # directory.
                "dea_dir" : "dea",
                # Which statistics to compute
                "statistics" : ["p_values", "q_values",
                                "log2_fold_changes"],
                # The resolution of the p-values' calculation (set it
                # to 'None' for an exact calculation)
                "resolution" : 1e4,
                # The family-wise error rate for the q-values
                "alpha" : 0.05,
                # The method used to adjust the p-values
                "method" : "fdr_bh",
                # The device the p-values are computed on
                "device" : "cpu"},
           # Whether to skip the samples that already have results
           resume = True)

Everything in ``config_dea`` that is not one of the ensemble's own keys is passed on to :func:`bulkdgd.analysis.dea.get_statistics`.

One of those keys is filled in for you. The statistics are computed against a scaled predicted mean, and the scaling is a property of the ensemble's model, which the function computing the statistics cannot know. ``scaling_factor`` is therefore taken from the ensemble's own model configuration (``"median"`` for the shipped ensemble) unless you pass it yourself. Passing the wrong one makes every predicted mean wrong by the ratio of the two, about three, without any error.

The results are written as one file per sample per member, under ``{results_dir}/dea``, as ``dea_{sample}.parquet``. An interrupted run therefore resumes at the first sample that has no file, not at the first member. Each file holds one row per gene, with the columns ``p_value``, ``q_value``, ``log2_fold_change``, the ``obs_counts`` the sample was scored on, and the ``dgd_mean`` and ``dgd_r`` the statistics were computed from. Everything that reads these results looks the sample up by name and reads whichever form is on disk, including ``.csv`` files.

.. note::

   One file per sample per member is fifteen times the file count of a single-model run, which quickly adds up to many small files. ``"zip_results" : True`` in ``config_dea`` packs each member's results into one archive once they are all written, and removes the loose files. The archive is written, closed and read back before anything is removed, and everything downstream reads either form.

How the members' calls are combined
-----------------------------------

``consensus`` turns fifteen sets of per-sample calls into one tiered list of genes per group of samples. It rests on three definitions:

* A model **calls** a gene for a sample when the gene's q-value is below ``q_val`` and the absolute value of its log2-fold change is above ``log2_fold_change``. Both comparisons are strict.

* A model **considers** a gene for a group of samples when it calls the gene in at least a ``recurrence`` share of that group's samples. Recurrence is computed within a group, and never across groups.

* A gene's **tier** is how many of the ensemble's models consider it: the number of models that agree on it. The **consensus recurrence** reported for a gene of tier *K* is the *K*-th largest of its per-model recurrences, which is the level at which *K* models agree. Genes below ``min_tier`` are left out.

For example, with three models and ten samples in one group:

.. code-block:: text

   gene    samples called in    recurrences, largest first   tier   consensus_recurrence
   GENE_A  10/10, 8/10, 4/10              1.00, 0.80, 0.40      3                   0.40
   GENE_B   5/10, 2/10, 1/10              0.50, 0.20, 0.10      2                   0.20
   GENE_C   3/10, 1/10, 0/10              0.30, 0.10, 0.00      1                      -

At the default ``recurrence`` of ``0.20``, all three of ``GENE_A``'s models consider it, so it is tier 3, and the third-largest recurrence, ``0.40``, is the level at which three models agree on it. Two of ``GENE_B``'s three do, so it is tier 2 at ``0.20``. Only one of ``GENE_C``'s does, so at the default ``min_tier`` of ``2`` it is dropped.

Samples that any one model has no results for are dropped for the whole ensemble, and the number dropped is reported. This keeps tiers comparable between genes, since every gene is then scored by the same models.

.. code-block:: python

   # The samples' metadata, indexed by the samples' names. It must
   # carry the column the samples are grouped by, and the column they
   # are filtered on, if any.
   df_metadata = pd.read_csv("metadata.csv", index_col = 0)

   # The genes' symbols, mapped from the genes' Ensembl IDs. They are
   # optional, and only cosmetic: a gene with no entry keeps its
   # Ensembl ID as its symbol.
   genes_symbols = \
       pd.read_csv("genes_symbols.csv",
                   index_col = 0).iloc[:, 0].to_dict()

   # Draw the consensus.
   dfs_consensus = \
       ensemble.consensus(
           # The data frame with the samples' metadata
           df_metadata = df_metadata,
           # The configuration for the consensus
           config_consensus = \
               {# The metadata column the samples are grouped by, and
                # within which a gene's recurrence is computed
                "group_column" : "cancer_type",
                # The name the group is given in the output
                "group_name" : "cancer_type",
                # A metadata column the samples are filtered on before
                # anything else, and the value they must have in it -
                # here, keeping the primary tumours and leaving the
                # metastatic ones out
                "filter_column" : "sample_type",
                "filter_value" : "Primary Tumor",
                # Where a member's results are read from
                "dea_dir" : "dea",
                # The thresholds a gene must pass to be called for a
                # sample
                "q_val" : 0.05,
                "log2_fold_change" : 1.0,
                # The share of a group's samples a model must call a
                # gene in to consider it
                "recurrence" : 0.20,
                # The tier below which a gene is left out
                "min_tier" : 2,
                # The genes' symbols, mapped from the genes' names
                "genes_symbols" : genes_symbols,
                # How many processes to score the samples with
                "n_processes" : 8})

The result is a dictionary mapping each group to a data frame with one row per gene, with the genes the most models agree on first. The rows below continue the three-model example; a run of the shipped ensemble has fifteen recurrences and tiers up to 15.

.. code-block:: text

     cancer_type             gene  symbol  tier  consensus_recurrence         rec_sorted
   0        BRCA  ENSG00000141510    TP53     3                   0.40  1.000|0.800|0.400
   1        BRCA  ENSG00000012048   BRCA1     2                   0.20  0.500|0.200|0.100

``rec_sorted`` lists every per-model recurrence, largest first, so a tier can be recomputed at a different ``recurrence`` without reading the per-sample files again. ``symbol`` falls back to the gene's name when ``genes_symbols`` has no entry for it.

.. code-block:: python

   # Write each group's consensus.
   for group, df_consensus in dfs_consensus.items():

       df_consensus.to_csv(f"consensus_{group}.csv",
                           index = False,
                           sep = ",")

.. note::

   ``consensus`` raises an error if any member has no results at all, since such a member would drop every sample and leave the consensus empty.

Gene set enrichment
-------------------

``gsea`` scores, for each member and each sample, how enriched a set of genes of interest is among the genes that member calls for that sample. It reads the differential expression back from disk, packed or loose, so it needs no model either.

The gene sets are a dictionary mapping a name to a list of genes. They are intersected with the index of the per-sample statistics, which holds Ensembl IDs, so the sets must be Ensembl IDs too:

.. code-block:: python

   # The sets of genes of interest, as Ensembl IDs.
   genes_sets = \
       {"brca_drivers" : \
           [line.strip() for line in open("brca_drivers.txt")],
        "housekeeping" : \
           [line.strip() for line in open("housekeeping.txt")]}

   # Compute the enrichment scores with every member.
   results_gsea = \
       ensemble.gsea(
           # The sets of genes of interest
           genes_sets = genes_sets,
           # The configuration for the analysis
           config_gsea = \
               {# Where a member's differential expression is read
                # from
                "dea_dir" : "dea",
                # Where a member's enrichment scores are written
                "gsea_dir" : "gsea",
                # The thresholds a gene must pass to count as
                # significant
                "p_val" : 0.05,
                "q_val" : 0.05,
                "log2_fold_change" : 1.0},
           # Whether to skip the members that already have their
           # enrichment scores
           resume = True)

Everything in ``config_gsea`` that is not one of the ensemble's own keys is passed on to :func:`bulkdgd.analysis.dea.get_significant_genes`.

Each member writes one ``e_scores.csv`` into ``{results_dir}/gsea``, covering every sample it has results for, with the columns ``sample``, ``genes_set``, ``num_genes_in_set``, ``num_genes_significant`` and ``e_score``. A score of 1 is what a set of that size would get by chance among that many significant genes; above 1 is enrichment. Comparing a set's score across the fifteen members shows how much of the enrichment does not depend on the seed:

.. code-block:: python

   # Read every member's enrichment scores back, keeping track of
   # which member each row came from.
   dfs = []

   for name, result in results_gsea.items():

       # A member that failed has no outputs, so skip it - and know
       # that the spread below is over one member fewer.
       if result["status"] == "failed":
           continue

       df = pd.read_csv(result["outputs"]["e_scores"])

       df.insert(0, "member", name)

       dfs.append(df)

   df_e_scores = pd.concat(dfs, ignore_index = True)

   # The spread of each set's score across the members, per sample.
   df_e_scores.groupby(["sample", "genes_set"])["e_score"].describe()

.. note::

   The thresholds are **not** shared with the consensus, and their defaults differ: :func:`bulkdgd.analysis.dea.get_significant_genes` defaults to a ``log2_fold_change`` of ``2``, while the consensus defaults to ``1.0``. It also filters on ``p_val``, which the consensus ignores, and it compares inclusively (``<=``, ``>=``) where the consensus compares strictly (``<``, ``>``), so a gene exactly on a threshold is significant here but not called there. Pass the same thresholds to both if the two answers should refer to the same genes.

When to use the ensemble, and when one model is enough
------------------------------------------------------

The ensemble costs fifteen times a single run. Use it for:

* **A per-gene claim about a cohort that will be followed up**, such as a driver list, a signature, or a set of genes for experimental validation. One model's list is one draw from a distribution over training seeds, and a tier is how many independent draws agree. For the seed the *representation search* starts from, the calls several seeds agree on overlap a cancer-matched driver catalogue more often than the calls only one of them makes (see :ref:`rep_multiseed`).

* **Results to be published**, for the same reason: a result from one seed may not reproduce with another.

* **Comparing methods**, where a difference such as "the model found this and DESeq2 did not" must hold when the model is re-fitted.

A single model is the right choice when:

* **The result is per-sample.** The consensus is a statement about a group, drawn from recurrences within it, and says nothing about one sample. A per-sample p-value, an imputation, or a representation used as a feature is a single-model task.

* **The cohort has few samples per group.** A recurrence is a share of a group's samples: at ``0.20``, a group of five calls a gene "recurrent" based on a single sample, so with few samples per group a tier counts single samples, and a group of one has no recurrence. Check the group sizes before interpreting the tiers.

* **You are exploring**, while the analysis still changes and 27 GiB of decoders plus fifteen representation searches is a costly way to find, for instance, a preprocessing error.

The two can be combined: the representations are the expensive stage and the only one that needs a decoder, so an exploratory pass with ``BulkDGD()`` on one seed and a final pass with the full ensemble can share their preprocessing and their gene set.
