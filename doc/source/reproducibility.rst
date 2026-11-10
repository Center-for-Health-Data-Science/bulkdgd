``reproducibility`` - seeding a run
===================================

.. automodule:: bulkdgd.reproducibility

   .. autofunction:: bulkdgd.reproducibility.set_seeds

   .. autofunction:: bulkdgd.reproducibility.get_seeds_state

Seeding has to happen before the model is built
-----------------------------------------------

:func:`bulkdgd.reproducibility.set_seeds` must be called **before** :class:`bulkdgd.core.models.BulkDGD` is constructed, and not merely before it is trained: building the model draws the decoder's weights.

.. code-block:: python

   import bulkdgd
   from bulkdgd.core.models import BulkDGD

   # Right: the decoder's weights are drawn from a seeded generator.
   bulkdgd.set_seeds(42)
   model = BulkDGD(**config_model)

   # Wrong: the weights were drawn before the seed was set.
   model = BulkDGD(**config_model)
   bulkdgd.set_seeds(42)

Set the seed once, at the top, rather than before each step that draws.

A model built from fitted parameters (a bare ``BulkDGD()``, or a configuration naming a ``"decoder_pth_file"``) has its drawn weights overwritten by the checkpoint, but the seed still affects the search for a representation, so set it before the model is built in both cases.

What is seeded
--------------

:func:`bulkdgd.reproducibility.set_seeds` seeds :mod:`torch`'s global generator (the decoder's weights, the representations' initialization, the training noise, and the data loader's shuffling), the generators of every GPU, :mod:`numpy`'s global generator (which a ``random_state`` of ``None`` falls back on in scikit-learn-style estimators), and the standard library's :mod:`random`.

It returns a record of what it seeded. Save it with the run's results.

A note on the Gaussian mixture model's ``random_state``
-------------------------------------------------------

With a ``tgmm`` latent space, ``"random_state"`` in the model's ``"latent_options"`` (see :doc:`model_config_options`) reseeds the global generator when the mixture's components are placed. Leave it unset and call :func:`bulkdgd.reproducibility.set_seeds` once instead.

What seeding does not fix
-------------------------

A seeded run repeats only if the operations it runs are deterministic, and several are not.

``deterministic = True`` asks :mod:`torch` for the deterministic version of those operations, and makes it raise an error for operations that have none. It is slower, and off by default.

``CUBLAS_WORKSPACE_CONFIG`` must be set before the first matrix multiplication. With ``deterministic = True``, :func:`bulkdgd.reproducibility.set_seeds` sets it if it is not already set, which is too late if something has already used cuBLAS, so set it in the job's environment.
