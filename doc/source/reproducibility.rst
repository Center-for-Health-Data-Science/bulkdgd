``reproducibility`` - seeding a run
===================================

.. automodule:: bulkdgd.reproducibility

   .. autofunction:: bulkdgd.reproducibility.set_seeds

   .. autofunction:: bulkdgd.reproducibility.get_seeds_state

Seeding has to happen before the model is built
-----------------------------------------------

:func:`bulkdgd.reproducibility.set_seeds` must be called **before** :class:`bulkdgd.core.model.BulkDGD` is constructed, and not merely before it is trained.

Constructing the model is already random. :class:`torch.nn.Linear` draws its weights in its own ``__init__``, through ``reset_parameters``, so by the time ``BulkDGD(...)`` returns, every weight of the decoder has already been drawn from the generator's current state. Seeding afterwards does not affect those weights: the log reports the run as seeded, but the run does not reproduce.

.. code-block:: python

   import bulkdgd
   from bulkdgd.core.model import BulkDGD

   # Right: the decoder's weights are drawn from a seeded generator.
   bulkdgd.set_seeds(42)
   model = BulkDGD(**config_model)

   # Wrong: the weights were drawn before the seed was set.
   model = BulkDGD(**config_model)
   bulkdgd.set_seeds(42)

Building a model also *consumes* the generator's stream, so anything drawn afterwards (the representations' initialization, the order of the batches, the noise added during training) continues from where the decoder left off. Set the seed once, at the top, rather than before each step that draws.

This applies to a model built from an architecture. A model built from fitted parameters (a bare ``BulkDGD()``, or a configuration naming a ``"decoder_pth_file"``) has its drawn weights overwritten by the checkpoint, so the seed does not affect them. The seed still affects everything drawn afterwards, which for such a model is the search for a representation, so set it before the model is built in both cases.

The same applies to :func:`torch.set_default_dtype`: a model built before the default dtype is changed is built in the old one.

What is seeded
--------------

:func:`bulkdgd.reproducibility.set_seeds` seeds :mod:`torch`'s global generator (the decoder's weights, the representations' initialization, the training noise, and the data loader's shuffling), the generators of every GPU, :mod:`numpy`'s global generator (which a ``random_state`` of ``None`` falls back on in scikit-learn-style estimators), and the standard library's :mod:`random`.

It returns a record of what it seeded. Save it with the run's results.

A note on the Gaussian mixture model's ``random_state``
-------------------------------------------------------

``"random_state"`` in the model's ``"latent_options"`` (see :doc:`model_config_options`) affects more than the mixture when the latent space is a ``tgmm``: that implementation calls :func:`torch.manual_seed` with it when it places the mixture's components, which reseeds the global generator for everything that comes after. It is usually better to leave it unset and call :func:`bulkdgd.reproducibility.set_seeds` once, which makes the placement reproducible along with everything else.

What seeding does not fix
-------------------------

A seeded run repeats only if the operations it runs are deterministic, and several are not. A sum reduced with atomics on a GPU adds its terms in the order its threads finish, and floating-point addition is not associative, so the same numbers can sum to slightly different results. Training is chaotic: a difference in the last bits of an early epoch gives a different model many epochs later.

``deterministic = True`` asks :mod:`torch` for the deterministic version of those operations, and makes it raise an error for operations that have none. It is off by default because it is slower, and because the error is only useful when reproducibility is required.

``CUBLAS_WORKSPACE_CONFIG`` must be set in the environment before cuBLAS is initialized, that is, before the first matrix multiplication, not merely before :func:`bulkdgd.reproducibility.set_seeds` is called. With ``deterministic = True``, :func:`bulkdgd.reproducibility.set_seeds` sets it if it is not already set, which is too late if something has already used cuBLAS. Set it in the job's environment instead.
