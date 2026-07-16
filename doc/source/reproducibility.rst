``reproducibility`` - seeding a run
===================================

.. automodule:: bulkdgd.reproducibility

   .. autofunction:: bulkdgd.reproducibility.set_seeds

   .. autofunction:: bulkdgd.reproducibility.get_seeds_state

Seeding has to happen before the model is built
-----------------------------------------------

:func:`bulkdgd.reproducibility.set_seeds` must be called **before** :class:`bulkdgd.core.model.BulkDGD` is constructed, and not merely before it is trained.

Constructing the model is already random. :class:`torch.nn.Linear` draws its weights in its own ``__init__``, through ``reset_parameters``, so by the time ``BulkDGD(...)`` returns, every weight of the decoder is a number that has already been drawn from whatever state the generator happened to be in. Seeding afterwards seeds nothing that has already happened, and a run seeded that way looks seeded - the log says so - while reproducing nothing.

.. code-block:: python

   import bulkdgd
   from bulkdgd.core.model import BulkDGD

   # Right: the decoder's weights are drawn from a seeded generator.
   bulkdgd.set_seeds(42)
   model = BulkDGD(**config_model)

   # Wrong: the weights were drawn before the seed was set.
   model = BulkDGD(**config_model)
   bulkdgd.set_seeds(42)

Building a model also *consumes* the generator's stream, so anything drawn afterwards - the representations' initialization, the order of the batches, the noise added during training - continues from where the decoder left off. This is why the seed is set once, at the top, rather than before each thing that draws.

The same applies to :func:`torch.set_default_dtype`: a model built before the default dtype is changed is built in the old one.

What is seeded
--------------

:func:`bulkdgd.reproducibility.set_seeds` seeds :mod:`torch`'s global generator (the decoder's weights, the representations' initialization, the training noise, and the data loader's shuffling), the generators of every GPU, :mod:`numpy`'s global generator (which is what a ``random_state`` of ``None`` falls back on in anything scikit-learn-shaped), and the standard library's :mod:`random`.

It returns a record of what it seeded, which is meant to be written down beside the run's results. A seed that is set and not recorded is a seed nobody has.

A note on the Gaussian mixture model's ``random_state``
-------------------------------------------------------

``"random_state"`` in the model's ``"latent_options"`` (see :doc:`model_config_options`) reaches further than its name suggests when the latent space is a ``tgmm``: that implementation calls :func:`torch.manual_seed` with it when it places the mixture's components, which reseeds the global generator for everything that comes after. It is therefore usually better to leave it unset and to call :func:`bulkdgd.reproducibility.set_seeds` once, which makes the placement reproducible along with everything else.

What seeding does not fix
-------------------------

A seeded run repeats only if the operations it runs are themselves deterministic, and several are not. A sum reduced with atomics on a GPU adds its terms in whatever order its threads finish in, and floating-point addition is not associative, so the same numbers can add up to slightly different numbers. Training is chaotic: a difference in the last bits of an early epoch is a different model many epochs later.

``deterministic = True`` asks :mod:`torch` for the deterministic version of those operations, and makes it raise where there is not one. It is off by default because it is slower, and because raising is the right behaviour only when reproducibility is what is being asked for.

Note that ``CUBLAS_WORKSPACE_CONFIG`` has to be set in the environment before cuBLAS is initialized - which is to say before the first matrix multiplication, and not merely before :func:`bulkdgd.reproducibility.set_seeds` is called. :func:`bulkdgd.reproducibility.set_seeds` sets it if it is not already set, which is too late if something has already used cuBLAS. Set it in the environment of the job to be sure of it.
