core.outputmodules
==================

.. automodule:: bulkdgd.core.outputmodules

   .. autoclass:: bulkdgd.core.outputmodules.OutputModuleBase
      :members: __init__, input_dim, output_dim, activation

   .. autoclass:: bulkdgd.core.outputmodules.OutputModuleNBFeatureDispersion
      :members: __init__, input_dim, output_dim, activation, log_r, rescale, log_prob_mass, forward, log_prob, loss, sample

   .. autoclass:: bulkdgd.core.outputmodules.OutputModuleNBFullDispersion
      :members: __init__, input_dim, output_dim, activation, rescale, log_prob_mass, forward, log_prob, loss, sample

   .. autoclass:: bulkdgd.core.outputmodules.OutputModulePoisson
      :members: __init__, input_dim, output_dim, activation, rescale, log_prob_mass, forward, log_prob, loss, sample
