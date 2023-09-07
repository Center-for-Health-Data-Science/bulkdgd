core.decoder
============

.. automodule:: core.decoder

   .. rubric:: Classes

   .. autosummary::
   
      Decoder
      NBLayer

   .. autoclass:: core.decoder.Decoder
      :members: __init__, forward

   .. autoclass:: core.decoder.NBLayer
      :members: __init__, dim, log_r, activation, log_prob_mass, rescale, forward, log_prob, loss, sample