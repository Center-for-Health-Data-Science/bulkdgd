The core subpackage
===================

core.dataclasses
----------------

.. autoclass:: core.dataclasses.GeneExpressionDataset
   :members: __init__, df, n_samples, n_genes, mean_exp

core.decoder
------------

.. autoclass:: core.decoder.NBLayer
   :members: __init__, dim, scaling_type, log_r, activation, reduction, forward, log_prob_mass, rescale, loss, log_prob, sample
.. autoclass:: core.decoder.Decoder
   :members: __init__, forward

core.latent
-----------

.. autoclass:: core.latent.GaussianMixtureModel
   :members: __init__, dim, n_comp, softball_params, logbeta_params, alpha, dirichlet_constant, weight, mean_prior, mean, logbeta, logbeta_dim, logbeta_factor, logbeta_prior, get_prior_log_prob, forward, log_prob, get_mixture_probs, get_distribution, sample, component_sample, sample_probs, sample_new_points
.. autoclass:: core.latent.RepresentationLayer
   :members: __init__, n_samples, dim, mean, stddev, z, forward, rescale

core.priors
-----------

.. autoclass:: core.priors.softball
   :members: __init__, dim, radius, sharpness, sample, log_prob
.. autoclass:: core.priors.gaussian
   :members: __init__, dim, mean, stddev, dist, sample, log_prob
