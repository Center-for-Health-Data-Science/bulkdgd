The core subpackage
===================

core.dataclasses
----------------

.. autoclass:: core.dataclasses.GeneExpressionDataset
   :members: __init__, df, n_samples, n_genes, mean_exp

core.decoder
------------

.. autoclass:: core.decoder.NBLayer
   :members: __init__, dim, log_r, activation, log_prob_mass, rescale, forward, loss, log_prob, sample

.. autoclass:: core.decoder.Decoder
   :members: __init__, forward

core.latent
-----------

.. autoclass:: core.latent.GaussianMixtureModel
   :members: __init__, dim, n_comp, log_var_params, alpha, dirichlet_constant, weight, mean_prior, mean, log_var, log_var_dim, log_var_factor, log_var_prior, get_prior_log_prob, forward, log_prob, get_mixture_probs, get_distribution, sample, component_sample, sample_probs, sample_new_points

.. autoclass:: core.latent.RepresentationLayer
   :members: __init__, n_samples, dim, mean, stddev, z, forward, rescale

core.priors
-----------

.. autoclass:: core.priors.SoftballPrior
   :members: __init__, dim, radius, sharpness, sample, log_prob

.. autoclass:: core.priors.GaussianPrior
   :members: __init__, dim, mean, stddev, dist, sample, log_prob
