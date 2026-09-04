#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-

#    _templates.py
#
#    Templates for the different configurations.
#
#    Copyright (C) 2026 Valentina Sora 
#                       <sora.valentina1@gmail.com>
#
#    This program is free software: you can redistribute it and/or
#    modify it under the terms of the GNU General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public
#    License along with this program. 
#    If not, see <http://www.gnu.org/licenses/>.


#######################################################################


# Set the module's description.
__doc__ = "Templates for the different configurations."


#######################################################################


# Import from the standard library.
import copy

# Import from bulkdgd.
from bulkdgd import _internals
from bulkdgd import core
from . import metrics


#######################################################################


def _override_items(d, paths2values):
    """Return a copy of ``d`` with the value at the end of each "key
    path" REPLACED by the given value.

    Parameters
    ----------
    d : :class:`dict`
        The template, or template section, to copy.

    paths2values : :class:`dict`
        A mapping from "key path" - a tuple of keys leading to the key
        to be set - to the value that key should take.

    Returns
    -------
    :class:`dict`
        The copy, with the values replaced.
    """

    # Create a copy of the dictionary.
    new_d = copy.deepcopy(d)

    # For each key path and the value the key should take
    for key_path, value in paths2values.items():

        # Start at the top of the copy.
        current = new_d

        # Walk down to the dictionary holding the last key, creating
        # the intermediate levels if the template does not have them.
        for key in key_path[:-1]:

            current = current.setdefault(key, {})

        # Set the value, whether or not the key was already there.
        current[key_path[-1]] = value

    # Return the copy.
    return new_d


#######################################################################


# Set the template for the options of the 'lgmm' latent type in the
# model configuration.
_MODEL_LGMM_OPTIONS = {

    # The number of components in the Gaussian mixture model.     
    "n_components" : {
        
        "type" : (int,),
        "condition" : lambda v: v > 0,
        "message" : "must be a positive integer",
        "default" : 45,
        },

    # The type of covariance to use in the Gaussian mixture model.       
    "covariance_type" : {
        "type" : (str,),
        "choices" :  ["fixed",  "isotropic",  "diagonal"],
        "default" : "diagonal",
        },

    # The type of prior distribution to use for the means of the
    # Gaussian components in the Gaussian mixture model.
    "means_prior_type" : {
        "type" : (str,),
        "choices" :  ["softball"],
        "default" : "softball",
        },
    
    # The options for the prior distribution of the means of the
    # Gaussian components in the Gaussian mixture model.
    "means_prior_options" : {
        
        "switch" : {
            "option" : "means_prior_type",
            "cases" : {
                "softball" : {
                    "radius" : {
                        "type" : (float, int),
                        "condition" : lambda v: v > 0,
                        "message" : "must be a positive number",
                        },
                    "sharpness" : {
                        "type" : (float, int),
                        "condition" : lambda v: v > 0,
                        "message" : "must be a positive number"},
                        },
                    },
                },
            },

    # The type of prior distribution to use for the weights of the
    # Gaussian components in the Gaussian mixture model.       
    "weights_prior_type" : {
        "type" : (str,),
        "choices" : ["dirichlet"],
        "default" : "dirichlet",
        },

    # The options for the prior distribution of the weights of the
    # Gaussian components in the Gaussian mixture model.
    "weights_prior_options" : {
        "switch" : {
            "option" :
                "weights_prior_type",
            "cases" : {
                "dirichlet" : {
                    "alpha" : {
                        "type" : (float, int),
                        "condition" : lambda v: v > 0,
                        "message" : "must be a positive number",
                        },
                    },
                },
            },
        },

    # The type of prior distribution to use for the log variances of
    # the Gaussian components in the Gaussian mixture model.
    "log_var_prior_type" : {
        "type" : (str,),
        "choices" : ["gaussian"],
        "default" : "gaussian",
        },

    # The options for the prior distribution of the log variances of
    # the Gaussian components in the Gaussian mixture model.
    "log_var_prior_options" : {
        "switch" : {
            "option" : "log_var_prior_type",
            "cases" : {
                "gaussian" : {
                    "mean" : {
                        "type" : (float, int),
                        },
                    "stddev" : {
                        "type" : (float, int),
                        "condition" : lambda v: v > 0,
                        "message" : "must be a positive number",
                        },
                    },
                },
            },
        },
    }


#---------------------------------------------------------------------#


# Set the template for the options of the 'tgmm' latent type in the
# model configuration.
_MODEL_TGMM_OPTIONS = {
    
    # The number of components in the Gaussian mixture model.
    "n_components" : {
        "type" : (int,),
        "condition" : lambda v: v > 0,
        "message" :"must be a positive integer",
        "default" : 35,
        },

    # The type of covariance to use in the Gaussian mixture model.  
    "covariance_type" : {
        "type" : (str,),
        "choices" : core.latents.GaussianMixtureModelTGMM.COVARIANCE_TYPES,
        "default" : "spherical",
        },

    # The method to use for initializing the means of the Gaussian
    # components in the Gaussian mixture model.
    "init_means" : {
        "type" : (str,),
        "choices" : \
            core.latents.GaussianMixtureModelTGMM.INIT_MEANS_METHODS,
        "default" : "maxdist",
        },
    
    # The method to use for initializing the weights of the Gaussian
    # components in the Gaussian mixture model.
    "init_weights" : {
        "type" : (str,),
        "choices" : \
            core.latents.GaussianMixtureModelTGMM.INIT_WEIGHTS_METHODS,
        "default" : "uniform",
        },
    
    # The method to use for initializing the covariances of the
    # Gaussian components in the Gaussian mixture model.
    "init_covariances" : {
        "type" : (str,),
        "choices" : 
            core.latents.GaussianMixtureModelTGMM.INIT_COVARIANCES_METHODS,
        "default" : "empirical",
        },
    
    # The tolerance for convergence in the Gaussian mixture model.
    "tol" : {
        "type" : (float, int),
        "condition" : lambda v: v > 0,
        "message" : "must be a positive number",
        "default" : 1e-4,
        },
    
    # The regularization term.
    "reg_covar" : {
        "type" : (float, int),
        "condition" : lambda v: v >= 0,
        "message" : "must be a non-negative number",
        "default" : 1e-6,
        },
    
    # The number of initializations to perform.
    "n_init" : {
        "type" : (int,),
        "condition" : lambda v: v > 0,
        "message" : "must be a positive integer",
        "default" : 1},
    
    # The random state to use.
    "random_state" : {
        "type" : (int,),
        "default" : None,
        },
    
    # Whether to use the CEM algorithm for fitting the Gaussian mixture
    # model.
    "cem" : {
        "type" : (bool,),
        "default" : False,
        },
    }


#---------------------------------------------------------------------#


# Set the template for the options of the Gaussian mixture model
# fitted to the representations after training.
_MODEL_GMM_FINAL_OPTIONS = {

    # A model that does not ask for a final mixture is trained exactly
    # as before and writes no 'gmm_final.pth'.
    "__optional__" : True,

    # The type of covariance the final Gaussian mixture model should
    # have.
    "covariance_type" : {
        "type" : (str,),
        "choices" : \
            core.latents.GaussianMixtureModelTGMM.COVARIANCE_TYPES,
        },

    # How far each per-component covariance is pulled back towards the
    # one shared by all of them.
    "shrinkage" : {
        "type" : (float, int),
        "condition" : lambda v: 0.0 <= v <= 1.0,
        "message" : "must be between 0.0 and 1.0",
        "default" : 0.0,
        },

    # The value added to the diagonal of the covariance.
    "reg_covar" : {
        "type" : (float, int, type(None)),
        "condition" : lambda v: v >= 0,
        "message" : "must be a non-negative number",
        "default" : None,
        },

    }


#---------------------------------------------------------------------#


# Set the template for the decoder's options in the model
# configuration.
_MODEL_DECODER_OPTIONS = {

    # The number of units in the hidden layers of the decoder.
    "n_units_hidden_layers" : {
        "type" : (list,),
        "condition" :
            lambda v: len(v) > 0 and all(isinstance(n, int) \
                      and n > 0 for n in v),
        "message" : "must be a non-empty list of positive integers",
        },
    
    # The activation functions to use in the hidden layers of the
    # decoder.
    "activations" : {
        "type" : (list,),
        "choices" : core.decoders.Decoder.ACTIVATIONS,
        },
    
    # The type of normalization to use in the hidden layers of the
    # decoder.
    "dropout" : {
        "type" : (float, int),
        "condition" : lambda v: 0 <= v <= 1,
        "message" : "must be a float between 0 and 1",
        "default" : 0,
        },
    
    # The type of output module to use.
    "output_module_name" : {
        "type" : (str,),
        "choices" : list(core.outputmodules.OUTPUT_MODULES.keys()),
        },
    
    # The options for the output module.
    "output_module_options" : {
        "switch" : {
            "option": "decoder_options.output_module_name",
            "cases" : {
                "poisson" : {
                    "activation" : {
                        "type" : (str,),
                        "choices" : 
                            core.outputmodules.
                                 OutputModulePoisson.
                                    ACTIVATION_FUNCTIONS,
                        },
                    },
            
                "nb_feature_dispersion" : {
                    "activation" : {
                        "type" : (str,),
                        "choices" : 
                            core.outputmodules.
                                 OutputModuleNBFeatureDispersion.
                                    ACTIVATION_FUNCTIONS,
                        },
                    "r_init" : {
                        "type" : (float, int),
                        "condition" : lambda v: v > 0,
                        "message" : "must be a positive number",
                        "default" : 2,
                        },
                    },
            
                "nb_full_dispersion" : {

                    "activation" : {
                        "type" : (str,),
                        "choices" :
                            core.outputmodules.
                                OutputModuleNBFullDispersion.
                                    ACTIVATION_FUNCTIONS,
                        },
                    },

                },
            },
        },
    }


#---------------------------------------------------------------------#


# Set the template for the data loader's options in the training
# configuration.
_TRAIN_DATA_LOADER = {
    
    # The options for the data loader for training data.
    "train" : {
        "batch_size" : {
            "type": (int,),
            "condition": lambda v: v > 0,
            "message": "must be a positive integer",
            "default": 64,
            },
        "shuffle" : {
            "type": (bool,),
            "default": True,
            },
        },
    
    # The options for the data loader for test data.
    "test" : {
        "batch_size" : {
            "type": (int,),
            "condition": lambda v: v > 0,
            "message": "must be a positive integer",
            "default": 64,
            },
        "shuffle" : {
            "type": (bool,),
            "default": False,
            },
        },
    }


#---------------------------------------------------------------------#


# Set the template for the Adam optimizer's options.
_OPTIMIZER_ADAM = {
    
    # The learning rate for the optimizer.
    "lr" : {
        "type": (float, int),
        "condition": lambda v: v > 0,
        "message": "must be a positive number",
        "default": 0.001,
        },
    
    # The weight decay for the optimizer.
    "weight_decay" : {
        "type": (float, int),
        "condition": lambda v: v >= 0,
        "message": "must be non-negative",
        "default": 0.0,
        },
    
    # The beta parameters for the optimizer.
    "betas" : {
        "type": (list,),
        "condition": lambda v: len(v) == 2 and \
            all(isinstance(x, (float, int)) for x in v),
        "message": "must be a list of two floats",
        "default": [0.9, 0.999],
        },
    }


#---------------------------------------------------------------------#


# Set the template for the AdamW optimizer's options.
_OPTIMIZER_ADAMW = {
    
    # The learning rate for the optimizer.
    "lr" : {
        "type": (float, int),
        "condition": lambda v: v > 0,
        "message": "must be a positive number",
        "default": 0.001,
        },
    
    # The weight decay for the optimizer.
    "weight_decay" : {
        "type": (float, int),
        "condition": lambda v: v >= 0,
        "message": "must be non-negative",
        "default": 0.0,
        },
    
    # The beta parameters for the optimizer.
    "betas" : {
        "type": (list,),
        "condition": lambda v: len(v) == 2 and \
            all(isinstance(x, (float, int)) for x in v),
        "message": "must be a list of two floats",
        "default": [0.9, 0.999],
        },
    }


#---------------------------------------------------------------------#


# Set the template for the options of the L-BFGS optimizer.
_OPTIMIZER_LBFGS = {

    # The learning rate for the optimizer.
    "lr" : {
        "type": (float, int),
        "condition": lambda v: v > 0,
        "message": "must be a positive number",
        "default": 1.0,
        },

    # How many iterations the optimizer takes per step.
    "max_iter" : {
        "type": (int,),
        "condition": lambda v: v > 0,
        "message": "must be a positive integer",
        "default": 100,
        },

    # How many past updates are kept to approximate the curvature.
    "history_size" : {
        "type": (int,),
        "condition": lambda v: v > 0,
        "message": "must be a positive integer",
        "default": 20,
        },

    # The line search to use, or null for a fixed step.
    "line_search_fn" : {
        "type": (str, type(None)),
        "choices": ["strong_wolfe", None],
        "default": "strong_wolfe",
        },

    # The gradient below which the optimizer stops.
    "tolerance_grad" : {
        "type": (float, int),
        "condition": lambda v: v > 0,
        "message": "must be a positive number",
        "default": 1e-9,
        },

    # The change in the parameters below which it stops.
    "tolerance_change" : {
        "type": (float, int),
        "condition": lambda v: v > 0,
        "message": "must be a positive number",
        "default": 1e-12,
        },
    }


#---------------------------------------------------------------------#


# Set the template for the optimizer's options.
_OPTIMIZER =  {

    # The type of optimizer to use.
    "optimizer_type" : {
        "type": (str,),
        "choices": ["adam", "adamw", "lbfgs"],
        "default": "adamw"},

    # The options for the optimizer.
    "optimizer_options" : {
        "switch" : {
            "option" : "optimizer_type",
            "cases" : {
                "adam" : _OPTIMIZER_ADAM,
                "adamw" : _OPTIMIZER_ADAMW,
                "lbfgs" : _OPTIMIZER_LBFGS,
                },
            },
        },

    # The norm to which the gradients are clipped before the optimizer
    # takes its step.
    "grad_clipping_max_norm" : {
        "type": (float, int, type(None)),
        "condition": lambda v: v is None or v > 0,
        "message": "must be a positive number, or null to not clip",
        "default": None,
        },
    }


#---------------------------------------------------------------------#


# Set the template for the learning rate scheduler's options for the
# OneCycleLR scheduler.
_LR_SCHEDULER_ONE_CYCLE = {
    
    # The maximum learning rate for the learning rate scheduler.
    "max_lr" : {
        "type": (float, int),
        "condition": lambda v: v > 0,
        "message": "must be a positive number",
        "default": 0.01,
        },
    
    # The percentage of the cycle to use for increasing the learning
    # rate.
    "pct_start" : {
        "type": (float, int),
        "condition": lambda v: 0 <= v <= 1,
        "message": "must be a number between 0 and 1",
        "default": 0.25,
        },
    
    # The annealing strategy to use for the learning rate scheduler.
    "anneal_strategy" : {
        "type": (str,),
        "choices": ["cos", "linear"],
        "default": "cos",
        },
    
    # Whether to use momentum cycling in the learning rate scheduler. 
    "cycle_momentum" : {
        "type": (bool,),
        "default": True,
        },
    
    # The base momentum for the learning rate scheduler.
    "base_momentum" : {
        "type": (float, int),
        "condition": lambda v: 0 <= v <= 1,
        "message": "must be a number between 0 and 1",
        "default": 0.85,
        },
    
    # The maximum momentum for the learning rate scheduler.
    "max_momentum" : {
        "type": (float, int),
        "condition": lambda v: 0 <= v <= 1,
        "message": "must be a number  between 0 and 1",
        "default": 0.9,
        },
    
    # The division factor for the learning rate scheduler.
    "div_factor" : {
        "type": (float, int),
        "condition": lambda v: v > 0,
        "message": "must be a positive number",
        "default": 25.0,
        },
    
    # The final division factor for the learning rate scheduler.
    "final_div_factor" : {
        "type": (float, int),
        "condition": lambda v: v > 0,
        "message": "must be a positive number",
        "default": 1000.0,
        },
    
    # Whether to use the three-phase version of the learning rate
    # scheduler.
    "three_phase" : {
        "type": (bool,),
        "default": False,
        },
    }


#---------------------------------------------------------------------#


# Set the template for the learning rate scheduler's options for the
# CosineAnnealingLR scheduler.
_LR_SCHEDULER_COSINE = {

    # The minimum learning rate the schedule anneals down to.
    "eta_min" : {
        "type": (float, int),
        "condition": lambda v: v >= 0,
        "message": "must be a non-negative number",
        "default": 0.0,
        },
    }


#---------------------------------------------------------------------#


# Set the template for learning rate scheduler's options.
_LR_SCHEDULER = {

    # The type of learning rate scheduler to use.
    "lr_scheduler_type" : {
        "type": (str, type(None)),
        "choices": ["one_cycle", "cosine"],
        "default": None,
        },

    # The options for the learning rate scheduler.
    "lr_scheduler_options" : {
        "switch" : {
            "option" : "lr_scheduler_type",
            "cases" : {
                "one_cycle": _LR_SCHEDULER_ONE_CYCLE,
                "cosine": _LR_SCHEDULER_COSINE,
                },
            },
        },
    }


#---------------------------------------------------------------------#


# Set the template for the options for removing collapsed components
# in the training configuration.
_COMPONENTS_REMOVAL = {
    
    # The type of removal to use for the collapsed components.
    "components_removal_type" : {
        "type": (str, type(None)),
        "choices": ["weight_threshold"],
        "default": None,
        },
    
    # The options for the removal of the collapsed components.
    "components_removal_options" : {
        "switch" : {
            "option" : "components_removal_type",
            "cases" : {
                
                "weight_threshold" : {
                    "threshold" : {
                        "type": (float, int),
                        "condition": lambda v: v >= 0,
                        "message": "must be a non-negative number",
                        "default": 1e-8,
                        },
                    },
                },
            },
        },
    }


#---------------------------------------------------------------------#


# Set the template for the options of the 'tgmm' latent type in the
# training configuration.
_TRAIN_TGMM = {

    # The options for the calculation of the loss.
    "loss_calculation" : {
        "lambda" : {
            "type": (float, int),
            "condition": lambda v: v >= 0,
            "message": "must be a non-negative number",
            "default": 1.0,
            },
        },

    # The type of model selection to use for selecting the best model
    # during training.
    "model_selection_type" : {
        "type": (str, type(None)),
        "choices": ["metric"],
        "default": None,
        },
    
    # The options for the model selection to use for selecting the best
    # model during training.
    "model_selection_options" : {
        "switch" : {
            "option" : "model_selection_type",
            "cases" : {

                "metric" : {

                    "metric" : {
                        "type": (str,),
                        "choices": [
                            "bic",
                            "silhouette_score",
                            "calinski_harabasz_score",
                            "davies_bouldin_score"],
                        "default": "bic",
                        },

                    # How far from the current number of components to
                    # look.
                    "step" : {
                        "type": (int,),
                        "condition": lambda v: v >= 1,
                        "message": "must be a positive integer",
                        "default": 1,
                        },
                    },
                },
            },
        },

    # The epoch at which to start fitting the Gaussian mixture model
    # during training.
    "fitting" : {

        "first_epoch" : {
            "type": (int,),
            "condition": lambda v: v >= 0,
            "message": "must be a non-negative integer",
            "default": 25,
            },
        
        # Whether to refit the Gaussian mixture model at the end of
        # the training period.
        "refit_final" : {
            "type": (bool,),
            "default": True,
            },
        
        # The interval (in epochs) at which to refit the Gaussian
        # mixture model during training.
        "refit_interval" : {
            "type": (int,),
            "condition": lambda v: v >= 0,
            "message": "must be a non-negative integer",
            "default": 0,
            },
        
        # The maximum number of iterations for fitting the Gaussian
        # mixture model during the first epoch of fitting.
        "max_iter_first_epoch" : {
            "type": (int,),
            "condition": lambda v: v > 0,
            "message": "must be a positive integer",
            "default": 1000,
            },
        
        # The maximum number of iterations for fitting the Gaussian
        # mixture model during the epochs of refitting.
        "max_iter_full_refit" : {
            "type": (int,),
            "condition": lambda v: v > 0,
            "message": "must be a positive integer",
            "default": 100,
            },
        
        # The maximum number of iterations for fitting the Gaussian
        # mixture model during the epochs of refitting with warm
        # initialization.
        "max_iter_warm_refit" : {
            "type": (int,),
            "condition": lambda v: v > 0,
            "message": "must be a positive integer",
            "default": 100,
            },
        
        # The maximum number of iterations for fitting the Gaussian
        # mixture model during the final refitting.
        "max_iter_final_refit" : {
            "type": (int,),
            "condition": lambda v: v > 0,
            "message": "must be a positive integer",
            "default": 1000,
            },
        },

    # The options for removing collapsed components.
    **_COMPONENTS_REMOVAL,
    
    }


#---------------------------------------------------------------------#


# Set the template for the options of the 'lgmm' latent type in the
# training configuration.
_TRAIN_LGMM = {

    # The options for the optimizer used to train the latent space.
    **_internals.recursive_add_items(
        d = _OPTIMIZER,
        paths2values = {
            ("optimizer_options",
             "switch",
             "cases",
             "adam",
             "lr",
             "default") : 0.01,
            ("optimizer_options",
             "switch",
             "cases",
             "adamw",
             "lr",
             "default") : 0.01,
            }),

    # The options for the learning rate scheduler used to train the
    # latent space.
    **_LR_SCHEDULER,

    # The options for removing collapsed components.
    **_COMPONENTS_REMOVAL,

    }


#---------------------------------------------------------------------#


# Set the template for the options for the per-sample training
# diagnostics.
_TRAIN_DIAGNOSTICS = {

    "__optional__" : True,

    # Whether to record each sample's contribution to the decoder's
    # gradient norm, once per epoch.
    "per_sample_grad_norm" : {
        "type": (bool,),
        "default": False,
        },

    # Save the decoder's state every this many epochs, so that a
    # TracIn pass can be run offline afterwards. Zero disables it.
    "checkpoint_every" : {
        "type": (int,),
        "default": 0,
        },

    # Where the per-epoch records and the checkpoints are written,
    # relative to the model's working directory.
    "output_dir" : {
        "type": (str,),
        "default": "diagnostics",
        },
    }


# Set the template for the options for fitting the Gaussian mixture
# model that describes the latent space after training.
_TRAIN_GMM_FINAL = {

    # A training configuration that says nothing about the final
    # mixture is not asking for one.
    "__optional__" : True,

    # What the post-training refit is allowed to change:
    # 'covariance_only' refits only the covariance in a closed-form
    # M-step; 'full_em' also re-estimates the means and weights.
    "fit" : {
        "type" : (str,),
        "choices" : ["covariance_only", "full_em"],
        "default" : "covariance_only",
        },

    # The maximum number of iterations, used only by 'full_em'.
    "max_iter" : {
        "type" : (int,),
        "condition" : lambda v: v > 0,
        "message" : "must be a positive integer",
        "default" : 1000,
        },

    }


#---------------------------------------------------------------------#


# Set the template for the decoder training options.
_TRAIN_DECODER = {

    # The options for the optimizer used to train the decoder.
    **_override_items(
        d = _OPTIMIZER,
        paths2values = {
            ("optimizer_options",
             "switch",
             "cases",
             "adam",
             "lr",
             "default") : 0.01,
            ("optimizer_options",
             "switch",
             "cases",
             "adamw",
             "lr",
             "default") : 0.01,
            ("optimizer_options",
             "switch",
             "cases",
             "adamw",
             "weight_decay",
             "default") : 0.1,
            }),

    # The options for the learning rate scheduler used to train the
    # decoder.
    **_override_items(
        d = _LR_SCHEDULER,
        paths2values = {
            ("lr_scheduler_type",
             "default") : "one_cycle",
            }),

    }


#---------------------------------------------------------------------#


# Set the template for the representations training options.
_TRAIN_REPRESENTATIONS = {

    # The type of noise to add to the representations during training.
    "train_noise_type" : {
        "type": (str, type(None)),
        "choices": ["gaussian", "none"],
        "default": "gaussian",
        },

    # The options for the noise to add to the representations during
    # training.
    "train_noise_options" : {
        "switch" : {
            "option" : "train_noise_type",
            "cases" : {
                "gaussian" : {

                    # The base scale of the Gaussian noise.
                    "scale" : {
                        "type": (float, int),
                        "condition": lambda v: v >= 0,
                        "message": "must be a non-negative number",
                        "default": 0.1,
                        },

                    # The scale multiplier at the start of training.
                    "start" : {
                        "type": (float, int),
                        "condition": lambda v: v >= 0,
                        "message": "must be a non-negative number",
                        "default": 1.0,
                        },

                    # The scale multiplier at the end of training.
                    "end" : {
                        "type": (float, int),
                        "condition": lambda v: v >= 0,
                        "message": "must be a non-negative number",
                        "default": 0.01,
                        },

                    # The probability mass the noise keeps within the
                    # component's radius.
                    "within_radius_prob" : {
                        "type": (float, int),
                        "condition": lambda v: 0 <= v <= 1,
                        "message": "must be a number between 0 and 1",
                        "default": 0.95,
                        },

                    # The final multiplier on the noise.
                    "gain" : {
                        "type": (float, int),
                        "condition": lambda v: v >= 0,
                        "message": "must be a non-negative number",
                        "default": 4.0,
                        },
                    },
                },
            },
        },

    # The options for the optimizer used to train the representations.
    **_override_items(
        d = _OPTIMIZER,
        paths2values = {
            ("optimizer_options",
             "switch",
             "cases",
             "adam",
             "lr",
             "default") : 0.01,
            ("optimizer_options",
             "switch",
             "cases",
             "adamw",
             "lr",
             "default") : 0.01,
            }),

    # The options for the learning rate scheduler used to train the
    # representations.
    **_override_items(
        d = _LR_SCHEDULER,
        paths2values = {
            ("lr_scheduler_type",
             "default") : "one_cycle",
            }),

    }


#---------------------------------------------------------------------#


# Set the template for the loss options.
_LOSS_OPTIONS = {

    # The type of reduction to use for the loss.
    "reduction_type" : {
        "type": (str,),
        "choices": ["mean", "sum"],
        "default": "sum",
        },

    # The options for the normalization of the loss for the latent
    # space.
    "latent" : {
        "norm_type" : {
            "type": (str,),
            "choices": \
                ["none", "n_samples", "n_samples * latent_dim"],
            "default": "n_samples"},
        "lambda" : {
            "type": (float, int),
            "condition": lambda v: v >= 0,
            "message": "must be a non-negative number",
            "default": 1.0,
            },
        },

    # The options for the normalization of the loss for the decoder.
    "decoder" : {
        "norm_type" : {
            "type": (str,),
            "choices": \
                ["none", "n_samples", "n_samples * n_genes"],
            "default": "n_samples",
            },
        },

    # The options for the normalization of the loss for the total loss.
    "total" : {
        "norm_type" : {
            "type": (str,),
            "choices": \
                ["none", "n_samples", "n_samples * n_genes"],
            "default": "n_samples",
            },
        },
    }


#---------------------------------------------------------------------#


# Set the template for the reporting options.
_REPORTING_OPTIONS = {

    # The options for the loss.
    "loss" : {

        # The options for the normalization of the loss for the latent
        # space.
        "latent" : {
            "norm_type" : {
                "type": (str,),
                "choices": \
                    ["none", "n_samples", "n_samples * latent_dim"],
                "default": "n_samples",
                },
            },

        # The options for the normalization of the loss for the
        # decoder.
        "decoder" : {
            "norm_type" : {
                "type": (str,),
                "choices": \
                    ["none", "n_samples", "n_samples * n_genes"],
                "default": "n_samples",
                },
            },

        # The options for the normalization of the loss for the total
        # loss.
        "total" : {
            "norm_type" : {
                "type": (str,),
                "choices": \
                    ["none", "n_samples", "n_samples * n_genes"],
                "default": "n_samples",
                },
            },
        },

    # The options for the metrics to calculate during training.
    "metrics" : {

        # The options for the metrics to calculate for the latent
        # space.
        "latent" : {
            "type" : (list,),
            "choices" : [*list(metrics.UNSUPERVISED_METRICS.keys()),
                         *list(metrics.SUPERVISED_METRICS.keys())],
            "default" : ["bic",
                         "silhouette_score",
                         "davies_bouldin_score",
                         "calinski_harabasz_score",
                         "adjusted_rand_index_score",
                         "adjusted_mutual_info_score"],
            },
        },
    
    # The options for the optional outputs.
    "optional_outputs" : {

        # The options for the model to output at the end of each epoch
        # during training.
        "model_epoch" : {
            "enabled" : {
                "type": (bool,),
                "default": False,
                },
            "stride" : {
                "type": (int,),
                "condition": lambda v: v > 0,
                "message": "must be a positive integer",
                "default": 1,
                },
            "dir" : {
                "type": (str, type(None)),
                "default": None,
                },
            },

        # The options for the representations to output at the end of
        # each epoch during training.
        "representations_epoch" : {
            "enabled" : {
                "type": (bool,),
                "default": False,
                },
            "stride" : {
                "type": (int,),
                "condition": lambda v: v > 0,
                "message": "must be a positive integer",
                "default": 1,
                },
            "dir" : {
                "type": (str, type(None)),
                "default": None,
                },
            },
        
        # The options for the latent probabilities to output at the
        # end of each epoch during training.
        "latent_probs_epoch" : {
            "enabled" : {
                "type": (bool,),
                "default": False,
                },
            "stride" : {
                "type": (int,),
                "condition": lambda v: v > 0,
                "message": "must be a positive integer",
                "default": 1,
                },
            "dir" : {
                "type": (str, type(None)),
                "default": None,
                },
            },
        
        # The options for the latent means to output at the end of each
        # epoch during training.
        "latent_means_epoch" : {
            "enabled" : {
                "type": (bool,),
                "default": False,
                },
            "stride" : {
                "type": (int,),
                "condition": lambda v: v > 0,
                "message": "must be a positive integer",
                "default": 1,
                },
            "dir" : {
                "type": (str, type(None)),
                "default": None,
                },
            },
        
        # The options for the gene-level saliency maps to output at the
        # end of each epoch during training.
        "genes_saliency_maps_epoch" : {
            "enabled" : {
                "type": (bool,),
                "default": False,
                },
            "stride" : {
                "type": (int,),
                "condition": lambda v: v > 0,
                "message": "must be a positive integer",
                "default": 1,
                },
            "dir" : {
                "type": (str, type(None)),
                "default": None,
                },
            },
        
        # The options for the pathway-level saliency maps to output at
        # the end of each epoch during training.
        "pathways_saliency_maps_epoch" : {
            "enabled" : {
                "type": (bool,),
                "default": False,
                },
            "stride" : {
                "type": (int,),
                "condition": lambda v: v > 0,
                "message": "must be a positive integer",
                "default": 1,
                },
            "dir" : {
                "type": (str, type(None)),
                "default": None,
                },
            },
        },
    }


#---------------------------------------------------------------------#


# Set the template for the options for the optimizations.
_REP_OPTIMIZATION = {
    
    # The number of epochs for the optimization.
    "epochs" : {
        "type": (int,),
        "condition": lambda v: v > 0,
        "message": "must be a positive integer",
        "default": 50,
        },
    
    # Whether to use automatic learning rate for the optimization.
    "auto_lr" : {
        "type": (bool,),
        "default": False,
        },

    # The type of noise to add to the representations while they are
    # being optimized.
    "noise_type" : {
        "type": (str, type(None)),
        "choices": ["gaussian", "none"],
        "default": "none",
        },

    # The options for that noise, with the same meanings they have in
    # '_TRAIN_REPRESENTATIONS'.
    "noise_options" : {
        "switch" : {
            "option" : "noise_type",
            "cases" : {
                "gaussian" : {

                    "scale" : {
                        "type": (float, int),
                        "condition": lambda v: v >= 0,
                        "message": "must be a non-negative number",
                        "default": 0.0,
                        },

                    "start" : {
                        "type": (float, int),
                        "condition": lambda v: v >= 0,
                        "message": "must be a non-negative number",
                        "default": 1.0,
                        },

                    "end" : {
                        "type": (float, int),
                        "condition": lambda v: v >= 0,
                        "message": "must be a non-negative number",
                        "default": 0.01,
                        },

                    "within_radius_prob" : {
                        "type": (float, int),
                        "condition": lambda v: 0 <= v <= 1,
                        "message": "must be a number between 0 and 1",
                        "default": 0.95,
                        },

                    "gain" : {
                        "type": (float, int),
                        "condition": lambda v: v >= 0,
                        "message": "must be a non-negative number",
                        "default": 1.0,
                        },
                    },
                },
            },
        },

    # The options for the optimizer.
    **_override_items(
        d = _OPTIMIZER,
        paths2values = \
            {("optimizer_options",
                "switch",
                "cases",
                "adam",
                "lr",
                "default") : 0.01,
                ("optimizer_options",
                "switch",
                "cases",
                "adamw",
                "lr",
                "default") : 0.01,
            }),
    }


#---------------------------------------------------------------------#


# Set the template for the options of the optimizers in the 
# representations configuration for the 'two_opt' scheme when
# the latent space is the legacy Gaussian mixture model.
_REP_TWO_OPT_LGMM = {

    # How much of a sample the model is allowed to give up on when
    # finding a representation.
    "contamination" : {
        "type": (float, int),
        "condition": lambda v: 0.0 <= v < 1.0,
        "message": "must be in [0, 1)",
        "default": 0.0,
        },

    # The r-value of the outlier component of that mixture.
    "contamination_r" : {
        "type": (float, int),
        "condition": lambda v: v > 0,
        "message": "must be a positive number",
        "default": 0.05,
        },


    # The reduction method to use for the loss.
    "loss_reduction_type" : {
        "type": (str,),
        "choices": ["mean", "sum"],
        "default": "sum",
        },
    
    # A data-driven starting point for the search, replacing one of
    # the mixture draws. Optional, and absent by default.
    "warm_start" : {

        "__optional__" : True,

        # The fitted predictor, written by
        # 'warmstart.fit_from_model_dir'.
        "pth_file" : {
            "type": (str, type(None)),
            "default": None,
            },
        },

    # The options for the first optimization of the representations.
    "optimization_1" : \
        _internals.recursive_add_items(
            d = _REP_OPTIMIZATION,
            paths2values = \
                {("epochs",
                  "default") : 10}),

    # The options for the second optimization of the representations.
    "optimization_2" : _REP_OPTIMIZATION,
    
    }

#---------------------------------------------------------------------#


# Set the template for the options of the optimizers in the
# representations configuration for the 'two_opt' scheme when
# the latent space is the TorchGMM wrapper.
_REP_TWO_OPT_TGMM = {

    # How much of a sample the model is allowed to give up on when
    # finding a representation.
    "contamination" : {
        "type": (float, int),
        "condition": lambda v: 0.0 <= v < 1.0,
        "message": "must be in [0, 1)",
        "default": 0.0,
        },

    # The r-value of the outlier component of that mixture.
    "contamination_r" : {
        "type": (float, int),
        "condition": lambda v: v > 0,
        "message": "must be a positive number",
        "default": 0.05,
        },


    # The reduction method to use for the loss.
    "loss_reduction_type" : {
        "type": (str,),
        "choices": ["mean", "sum"],
        "default": "sum",
        },

    # The options for calculating the loss of the latent space.
    "latent_loss_calculation" : {
        "lambda" : {
            "type": (float, int),
            "condition": lambda v: v >= 0,
            "message": "must be a non-negative number",
            "default": 1.0,
            },
        },

    # A data-driven starting point for the search, replacing one of
    # the mixture draws.
    "warm_start" : {

        "__optional__" : True,

        # The fitted predictor, written by
        # 'warmstart.fit_from_model_dir'.
        "pth_file" : {
            "type": (str, type(None)),
            "default": None,
            },
        },

    # The options for the first optimization of the representations.
    "optimization_1" : \
        _override_items(
            d = _REP_OPTIMIZATION,
            paths2values = \
                {("epochs",
                  "default") : 300}),

    # The options for the second optimization of the representations.

    "optimization_2" : \
        _override_items(
            d = _REP_OPTIMIZATION,
            paths2values = \
                {("epochs",
                  "default") : 500}),

    }


#######################################################################
 

# Set the template for the model's configuration.
CONFIG_MODEL = {
    
    # The path to the file containing the genes to use for the model.
    "genes_txt_file" : {
        "type" : (str,),
        },
    
    # The dimension of the latent space.
    "latent_dim" : {
        "type" : (int,),
        "condition" : lambda v: v > 0,
        "message" : "must be a positive integer",
        "default" : 64,
        },
    
    # The type of latent space to use in the model.
    "latent_type" : {
        "type" : (str,),
        "choices" : ["lgmm", "tgmm"],
        "default" : "tgmm",
        },
    
    # The options for the latent space in the model.
    "latent_options" : {
        "switch" : {
            "option" : "latent_type",
            "cases" : {
                "lgmm" : _MODEL_LGMM_OPTIONS,
                "tgmm" : _MODEL_TGMM_OPTIONS,
                },
            },
        },
    
    # The options for the Gaussian mixture model fitted to the
    # representations after training.
    "gmm_final" : _MODEL_GMM_FINAL_OPTIONS,

    # The options for the decoder in the model.
    "decoder_options" : _MODEL_DECODER_OPTIONS,

    # How the scaling factor of a sample is computed - the number the
    # decoder's predicted means are multiplied by to put them on the
    # scale of the sample's own counts.
    "scaling_factor" : {
        "type" : (str,),
        "choices" : ["mean", "median"],
        "default" : "mean",
        },

    # The precision the model's parameters are built in.
    "dtype" : {
        "type" : (str,),
        "choices" : ["float32", "float64"],
        "default" : "float64",
        },

    }


# Set the template for the training configuration.
CONFIG_TRAIN = {
    
    # The number of epochs for training the model.
    "n_epochs" : {
        "type": (int,),
        "condition": lambda v: v > 0,
        "message": "must be a positive integer",
        "default": 200,
        },
    
    # The reduction method to use for the loss.
    "loss_reduction_type" : {
        "type": (str,),
        "choices": ["mean", "sum"],
        "default": "sum",
        },

    # The options for the data loaders for training and test data.
    "data_loader_options" : _TRAIN_DATA_LOADER,

    # The options for reporting during training.
    "reporting_options" : _REPORTING_OPTIONS,
    
    # The type of latent space used in the model.
    "latent_type" : {
        "type": (str,),
        "choices": ["lgmm", "tgmm"],
        "default": "tgmm",
        },

    # The options for the latent space in the training configuration.
    "latent_training_options" : {
        "switch" : {
            "option" : "latent_type",
            "cases" : {
                "lgmm" : _TRAIN_LGMM,
                "tgmm" : _TRAIN_TGMM,
                },
            },
        },
    
    # The options for fitting the Gaussian mixture model that is
    # fitted to the representations after training.
    "gmm_final_training_options" : _TRAIN_GMM_FINAL,

    # The options for the per-sample training diagnostics. Optional.
    "training_diagnostics" : _TRAIN_DIAGNOSTICS,

    # Where to write the per-epoch learning rates, indexed by epoch.
    "output_lr_file" : {
        "type": (str, type(None)),
        "default": None,
        },

    # The options for the decoder in the training configuration.
    "decoder_training_options" : _TRAIN_DECODER,

    # The options for the representations in the training
    # configuration.
    "representations_training_options" : _TRAIN_REPRESENTATIONS,

    # The type of early stopping to use during training.
    "early_stopping_type" : {
        "type": (str, type(None)),
        "choices": ["loss"],
        "default": None,
        },
    
    # The options for early stopping during training.
    "early_stopping_options" : {
        "patience" : {
            "type": (int,),
            "condition": lambda v: v > 0,
            "message": "must be a positive integer",
            "default": 10},
        },
    }


# Set the template for the configuration to find the representations
# for a new set of samples.
CONFIG_REP = {
    
    # The type of scheme to use for finding the representations for a
    # new set of samples.
    "scheme_type" : {
        "type": (str,),
        "choices": ["two_opt", "two_opt_multiseed"],
        "default": "two_opt",
        },

    # The type of latent space used in the model.
    "latent_type" : {
        "type": (str,),
        "choices": ["lgmm", "tgmm"],
        "default": "tgmm",
        },

    # The number of initial representations to sample per component of
    # the latent space.
    "n_rep_per_comp" : {
        "type": (int,),
        "condition": lambda v: v > 0,
        "message": "must be a positive integer",
        "default": 1,
        },

    # The options for the data loader for the new set of samples.
    "data_loader_options" : {
        "batch_size" : {
            "type": (int,),
            "condition": lambda v: v > 0,
            "message": "must be a positive integer",
            "default": 16,
            },
        "shuffle" : {
            "type": (bool,),
            "default": False,
            },
        },

    # The options for reporting during the optimization(s).
    "reporting_options" : {
        "loss" : _LOSS_OPTIONS,
        },
    
    # The options for the specific optimization scheme.
    "scheme_options" : {
        "switch" : {
            "option" : "scheme_type",
            "cases" : {
                "two_opt" : {
                    "switch" : {
                        "option" : "latent_type",
                        "cases" : {
                            "lgmm" : _REP_TWO_OPT_LGMM,
                            "tgmm" : _REP_TWO_OPT_TGMM,
                            },
                        },
                    },

                # The multi-seed scheme takes the same options as the
                # scheme it runs.
                "two_opt_multiseed" : {
                    "switch" : {
                        "option" : "latent_type",
                        "cases" : {
                            "lgmm" : _REP_TWO_OPT_LGMM,
                            "tgmm" : _REP_TWO_OPT_TGMM,
                            },
                        },
                    },
                },
            },
        },
    }