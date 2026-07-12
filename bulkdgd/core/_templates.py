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


# Import from bulkdgd.
from bulkdgd import _internals
from bulkdgd import core
from . import metrics


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
        "default": (0.9, 0.999),
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
        "default": (0.9, 0.999),
        },
    }


#---------------------------------------------------------------------#


# Set the template for the optimizer's options.
_OPTIMIZER =  {

    # The type of optimizer to use.
    "optimizer_type" : {
        "type": (str,),
        "choices": ["adam", "adamw"],
        "default": "adamw"},

    # The options for the optimizer.
    "optimizer_options" : {
        "switch" : {
            "option" : "optimizer_type",
            "cases" : {
                "adam" : _OPTIMIZER_ADAM,
                "adamw" : _OPTIMIZER_ADAMW
                },
            },
        },

    # The norm to which the gradients are clipped before the optimizer
    # takes its step.
    #
    # If not set, the gradients are not clipped, which is what happened
    # before this option existed.
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


# Set the template for learning rate scheduler's options.
_LR_SCHEDULER = {
    
    # The type of learning rate scheduler to use.
    "lr_scheduler_type" : {
        "type": (str, type(None)),
        "choices": ["one_cycle"],
        "default": None,
        },
    
    # The options for the learning rate scheduler.
    "lr_scheduler_options" : {
        "switch" : {
            "option" : "lr_scheduler_type",
            "cases" : {
                "one_cycle": _LR_SCHEDULER_ONE_CYCLE,
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
                    "type": (str,),
                    "choices": [
                        "bic",
                        "silhouette_score",
                        "calinski_harabasz_score",
                        "davies_bouldin_score"],
                    "default": "bic",
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


# Set the template for the decoder training options.
_TRAIN_DECODER = {

    # The options for the optimizer used to train the decoder.
    **_internals.recursive_add_items(
        d = _OPTIMIZER,
        paths2values = {
            ("optimizer_options",
             "switch",
             "cases",
             "adam",
             "lr",
             "default") : 0.001,
            ("optimizer_options",
             "switch",
             "cases",
             "adamw",
             "lr",
             "default") : 0.001,
            }),

    # The options for the learning rate scheduler used to train the
    # decoder.
    **_LR_SCHEDULER,
    
    }


#---------------------------------------------------------------------#


# Set the template for the representations training options.
_TRAIN_REPRESENTATIONS = {

    # The type of noise to add to the representations during training.
    "train_noise_type" : {
        "type": (str, type(None)),
        "choices": ["gaussian"],
        "default": "none",
        },

    # The options for the noise to add to the representations during
    # training.
    "train_noise_options" : {
        "switch" : {
            "option" : "train_noise_type",
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

    # The options for the optimizer used to train the representations.
    **_internals.recursive_add_items(
        d = _OPTIMIZER,
        paths2values = {
            ("optimizer_options",
             "switch",
             "cases",
             "adam",
             "lr",
             "default") : 0.001,
            ("optimizer_options",
             "switch",
             "cases",
             "adamw",
             "lr",
             "default") : 0.001,
            }),

    # The options for the learning rate scheduler used to train the
    # representations.
    **_LR_SCHEDULER,
    
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
            "default": "none"},
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
            "default": "none",
            },
        },
    
    # The options for the normalization of the loss for the total loss.
    "total" : {
        "norm_type" : {
            "type": (str,),
            "choices": \
                ["none", "n_samples", "n_samples * n_genes"],
            "default": "none",
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
                "default": "none",
                },
            },
        
        # The options for the normalization of the loss for the
        # decoder.
        "decoder" : {
            "norm_type" : {
                "type": (str,),
                "choices": \
                    ["none", "n_samples", "n_samples * n_genes"],
                "default": "none",
                },
            },
        
        # The options for the normalization of the loss for the total
        # loss.
        "total" : {
            "norm_type" : {
                "type": (str,),
                "choices": \
                    ["none", "n_samples", "n_samples * n_genes"],
                "default": "none",
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
            "default" : ["silhouette_score"],
            },
        },
    
    # The options for the optional outputs.
    "optional_outputs" : {

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
    
    # The options for the optimizer (spread flat, matching how
    # 'model.BulkDGD._get_representations_one_opt' and
    # '_get_representations_two_opt' actually read 'optimizer_type'/
    # 'optimizer_options', and matching '_TRAIN_DECODER''s pattern --
    # not nested under an 'optimizer' key).
    **_internals.recursive_add_items(
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
# representations configuration for the 'one_opt' scheme when
# the latent space is the legacy Gaussian mixture model.
_REP_ONE_OPT_LGMM = {

    # The reduction method to use for the loss.
    "loss_reduction_type" : {
        "type": (str,),
        "choices": ["mean", "sum"],
        "default": "sum",
        },

    # The options for the optimization of the representations.
    "optimization" : _REP_OPTIMIZATION,
    
    }


# Set the template for the options of the optimizers in the 
# representations configuration for the 'one_opt' scheme when
# the latent space is the TorchGMM wrapper.
_REP_ONE_OPT_TGMM = {

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

    # The options for the optimization of the representations.
    "optimization" : _REP_OPTIMIZATION,

    }

#---------------------------------------------------------------------#


# Set the template for the options of the optimizers in the 
# representations configuration for the 'two_opt' scheme when
# the latent space is the legacy Gaussian mixture model.
_REP_TWO_OPT_LGMM = {

    # The reduction method to use for the loss.
    "loss_reduction_type" : {
        "type": (str,),
        "choices": ["mean", "sum"],
        "default": "sum",
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
    
    # The options for the decoder in the model.
    "decoder_options" : _MODEL_DECODER_OPTIONS,

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
        "choices": ["lgmm", "tgmm"]
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
        "choices": ["one_opt", "two_opt"],
        },

    # The type of latent space used in the model.
    "latent_type" : {
        "type": (str,),
        "choices": ["lgmm", "tgmm"],
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
            "default": 128,
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
                "one_opt" : {
                    "switch" : {
                        "option" : "latent_type",
                        "cases" : {
                            "lgmm" : _REP_ONE_OPT_LGMM,
                            "tgmm" : _REP_ONE_OPT_TGMM,
                            },
                        },
                    },
                "two_opt" : {
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