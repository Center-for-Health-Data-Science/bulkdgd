Configuration for the optimization scheme
=========================================

So far, bulkDGD implements two optimization schemes:

* ``one_opt``, which consists of only one round of optimization for the best representations found for the samples in latent space. The ``one_opt`` scheme is implemented in the YAML file ``bulkDGD/configs/representations/one_opt.yaml``.

* ``two_opt``, which consists of two consecutive rounds of optimizations. Indeed, multiple candidate representations per sample are found, optimized, and the best one for each sample is picked from the pool. Then, a second round of optimization (similar to the one run under the ``one_opt`` scheme) is performed on these selected representations. The ``two_opt`` scheme is implemented in the YAML file ``bulkDGD/configs/representations/two_opt.yaml``.

``one_opt`` scheme
------------------

When running :meth:`core.model.BulkDGDModel.get_representations` the ``config`` argument should be a dictionary structured as follows to use the ``one_opt`` scheme:

.. code-block:: python
   
   {# Set the name of the optimization scheme the configuration refers
    # to.
    #
    # Type: str.
    #
    # Options:
    # - 'one_opt' for the optimization scheme with only one round of
    #   optimization.
    # - 'two_opt' for the optimization scheme with two rounds of
    #   optimization.
    "scheme" : "one_opt",

    # Set how many representations to initialize per sample per
    # component of the Gaussian mixture model.
    #
    # Type: int.
    "n_rep_per_comp" : 1,

    # Set the options for the data loader. They are passed to the
    # 'torch.utils.data.DataLoader' constructor.
    "data_loader" : \

      {# Set how many samples per batch to load.
       #
       # Type: int.
       "batch_size" : 8},

    # Set the options to output the loss.
    "loss" : \

      {# Set the options to output the GMM loss.
       "gmm" : \
         
         # Set the method used to normalize the loss when reporting it
         # per epoch.
         #
         # Type: str.
         #
         # Options:
         # - 'none' means that the loss will not be normalized.
         # - 'n_samples' means that the loss will be normalized by the
         #   number of samples.
         {"norm_method" : "n_samples"},

       # Set the options to output the reconstruction loss.
       "recon" : \
         
         # Set the method used to normalize the loss when reporting it
         # per epoch.
         #
         # Type: str.
         #
         # Options:
         # - 'none' means that the loss will not be normalized.
         # - 'n_samples' means that the loss will be normalized by the
         #   number of samples.
         {"norm_method" : "n_samples"},

       # Set the options to output the total loss.
       "total" : \

         # Set the method used to normalize the loss when reporting it
         # per epoch.
         #
         # Type: str.
         #
         # Options:
         # - 'none' means that the loss will not be normalized.
         # - 'n_samples' means that the loss will be normalized by the
         #   number of samples.
         {"norm_method" : "n_samples"}},

   # Set the options for the optimization.
   "opt":
        
      {# Set the number of epochs the optimization should be run for.
       #
       # Type: int.
       "epochs" : 60,

       # Set the options for the optimizer.
       "optimizer" : \
        
          {# Set the name of the optimizer to be used.
           #
           # Type: str.
           #
           # Options:
           # - 'adam' for the Adam optimizer.
           "optimizer_name" : "adam",
           
           # Set the options to initialize the optimizer - they vary
           # according to the selected optimizer.
           #
           # For the 'adam' optimizer, they will be passed to the
           # 'torch.optim.Adam' constructor.
           "optimizer_options" : \

               # Set these options if 'optimizer_name' is 'adam'.

               {# Set the learning rate.
                #
                # Type: float.
                "lr" : 0.01,

                # Set the weight decay.
                #
                # Type: float.
                "weight_decay" : 0.0,

                # Set the betas.
                #
                # Type: list of float.
                "betas" : [0.5, 0.9],
               },
          },
      },

   }

``two_opt`` scheme
------------------

When running :meth:`core.model.BulkDGDModel.get_representations` the ``config`` argument should be a dictionary structured as follows to use the ``two_opt`` scheme:

.. code-block:: python

   {# Set the name of the optimization scheme the configuration refers
    # to.
    #
    # Type: str.
    #
    # Options:
    # - 'one_opt' for the optimization scheme with only one round of
    #   optimization.
    # - 'two_opt' for the optimization scheme with two rounds of
    #   optimization.
    "scheme" : "two_opt",

    # Set how many representations to initialize per sample per
    # component of the Gaussian mixture model.
    #
    # Type: int.
    "n_rep_per_comp" : 1,

    # Set the options for the data loader. They are passed to the
    # 'torch.utils.data.DataLoader' constructor.
    "data_loader" : \

      {# Set how many samples per batch to load.
       #
       # Type: int.
       "batch_size" : 8},

    # Set the options to output the loss.
    "loss" : \

      {# Set the options to output the GMM loss.
       "gmm" : \
         
         # Set the method used to normalize the loss when reporting it
         # per epoch.
         #
         # Type: str.
         #
         # Options:
         # - 'none' means that the loss will not be normalized.
         # - 'n_samples' means that the loss will be normalized by the
         #   number of samples.
         {"norm_method" : "n_samples"},

       # Set the options to output the reconstruction loss.
       "recon" : \
         
         # Set the method used to normalize the loss when reporting it
         # per epoch.
         #
         # Type: str.
         #
         # Options:
         # - 'none' means that the loss will not be normalized.
         # - 'n_samples' means that the loss will be normalized by the
         #   number of samples.
         {"norm_method" : "n_samples"},

       # Set the options to output the total loss.
       "total" : \

         # Set the method used to normalize the loss when reporting it
         # per epoch.
         #
         # Type: str.
         #
         # Options:
         # - 'none' means that the loss will not be normalized.
         # - 'n_samples' means that the loss will be normalized by the
         #   number of samples.
         {"norm_method" : "n_samples"}},

   # Set the options for the first optimization.
   "opt1":
        
      {# Set the number of epochs the optimization should be run for.
       #
       # Type: int.
       "epochs" : 10,

       # Set the options for the optimizer.
       "optimizer" : \
        
          {# Set the name of the optimizer to be used.
           #
           # Type: str.
           #
           # Options:
           # - 'adam' for the Adam optimizer.
           "optimizer_name" : "adam",
           
           # Set the options to initialize the optimizer - they vary
           # according to the selected optimizer.
           #
           # For the 'adam' optimizer, they will be passed to the
           # 'torch.optim.Adam' constructor.
           "optimizer_options" : \

               # Set these options if 'optimizer_name' is 'adam'.

               {# Set the learning rate.
                #
                # Type: float.
                "lr" : 0.01,

                # Set the weight decay.
                #
                # Type: float.
                "weight_decay" : 0.0,

                # Set the betas.
                #
                # Type: list of float.
                "betas" : [0.5, 0.9],
               },
          },
      },

   # Set the options for the second optimization.
   "opt2":
        
      {# Set the number of epochs the optimization should be run for.
       #
       # Type: int.
       "epochs" : 50,

       # Set the options for the optimizer.
       "optimizer" : \
        
          {# Set the name of the optimizer to be used.
           #
           # Type: str.
           #
           # Options:
           # - 'adam' for the Adam optimizer.
           "optimizer_name" : "adam",
           
           # Set the options to initialize the optimizer - they vary
           # according to the selected optimizer.
           #
           # For the 'adam' optimizer, they will be passed to the
           # 'torch.optim.Adam' constructor.
           "optimizer_options" : \

               # Set these options if 'optimizer_name' is 'adam'.

               {# Set the learning rate.
                #
                # Type: float.
                "lr" : 0.01,

                # Set the weight decay.
                #
                # Type: float.
                "weight_decay" : 0.0,

                # Set the betas.
                #
                # Type: list of float.
                "betas" : [0.5, 0.9],
               },
          },
      },

   }     
     
