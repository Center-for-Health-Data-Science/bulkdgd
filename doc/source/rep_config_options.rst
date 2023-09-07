Options used to configure the optimization scheme
=================================================

So far, bulkDGD implements two optimization schemes:

* ``one_opt``, which consists of only one round of optimization for the best representations found for the samples in latent space.

* ``two_opt``, which consists of two consecutive rounds of optimizations. Indeed, multiple candidate representations per sample are found, optimized, and the best one for each sample is picked from the pool. Then, a second round of optimization (similar to the one run under the ``one_opt`` scheme) is performed on these selected representations.

``one_opt`` scheme
------------------

When running :meth:`core.model.DGDModel.get_representations` with ``method`` set to ``one_opt``, the ``config`` argument should be a dictionary structured as follows:

.. code-block:: python
   
   {"optimization" : \
        
      {# Number of epochs the optimization should be run for
       "epochs" : 60,
        
       # Name of the optimizer to be used - so far, only 'adam'
       # has been implemented
       "optimizer_name" : "adam",
        
       # Options to set up the optimizer - they can vary
       # according to the selected optimizer
       "optimizer_options" : \

           {# Learning rate
            lr: 0.01,

            # Weight decay
            weight_decay: 0,

            # Betas
            betas: [0.5, 0.9],
           },
      },
   }

``two_opt`` scheme
------------------

When running :meth:`core.model.DGDModel.get_representations` with ``method`` set to ``two_opt``, the ``config`` argument should be a dictionary structured as follows:

.. code-block:: python
   
   {"optimization" : \
        
      {# Options regarding the first optimization round
       "opt1" : 
           
           {# Number of epochs the optimization should be run for
            "epochs" : 10,
           
            # Name of the optimizer to be used - so far, only 'adam'
            # has been implemented
            "optimizer_name" : "adam",
           
            # Options to set up the optimizer - they can vary
            # according to the selected optimizer
            "optimizer_options" : \

               {# Learning rate
                lr: 0.01,

                # Weight decay
                weight_decay: 0,

                # Betas
                betas: [0.5, 0.9],
               },
            },
       },

      {# Options regarding the second optimization round
       "opt2" : 
           
           {# Number of epochs the optimization should be run for
            "epochs" : 50,
           
            # Name of the optimizer to be used - so far, only 'adam'
            # has been implemented
            "optimizer_name" : "adam",
           
            # Options to set up the optimizer - they can vary
            # according to the selected optimizer
            "optimizer_options" : \

               {# Learning rate
                lr: 0.01,

                # Weight decay
                weight_decay: 0,

                # Betas
                betas: [0.5, 0.9],
               },
            },
       },
   }