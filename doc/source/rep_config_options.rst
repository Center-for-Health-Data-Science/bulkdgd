Options used to configure the optimization scheme
=================================================

So far, bulkDGD implements two optimization schemes:

* ``one_opt``, which consists of only one round of optimization for the best representations found for the samples in latent space. The ``one_opt`` scheme is implemented in the YAML file ``bulkDGD/ioutil/configs/representations/one_opt.yaml``.

* ``two_opt``, which consists of two consecutive rounds of optimizations. Indeed, multiple candidate representations per sample are found, optimized, and the best one for each sample is picked from the pool. Then, a second round of optimization (similar to the one run under the ``one_opt`` scheme) is performed on these selected representations. The ``two_opt`` scheme is implemented in the YAML file ``bulkDGD/ioutil/configs/representations/two_opt.yaml``.

``one_opt`` scheme
------------------

When running :meth:`core.model.DGDModel.get_representations` with ``method`` set to ``one_opt``, the ``config`` argument should be a dictionary structured as follows:

.. code-block:: python
   
   {"optimization" : \
        
      {# Set the number of epochs the optimization should be run for.
       "epochs" : 60,
        
       # Set the optimizer to be used - so far, only 'adam'
       # has been implemented.
       "optimizer_name" : "adam",
        
       # Set the options to initialize the optimizer - they vary
       # according to the selected optimizer.
       "optimizer_options" : \

           {# Set the learning rate.
            lr: 0.01,

            # Set the weight decay.
            weight_decay: 0,

            # Set the betas.
            betas: [0.5, 0.9],
           },
      },
   }

``two_opt`` scheme
------------------

When running :meth:`core.model.DGDModel.get_representations` with ``method`` set to ``two_opt``, the ``config`` argument should be a dictionary structured as follows:

.. code-block:: python
   
   {"optimization" : \
        
      {# Set the options regarding the first optimization round.
       "opt1" : 
           
           {# Set the number of epochs the optimization should be run
            # for.
            "epochs" : 10,
           
            # Set the optimizer to be used - so far, only 'adam'
            # has been implemented.
            "optimizer_name" : "adam",
           
            # Set the options to initialize the optimizer - they
            # vary according to the selected optimizer.
            "optimizer_options" : \

               {# Set the learning rate.
                lr: 0.01,

                # Set the weight decay.
                weight_decay: 0,

                # Set the betas.
                betas: [0.5, 0.9],
               },
            },
       },

      {# Set the options regarding the second optimization round.
       "opt2" : 
           
           {# Set the number of epochs the optimization should be
            # run for.
            "epochs" : 50,
           
            # Set the optimizer to be used - so far, only 'adam'
            # has been implemented.
            "optimizer_name" : "adam",
           
            # Set the options to initialize the optimizer - they
            # vary according to the selected optimizer.
            "optimizer_options" : \

               {# Set the learning rate.
                lr: 0.01,

                # Set the weight decay.
                weight_decay: 0,

                # Set the betas.
                betas: [0.5, 0.9],
               },
            },
       },
   }
