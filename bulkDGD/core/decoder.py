#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-

#    decoder.py
#
#    This module contains the classes defining the decoder
#    (:class:`core.decoder.Decoder`).
#
#    The code was originally developed by Viktoria Schuster,
#    Inigo Prada Luengo, and Anders Krogh.
#    
#    Valentina Sora modified and complemented it for the purposes
#    of this package.
#
#    Copyright (C) 2024 Valentina Sora 
#                       <sora.valentina1@gmail.com>
#                       Viktoria Schuster
#                       <viktoria.schuster@sund.ku.dk>
#                       Inigo Prada Luengo
#                       <inlu@diku.dk>
#                       Anders Krogh
#                       <akrogh@di.ku.dk>
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
__doc__ = \
    "This module contains the classes defining the decoder " \
    "(:class:`core.decoder.Decoder`)."


#######################################################################


# Import from the standard library.
import logging as log
# Import from third-party packages.
import torch
import torch.nn as nn
# Import from 'bulkDGD'.
from . import outputmodules


#######################################################################


# Get the module's logger.
logger = log.getLogger(__name__)


########################## PUBLIC FUNCTIONS ###########################


def reshape_scaling_factors(x,
                            out_dim):
    """Reshape the scaling factor (a tensor).

    Parameters
    ----------
    x : ``torch.Tensor``
        The input tensor to be reshaped.

    out_dim : ``int``
        The dimensionality of the output tensor.

    Returns
    -------
    x : ``torch.Tensor``
        The reshaped tensor.
    """
    
    # Get the dimensionality of the input tensor
    start_dim = len(x.shape)

    # For each extra dimension that the output tensor needs to
    # have with respect to the input tensor
    for i in range(out_dim - start_dim):

        # Add a singleton dimension
        x = x.unsqueeze(1)
    
    # Return the reshaped tensor
    return x


class Decoder(nn.Module):
    
    """
    Class implementing the decoder.
    """


    ######################## CLASS ATTRIBUTES #########################


    # Set the names of the available output modules.
    OUTPUT_MODULES = \
        ["nb_feature_dispersion", "nb_full_dispersion"]


    ######################### INITIALIZATION ##########################


    def __init__(self,
                 n_units_input_layer,
                 n_units_hidden_layers,
                 output_module_name,
                 output_module_options):
        """Initialize an instance of the neural network representing
        the decoder.

        Parameters
        ----------
        n_units_input_layer : ``int``
            The mumber of neurons in the input layer.

        n_units_hidden_layers : ``list``
            The number of units in each of the hidden layers. As many
            hidden layers as the number of items in the list will be
            created.

        output_module_name : ``str``, {``"nb_feature_dispersion"``, \
            ``"nb_full_dispersion"``}
            The name of the output module that will be set. Available
            output modules are:

            - ``"nb_feature_dispersion"`` for negative binomial
              distributions with means learned per gene per sample and
              r-values learned per gene.

            - ``"nb_full_dispersion"`` for negative binomial
              distributions with both means and r-values learned per
              gene per sample.

        output_module_options : ``dict``
            A dictionary of options for setting up the output module.

            For the ``"nb_feature_dispersion"`` output module, the
            following options must be provided:

            - ``"output_dim"``: the dimensionality of each output 
              layer of the output module.

            - ``"activation"``: the name of the activation function to
              be used in the output module.

            - ``"r_init"``: the initial r-value for the negative
               binomial distributions modeling the genes' counts.

            For the ``"nb_full_dispersion"`` output module, the
            following options must be provided:

            - ``"output_dim"``: the dimensionality of each output 
              layer of the output module.

            - ``"activation"``: the name of the activation function to
              be used in the output module.
        """

        # Initialize the class.
        super().__init__()

        #-------------------------------------------------------------#

        # Set the layers.
        self.main = \
            self._get_layers(\
                n_units_input_layer = n_units_input_layer,
                n_units_hidden_layers = n_units_hidden_layers)

        #-------------------------------------------------------------#

        # Set the output module.
        self.nb = \
            self._get_nb(\
                n_units_last_hidden = n_units_hidden_layers[-1],
                output_module_name = output_module_name,
                output_module_options = output_module_options)


    def _get_layers(self,
                    n_units_input_layer,
                    n_units_hidden_layers):
        """Get the decoder's layers.

        Parameters
        ----------
        n_units_input_layer : ``int``
            The mumber of neurons in the input layer.

        n_units_hidden_layers : ``list``
            The number of units in each of the hidden layers. As many
            hidden layers as the number of items in the list will be
            created.

        Returns
        -------
        list_layers : ``torch.nn.ModuleList``
            The list of layers.
        """

        # Create a 'ModuleList' to store the layers.
        layers = nn.ModuleList()

        # Get number of groups of connections (one group of connections
        # connect two layers, so they are one less than the total
        # number of layers).
        n_connects = len(n_units_hidden_layers)

        #-------------------------------------------------------------#

        # For each group of connections
        for n_connect in range(n_connects):

            # If it is the first hidden layer
            if n_connect == 0:

                # Add full connections between the input layer and
                # the first hidden layer using the 'ReLu' activation
                # function.
                layers.extend(\
                    [nn.Linear(n_units_input_layer,
                               n_units_hidden_layers[n_connect]),
                     nn.ReLU(True)])
                
                # Set the previous number of units (used in the
                # next step of the loop) as the number of units in
                # the first hidden layer.
                prev_n_units = n_units_hidden_layers[n_connect]

                # Go to the next step.
                continue

            #---------------------------------------------------------#

            # If it is the last hidden layer
            elif n_connect == n_connects-1:

                # Add a linear layer.
                layers.append(\
                    nn.Linear(\
                        prev_n_units,
                        n_units_hidden_layers[n_connect]))

                # Return the list of layers.
                return layers

            #---------------------------------------------------------#

            # If it is an intermediate hidden layer
            else:
                
                # Add full connections between the previous hidden
                # layer and the current hidden layer using the 'ReLu'
                # activation function.
                layers.extend(\
                    [nn.Linear(\
                        prev_n_units,
                        n_units_hidden_layers[n_connect]),
                     nn.ReLU(True)])

                # Set the previous number of units (used in the next
                # step of the loop) as the number of units in the
                # current hidden layer.
                prev_n_units = n_units_hidden_layers[n_connect]


    def _get_nb(self,
                n_units_last_hidden,
                output_module_name,
                output_module_options):
        """Get the 'negative binomial' layer.

        Parameters
        ----------
        n_units_last_hidden : ``int``
            The number of units in the last hidden layer.

        output_module_name : ``str``
            The name of the output module that will be set. Available
            output modules are:

            - ``"nb_feature_dispersion"`` for negative binomial
              distributions with means learned per gene per sample and
              r-values learned per gene.

            - ``"nb_full_dispersion"`` for negative binomial
              distributions with both means and r-values learned per
              gene per sample.

        output_module_options : ``dict``
            A dictionary of options for setting up the output module.

            For the ``"nb_feature_dispersion"`` output module, the
            following options must be provided:

            - ``"dim"``: the dimensionality of each output layer of
              the output module.

            - ``"activation"``: the name of the activation function to
              be used in the output module.

            - ``"r_init"``: the initial r-value for the negative
               binomial distributions modeling the genes' counts.

            For the ``"nb_full_dispersion"`` output module, the
            following options must be provided:

            - ``"dim"``: the dimensionality of each output layer of
              the output module.

            - ``"activation"``: the name of the activation function to
              be used in the output module.

        Returns
        -------
        nb : ``bulkDGD.core.outputmodules.OutputModuleNB``
            An instance of a subclass of
            :class:`core.outputmodules.OutputModuleNB` depending on the
            chosen output module.
        """

        # If the output module is 'nb_feature_dispersion'
        if output_module_name == "nb_feature_dispersion":

            # Return it.
            return outputmodules.OutputModuleNBFeatureDispersion(\
                        input_dim = n_units_last_hidden,
                        **output_module_options)

        #-------------------------------------------------------------#

        # If the output module is 'nb_full_dispersion'
        elif output_module_name == "nb_full_dispersion":

            # Return it.
            return outputmodules.OutputModuleNBFullDispersion(\
                        input_dim = n_units_last_hidden,
                        **output_module_options)

        #-------------------------------------------------------------#

        # Otherwise
        else:

            # Get the available output modules.
            output_modules = \
                ", ".join([f"'{om}'" for om in self.OUTPUT_MODULES])

            # Raise an error.
            errstr = \
                "Invalid name for the output module passed: " \
                f"'{output_module_name}'. Available output modules " \
                f"are: '{output_modules}'."
            raise ValueError(errstr)


    ######################### PUBLIC METHODS ##########################


    def forward(self,
                z):
        """Forward pass through the neural network.

        Parameters
        ----------
        z : ``torch.Tensor``
            A tensor holding the representations to pass through the
            decoder.

        Returns
        -------
        y : ``torch.Tensor``
            A tensor holding the outputs of the decoder for the
            given representations.
        """

        # For each layer of the neural network
        for i in range(len(self.main)):
            
            # Pass through the layer and find the intermediate (or
            # final) representations.
            z = self.main[i](z)

        # Return the final result.
        return self.nb(z)
