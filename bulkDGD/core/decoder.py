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
    x : :class:`torch.Tensor`
        The input tensor to be reshaped.

    out_dim : :class:`int`
        The dimensionality of the output tensor.

    Returns
    -------
    x : :class:`torch.Tensor`
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
        ["nb_feature_dispersion", "nb_full_dispersion", "poisson"]

    # Set the names of the supported activation functions.
    ACTIVATIONS = \
        ["relu", "elu"]


    ######################### INITIALIZATION ##########################


    def __init__(self,
                 n_units_input_layer,
                 n_units_hidden_layers,
                 activations,
                 output_module_name,
                 output_module_options):
        """Initialize an instance of the neural network representing
        the decoder.

        Parameters
        ----------
        n_units_input_layer : :class:`int`
            The number of neurons in the input layer.

        n_units_hidden_layers : :class:`list`
            The number of units in each of the hidden layers. As many
            hidden layers as the number of items in the list will be
            created.

        activations : :class:`list`
            A list containing the names of the activation functions to
            use in each hidden layer. Available activation functions
            are:

            - ``"relu"`` : the ReLU function.

            - ``"elu"`` : the ELU function.

        output_module_name : :class:`str`, \
            {``"nb_feature_dispersion"``, ``"nb_full_dispersion"``, \
            ``"poisson"``}
            The name of the output module that will be set. Available
            output modules are:

            - ``"nb_feature_dispersion"`` for negative binomial
              distributions with means learned per gene per sample and
              r-values learned per gene.

            - ``"nb_full_dispersion"`` for negative binomial
              distributions with both means and r-values learned per
              gene per sample.

            - ``"poisson"`` for Poisson distributions with means
              learned per gene per sample.

        output_module_options : :class:`dict`
            A dictionary of options for setting up the output module.

            For the ``"nb_feature_dispersion"`` output module, the
            following options must be provided:

            - ``"activation"`` : the name of the activation function to
              be used in the output module.

            - ``"r_init"`` : the initial r-value for the negative
              binomial distributions modeling the genes' counts.

            For the ``"nb_full_dispersion"`` output module, the
            following options must be provided:

            - ``"activation"`` : the name of the activation function to
              be used in the output module.

            For the ``"poisson"`` output module, the following options
            must be provided:

            - ``"activation"`` : the name of the activation function to
              be used in the output module.
        """

        # Initialize the class.
        super().__init__()

        #-------------------------------------------------------------#

        # Set the layers.
        self.main = \
            self._get_layers(\
                n_units_input_layer = n_units_input_layer,
                n_units_hidden_layers = n_units_hidden_layers,
                activations = activations)

        #-------------------------------------------------------------#

        # Set the output module.
        self.nb = \
            self._get_output_module(\
                n_units_last_hidden = n_units_hidden_layers[-1],
                output_module_name = output_module_name,
                output_module_options = output_module_options)


    def _get_layers(self,
                    n_units_input_layer,
                    n_units_hidden_layers,
                    activations):
        """Get the decoder's layers.

        Parameters
        ----------
        n_units_input_layer : :class:`int`
            The number of neurons in the input layer.

        n_units_hidden_layers : :class:`list`
            The number of units in each of the hidden layers. As many
            hidden layers as the number of items in the list will be
            created.

        activations : :class:`list`
            A list containing the names of the activation functions to
            use in each hidden layer. Available activation functions
            are:

            - ``"relu"`` : the ReLU function.

            - ``"elu"`` : the ELU function.

        Returns
        -------
        list_layers : :class:`torch.nn.ModuleList`
            The list of layers.
        """

        # If the number of hidden layers does not correspond to the
        # length of activation functions
        if len(n_units_hidden_layers) != len(activations):

            # Raise an error.
            errstr = \
                "The number of hidden layers should be equal to " \
                "the number of activation functions passed."
            raise ValueError(errstr)

        #-------------------------------------------------------------#

        # Create a 'ModuleList' to store the layers.
        layers = nn.ModuleList()

        # Get the number of linear modules.
        n_modules = len(n_units_hidden_layers)

        # Initialize the 'previous number of units' to None.
        prev_n_units = None

        #-------------------------------------------------------------#

        # For each linear module
        for n_module in range(n_modules):

            # If we are at the first module
            if n_module == 0:
                
                # The number of input units will be the number of
                # units in the input layer.
                n_units_in = n_units_input_layer

            # Otherwise
            else:

                # The number of input units will be the number of
                # units in the previous hidden layer.
                n_units_in = prev_n_units

            #---------------------------------------------------------#

            # The number of output units will be the ones in the
            # first hidden layer.
            n_units_out = n_units_hidden_layers[n_module]

            #---------------------------------------------------------#

            # Get the name of the activation function to be used in the
            # current module.
            activation_name = activations[n_module]

            # If the activation function is a ReLU
            if activation_name == "relu":

                # Set it.
                activation = nn.ReLU(inplace = True)

            # If the activation function is an ELU
            elif activation_name == "elu":

                # Set it.
                activation = nn.ELU(inplace = True)

            # Otherwise
            else:

                # Get the names of the supported activation functions.
                supported_activations = \
                    ", ".join([f"'{a}'" for a in self.ACTIVATIONS])

                # Raise an error.
                errstr = \
                    "Unsupported activation function " \
                    f"'{activation}' provided. Supported " \
                    "activation functions are: " \
                    f"{supported_activations}."
                raise ValueError(errstr)

            #---------------------------------------------------------#

            # Add the linear module and corresponding activation
            # function to the list of modules.
            layers.extend(\
                [nn.Linear(in_features = n_units_in,
                           out_features = n_units_out,
                           bias = True),
                 activation])

            # Inform the user about the decoder's hidden layers.
            infostr = \
                f"The decoder's hidden layer # {n_module+1} was " \
                f"set. Input features: {n_units_in}. Output " \
                f"features: {n_units_out}. Activation function: " \
                f"'{activation.__class__.__name__}'."
            logger.info(infostr)

            #---------------------------------------------------------#

            # Set the previous number of units (used in the next
            # step of the loop) as the number of units in the
            # current hidden layer.
            prev_n_units = n_units_hidden_layers[n_module]

        #-------------------------------------------------------------#

        # Return the modules/layers.
        return layers


    def _get_output_module(self,
                           n_units_last_hidden,
                           output_module_name,
                           output_module_options):
        """Get the decoder's output module.

        Parameters
        ----------
        n_units_last_hidden : :class:`int`
            The number of units in the last hidden layer.

        output_module_name : :class:`str`
            The name of the output module that will be set. Available
            output modules are:

            - ``"nb_feature_dispersion"`` for negative binomial
              distributions with means learned per gene per sample and
              r-values learned per gene.

            - ``"nb_full_dispersion"`` for negative binomial
              distributions with both means and r-values learned per
              gene per sample.

            - ``"poisson"`` for Poisson distributions with means
              learned per gene per sample.

        output_module_options : :class:`dict`
            A dictionary of options for setting up the output module.

            For the ``"nb_feature_dispersion"`` output module, the
            following options must be provided:

            - ``"activation"`` : the name of the activation function to
              be used in the output module.

            - ``"r_init"`` : the initial r-value for the negative
               binomial distributions modeling the genes' counts.

            For the ``"nb_full_dispersion"`` output module, the
            following options must be provided:

            - ``"activation"`` : the name of the activation function to
              be used in the output module.

            For the ``"poisson"`` output module, the following options
            must be provided:

            - ``"activation"`` : the name of the activation function to
              be used in the output module.

        Returns
        -------
        out_module : \
            :class:`bulkDGD.core.outputmodules.OutputModuleBase`
            An instance of a subclass of
            :class:`core.outputmodules.OutputModuleBase` depending on
            the chosen output module.
        """

        # If the output module is 'nb_feature_dispersion'
        if output_module_name == "nb_feature_dispersion":

            # Get the output module's class.
            out_module_class = \
                outputmodules.OutputModuleNBFeatureDispersion

        #-------------------------------------------------------------#

        # If the output module is 'nb_full_dispersion'
        elif output_module_name == "nb_full_dispersion":

            # Get the output module's class.
            out_module_class = \
                outputmodules.OutputModuleNBFullDispersion

        #-------------------------------------------------------------#

        # If the output module is 'poisson'
        elif output_module_name == "poisson":

            # Get the output module's class.
            out_module_class = \
                outputmodules.OutputModulePoisson

        #-------------------------------------------------------------#

        # Otherwise
        else:

            # Get the available output modules.
            output_modules = \
                ", ".join([f"'{om}'" for om in self.OUTPUT_MODULES])

            # Raise an error.
            errstr = \
                "An invalid name for the output module was passed: " \
                f"'{output_module_name}'. Available output modules " \
                f"are: '{output_modules}'."
            raise ValueError(errstr)

        #-------------------------------------------------------------#

        # Set the output module.
        out_module = \
            out_module_class(input_dim = n_units_last_hidden,
                             **output_module_options)

        #-------------------------------------------------------------#

        # Set a string with the options used for the output module.
        out_module_opts_str = \
            ", ".join([f"{opt} = '{val}'" \
                       if isinstance(val, str) \
                       else f"{opt} = {val}" \
                       for opt, val in output_module_options.items()])

        # Inform the user about the output module.
        infostr = \
            "The decoder's output module was set. Module " \
            f"'{out_module.__class__.__name__}' " \
            f"({out_module_opts_str})."
        logger.info(infostr)

        #-------------------------------------------------------------#

        # Return the output module.
        return out_module


    ######################### PUBLIC METHODS ##########################


    def forward(self,
                z):
        """Forward pass through the neural network.

        Parameters
        ----------
        z : :class:`torch.Tensor`
            A tensor holding the representations to pass through the
            decoder.

        Returns
        -------
        y : :class:`torch.Tensor`
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
