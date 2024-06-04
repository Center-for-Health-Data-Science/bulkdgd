#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-

#    dataclasses.py
#
#    This module contains the classses defining the structure of the
#    datasets to be used with the DGD model.
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
    "This module contains the classses defining the structure of " \
    "the datasets to be used with the DGD model."


#######################################################################


# Import from the standard library.
import logging as log
# Import from third-party packages.
import numpy as np
import torch


#######################################################################


# Get the module's logger.
logger = log.getLogger(__name__)


#######################################################################


class GeneExpressionDataset(object):

    """
    Class implementing a dataset containing gene expression data
    for multiple samples.

    This class is designed so that it can be used with the
    ``torch.utils.data.DataLoader`` utility, if needed.

    The class needs to implement a ``__getitem__`` and a ``__len__``
    method.
    """


    ######################### INITIALIZATION ##########################


    def __init__(self,
                 df):
        """Initialize an instance of the class.

        Parameters
        ----------
        df : ``pandas.DataFrame``
            A data frame whose rows must represent samples,
            and columns must represent genes.

            Therefore, each cell of the data frame represents the
            expression of the gene on the column in the sample on
            the row.

            For example:

            .. code-block:: shell
            
               ,gene_1,gene_2,gene_3,gene_4
               sample_1,123,12,2342,145
               sample_2,189,184,2397,1980
               sample_3,978,9467,563,23
        """

        # Get the samples' names.
        self._samples = df.index

        # Get the genes' names.
        self._genes = df.columns

        # Get the expression data for all samples and the
        # mean gene expression for each sample .
        self._data_exp, self._mean_exp = self._get_exp(df = df)


    def _get_exp(self,
                 df):
        """Return the gene expression for all samples and the
        mean gene expression for each sample.

        Parameters
        ----------
        df : ``pandas.DataFrame``
            A data frame containing the gene expression data.

        Returns
        -------
        data_exp : ``torch.Tensor``
            The gene expression for all samples.

            This is a 2D tensor where:

            - The first dimension has a length equal to the number of
              samples in the dataset.

            - The second dimension has a length equal to the number of
              genes whose expression is reported in the dataset.

        mean_exp : ``torch.Tensor``
            The mean gene expression for each sample.
            
            This is a 1D tensor whose length is equal to the
            number of samples in the dataset.
        """

        # Get the gene expression for all samples.
        data_exp = \
            torch.Tensor(\
                np.array(df.loc[self.samples, self.genes].values,
                         dtype = "float64"))

        # Get the mean gene expression for each sample.
        mean_exp = \
            torch.mean(data_exp,
                       dim = 1).unsqueeze(1)

        # Return the two tensors.
        return data_exp, mean_exp


    ########################### PROPERTIES ############################


    @property
    def samples(self):
        """The names/IDs/indexes of the samples in the dataset.
        """
        return self._samples


    @samples.setter
    def samples(self,
                value):
        """Raise an error if the user tries to modify the value of
        ``samples`` after initialization.
        """

        errstr = \
            "The value of 'samples' is set at initialization and " \
            "depends on the input dataset. Therefore, it cannot " \
            "be changed."
        raise ValueError(errstr)


    @property
    def genes(self):
        """The names of the genes included in the dataset.
        """
        return self._genes


    @genes.setter
    def genes(self,
              value):
        """Raise an error if the user tries to modify the value of
        ``genes`` after initialization.
        """
    
        errstr = \
            "The value of 'genes' is set at initialization and " \
            "depends on the input dataset. Therefore, it cannot " \
            "be changed."
        raise ValueError(errstr)


    @property
    def data_exp(self):
        """A 2D tensor where:

        * The first dimension has a length equal to the number of
          samples in the dataset.

        * The second dimension has a length equal to the number of
          genes whose expression is reported in the dataset.
        """
        
        return self._data_exp
    

    @data_exp.setter
    def data_exp(self,
                 value):
        """Raise an error if the user tries to modify the value of
        ``data_exp``.
        """

        errstr = \
            "The value of 'data_exp' is set at initialization and " \
            "depends on the input dataset. Therefore, it cannot be " \
            "changed."
        raise ValueError(errstr)   


    @property
    def mean_exp(self):
        """A 1D tensor with length equal to the number of samples in
        the dataset containing the mean gene expression for each
        sample.
        """
        
        return self._mean_exp


    @mean_exp.setter
    def mean_exp(self,
                 value):
        """Raise an error if the user tries to modify the value of
        ``mean_exp`` after initialization.
        """
        
        errstr = \
            "The value of 'mean_exp' is set at initialization and " \
            "depends on the input dataset. Therefore, it cannot be " \
            "changed."
        raise ValueError(errstr)


    ######################### DUNDER METHODS ##########################


    def __getitem__(self,
                    idx = None):
        """Get items from the dataset.
        
        Parameters
        ----------
        idx : ``list`` or ``torch.Tensor``, optional
            If passed, a list of indexes of the samples to get from
            the dataset.

        Returns
        -------
        data : ``numpy.ndarray``
            An array containing the data for the selected samples.

        mean_expr : ``numpy.ndarray``
            An array with the mean gene expression for each sample.

        idx : ``list``
            A list of indexes of the samples that are returned.
        """

        # If no index is passed
        if idx is None:

            # The index will encompass all items in the dataset.
            idx = np.arange(self.__len__()).tolist()
        
        # If the index is a tensor
        elif torch.is_tensor(idx):

            # Convert it to a list.
            idx = idx.tolist()

        # Return the data for the sample(s) of interest, its (their)
        # mean gene expression, and its (their) index(es).
        return (self.data_exp[idx], self.mean_exp[idx], idx)


    def __len__(self):
        """Get the length of the dataset, which corresponds to the
        number of samples.
        """

        # Return the number of samples.
        return len(self._samples)


    ######################### PUBLIC METHODS ##########################


    def get_tot_expr_per_gene(self):
        """Get the total expression of all genes across the samples in
        the dataset.

        Returns
        -------
        ``torch.Tensor``
            The total expression of all genes across all samples.
        """

        # Return the total expression of all genes across all samples.
        return torch.from_numpy(self.df.sum(axis = 0).to_numpy())
