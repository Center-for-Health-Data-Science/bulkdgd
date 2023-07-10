#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-

#    dataclasses.py
#
#    Module containing the classses defining the structure of the
#    datasets to be used in the DGD model.
#
#    The code was originally developed by Viktoria Schuster,
#    Inigo Prada Luengo, Yuhu Liang, and Anders Krogh.
#    
#    Valentina Sora rearranged it for the purposes of this package.
#    Therefore, only functions/methods needed for the purposes
#    of this package were retained in the code.
#
#    Copyright (C) 2023 Valentina Sora 
#                       <sora.valentina1@gmail.com>
#                       Viktoria Schuster
#                       <viktoria.schuster@sund.ku.dk>
#                       Inigo Prada Luengo
#                       <inlu@diku.dk>
#                       Yuhu Liang
#                       <yuhu.liang@di.ku.dk>
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


# Description of the module
__doc__ = \
    "Module containing the classses defining the structure of " \
    "the datasets to be used in the DGD model."


# Third-party packages
import numpy as np
import torch


class GeneExpressionDataset(object):

    """
    Class implementing a dataset containing gene expression data
    for multiple samples, so that one can take advantage of the
    ``torch.utils.data.DataLoader`` utility for a map-style dataset.

    The class needs to implement a ``__getitem__`` and a ``__len__``
    method.
    """

    def __init__(self,
                 df):
        """Initialize an instance of the class.

        Parameters
        ----------
        df : ``pandas.DataFrame``
            A data frame whose rows must represent samples,
            and columns must represent genes. Therefore, each
            cell of the data frame represents the expression
            of the gene on the column in the sample on the row.

            For example:

            .. code-block:: shell
            
               ,gene_1,gene_2,gene_3,gene_4
               sample_1,123,12,2342,145
               sample_2,189,184,2397,1980
               sample_3,978,9467,563,23
        """

        # Get the data frame
        self._df = df

        # Get the number of samples
        self._n_samples = df.shape[0]

        # Get the number of genes
        self._n_genes = df.shape[1]

        # Get the mean gene expression for each sample
        self._mean_exp = self._get_mean_exp(df = self.df)


    #-------------------- Initialization methods ---------------------#


    def _get_mean_exp(self,
                      df):
        """Return the mean gene expression for each sample.

        Parameters
        ----------
        df : ``pandas.DataFrame``
            A data frame containing the gene espression data.

        Returns
        -------
        ``torch.Tensor``
            A tensor containing the mean gene expression for each
            sample.
        """

        return torch.mean(torch.Tensor(df.to_numpy()),
                          dim = -1).unsqueeze(1)


    #-------------------------- Properties ---------------------------#


    @property
    def df(self):
        """The original data frame from which the dataset is
        constructed.
        """
        
        return self._df


    @df.setter
    def df(self,
           value):
        """Raise an exception if the user tries to modify
        the value of ``df`` after initialization.
        """
        
        errstr = \
            "The value of 'df' cannot be changed " \
            "after initialization."
        raise ValueError(errstr)


    @property
    def n_samples(self):
        """An integer representing the number of samples in the
        data frame.
        """
        
        return self._n_samples


    @n_samples.setter
    def n_samples(self,
                  value):
        """Raise an exception if the user tries to modify
        the value of ``n_samples`` after initialization.
        """
        
        errstr = \
            "The value of 'n_samples' cannot be changed " \
            "after initialization."
        raise ValueError(errstr)


    @property
    def n_genes(self):
        """An integer representing the number of genes in
        the data frame.
        """
        
        return self._n_genes


    @n_genes.setter
    def n_genes(self,
                value):
        """Raise an exception if the user tries to modify
        the value of ``n_genes`` after initialization.
        """
        
        errstr = \
            "The value of 'n_genes' cannot be changed " \
            "after initialization."
        raise ValueError(errstr)


    @property
    def mean_exp(self):
        """A one-dimensional ``torch.Tensor`` with length equal
        to the number of samples in the dataset containing the
        mean gene expression for each sample.
        """
        
        return self._mean_exp


    @mean_exp.setter
    def mean_exp(self,
                 value):
        """Raise an exception if the user tries to modify
        the value of ``mean_exp`` after initialization.
        """
        
        errstr = \
            "The value of 'mean_exp' cannot be changed " \
            "after initialization."
        raise ValueError(errstr)


    #----------------------- "Dunder" methods ------------------------#


    def __getitem__(self,
                    idx = None):
        """Get items from the dataset.
        
        Parameters
        ----------
        idx : ``list``, optional
            If passed, a list of indexes of the samples to get from
            the dataset.

        Returns
        -------
        ``tuple``
            A length-2 tuple with a ``numpy.ndarray`` containing the
            data for the selected samples, a ``numpy.ndarray`` with
            the mean gene expression for each sample, and the
            ``list`` of indexes of the samples.
        """

        # If no index is passed
        if idx is None:

            # The index will encompass all items
            # in the dataset
            idx = np.arange(self.__len__()).tolist()
        
        # If the index is a tensor
        elif torch.is_tensor(idx):

            # Convert it to a list
            idx = idx.tolist()
        
        # Get the data of interest from the data frame
        # as a numpy array
        data = \
            np.array(self.df.iloc[idx, 0:self.n_genes],
                     dtype = "float64")

        # Return the data for the sample(s) of interest,
        # its (their) mean gene expression, and its (their)
        # index(es)
        return (data, self.mean_exp[idx], idx)


    def __len__(self):
        """Get the length of the dataset from the lenght of the
        associated data frame.
        """

        # Return the length of the associated data frame
        return len(self.df)