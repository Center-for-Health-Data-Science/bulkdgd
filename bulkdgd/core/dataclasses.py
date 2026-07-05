#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-

#    dataclasses.py
#
#    This module contains the classes defining the structure of the
#    datasets to be used with the
#    :class:`core.model.BulkDGD`.
#
#    The code was originally developed by Viktoria Schuster,
#    Inigo Prada Luengo, and Anders Krogh.
#
#    Valentina Sora modified and complemented it for the purposes
#    of this package.
#
#    Copyright (C) 2026 Valentina Sora
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
    "This module contains the classes defining the structure of " \
    "the datasets to be used with the " \
    ":class:`core.model.BulkDGD`."


#######################################################################


# Import from the standard library.
import logging as log
from typing import Optional, Union, Tuple

# Import from third-party libraries.
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
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
    :class:`torch.utils.data.DataLoader` utility, if needed.
    """


    ######################### INITIALIZATION ##########################


    def __init__(self,
                 df: pd.DataFrame,
                 labels: Optional[list[str]] = None) -> None:
        """Initialize an instance of the class.

        Parameters
        ----------
        df : :class:`pandas.DataFrame`
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
        
        labels : :class:`list`, optional
            A list of labels for the samples.
        """

        # Get the samples' names.
        self._samples = df.index

        #-------------------------------------------------------------#

        # If labels were passed.
        if labels is not None:

            # Set the label encoder.
            self._label_encoder = LabelEncoder()

            # Fit the label encoder.
            self._label_encoder.fit(labels)

            # Set the labels.
            self._labels = \
                torch.LongTensor(self._label_encoder.transform(labels))
        
        # Otherwise
        else:

            # Set the labels to None.
            self._labels = None

        #-------------------------------------------------------------#

        # Get the genes' names (all remaining columns).
        self._genes = df.columns

        #-------------------------------------------------------------#

        # Get the expression data for all samples and the
        # mean gene expression for each sample.
        self._data_exp, self._mean_exp = self._get_exp(df = df)


    def _get_exp(self,
                 df: pd.DataFrame) -> (torch.Tensor, torch.Tensor):
        """Return the gene expression for all samples and the
        mean gene expression for each sample.

        Parameters
        ----------
        df : :class:`pandas.DataFrame`
            A data frame containing the gene expression data.

        Returns
        -------
        data_exp : :class:`torch.Tensor`
            The gene expression for all samples.

            This is a 2D tensor where:

            - The first dimension has a length equal to the number of
              samples in the dataset.

            - The second dimension has a length equal to the number of
              genes whose expression is reported in the dataset.

        mean_exp : :class:`torch.Tensor`
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


    @property
    def labels(self):
        """The labels for the samples in the dataset, or :obj:`None`
        if no labels were provided.
        """

        return self._labels


    @labels.setter
    def labels(self,
               value):
        """Raise an error if the user tries to modify the value of
        ``labels`` after initialization.
        """

        errstr = \
            "The value of 'labels' is set at initialization and " \
            "depends on the input dataset. Therefore, it cannot be " \
            "changed."
        raise ValueError(errstr)


    ######################### DUNDER METHODS ##########################


    def __getitem__(self,
                    idx: Optional[Union[list, torch.Tensor]] = None) \
                        -> Tuple[torch.Tensor, torch.Tensor, \
                                 list[str], torch.Tensor]:
        """Get items from the dataset.
        
        Parameters
        ----------
        idx : :class:`list` or :class:`torch.Tensor`, optional
            If passed, a list of indexes of the samples to get from
            the dataset.

        Returns
        -------
        data : :class:`torch.Tensor`
            An array containing the data for the selected samples.

        mean_expr : :class:`torch.Tensor`
            An array with the mean gene expression for each sample.

        idx : :class:`list`
            A list of indexes of the samples that are returned.
        
        labels : :class:`torch.Tensor`
            If labels are available, a tensor containing the labels for
            the selected samples.
        """

        # If no index is passed
        if idx is None:

            # The index will encompass all items in the dataset.
            idx = np.arange(self.__len__()).tolist()
        
        # If the index is a tensor
        elif torch.is_tensor(idx):

            # Convert it to a list.
            idx = idx.tolist()

        # If labels are available
        if self._labels is not None:

            # Preserve the original integer dtype used for labels.
            labels_out = self._labels[idx]

            # Return the data, mean expression, indexes, and labels.
            return (self.data_exp[idx], self.mean_exp[idx], idx,
                    labels_out)

        # Return the data for the sample(s) of interest, its (their)
        # mean gene expression, and its (their) index(es).
        return (self.data_exp[idx], self.mean_exp[idx], idx)


    def __len__(self) -> int:
        """Get the length of the dataset, which corresponds to the
        number of samples.
        """

        # Return the number of samples.
        return len(self._samples)


    ######################### PUBLIC METHODS ##########################


    def get_tot_expr_per_gene(self) -> torch.Tensor:
        """Get the total expression of all genes across the samples in
        the dataset.

        Returns
        -------
        tot_expr : :class:`torch.Tensor`
            The total expression of all genes across all samples.
        """

        # Return the total expression of all genes across all samples.
        return torch.sum(self.data_exp, dim = 0)
