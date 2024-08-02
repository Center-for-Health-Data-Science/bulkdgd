#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-

#    util.py
#
#    Utilities to generate customized lists of genes.
#
#    Copyright (C) 2024 Valentina Sora 
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
__doc__ = "Utilities to generate customized lists of genes."


#######################################################################


# Import from the standard library.
from io import StringIO
import logging as log
# Import from third-party packages.
import pandas as pd
import requests as rq
# Import from 'bulkDGD'.
from . import defaults


#######################################################################


# Get the module's logger.
logger = log.getLogger(__name__)


#######################################################################


def get_genes_attributes(attributes,
                         filters = None,
                         dataset = "hsapiens_gene_ensembl"):
    """Get genes' attributes from the Ensembl database.

    Parameters
    ----------
    attributes : ``list``
        The attributes to retrieve for the genes.

    filters : ``dict``, optional
        The filters and corresponding values to filter the genes.

    dataset : ``str``, ``"hsapiens_gene_ensembl"``
        The dataset the genes belong to.

    Returns
    -------
    df_annot : ``pandas.DataFrame``
        A data frame containing the attributes fetched from the Ensembl
        database for the genes of interest.
    """

    # Initialize an empty string that will contain the filters and
    # attributes for the query.
    ensembl_query_fill = ""

    #-----------------------------------------------------------------#

    # For each filter and associated values
    for filt, values in \
        (filters if filters is not None else {}).items():

        # Concatenate the values into a string.
        filt_values = ",".join(values)

        # Add the filter and its values to the query.
        ensembl_query_fill += \
            f'<Filter name = "{filt}" value = "{filt_values}"/>'

    #-----------------------------------------------------------------#

    # For each attribute
    for attr in attributes:

        # Add it to the query.
        ensembl_query_fill += \
            f'<Attribute name = "{attr}" />'

    #-----------------------------------------------------------------#

    # Get the full XML query for the Ensembl database.
    ensembl_query = \
        defaults.ENSEMBL_XML_QUERY.format(dataset,
                                          ensembl_query_fill)

    #-----------------------------------------------------------------#

    # Get the data.
    data = rq.get(ensembl_query).text

    # If there was an error with getting the data
    if data.startswith("Query ERROR"):

        # Raise an exception.
        raise rq.exceptions.RequestException(data)

    #-----------------------------------------------------------------#
            
    # Convert the data into a data frame.
    df = pd.read_table(StringIO(data),
                       sep = ",",
                       header = 0,
                       low_memory = False)

    #-----------------------------------------------------------------#

    # Return the data frame.
    return df
