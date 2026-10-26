#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-

#    __init__.py
#
#    The BulkDGD models, their registry, and the function building
#    the model a configuration is for.
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


# Import everything from the modules.
from .base import *
from .vanilla import *


#######################################################################


# Set the available models.
MODELS = {

    # The model with one point per sample in latent space.
    "vanilla" : BulkDGD,

    }


#######################################################################


def get_model(config_model: dict[str, object],
              **kwargs) -> BulkDGDBase:
    """Build the model a configuration is for.

    Parameters
    ----------
    config_model : :class:`dict`
        The model's configuration. Its ``"model_type"`` picks the
        model (by default, ``"vanilla"``).

    **kwargs
        The other options of the model's class (as ``device``).

    Returns
    -------
    model : :class:`bulkdgd.core.models.BulkDGDBase`
        The model.
    """

    # Get the model's type (by default, the vanilla model).
    model_type = config_model.get("model_type",
                                  "vanilla")

    # If the type is not available
    if model_type not in MODELS:

        # Raise an error.
        errstr = \
            f"Unsupported model type '{model_type}'. The supported " \
            f"types are: {', '.join(MODELS)}."
        raise ValueError(errstr)

    # Build and return the model.
    return MODELS[model_type](**config_model,
                              **kwargs)
