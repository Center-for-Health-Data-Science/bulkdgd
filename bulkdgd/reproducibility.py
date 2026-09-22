#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-

#    reproducibility.py
#
#    Seed the generators the package draws from, so that a run can be
#    repeated.
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


#######################################################################


# Set the module's description.
__doc__ = \
    "Seed the generators the package draws from, so that a run can " \
    "be repeated."


#######################################################################


# Import from the standard library.
import logging as log
import os
import random
from typing import Optional

# Import from third-party libraries.
import numpy as np
import torch


#######################################################################


# Get the module's logger.
logger = log.getLogger(__name__)


#######################################################################


def set_seeds(seed: int,
              deterministic: bool = False) -> dict:
    """Seed every generator the package draws from. Call it before
    building the model, whose initialization is already random.

    Parameters
    ----------
    seed : :class:`int`
        The seed.

    deterministic : :class:`bool`, ``False``
        Whether to also use :mod:`torch`'s deterministic algorithms
        (slower; operations without one raise an error).

    Returns
    -------
    seeds : :class:`dict`
        The seeds used for each generator.
    """

    #-----------------------------------------------------------------#

    # Seed the standard library's generator.
    random.seed(seed)

    # Seed numpy's global generator (used by scikit-learn when
    # 'random_state' is None).
    np.random.seed(seed)

    # Seed torch's global generator (decoder's weights, representations,
    # noise and data loader's shuffling).
    torch.manual_seed(seed)

    # If a GPU is available
    if torch.cuda.is_available():

        # Seed the generators of every GPU.
        torch.cuda.manual_seed_all(seed)

    #-----------------------------------------------------------------#

    # Build the record of what was seeded.
    seeds = {"seed" : int(seed),
             "python_random" : int(seed),
             "numpy" : int(seed),
             "torch" : int(seed),
             "torch_cuda" : \
                int(seed) if torch.cuda.is_available() else None,
             "deterministic_algorithms" : bool(deterministic)}

    #-----------------------------------------------------------------#

    # If deterministic algorithms were asked for.
    if deterministic:

        # Use torch's deterministic algorithms.
        torch.use_deterministic_algorithms(True)

        # Make cuBLAS deterministic (no effect if cuBLAS is already
        # initialized).
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

        # Make cuDNN use deterministic algorithms.
        torch.backends.cudnn.deterministic = True

        # Stop cuDNN from picking its algorithms by timing them.
        torch.backends.cudnn.benchmark = False

        # Record the cuBLAS configuration.
        seeds["cublas_workspace_config"] = \
            os.environ.get("CUBLAS_WORKSPACE_CONFIG")

    #-----------------------------------------------------------------#

    # Log the seed.
    logger.info(
        f"The generators were seeded with {seed}"
        f"{' (deterministic algorithms)' if deterministic else ''}.")

    #-----------------------------------------------------------------#

    # Return the record.
    return seeds


def get_seeds_state() -> dict:
    """Get a number summarizing the state of each generator (not the
    full state, but enough to tell runs apart).

    Returns
    -------
    state : :class:`dict`
        A number per generator.
    """

    # Return a number for each generator.
    return {

        # Set torch's initial seed.
        "torch" : int(torch.random.initial_seed()),

        # Set the first word of numpy's state.
        "numpy" : int(np.random.get_state()[1][0]),

        # Set a hash of the start of the standard library's state.
        "python_random" : hash(random.getstate()[1][:4]),
    }
