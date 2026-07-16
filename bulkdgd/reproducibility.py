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
    """Seed every generator a run of the package draws from.

    A model is built before it is trained, and the building is already
    random: the decoder's weights are drawn, the Gaussian mixture
    model's components are placed, and the representations are
    initialized. So this has to be called **before** the model is
    constructed, and not before it is trained - by then the decoder's
    weights have already been drawn from an unseeded generator.

    What draws from what:

    - The decoder's weights come from :mod:`torch`'s global generator,
      through :class:`torch.nn.Linear`'s own initialization.

    - The representations of the training and the test samples are
      initialized with :func:`torch.randn`, from the same generator.

    - The noise added to the representations during the training comes
      from it as well.

    - The data loader shuffles the training samples with it, unless it
      is given a generator of its own.

    - The Gaussian mixture model places its components with it, when
      its ``random_state`` is left unset. When it is set, ``tgmm``
      calls :func:`torch.manual_seed` itself, which reseeds the global
      generator for everything that comes after it - so it is better to
      leave ``random_state`` unset and to seed here, once.

    - :mod:`numpy` and :mod:`random` are seeded too. Little is drawn
      from them, but 'little' is not 'nothing', and a generator that is
      not seeded is a generator that makes a run unrepeatable.

    Parameters
    ----------
    seed : :class:`int`
        The seed.

    deterministic : :class:`bool`, ``False``
        Whether to also ask :mod:`torch` for deterministic algorithms.

        Seeding makes a run repeat itself only if the operations it
        runs are themselves deterministic, and several are not: an
        operation that sums with atomics on a GPU adds its terms in
        whatever order the threads finish, and floating-point addition
        is not associative, so the same seed gives a slightly different
        number. This turns those operations into their deterministic
        versions where they have one, and makes them raise where they
        do not.

        It is off by default because it is slower, and because an
        operation without a deterministic version raises rather than
        falls back - which is the right behaviour when reproducibility
        is what is being asked for, and the wrong one when it is not.

    Returns
    -------
    seeds : :class:`dict`
        What was seeded and with what, to be recorded with the run's
        results. A seed that is not written down is a seed nobody has.
    """

    #-----------------------------------------------------------------#

    # Seed the standard library's generator.
    random.seed(seed)

    # Seed numpy's global generator. This is what a 'random_state' of
    # 'None' falls back on in anything scikit-learn-shaped.
    np.random.seed(seed)

    # Seed torch's global generator. This covers the decoder's weights,
    # the representations' initialization, the noise, and the data
    # loader's shuffling.
    torch.manual_seed(seed)

    # Seed the generators of every GPU. 'torch.manual_seed' already
    # does this, but saying so is cheaper than finding out that a
    # version of torch changed its mind.
    if torch.cuda.is_available():

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

        # Ask for them. An operation that has no deterministic version
        # will raise rather than quietly use the other one.
        torch.use_deterministic_algorithms(True)

        # cuBLAS needs to be told separately, and it has to be told
        # through the environment before it is initialized - which is
        # to say before the first matrix multiplication, not before
        # this call. If it is already up, this does nothing, and the
        # run is not deterministic although it was asked to be.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

        # cuDNN picks its algorithms by timing them, and the fastest
        # one is not the same one every time.
        torch.backends.cudnn.deterministic = True

        torch.backends.cudnn.benchmark = False

        # Record it.
        seeds["cublas_workspace_config"] = \
            os.environ.get("CUBLAS_WORKSPACE_CONFIG")

    #-----------------------------------------------------------------#

    # Tell the user. A run whose seed is not in its log is a run whose
    # seed has to be guessed at afterwards.
    logger.info(
        f"The generators were seeded with {seed}"
        f"{' (deterministic algorithms)' if deterministic else ''}.")

    #-----------------------------------------------------------------#

    # Return the record.
    return seeds


def get_seeds_state() -> dict:
    """Get the state of the generators, as a set of numbers that can be
    written down.

    This is not the state itself - the state of a Mersenne twister is
    624 words, and writing it into a results file helps nobody. It is
    enough to tell two runs apart, and to notice that a run which was
    supposed to have been seeded was not.

    Returns
    -------
    state : :class:`dict`
        A number per generator.
    """

    # Return a number drawn from each generator, without disturbing it.
    return {

        "torch" : int(torch.random.initial_seed()),

        "numpy" : int(np.random.get_state()[1][0]),

        "python_random" : hash(random.getstate()[1][:4]),
    }
