#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-

#    base.py
#
#    This module contains what the BulkDGD models share.
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
__doc__ = "This module contains what the BulkDGD models share."


#######################################################################


# Import from the standard library.
import logging as log
from typing import Optional, Union

# Import from third-party libraries.
import torch


#######################################################################


# Get the module's logger.
logger = log.getLogger(__name__)


#######################################################################


def clip_grads(optimizer: torch.optim.Optimizer,
               max_norm: Optional[Union[float, int]]) -> None:
    """Clip an optimizer's gradients to a maximum norm.

    Parameters
    ----------
    optimizer : :class:`torch.optim.Optimizer`
        The optimizer whose parameters' gradients should be clipped.

    max_norm : :class:`float` or :class:`int`, optional
        The norm to clip the gradients to. If :obj:`None`, the gradients
        are not clipped.
    """

    # If no maximum norm was set, the gradients are not clipped.
    if max_norm is None:
        return

    # Get the parameters the optimizer steps that have a gradient.
    params = \
        [p for group in optimizer.param_groups \
         for p in group["params"] if p.grad is not None]

    # If there are none, there is nothing to clip.
    if not params:
        return

    #-----------------------------------------------------------------#

    # Zero out the non-finite gradients.
    for p in params:

        # If the parameter's gradient is not entirely finite
        if not torch.isfinite(p.grad).all():

            # Warn the user.
            logger.warning(\
                "A non-finite gradient was found and zeroed.")

            # Zero the gradient.
            p.grad = torch.zeros_like(p.grad)

    #-----------------------------------------------------------------#

    # Clip the gradients and get their norm before clipping.
    total_norm = \
        torch.nn.utils.clip_grad_norm_(parameters = params,
                                       max_norm = max_norm)

    # Log the gradient norm before clipping.
    logger.debug(
        f"Gradient norm before clipping: {float(total_norm):.4f} "
        f"(clipped to {max_norm}).")
