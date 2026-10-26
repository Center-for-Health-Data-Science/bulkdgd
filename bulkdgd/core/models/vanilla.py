#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-

#    vanilla.py
#
#    This module contains the class implementing the BulkDGD model with
#    one point per sample in latent space
#    (:class:`core.models.BulkDGD`).
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
    "This module contains the class implementing the BulkDGD model " \
    "with one point per sample in latent space " \
    "(:class:`core.models.BulkDGD`)."


#######################################################################


# Import from the standard library.
import logging as log
import math
from types import SimpleNamespace
from typing import Optional

# Import from third-party libraries.
from scipy.stats import chi2
import torch

# Import from 'bulkdgd'.
from .. import (
    representations,
    _util)
from . import base


#######################################################################


# Get the module's logger.
logger = log.getLogger(__name__)


#######################################################################


class BulkDGD(base.BulkDGDBase):

    """
    Class implementing the BulkDGD model, with one point per sample in
    latent space and optional noise on the points.
    """

    ############################# HELPERS #############################


    @staticmethod
    def _cosine_anneal(start: float,
                       end: float,
                       progress: float) -> float:
        """Get a multiplier cosine-annealed from ``start`` to ``end``.

        Parameters
        ----------
        start : :class:`float`
            The multiplier at the start.

        end : :class:`float`
            The multiplier at the end.

        progress : :class:`float`
            The fraction of the annealing elapsed.

        Returns
        -------
        multiplier : :class:`float`
            The multiplier.
        """

        # Return the multiplier.
        return end + (start - end) * 0.5 * \
            (1 + math.cos(math.pi * progress))


    def _draw_kept_noise(self,
                         noise_type: str,
                         noise_options: dict[str, object],
                         noise_scale: float,
                         generator: torch.Generator,
                         optimizer: torch.optim.Optimizer,
                         rep_layer: representations.RepresentationLayer,
                         samples_ixs: torch.Tensor) -> torch.Tensor:
        """Draw the noise kept in a batch's stored representations.

        Parameters
        ----------
        noise_type : :class:`str`, {``"gaussian"``, ``"langevin"``}
            The type of noise.

        noise_options : :class:`dict`
            The noise's options.

        noise_scale : :class:`float`
            The noise's scale (Gaussian) or the temperature's
            multiplier (Langevin) at the current epoch.

        generator : :class:`torch.Generator`
            The noise's own random generator.

        optimizer : :class:`torch.optim.Optimizer`
            The optimizer of the representations.

        rep_layer : \
            :class:`bulkdgd.core.representations.RepresentationLayer`
            The layer holding the representations.

        samples_ixs : :class:`torch.Tensor`
            The indexes of the batch's samples.

        Returns
        -------
        noise : :class:`torch.Tensor`
            The noise, shaped as the batch's representations.
        """

        # Get the batch's representations.
        z = rep_layer.z[samples_ixs]

        # Draw standard normal values from the noise's generator.
        draw = torch.randn(z.shape,
                           generator = generator,
                           device = z.device,
                           dtype = z.dtype)

        # If the noise is Gaussian
        if noise_type == "gaussian":

            # Get the radius of the hypersphere within which the
            # specified fraction of samples lie.
            radius = \
                float(chi2.ppf(noise_options["within_radius_prob"],
                               self.latent.dim)) ** 0.5

            # Return the scaled noise.
            return noise_scale * noise_options["gain"] * draw / radius

        # Get the optimizer's state for the representations.
        rep_state = optimizer.state.get(rep_layer.z, {})

        # If the optimizer has not taken a step yet
        if "exp_avg_sq" not in rep_state:

            # No noise is kept before the first step.
            return torch.zeros_like(z)

        # Get the optimizer's options.
        rep_group = optimizer.param_groups[0]

        # Get the bias-corrected second moments of the batch's
        # gradients.
        v_hat = \
            rep_state["exp_avg_sq"][samples_ixs] / \
                (1 - rep_group["betas"][1] ** float(rep_state["step"]))

        # Get each coordinate's step size.
        step_size = rep_group["lr"] / (v_hat.sqrt() + rep_group["eps"])

        # Return the noise, with a variance of twice the step size
        # times the temperature.
        return draw * \
            torch.sqrt(2.0 * noise_options["temperature"] * \
                           noise_scale * step_size)


    ######################### TRAINING HOOKS ##########################


    def _validate_config_train(self,
                               config_train: dict[str, object]) -> \
                                    list[str]:
        """Check the parsed training configuration against the model.

        Parameters
        ----------
        config_train : :class:`dict`
            The parsed training configuration.

        Returns
        -------
        errors : :class:`list`
            The errors found (none: every option fits the model).
        """

        # Return no errors.
        return []


    def _make_rep_layer(self,
                        n_samples: int,
                        representations_options: dict[str, object]) \
                            -> representations.RepresentationLayer:
        """Build a representation layer for ``n_samples`` samples, as
        the training starts.

        Parameters
        ----------
        n_samples : :class:`int`
            The number of samples to build representations for.

        representations_options : :class:`dict`
            The training's options for the representations.

        Returns
        -------
        rep_layer : \
            :class:`bulkdgd.core.representations.RepresentationLayer`
            The representation layer.
        """

        # Get the scale factor for the initialization of the
        # representations.
        init_rep_scale = \
            representations_options.get("init_rep_scale", 0.0)

        # Get the distribution the initial representations are drawn
        # from (by default, a scaled normal).
        init_rep_dist = \
            representations_options.get("init_rep_dist", None)

        # Get the model's precision.
        dtype = self._DTYPES_TORCH[self._dtype]

        # If no distribution was given
        if init_rep_dist is None:

            # Return a layer of scaled standard normal draws.
            return representations.RepresentationLayer(\
                values = init_rep_scale * torch.randn(\
                    size = (n_samples, self.latent.dim),
                    dtype = dtype)).to(self.device)

        # Copy the distribution's options, setting the shape from the
        # data and the model.
        options = \
            dict(representations_options.get("init_rep_dist_options",
                                             {}))
        options["n_samples"] = n_samples
        options["dim"] = self.latent.dim

        # Return the representation layer.
        return representations.RepresentationLayer(\
            dist = init_rep_dist,
            dist_options = options,
            device = self.device).to(dtype = dtype)


    def _setup_rep_draws(self,
                         state: SimpleNamespace) -> None:
        """Set up the noise on the training and test samples'
        representations: the noise the decoder sees and the noise
        kept in the stored representations.

        Parameters
        ----------
        state : :class:`types.SimpleNamespace`
            The training's state, updated in place.

        Returns
        -------
        None
        """

        # Get the options for training the representations.
        representations_options = state.representations_options

        # For the training and the test samples
        for phase in (state.train, state.test):

            # Get the type of noise and its options.
            noise_type = \
                representations_options[f"{phase.name}_noise_type"]
            noise_options = \
                representations_options.get(
                    f"{phase.name}_noise_options")

            # Keep the type, with no noise, no persistence and no own
            # generator by default.
            phase.noise_type = noise_type
            phase.noise_scale = 0.0
            phase.noise_persist = False
            phase.noise_generator = None

            # If the noise is Gaussian
            if noise_type == "gaussian":

                # Get the noise scale (zero disables the noise), the
                # starting and ending multipliers (for cosine
                # annealing), and the gain factor.
                phase.noise_scale_base = noise_options["scale"]
                phase.noise_start = noise_options["start"]
                phase.noise_end = noise_options["end"]
                phase.noise_gain = noise_options["gain"]

                # Get the radius holding the given fraction of the
                # noise's mass.
                phase.noise_radius = \
                    float(chi2.ppf(noise_options["within_radius_prob"],
                                   self.latent.dim)) ** 0.5

                # Get whether the noise stays in the stored
                # representations.
                phase.noise_persist = noise_options["persist"]

            # If the noise follows the optimizer's step sizes
            elif noise_type == "langevin":

                # Get the temperature the representations are sampled
                # at, its multipliers at the start and end of the
                # annealing, and the fraction of training it spans.
                phase.noise_temperature = noise_options["temperature"]
                phase.noise_start = noise_options["start"]
                phase.noise_end = noise_options["end"]
                phase.noise_end_fraction = noise_options["end_fraction"]

                # The noise stays in the stored representations.
                phase.noise_persist = True

            # If the test samples get noise
            if not phase.is_train \
            and noise_type in ("gaussian", "langevin"):

                # Get the seed of the noise's generator, by default the
                # seed torch was given.
                seed = noise_options["seed"]
                if seed is None:
                    seed = torch.initial_seed()

                # Create the noise's own generator.
                phase.noise_generator = \
                    torch.Generator(device = self.device)
                phase.noise_generator.manual_seed(int(seed))

        #-------------------------------------------------------------#

        # Get the type of noise kept in the stored representations,
        # apart from the noise the decoder sees, and its options.
        state.kept_noise_type = \
            representations_options.get("kept_noise_type", "none")
        state.kept_noise_options = \
            representations_options.get("kept_noise_options")

        # Set no kept noise by default.
        state.kept_noise_scale = 0.0

        # If any noise is kept
        if state.kept_noise_type in ("gaussian", "langevin"):

            # Get the seed of its generator, by default one past the
            # seed torch was given.
            seed = state.kept_noise_options["seed"]
            if seed is None:
                seed = torch.initial_seed() + 1

            # Create the kept noise's own generator.
            state.kept_noise_generator = \
                torch.Generator(device = self.device)
            state.kept_noise_generator.manual_seed(int(seed))


    def _update_rep_draws(self,
                          state: SimpleNamespace,
                          epoch: int) -> None:
        """Get the noise scales of an epoch.

        Parameters
        ----------
        state : :class:`types.SimpleNamespace`
            The training's state, updated in place.

        epoch : :class:`int`
            The epoch.

        Returns
        -------
        None
        """

        # Get the number of epochs' last index (at least one).
        last = max(state.n_epochs - 1, 1)

        # For the training and the test samples
        for phase in (state.train, state.test):

            # If the noise is Gaussian and enabled
            if phase.noise_type == "gaussian" \
            and phase.noise_scale_base > 0:

                # If these are the training samples
                if phase.is_train:

                    # Get the multiplier, cosine-annealed between the
                    # start and end values.
                    noise_scale = \
                        self._cosine_anneal(\
                            start = phase.noise_start,
                            end = phase.noise_end,
                            progress = (epoch - 1) / last)

                # Otherwise
                else:

                    # Get the multiplier, with the test samples' own
                    # order of operations.
                    noise_scale = \
                        phase.noise_end + \
                            (phase.noise_start - phase.noise_end) * \
                                0.5 * \
                                    (1 + math.cos(math.pi * \
                                        (epoch - 1) / last))

                # Multiply it by the base scale.
                phase.noise_scale = \
                    noise_scale * phase.noise_scale_base

            # If the noise follows the optimizer's step sizes
            elif phase.noise_type == "langevin":

                # Get the temperature's multiplier, cosine-annealed
                # over the fraction of training the annealing spans.
                phase.noise_scale = \
                    self._cosine_anneal(\
                        start = phase.noise_start,
                        end = phase.noise_end,
                        progress = \
                            min((epoch - 1) / last / \
                                    phase.noise_end_fraction,
                                1.0))

                # Inject no noise when the temperature is zero.
                if phase.noise_temperature == 0:
                    phase.noise_scale = 0.0

            # Otherwise
            else:

                # No noise will be injected.
                phase.noise_scale = 0.0

        #-------------------------------------------------------------#

        # Get the kept noise's options.
        options = state.kept_noise_options

        # If Gaussian noise is kept in the stored representations
        if state.kept_noise_type == "gaussian":

            # Get its scale, cosine-annealed between the start and end
            # values.
            state.kept_noise_scale = \
                self._cosine_anneal(\
                    start = options["start"],
                    end = options["end"],
                    progress = (epoch - 1) / last) \
                * options["scale"]

        # If the kept noise follows the optimizer's step sizes
        elif state.kept_noise_type == "langevin":

            # Get the temperature's multiplier, cosine-annealed over the
            # fraction of training the annealing spans.
            state.kept_noise_scale = \
                self._cosine_anneal(\
                    start = options["start"],
                    end = options["end"],
                    progress = \
                        min((epoch - 1) / last / \
                                options["end_fraction"],
                            1.0))

            # Keep no noise when the temperature is zero.
            if options["temperature"] == 0:
                state.kept_noise_scale = 0.0


    def _draw_latent(self,
                     state: SimpleNamespace,
                     phase: SimpleNamespace,
                     samples_ixs: torch.Tensor) -> \
                        tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Get a batch's representations as the decoder sees them:
        the points, plus the epoch's noise.

        Parameters
        ----------
        state : :class:`types.SimpleNamespace`
            The training's state.

        phase : :class:`types.SimpleNamespace`
            The set of samples.

        samples_ixs : :class:`torch.Tensor`
            The indexes of the batch's samples.

        Returns
        -------
        z : :class:`torch.Tensor`
            The representations decoded.

        noise : :class:`torch.Tensor` or :obj:`None`
            The noise added, or :obj:`None` if there is none.
        """

        # Get the representations for the current samples.
        z = phase.rep_layer(ixs = samples_ixs).to(self.device)

        # If noise injection is not enabled
        if not phase.noise_scale > 0:

            # Return the points.
            return z, None

        #-------------------------------------------------------------#

        # If the noise is Gaussian and drawn from the global stream
        # (the training samples)
        if phase.noise_type == "gaussian" \
        and phase.noise_generator is None:

            # Get the base noise.
            base_noise = torch.randn_like(z) / phase.noise_radius

            # Scale it to get the final noise to inject.
            noise = \
                phase.noise_scale * base_noise * phase.noise_gain

        # If the noise is Gaussian and drawn from its own generator
        # (the test samples)
        elif phase.noise_type == "gaussian":

            # Get the noise.
            noise = \
                phase.noise_scale * phase.noise_gain * \
                    torch.randn(z.shape,
                                generator = phase.noise_generator,
                                device = z.device,
                                dtype = z.dtype) / phase.noise_radius

        # Otherwise (the noise follows the optimizer)
        else:

            # Get the noise.
            noise = \
                self._draw_langevin_noise(phase = phase,
                                          z = z,
                                          samples_ixs = samples_ixs)

        # Return the representations with the noise added, and the
        # noise.
        return z + noise, noise


    def _draw_langevin_noise(self,
                             phase: SimpleNamespace,
                             z: torch.Tensor,
                             samples_ixs: torch.Tensor) -> \
                                torch.Tensor:
        """Draw the noise scaled to the optimizer's step sizes for a
        batch's representations.

        Parameters
        ----------
        phase : :class:`types.SimpleNamespace`
            The set of samples.

        z : :class:`torch.Tensor`
            The batch's representations.

        samples_ixs : :class:`torch.Tensor`
            The indexes of the batch's samples.

        Returns
        -------
        noise : :class:`torch.Tensor`
            The noise.
        """

        # Get the optimizer's state for the representations.
        rep_state = phase.optimizer.state.get(phase.rep_layer.z, {})

        # If the optimizer has not taken a step yet
        if "exp_avg_sq" not in rep_state:

            # No noise is drawn before the first step.
            return torch.zeros_like(z)

        # Get the optimizer's options.
        rep_group = phase.optimizer.param_groups[0]

        # Get the bias-corrected second moments of the batch's
        # gradients.
        v_hat = \
            rep_state["exp_avg_sq"][samples_ixs] / \
                (1 - rep_group["betas"][1] ** float(rep_state["step"]))

        # Get each coordinate's step size.
        step_size = rep_group["lr"] / (v_hat.sqrt() + rep_group["eps"])

        # Draw the standard normal values, from the global stream or
        # from the noise's own generator.
        draw = \
            torch.randn_like(z) if phase.noise_generator is None \
            else torch.randn(z.shape,
                             generator = phase.noise_generator,
                             device = z.device,
                             dtype = z.dtype)

        # Return the noise, with a variance of twice the step size
        # times the temperature.
        return draw * \
            torch.sqrt(2.0 * phase.noise_temperature * \
                           phase.noise_scale * step_size)


    def _rep_loss_terms(self,
                        state: SimpleNamespace,
                        phase: SimpleNamespace,
                        samples_ixs: torch.Tensor) -> \
                            Optional[torch.Tensor]:
        """Get the representations' own terms of a batch's loss.

        Parameters
        ----------
        state : :class:`types.SimpleNamespace`
            The training's state.

        phase : :class:`types.SimpleNamespace`
            The set of samples.

        samples_ixs : :class:`torch.Tensor`
            The indexes of the batch's samples.

        Returns
        -------
        loss : :obj:`None`
            None: points have no terms of their own.
        """

        # Return no terms.
        return None


    def _after_backward(self,
                        state: SimpleNamespace,
                        phase: SimpleNamespace,
                        samples_ixs: torch.Tensor,
                        draw: Optional[torch.Tensor]) -> None:
        """Add the noise kept in a batch's stored representations,
        after the backward step.

        Parameters
        ----------
        state : :class:`types.SimpleNamespace`
            The training's state.

        phase : :class:`types.SimpleNamespace`
            The set of samples.

        samples_ixs : :class:`torch.Tensor`
            The indexes of the batch's samples.

        draw : :class:`torch.Tensor` or :obj:`None`
            The noise the decoder saw, or :obj:`None` if none.

        Returns
        -------
        None
        """

        # If the noise stays in the stored representations
        if phase.noise_scale > 0 and phase.noise_persist:

            # Add it to them.
            with torch.no_grad():
                phase.rep_layer.z[samples_ixs] += draw

        # If a separate noise is kept in them
        if state.kept_noise_scale > 0:

            # Add it to them.
            with torch.no_grad():
                phase.rep_layer.z[samples_ixs] += \
                    self._draw_kept_noise(\
                        noise_type = state.kept_noise_type,
                        noise_options = state.kept_noise_options,
                        noise_scale = state.kept_noise_scale,
                        generator = state.kept_noise_generator,
                        optimizer = phase.optimizer,
                        rep_layer = phase.rep_layer,
                        samples_ixs = samples_ixs)


    def _epoch_log_extras(self,
                          state: SimpleNamespace) -> str:
        """Get what the epoch's log line reports about the training
        samples' noise.

        Parameters
        ----------
        state : :class:`types.SimpleNamespace`
            The training's state.

        Returns
        -------
        text : :class:`str`
            The text added to the log line (empty without noise).
        """

        # Get the training samples.
        train = state.train

        # If the noise is Gaussian and enabled
        if train.noise_type == "gaussian" \
        and train.noise_scale_base > 0:

            # Return the noise scale.
            return f", noise scale {train.noise_scale:.6f}"

        # If the noise follows the optimizer's step sizes
        if train.noise_type == "langevin":

            # Return the temperature.
            return \
                ", noise temperature " \
                f"{train.noise_temperature * train.noise_scale:.6f}"

        # Return nothing.
        return ""


    def _latent_fit_inputs(self,
                           rep_layer: \
                            representations.RepresentationLayer) -> \
                                tuple[torch.Tensor, None]:
        """Get what the mixture is fitted to.

        Parameters
        ----------
        rep_layer : \
            :class:`bulkdgd.core.representations.RepresentationLayer`
            The training samples' representations.

        Returns
        -------
        points : :class:`torch.Tensor`
            The representations.

        variances : :obj:`None`
            None: points have no widths.
        """

        # Return the representations, detached.
        return rep_layer().detach().to(self.device), None


    def _save_rep_outputs(self,
                          epoch: int,
                          rep_layer_train: \
                            representations.RepresentationLayer,
                          rep_layer_test: \
                            representations.RepresentationLayer,
                          samples_names_train: list[str],
                          samples_names_test: list[str],
                          save_dir: Optional[str]) -> None:
        """Save the training and test samples' representations at an
        epoch.

        Parameters
        ----------
        epoch : :class:`int`
            The epoch.

        rep_layer_train : \
            :class:`bulkdgd.core.representations.RepresentationLayer`
            The training samples' representations.

        rep_layer_test : \
            :class:`bulkdgd.core.representations.RepresentationLayer`
            The test samples' representations.

        samples_names_train : :class:`list`
            The training samples' names.

        samples_names_test : :class:`list`
            The test samples' names.

        save_dir : :class:`str` or :obj:`None`
            The folder the files are written to.

        Returns
        -------
        None
        """

        # For each set of samples, its prefix and names
        for prefix, rep_layer, samples_names in \
            (("train", rep_layer_train, samples_names_train),
             ("test", rep_layer_test, samples_names_test)):

            # Save the set's current representations.
            _util.save_rep_epoch(\
                epoch = epoch,
                prefix = prefix,
                latent_dim = self.latent.dim,
                save_dir = save_dir,
                rep_layer = rep_layer,
                samples_names = samples_names)


    ########################## SEARCH HOOKS ###########################


    def _validate_config_rep(self,
                             config_rep: dict[str, object]) -> \
                                list[str]:
        """Check the parsed search configuration against the model.

        Parameters
        ----------
        config_rep : :class:`dict`
            The parsed search configuration.

        Returns
        -------
        errors : :class:`list`
            The errors found (none: every option fits the model).
        """

        # Return no errors.
        return []


    def _make_search_layer(self,
                           values: torch.Tensor) -> \
                            representations.RepresentationLayer:
        """Build a search's representation layer.

        Parameters
        ----------
        values : :class:`torch.Tensor`
            The representations.

        Returns
        -------
        rep_layer : \
            :class:`bulkdgd.core.representations.RepresentationLayer`
            The representation layer, on the model's device.
        """

        # Return the layer.
        return representations.RepresentationLayer(\
                    values = values,
                    device = self.device)


    def _setup_search_draws(self,
                            noise_type: Optional[str],
                            noise_options: Optional[dict],
                            opt_num: int) -> SimpleNamespace:
        """Set up the noise on a search's representations.

        Parameters
        ----------
        noise_type : :class:`str` or :obj:`None`
            The type of noise (only ``"gaussian"``); anything else
            disables it.

        noise_options : :class:`dict` or :obj:`None`
            The noise options: ``scale``, ``start``/``end`` (the
            cosine-annealed scale multipliers), ``within_radius_prob``
            and ``gain``.

        opt_num : :class:`int`
            The number of the optimization round.

        Returns
        -------
        draws : :class:`types.SimpleNamespace`
            The noise's settings.
        """

        # Get the noise options.
        noise_options = noise_options or {}

        # Get the base noise scale (zero disables the noise), and no
        # noise at the start.
        draws = \
            SimpleNamespace(\
                noise_scale_base = \
                    float(noise_options.get("scale", 0.0))
                    if noise_type == "gaussian" else 0.0,
                noise_scale = 0.0)

        # If the noise is disabled
        if not draws.noise_scale_base > 0:

            # Return the settings.
            return draws

        #-------------------------------------------------------------#

        # Get the annealing schedule and the gain.
        draws.noise_start = float(noise_options["start"])
        draws.noise_end = float(noise_options["end"])
        draws.noise_gain = float(noise_options["gain"])

        # Get the radius holding the given fraction of the mass.
        draws.noise_radius = \
            float(chi2.ppf(
                float(noise_options["within_radius_prob"]),
                self.latent.dim)) ** 0.5

        # Inform the user about the noise.
        logger.info(
            f"Optimization number {opt_num} will inject Gaussian "
            "noise into the representations (scale "
            f"{draws.noise_scale_base}, annealed from "
            f"{draws.noise_start} to {draws.noise_end}).")

        # Return the settings.
        return draws


    def _update_search_draws(self,
                             draws: SimpleNamespace,
                             epoch: int,
                             epochs: int) -> None:
        """Get a search epoch's noise scale.

        Parameters
        ----------
        draws : :class:`types.SimpleNamespace`
            The noise's settings, updated in place.

        epoch : :class:`int`
            The epoch.

        epochs : :class:`int`
            The number of epochs.

        Returns
        -------
        None
        """

        # If the noise is disabled
        if not draws.noise_scale_base > 0:

            # Stop here.
            return

        # Get the scale, cosine-annealed between 'start' and 'end',
        # times the base scale.
        draws.noise_scale = \
            self._cosine_anneal(\
                start = draws.noise_start,
                end = draws.noise_end,
                progress = (epoch - 1) / max(epochs - 1, 1)) \
            * draws.noise_scale_base


    def _search_objective(self,
                          spec: SimpleNamespace,
                          rep_layer: \
                            representations.RepresentationLayer,
                          draws: SimpleNamespace,
                          batch: SimpleNamespace) -> torch.Tensor:
        """Get the loss a search minimizes on a batch: the negative
        log-posterior of the candidates (plus the output module's
        dispersion regularization).

        Parameters
        ----------
        spec : :class:`types.SimpleNamespace`
            The search's shared settings.

        rep_layer : \
            :class:`bulkdgd.core.representations.RepresentationLayer`
            The layer holding every sample's candidates.

        draws : :class:`types.SimpleNamespace`
            The noise's settings.

        batch : :class:`types.SimpleNamespace`
            The batch.

        Returns
        -------
        loss : :class:`torch.Tensor`
            The batch's total loss (a single-value tensor).
        """

        # Get the batch's candidates.
        z = self._get_candidates(spec = spec,
                                 rep_layer = rep_layer,
                                 batch = batch)

        # If the noise is enabled
        if draws.noise_scale > 0:

            # Add it to the decoder's input (not to the optimized
            # representations).
            z = z + draws.noise_scale * draws.noise_gain * \
                    torch.randn_like(z) / draws.noise_radius

        # Get the reconstruction loss and the decoder's outputs.
        recon_loss, pred_means, pred_log_r_values = \
            self._get_candidates_recon_loss(spec = spec,
                                            batch = batch,
                                            z = z)

        #-------------------------------------------------------------#

        # If the reduction method is 'sum'
        if spec.loss_reduction_type == "sum":

            # Sum the reconstruction loss.
            recon_loss_final = recon_loss.sum().clone()

        # If the reduction method is 'mean'
        elif spec.loss_reduction_type == "mean":

            # Average the reconstruction loss.
            recon_loss_final = recon_loss.mean().clone()

        # Get the latent loss.
        latent_loss_final = \
            self._get_latent_loss(\
                z = z,
                reduction = spec.loss_reduction_type,
                latent_lambda = spec.latent_lambda)

        # Get the output module's dispersion regularization (zero
        # unless it shrinks the dispersion).
        dispersion_reg = \
            self.decoder.nb.dispersion_regularization(
                pred_means = pred_means,
                pred_log_r_values = pred_log_r_values,
                reduction = spec.loss_reduction_type)

        #-------------------------------------------------------------#

        # Return the total loss.
        return recon_loss_final + latent_loss_final + dispersion_reg


    def _selection_score(self,
                         spec: SimpleNamespace,
                         rep_layer: representations.RepresentationLayer,
                         batch: SimpleNamespace) -> \
                            tuple[torch.Tensor, torch.Tensor,
                                  torch.Tensor, torch.Tensor]:
        """Score a batch's candidates for the selection by their
        negative log-posterior (lower is better).

        Parameters
        ----------
        spec : :class:`types.SimpleNamespace`
            The search's shared settings.

        rep_layer : \
            :class:`bulkdgd.core.representations.RepresentationLayer`
            The layer holding every sample's candidates.

        batch : :class:`types.SimpleNamespace`
            The batch.

        Returns
        -------
        z : :class:`torch.Tensor`
            The candidates, shaped (batch * reps * components, dim).

        total_loss : :class:`torch.Tensor`
            Each candidate's score, shaped (batch * reps *
            components,).

        recon_loss : :class:`torch.Tensor`
            Each candidate's reconstruction loss, same shape.

        latent_loss : :class:`torch.Tensor`
            Each candidate's latent loss, same shape.
        """

        # Get the batch's candidates.
        z = self._get_candidates(spec = spec,
                                 rep_layer = rep_layer,
                                 batch = batch)

        # Get the reconstruction loss.
        recon_loss, _, _ = \
            self._get_candidates_recon_loss(spec = spec,
                                            batch = batch,
                                            z = z)

        # If the reduction method is 'sum'
        if spec.loss_reduction_type == "sum":

            # Get the reconstruction loss per candidate by summing over
            # the genes: shape (batch, reps, components).
            recon_loss_final = recon_loss.sum(-1).clone()

        # If the reduction method is 'mean'
        elif spec.loss_reduction_type == "mean":

            # Get the reconstruction loss per candidate by averaging
            # over the genes: shape (batch, reps, components).
            recon_loss_final = recon_loss.mean(-1).clone()

        # Flatten it to (batch * reps * components).
        recon_loss_final = \
            recon_loss_final.view(batch.n * \
                                  spec.n_rep_per_comp * \
                                  spec.n_components)

        # Get the latent loss per candidate.
        latent_loss = \
            self._get_latent_loss(z = z,
                                  reduction = None,
                                  latent_lambda = spec.latent_lambda)

        #-------------------------------------------------------------#

        # Return the candidates, their total loss and its parts.
        return z, recon_loss_final + latent_loss, recon_loss_final, \
            latent_loss
