#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-

#    ensemble.py
#
#    An ensemble of bulkDGD models differing only in the seed they were
#    trained with, and the tiered consensus drawn from them.
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
    """An ensemble of bulkDGD models differing only in the seed they
    were trained with, and the tiered consensus drawn from them."""


#######################################################################


# Import from the standard library.
import copy
import logging as log
import multiprocessing as mp
import os
import shutil
import time
import traceback
from typing import Optional
import zipfile

# Import from third-party libraries.
import pandas as pd

# Import from the package.
import torch
import yaml

# Import from 'bulkdgd'.
import bulkdgd
from bulkdgd import defaults
from bulkdgd.analysis import dea as analysis_dea
from bulkdgd.core.model import BulkDGD
from bulkdgd.ioutil import deaio
from bulkdgd.ioutil.tableio import load_table, save_table
from bulkdgd.reproducibility import set_seeds


#######################################################################


# Get the module's logger.
logger = log.getLogger(__name__)


#######################################################################


def _get_genes_called(item: tuple) -> tuple:
    """Get, for one sample, the genes each model of the ensemble calls
    for it.

    Parameters
    ----------
    item : :class:`tuple`
        A tuple containing the sample's name, the group the sample
        belongs to, the directories containing the models' results,
        the prefix the per-sample files are named with, the columns to
        read, and the thresholds a gene must pass to be called.

    Returns
    -------
    result : :class:`tuple` or :obj:`None`
        A tuple containing the group the sample belongs to and the
        list of genes each model calls for it, or :obj:`None` if any
        of the models has no results for the sample.
    """

    # Unpack the item (a single tuple, for the worker pool).
    sample, group, dea_dirs, prefix, usecols, q_val, log2_fold_change \
        = item

    #-----------------------------------------------------------------#

    # Initialize the list to store the genes each model calls.
    genes_called = []

    # For each model's results
    for dea_dir in dea_dirs:

        # Read the sample's statistics, packed or loose.
        df_stats = deaio.read_dea(dea_dir,
                                   sample,
                                   prefix = prefix,
                                   index_col = 0,
                                   header = 0,
                                   usecols = usecols)

        # If the model has no results for the sample
        if df_stats is None:

            # Return nothing, dropping the sample for every model.
            return None

        # Get the required columns the statistics are missing, by name
        # ('usecols' is ignored for Parquet files).
        missing = [c for c in ("q_value", "log2_fold_change")
                   if c not in df_stats.columns]

        # If any column is missing
        if missing:

            # Raise an error.
            errstr = \
                f"The statistics for sample '{sample}' have no " \
                f"{', '.join(repr(c) for c in missing)} column. The " \
                f"columns found were: " \
                f"{', '.join(repr(c) for c in df_stats.columns)}."
            raise KeyError(errstr)

        # Keep only the required columns.
        df_stats = df_stats[["q_value", "log2_fold_change"]]

        # Get the genes the model calls for the sample.
        genes = \
            df_stats.index[
                (df_stats["q_value"] < q_val) \
                & (df_stats["log2_fold_change"].abs() \
                    > log2_fold_change)]

        # Add them to the list.
        genes_called.append(list(genes))

    #-----------------------------------------------------------------#

    # Return the group the sample belongs to and the genes called.
    return group, genes_called


#######################################################################


class BulkDGDEnsemble:

    """A class implementing an ensemble of bulkDGD models differing
    only in the seed they were trained with.
    """

    ######################## CLASS ATTRIBUTES #########################


    # The keys each member of the ensemble must have.
    REQUIRED_KEYS = ("seed", "model_dir", "results_dir")

    #-----------------------------------------------------------------#

    # The file storing a trained latent space's parameters.
    GMM_PTH_FILE = "gmm.pth"

    # The file storing a trained decoder's parameters.
    DEC_PTH_FILE = "dec.pth"

    # The file storing the final Gaussian mixture model's parameters.
    GMM_FINAL_PTH_FILE = "gmm_final.pth"

    # The file recording what a member was seeded with.
    SEEDS_FILE = "seeds.yaml"

    #-----------------------------------------------------------------#

    # The files a member's training writes beside its parameters.
    LOSS_FILE = "loss.csv"
    TIME_FILE = "time.csv"

    #-----------------------------------------------------------------#

    # The files a member's representations are written to.
    REP_FILE = "representations.csv"
    PRED_MEANS_FILE = "pred_means.csv"
    PRED_R_VALUES_FILE = "pred_r_values.csv"

    #-----------------------------------------------------------------#

    # The file a member's enrichment scores are written to.
    E_SCORES_FILE = "e_scores.csv"

    #-----------------------------------------------------------------#

    # The default thresholds for calling a gene in a sample, the
    # default recurrence, and the default minimum tier.
    DEFAULT_Q_VAL = 0.05
    DEFAULT_LOG2_FOLD_CHANGE = 1.0
    DEFAULT_RECURRENCE = 0.20
    DEFAULT_MIN_TIER = 2


    ######################### INITIALIZATION ##########################


    def __init__(self,
                 config_model: Optional[dict[str, object]] = None,
                 config_ensemble: \
                     Optional[dict[str, dict[str, object]]] = None,
                 device: str = "cpu") -> None:
        """Initialize an instance of the class.

        Parameters
        ----------
        config_model : :class:`dict`, optional
            The configuration of the model the ensemble is made of,
            as a single :class:`bulkdgd.core.model.BulkDGD` takes it,
            shared by every member.

            For the available options, refer to the
            :ref:`model_config_options` page.

            If neither ``config_model`` nor ``config_ensemble`` is
            passed, the ensemble that ships with the package is used.

        config_ensemble : :class:`dict`, optional
            The configuration of the ensemble, mapping each member's
            name to a dictionary with these keys:

            * ``"seed"`` (:class:`int`) - the seed the member is
              trained with. No two members may share it.

            * ``"model_dir"`` (:class:`str`) - the directory where the
              member's trained parameters live.

            * ``"results_dir"`` (:class:`str`) - the directory where
              the analyses run with the member live.

        device : :class:`str`, ``"cpu"``
            The device the members are placed on when they are built.
        """

        # If neither configuration was passed
        if config_model is None and config_ensemble is None:

            # Use the ones of the ensemble shipped with the package.
            config_model, config_ensemble = self.shipped_config()

        # If only one configuration was passed
        elif config_model is None or config_ensemble is None:

            # Raise an error.
            errstr = \
                "'config_model' and 'config_ensemble' describe the " \
                "ensemble together and must be given together. Give " \
                "neither to use the ensemble that ships with the " \
                "package."
            raise ValueError(errstr)

        #-------------------------------------------------------------#

        # Save a copy of the model's configuration.
        self._config_model = copy.deepcopy(config_model)

        #-------------------------------------------------------------#

        # Check and save the ensemble's configuration.
        self._config_ensemble = \
            self._check_config_ensemble(
                config_ensemble = config_ensemble)

        #-------------------------------------------------------------#

        # Save the device the members are placed on.
        self._device = device

        #-------------------------------------------------------------#

        # Inform the user about the ensemble that was set.
        logger.info(
            f"The ensemble was successfully set ({self.n_models} "
            f"models, seeds: "
            f"{', '.join(str(s) for s in self.seeds)}).")

        #-------------------------------------------------------------#

        # If the model's configuration sets the mixture's own seed,
        # which every member would share
        if self._config_model.get(
                "latent_options", {}).get("random_state") is not None:

            # Warn the user.
            logger.warning(
                "The model's configuration sets "
                "'latent_options.random_state', which every member of "
                "the ensemble is built with. The members will be less "
                "different from each other than their seeds suggest. "
                "When it is unset, each member's own seed also seeds "
                "the mixture.")


    @staticmethod
    def shipped_config() -> tuple:
        """Get the configuration of the ensemble that ships with the
        package.

        Returns
        -------
        config_model : :class:`dict`
            The configuration of the model, shared by every member,
            without the paths to the trained parameters.

        config_ensemble : :class:`dict`
            The configuration of the ensemble, with one entry per
            shipped member.
        """

        # Import the loader of the shipped models.
        from bulkdgd.core import _util

        # Read the shared model configuration from the base member.
        config_model = _util.load_shipped_model(
            seed = defaults.BASE_SEED)

        # For each section pointing at a file of trained parameters
        for key in ("latent_options", "decoder_options"):

            # Copy the section.
            config_model[key] = dict(config_model[key])

        # Drop the files of trained parameters, which 'get_model'
        # sets for each member.
        config_model["latent_options"].pop("latent_pth_file", None)
        config_model["decoder_options"].pop("decoder_pth_file", None)

        # Put the results under the working directory.
        results_root = os.path.join(os.getcwd(),
                                    "bulkdgd_ensemble_results")

        # Build one entry per shipped member.
        config_ensemble = {
            seed : {"seed" : int(seed.removeprefix("seed")),
                    "model_dir" : defaults.model_dir(seed),
                    "results_dir" : os.path.join(results_root, seed)}
            for seed in defaults.ENSEMBLE_SEEDS}

        # Return the configurations.
        return config_model, config_ensemble


    def _check_config_ensemble(
            self,
            config_ensemble: dict[str, dict[str, object]]) -> \
                dict[str, dict[str, object]]:
        """Check the ensemble's configuration.

        Parameters
        ----------
        config_ensemble : :class:`dict`
            The ensemble's configuration.

        Returns
        -------
        config_ensemble : :class:`dict`
            The ensemble's configuration, checked.
        """

        # If the configuration is empty
        if not config_ensemble:

            # Raise an error.
            raise ValueError(
                "The ensemble's configuration is empty. It must "
                "contain at least one member.")

        #-------------------------------------------------------------#

        # Initialize an empty dictionary to store the seeds found so
        # far, and which member had them.
        seeds_found = {}

        # For each member of the ensemble
        for name, options in config_ensemble.items():

            # If the member's options are not a dictionary
            if not isinstance(options, dict):

                # Raise an error.
                raise TypeError(
                    f"The options for the member '{name}' must be a "
                    "dictionary.")

            #---------------------------------------------------------#

            # Get the keys the member is missing.
            keys_missing = \
                [key for key in self.REQUIRED_KEYS
                 if key not in options]

            # If the member is missing any key
            if keys_missing:

                # Raise an error.
                raise KeyError(
                    f"The member '{name}' is missing these required "
                    f"keys: {', '.join(keys_missing)}.")

            #---------------------------------------------------------#

            # Get the member's seed.
            seed = options["seed"]

            # If the seed is not an integer
            if not isinstance(seed, int) or isinstance(seed, bool):

                # Raise an error.
                raise TypeError(
                    f"The seed of the member '{name}' must be an "
                    "integer.")

            # If the seed was already used by another member
            if seed in seeds_found:

                # Raise an error.
                raise ValueError(
                    f"The members '{seeds_found[seed]}' and "
                    f"'{name}' have the same seed ({seed}). The "
                    "members of an ensemble differ in the seed they "
                    "are trained with, so each seed may appear only "
                    "once.")

            # Record which member had the seed.
            seeds_found[seed] = name

        #-------------------------------------------------------------#

        # Return a copy of the configuration.
        return copy.deepcopy(config_ensemble)


    @classmethod
    def from_existing(
            cls,
            config_model: dict[str, object],
            config_ensemble: dict[str, dict[str, object]],
            device: str = "cpu") -> "BulkDGDEnsemble":
        """Build an ensemble from models that were already trained,
        checking that their trained parameters exist without loading
        them.

        Parameters
        ----------
        config_model : :class:`dict`
            The configuration of the model the ensemble is made of.

        config_ensemble : :class:`dict`
            The configuration of the ensemble.

        device : :class:`str`, ``"cpu"``
            The device the members are placed on when they are built.

        Returns
        -------
        ensemble : :class:`BulkDGDEnsemble`
            The ensemble.
        """

        # Build the ensemble.
        ensemble = cls(config_model = config_model,
                       config_ensemble = config_ensemble,
                       device = device)

        #-------------------------------------------------------------#

        # Initialize an empty list to store the members whose trained
        # parameters are missing.
        members_missing = []

        # For each member of the ensemble
        for name, options in ensemble.config_ensemble.items():

            # Get the files the member must have (a shipped member's
            # decoder is downloaded when first used).
            pth_files = \
                (cls.GMM_PTH_FILE,) \
                if os.path.basename(options["model_dir"]) \
                    in defaults.ENSEMBLE_SEEDS \
                else (cls.GMM_PTH_FILE, cls.DEC_PTH_FILE)

            # For each file storing the trained parameters
            for pth_file in pth_files:

                # Get the path to the file.
                path = os.path.join(options["model_dir"], pth_file)

                # If the file is not there
                if not os.path.isfile(path):

                    # Add it to the list.
                    members_missing.append(f"'{name}' ({path})")

        #-------------------------------------------------------------#

        # If any member's parameters are missing
        if members_missing:

            # Raise an error.
            raise FileNotFoundError(
                "These members have no trained parameters: "
                f"{'; '.join(members_missing)}. Train them, or point "
                "the ensemble's configuration at the directories "
                "where they live.")

        #-------------------------------------------------------------#

        # Inform the user that the models were found.
        logger.info(
            f"The trained parameters of all {ensemble.n_models} "
            "members were found.")

        #-------------------------------------------------------------#

        # Return the ensemble.
        return ensemble


    ############################ PROPERTIES ###########################


    @property
    def config_model(self) -> dict[str, object]:
        """The configuration of the model the ensemble is made of.
        """

        return self._config_model


    @config_model.setter
    def config_model(self,
                     value) -> None:
        """Raise an exception if the user tries to modify the
        configuration of the model the ensemble is made of.

        Parameters
        ----------
        value
            The new value.
        """

        # Raise an error.
        raise ValueError(
            "The configuration of the model the ensemble is made of "
            "cannot be changed after the ensemble is initialized.")


    #-----------------------------------------------------------------#


    @property
    def config_ensemble(self) -> dict[str, dict[str, object]]:
        """The configuration of the ensemble.
        """

        return self._config_ensemble


    @config_ensemble.setter
    def config_ensemble(self,
                        value) -> None:
        """Raise an exception if the user tries to modify the
        configuration of the ensemble.

        Parameters
        ----------
        value
            The new value.
        """

        # Raise an error.
        raise ValueError(
            "The configuration of the ensemble cannot be changed "
            "after the ensemble is initialized. Pass a different "
            "configuration to the method you are calling to write "
            "its results elsewhere.")


    #-----------------------------------------------------------------#


    @property
    def names(self) -> list:
        """The names of the ensemble's members.
        """

        return list(self._config_ensemble.keys())


    #-----------------------------------------------------------------#


    @property
    def seeds(self) -> list:
        """The seeds the ensemble's members are trained with.
        """

        return [options["seed"]
                for options in self._config_ensemble.values()]


    #-----------------------------------------------------------------#


    @property
    def n_models(self) -> int:
        """The number of models the ensemble is made of.
        """

        return len(self._config_ensemble)


    #-----------------------------------------------------------------#


    @property
    def device(self) -> str:
        """The device the ensemble's members are placed on.
        """

        return self._device


    ######################### PRIVATE METHODS #########################


    def _get_config_ensemble(
            self,
            config_ensemble: dict[str, dict[str, object]] = None) -> \
                dict[str, dict[str, object]]:
        """Get the ensemble's configuration to be used: the one
        passed, if any, or the ensemble's own.

        Parameters
        ----------
        config_ensemble : :class:`dict`, optional
            The configuration to be used.

        Returns
        -------
        config_ensemble : :class:`dict`
            The configuration to be used.
        """

        # If no configuration was passed
        if config_ensemble is None:

            # Use the ensemble's own.
            return self._config_ensemble

        #-------------------------------------------------------------#

        # Otherwise, check the one passed.
        config_ensemble = \
            self._check_config_ensemble(
                config_ensemble = config_ensemble)

        # If it does not describe the same members
        if set(config_ensemble.keys()) != set(self.names):

            # Raise an error.
            raise ValueError(
                "The configuration passed describes different "
                "members than the ensemble's own "
                f"({', '.join(sorted(config_ensemble.keys()))} "
                f"against {', '.join(sorted(self.names))}). It may "
                "point the ensemble's members at different "
                "directories, but it may not change which members "
                "there are.")

        #-------------------------------------------------------------#

        # Return the configuration.
        return config_ensemble


    def _get_dea_dir(self,
                     options: dict[str, object],
                     dea_dir: str) -> str:
        """Get the directory containing a member's differential
        expression analysis' results.

        Parameters
        ----------
        options : :class:`dict`
            The member's options.

        dea_dir : :class:`str`
            The directory containing the results. If it is a relative
            path, it is taken relative to the member's results'
            directory.

        Returns
        -------
        dea_dir : :class:`str`
            The directory containing the results.
        """

        # If the directory is an absolute path
        if os.path.isabs(dea_dir):

            # Return it as it is.
            return dea_dir

        #-------------------------------------------------------------#

        # Otherwise, take it relative to the member's results.
        return os.path.join(options["results_dir"], dea_dir)


    ######################### PUBLIC METHODS ##########################


    def get_model(self,
                  name: str,
                  config_ensemble: dict[str, dict[str, object]] = \
                    None) -> BulkDGD:
        """Get one member of the ensemble, with its trained parameters
        loaded.

        Parameters
        ----------
        name : :class:`str`
            The member's name.

        config_ensemble : :class:`dict`, optional
            The configuration to be used. If not passed, the
            ensemble's own is used.

        Returns
        -------
        model : :class:`bulkdgd.core.model.BulkDGD`
            The member.
        """

        # Get the configuration to be used.
        config_ensemble = \
            self._get_config_ensemble(
                config_ensemble = config_ensemble)

        #-------------------------------------------------------------#

        # If the member is not in the ensemble
        if name not in config_ensemble:

            # Raise an error.
            raise KeyError(
                f"'{name}' is not a member of the ensemble. The "
                f"members are: {', '.join(self.names)}.")

        #-------------------------------------------------------------#

        # Get the member's options.
        options = config_ensemble[name]

        # Get a copy of the model's configuration.
        config_model = copy.deepcopy(self._config_model)

        # Point it at the member's trained latent space.
        config_model["latent_options"]["latent_pth_file"] = \
            os.path.join(options["model_dir"], self.GMM_PTH_FILE)

        # Get the file of the member's trained decoder.
        decoder_pth_file = \
            os.path.join(options["model_dir"], self.DEC_PTH_FILE)

        # If the decoder's file is missing and the member is one the
        # package ships
        if not os.path.isfile(decoder_pth_file) \
                and os.path.basename(options["model_dir"]) \
                    in defaults.ENSEMBLE_SEEDS:

            # Download the decoder's file.
            bulkdgd._internals.util.download_decoder_pth(
                dest_path = decoder_pth_file)

        # Point the configuration at the member's trained decoder.
        config_model["decoder_options"]["decoder_pth_file"] = \
            decoder_pth_file

        #-------------------------------------------------------------#

        # Return the model.
        return BulkDGD(device = self._device, **config_model)


    def status(self,
               config_ensemble: dict[str, dict[str, object]] = None,
               dea_dir: str = "dea",
               prefix: str = deaio.DEA_PREFIX) -> pd.DataFrame:
        """Report what each member of the ensemble already has on
        disk.

        Parameters
        ----------
        config_ensemble : :class:`dict`, optional
            The configuration to be used. If not passed, the
            ensemble's own is used.

        dea_dir : :class:`str`, ``"dea"``
            The directory containing a member's differential
            expression analysis' results. If it is a relative path, it
            is taken relative to the member's results' directory.

        prefix : :class:`str`, ``"dea_"``
            The prefix the per-sample files are named with.

        Returns
        -------
        df_status : :class:`pandas.DataFrame`
            A data frame with one row per member, reporting the seed
            the member is trained with, whether its trained parameters
            are there, how many samples it has results for, and
            whether the results are packed.
        """

        # Get the configuration to be used.
        config_ensemble = \
            self._get_config_ensemble(
                config_ensemble = config_ensemble)

        #-------------------------------------------------------------#

        # Initialize an empty list to store the members' status.
        rows = []

        # For each member of the ensemble
        for name, options in config_ensemble.items():

            # Get the directory containing the member's differential
            # expression analysis' results.
            dea_dir_member = \
                self._get_dea_dir(options = options,
                                  dea_dir = dea_dir)

            # Get the samples the member has results for.
            samples = deaio.list_samples(dea_dir_member,
                                         prefix = prefix)

            # Get the files the member must have (only the latent
            # space's for a shipped member).
            pth_files = \
                (self.GMM_PTH_FILE,) \
                if os.path.basename(options["model_dir"]) \
                    in defaults.ENSEMBLE_SEEDS \
                else (self.GMM_PTH_FILE, self.DEC_PTH_FILE)

            #---------------------------------------------------------#

            # Add the member's status to the list.
            rows.append(
                {"name" : name,
                 "seed" : options["seed"],
                 "trained" : \
                    all(os.path.isfile(
                            os.path.join(options["model_dir"],
                                         pth_file))
                        for pth_file in pth_files),
                 "n_samples_dea" : len(samples),
                 "dea_packed" : \
                    os.path.isfile(
                        os.path.join(dea_dir_member,
                                     deaio.DEA_ZIP_NAME))})

        #-------------------------------------------------------------#

        # Return a data frame with the members' status.
        return pd.DataFrame(rows).set_index("name")


    def _get_model_untrained(self) -> BulkDGD:
        """Build a member of the ensemble as it is before it is
        trained.

        Returns
        -------
        model : :class:`bulkdgd.core.model.BulkDGD`
            The untrained member.
        """

        # Get the model's configuration.
        config_model = copy.deepcopy(self._config_model)

        # Drop the files of trained parameters, if any.
        config_model.get("latent_options", {}).pop(
            "latent_pth_file", None)
        config_model.get("decoder_options", {}).pop(
            "decoder_pth_file", None)

        #-------------------------------------------------------------#

        # Return the model.
        return BulkDGD(device = self._device, **config_model)


    def _write_seeds(self,
                     model_dir: str,
                     seeds: dict) -> str:
        """Write down what a member was seeded with, beside the member
        it produced.

        Parameters
        ----------
        model_dir : :class:`str`
            The directory where the member's parameters live.

        seeds : :class:`dict`
            What the member was seeded with.

        Returns
        -------
        seeds_file : :class:`str`
            The file the seeds were written to.
        """

        # Get the path to the file.
        seeds_file = os.path.join(model_dir, self.SEEDS_FILE)

        #-------------------------------------------------------------#

        # Open the file.
        with open(seeds_file, "w") as f:

            # Write the seeds, the versions, and the device.
            yaml.safe_dump(
                {"seeds" : seeds,
                 "dtype" : self._config_model.get("dtype"),
                 # Cast the versions to strings, since 'yaml.safe_dump'
                 # cannot represent a 'TorchVersion'.
                 "bulkdgd_version" : str(bulkdgd.__version__),
                 "torch_version" : str(torch.__version__),
                 # The mixture's own seed, shared by every member.
                 "latent_random_state" : \
                    self._config_model.get(
                        "latent_options", {}).get("random_state"),
                 "device" : self._device},
                f,
                default_flow_style = False,
                sort_keys = False)

        #-------------------------------------------------------------#

        # Return the file.
        return seeds_file


    def _run_members(self,
                     config_ensemble: dict,
                     stage: str,
                     get_outputs_done: object,
                     run_member: object,
                     resume: bool,
                     return_data: bool) -> dict[str, dict]:
        """Run one stage over every member of the ensemble, one member
        at a time, recording the members that fail and running the
        others anyway.

        Parameters
        ----------
        config_ensemble : :class:`dict`
            The configuration to be used.

        stage : :class:`str`
            The name of the stage, for the messages.

        get_outputs_done : callable
            Given a member's name and options, the outputs it already
            has, or :obj:`None` if it does not have them.

        run_member : callable
            Given a member's name and options, run the stage for it,
            and return its outputs and its data.

        resume : :class:`bool`
            Whether to skip the members that are already done.

        return_data : :class:`bool`
            Whether to keep each member's data in what is returned.

        Returns
        -------
        results : :class:`dict`
            What happened to each member.
        """

        # Initialize an empty dictionary to store what happened to
        # each member.
        results = {}

        # For each member of the ensemble
        for name, options in config_ensemble.items():

            # Start the member's record.
            result = {"name" : name,
                      "seed" : options["seed"],
                      "status" : "pending",
                      "outputs" : {},
                      "error" : None,
                      "elapsed" : 0.0}

            #---------------------------------------------------------#

            # Get the outputs the member already has, if it is to be
            # skipped when it has them.
            outputs_done = \
                get_outputs_done(name, options) if resume else None

            # If the member is already done
            if outputs_done is not None:

                # Record it.
                result["status"] = "skipped"
                result["outputs"] = outputs_done
                results[name] = result

                # Inform the user that the member is skipped.
                logger.info(
                    f"[{stage}] '{name}' is already done - skipping "
                    "it. Pass 'resume = False' to run it again.")

                # Move on to the next member.
                continue

            #---------------------------------------------------------#

            # Take the time the member started at.
            time_start = time.time()

            # Inform the user that the member is starting.
            logger.info(f"[{stage}] '{name}' (seed "
                        f"{options['seed']}) started.")

            # Try to run the stage for the member
            try:

                # Run it.
                outputs, data = run_member(name, options)

                # Record that it is done.
                result["status"] = "done"
                result["outputs"] = outputs

                # If the data are to be kept
                if return_data:

                    # Keep them.
                    result["data"] = data

            # If anything went wrong
            except Exception:

                # Record the failure, with its traceback.
                result["status"] = "failed"
                result["error"] = traceback.format_exc()

                # Inform the user about the failure.
                logger.error(
                    f"[{stage}] '{name}' failed:\n"
                    f"{result['error']}")

            #---------------------------------------------------------#

            # Take the time the member took.
            result["elapsed"] = time.time() - time_start

            # Store the member's record.
            results[name] = result

        #-------------------------------------------------------------#

        # Get how many members ended in each state.
        n_done = sum(1 for r in results.values()
                     if r["status"] == "done")
        n_skipped = sum(1 for r in results.values()
                        if r["status"] == "skipped")
        n_failed = sum(1 for r in results.values()
                       if r["status"] == "failed")

        # Inform the user about how the stage went.
        logger.info(
            f"[{stage}] {n_done} members ran, {n_skipped} were "
            f"already done, and {n_failed} failed.")

        # If any member failed
        if n_failed:

            # Get the members that failed.
            names_failed = \
                [name for name, result in results.items()
                 if result["status"] == "failed"]

            # Warn the user.
            logger.warning(
                f"[{stage}] These members failed: "
                f"{', '.join(names_failed)}. Look at their 'error' "
                "before going on.")

        #-------------------------------------------------------------#

        # Return what happened to each member.
        return results


    def train(self,
              df_samples: pd.DataFrame,
              names_train: list,
              names_test: list,
              config_train: dict[str, object],
              config_ensemble: dict[str, dict[str, object]] = None,
              gmm_pth_file: str = None,
              dec_pth_file: str = None,
              gmm_final_pth_file: str = None,
              pathways: pd.DataFrame = None,
              labels_train: object = None,
              labels_test: object = None,
              resume: bool = True,
              return_data: bool = False) -> dict[str, dict]:
        """Train every member of the ensemble, seeding each with its
        own seed before it is built, and writing the seeds beside it.

        Parameters
        ----------
        df_samples : :class:`pandas.DataFrame`
            The samples to train on.

        names_train : :class:`list`
            The names of the samples to train on.

        names_test : :class:`list`
            The names of the samples to test on.

        config_train : :class:`dict`
            The configuration for the training, as a single
            :class:`bulkdgd.core.model.BulkDGD` takes it, shared by
            every member.

        config_ensemble : :class:`dict`, optional
            The configuration to be used. If not passed, the
            ensemble's own is used.

        gmm_pth_file : :class:`str`, optional
            The name of the file where a member's trained latent
            space's parameters are written, inside the member's own
            directory.

        dec_pth_file : :class:`str`, optional
            The name of the file where a member's trained decoder's
            parameters are written.

        gmm_final_pth_file : :class:`str`, optional
            The name of the file where a member's final Gaussian
            mixture model's parameters are written.

        pathways : :class:`pandas.DataFrame`, optional
            The pathways, if the saliency maps are to be computed.

        labels_train : optional
            The labels of the samples to train on.

        labels_test : optional
            The labels of the samples to test on.

        resume : :class:`bool`, ``True``
            Whether to skip the members that are already trained.

        return_data : :class:`bool`, ``False``
            Whether to keep what the training returned for each
            member. Everything is written to disk either way.

        Returns
        -------
        results : :class:`dict`
            What happened to each member: whether it ran, was
            skipped, or failed, what it wrote, and how long it took.
        """

        # Get the configuration to be used.
        config_ensemble = \
            self._get_config_ensemble(
                config_ensemble = config_ensemble)

        # Get the names of the files the parameters are written to.
        gmm_pth_file = gmm_pth_file or self.GMM_PTH_FILE
        dec_pth_file = dec_pth_file or self.DEC_PTH_FILE
        gmm_final_pth_file = \
            gmm_final_pth_file or self.GMM_FINAL_PTH_FILE

        #-------------------------------------------------------------#

        # Define what it means for a member to be already trained.
        def get_outputs_done(name,
                             options):
            """Get the files of a member's trained parameters, if they
            are all there.

            Parameters
            ----------
            name : :class:`str`
                The member's name.

            options : :class:`dict`
                The member's options.

            Returns
            -------
            paths : :class:`dict` or :obj:`None`
                The files, or :obj:`None` if any is missing.
            """

            # Get where the member's parameters would be.
            paths = \
                {"gmm" : os.path.join(options["model_dir"],
                                      gmm_pth_file),
                 "decoder" : os.path.join(options["model_dir"],
                                          dec_pth_file)}

            # If they are all there
            if all(os.path.isfile(path) for path in paths.values()):

                # Return them.
                return paths

            # Otherwise, return nothing.
            return None

        #-------------------------------------------------------------#

        # Define how a member is trained.
        def run_member(name,
                       options):
            """Train a member.

            Parameters
            ----------
            name : :class:`str`
                The member's name.

            options : :class:`dict`
                The member's options.

            Returns
            -------
            outputs : :class:`dict`
                The files that were written.

            data : :class:`tuple`
                What the training returned.
            """

            # Make the directory the member lives in.
            os.makedirs(options["model_dir"], exist_ok = True)

            # Seed everything with the member's seed, before the model
            # is built, since building it is random.
            seeds = \
                set_seeds(
                    seed = options["seed"],
                    deterministic = \
                        config_train.get("deterministic", False))

            # Write down what the member was seeded with.
            seeds_file = \
                self._write_seeds(model_dir = options["model_dir"],
                                  seeds = seeds)

            #---------------------------------------------------------#

            # Build the member, untrained.
            model = self._get_model_untrained()

            # Train it.
            data = \
                model.train(
                    df_samples = df_samples,
                    names_train = names_train,
                    names_test = names_test,
                    config_train = config_train,
                    gmm_pth_file = \
                        os.path.join(options["model_dir"],
                                     gmm_pth_file),
                    dec_pth_file = \
                        os.path.join(options["model_dir"],
                                     dec_pth_file),
                    gmm_final_pth_file = \
                        os.path.join(options["model_dir"],
                                     gmm_final_pth_file),
                    pathways = pathways,
                    labels_train = labels_train,
                    labels_test = labels_test)

            #---------------------------------------------------------#

            # Write down what the training produced.
            outputs = \
                self._write_train_outputs(
                    model_dir = options["model_dir"],
                    data = data)

            # Record the parameters and the seeds among the outputs.
            outputs["gmm"] = os.path.join(options["model_dir"],
                                          gmm_pth_file)
            outputs["decoder"] = os.path.join(options["model_dir"],
                                              dec_pth_file)
            outputs["seeds"] = seeds_file

            #---------------------------------------------------------#

            # Return the outputs and the data.
            return outputs, data

        #-------------------------------------------------------------#

        # Train the members.
        return self._run_members(
                    config_ensemble = config_ensemble,
                    stage = "train",
                    get_outputs_done = get_outputs_done,
                    run_member = run_member,
                    resume = resume,
                    return_data = return_data)


    def _write_train_outputs(self,
                             model_dir: str,
                             data: tuple) -> dict[str, str]:
        """Write down what a member's training produced.

        Parameters
        ----------
        model_dir : :class:`str`
            The directory where the member's parameters live.

        data : :class:`tuple`
            What the training returned.

        Returns
        -------
        outputs : :class:`dict`
            The files that were written.
        """

        # Unpack what the training returned.
        dfs_rep, dfs_pred_means, dfs_pred_r_values, df_loss, \
            dfs_metrics, df_time = data

        # Initialize an empty dictionary to store the files written.
        outputs = {}

        #-------------------------------------------------------------#

        # Define how one data frame is written.
        def write(df,
                  name):
            """Write a data frame into the member's directory, and
            record the file.

            Parameters
            ----------
            df : :class:`pandas.DataFrame`
                The data frame.

            name : :class:`str`
                The name of the file.
            """

            # Get the path to the file.
            path = os.path.join(model_dir, name)

            # Write the data frame.
            save_table(df,
                       path,
                       sep = ",",
                       index = True)

            # Record the file.
            outputs[name.rsplit(".", 1)[0]] = path

        #-------------------------------------------------------------#

        # Write the losses and the training times.
        write(df_loss, self.LOSS_FILE)
        write(df_time, self.TIME_FILE)

        #-------------------------------------------------------------#

        # For the representations and the predicted means
        for dfs, stem in ((dfs_rep, "representations"),
                          (dfs_pred_means, "pred_means")):

            # Write the training and the test samples' data frames.
            write(dfs[0], f"{stem}_train.parquet")
            write(dfs[1], f"{stem}_test.parquet")

        #-------------------------------------------------------------#

        # If there are predicted r-values (none for a Poisson output
        # module)
        if dfs_pred_r_values is not None:

            # If there is one data frame per split (full dispersion)
            if isinstance(dfs_pred_r_values, tuple):

                # Write the training and the test samples' r-values.
                write(dfs_pred_r_values[0], "pred_r_values_train.csv")
                write(dfs_pred_r_values[1], "pred_r_values_test.csv")

            # Otherwise (per-gene dispersion)
            else:

                # Write the r-values.
                write(dfs_pred_r_values, self.PRED_R_VALUES_FILE)

        #-------------------------------------------------------------#

        # If any metrics were computed
        if dfs_metrics is not None:

            # Write them.
            write(dfs_metrics[0], "metrics_train.csv")
            write(dfs_metrics[1], "metrics_test.csv")

        #-------------------------------------------------------------#

        # Return the files that were written.
        return outputs


    def find_representations(
            self,
            df_samples: pd.DataFrame,
            config_rep: dict[str, object],
            config_ensemble: dict[str, dict[str, object]] = None,
            get_saliency_map: bool = False,
            genes_mask: torch.Tensor = None,
            resume: bool = True,
            return_data: bool = False) -> dict[str, dict]:
        """Find the representations of a set of samples with every
        member of the ensemble.

        Parameters
        ----------
        df_samples : :class:`pandas.DataFrame`
            The samples to find the representations for.

        config_rep : :class:`dict`
            The configuration for the search, as a single
            :class:`bulkdgd.core.model.BulkDGD` takes it.

        config_ensemble : :class:`dict`, optional
            The configuration to be used. If not passed, the
            ensemble's own is used. Pass a different one to write the
            results elsewhere.

        get_saliency_map : :class:`bool`, ``False``
            Whether to compute the saliency maps.

        genes_mask : :class:`torch.Tensor`, optional
            The mask for the genes to be considered.

        resume : :class:`bool`, ``True``
            Whether to skip the members that already have the
            representations.

        return_data : :class:`bool`, ``False``
            Whether to keep what the search returned for each member.
            Everything is written to disk either way.

        Returns
        -------
        results : :class:`dict`
            What happened to each member.
        """

        # Get the configuration to be used.
        config_ensemble = \
            self._get_config_ensemble(
                config_ensemble = config_ensemble)

        #-------------------------------------------------------------#

        # Define what it means for a member to already have the
        # representations.
        def get_outputs_done(name,
                             options):
            """Get the files of a member's representations and
            predicted means, if they are all there.

            Parameters
            ----------
            name : :class:`str`
                The member's name.

            options : :class:`dict`
                The member's options.

            Returns
            -------
            paths : :class:`dict` or :obj:`None`
                The files, or :obj:`None` if any is missing.
            """

            # Get where they would be.
            paths = \
                {"representations" : \
                    os.path.join(options["results_dir"],
                                 self.REP_FILE),
                 "pred_means" : \
                    os.path.join(options["results_dir"],
                                 self.PRED_MEANS_FILE)}

            # If they are all there
            if all(os.path.isfile(path) for path in paths.values()):

                # Return them.
                return paths

            # Otherwise, return nothing.
            return None

        #-------------------------------------------------------------#

        # Define how a member's representations are found.
        def run_member(name,
                       options):
            """Find the representations of the samples with a member.

            Parameters
            ----------
            name : :class:`str`
                The member's name.

            options : :class:`dict`
                The member's options.

            Returns
            -------
            outputs : :class:`dict`
                The files that were written.

            data : :class:`tuple`
                The representations, the predicted means, the
                predicted r-values, and the times.
            """

            # Make the directory the results live in.
            os.makedirs(options["results_dir"], exist_ok = True)

            # Get the member, with its trained parameters.
            model = self.get_model(name = name,
                                   config_ensemble = config_ensemble)

            #---------------------------------------------------------#

            # Find the representations.
            df_rep, df_pred_means, df_pred_r_values, df_time = \
                model.get_representations(
                    df_samples = df_samples,
                    config_rep = config_rep,
                    get_saliency_map = get_saliency_map,
                    genes_mask = genes_mask)

            #---------------------------------------------------------#

            # Initialize an empty dictionary to store the files.
            outputs = {}

            # For the representations, the predicted means, and the
            # times
            for df, name_file, key in (
                    (df_rep, self.REP_FILE, "representations"),
                    (df_pred_means, self.PRED_MEANS_FILE,
                     "pred_means"),
                    (df_time, self.TIME_FILE, "time")):

                # Get the path to the file.
                path = os.path.join(options["results_dir"], name_file)

                # Write the data frame.
                save_table(df,
                           path,
                           sep = ",",
                           index = True)

                # Record the file.
                outputs[key] = path

            #---------------------------------------------------------#

            # If the output module has predicted r-values
            if df_pred_r_values is not None:

                # Get the path to the file.
                path = os.path.join(options["results_dir"],
                                    self.PRED_R_VALUES_FILE)

                # Write the r-values.
                save_table(df_pred_r_values,
                           path,
                           sep = ",",
                           index = True)

                # Record the file.
                outputs["pred_r_values"] = path

            #---------------------------------------------------------#

            # Return the outputs and the data.
            return outputs, (df_rep, df_pred_means, df_pred_r_values,
                             df_time)

        #-------------------------------------------------------------#

        # Find the representations with each member.
        return self._run_members(
                    config_ensemble = config_ensemble,
                    stage = "find_representations",
                    get_outputs_done = get_outputs_done,
                    run_member = run_member,
                    resume = resume,
                    return_data = return_data)


    def dea(self,
            df_samples: pd.DataFrame,
            config_dea: dict[str, object],
            config_ensemble: dict[str, dict[str, object]] = None,
            resume: bool = True,
            return_data: bool = False) -> dict[str, dict]:
        """Perform differential expression analysis with every member
        of the ensemble, writing one file per sample.

        Parameters
        ----------
        df_samples : :class:`pandas.DataFrame`
            The samples' observed counts.

        config_dea : :class:`dict`
            The configuration for the analysis. These keys belong to
            the ensemble:

            * ``"dea_dir"`` (:class:`str`, optional) - where a
              member's results are written. If it is a relative path,
              it is taken relative to the member's results'
              directory. It defaults to ``"dea"``.

            * ``"zip_results"`` (:class:`bool`, optional) - whether to
              pack a member's results into one archive, verified
              before the loose files are removed. It defaults to
              :obj:`False`.

            * ``"prefix"`` (:class:`str`, optional) - the prefix the
              per-sample files are named with. It defaults to
              ``"dea_"``.

            * ``"pred_means_file"``, ``"pred_r_values_file"``
              (:class:`str`, optional) - the files the predicted means
              and r-values are read from, inside a member's results'
              directory.

            Every other key is passed on to
            :func:`bulkdgd.analysis.dea.get_statistics`, with
            ``"scaling_factor"`` defaulting to the model's.

        config_ensemble : :class:`dict`, optional
            The configuration to be used. If not passed, the
            ensemble's own is used.

        resume : :class:`bool`, ``True``
            Whether to skip the samples that already have results.

        return_data : :class:`bool`, ``False``
            Whether to keep each sample's statistics. They are written
            to disk either way.

        Returns
        -------
        results : :class:`dict`
            What happened to each member.
        """

        # Get the configuration to be used.
        config_ensemble = \
            self._get_config_ensemble(
                config_ensemble = config_ensemble)

        #-------------------------------------------------------------#

        # Copy the configuration, to take the ensemble's options out
        # of it.
        config_dea = copy.deepcopy(config_dea)

        # Get where the results go, whether to pack them, and the
        # prefix the per-sample files are named with.
        dea_dir = config_dea.pop("dea_dir", "dea")
        zip_results = config_dea.pop("zip_results", False)
        prefix = config_dea.pop("prefix", deaio.DEA_PREFIX)

        # Get the files the predicted means and r-values are read
        # from.
        pred_means_file = \
            config_dea.pop("pred_means_file", self.PRED_MEANS_FILE)
        pred_r_values_file = \
            config_dea.pop("pred_r_values_file",
                           self.PRED_R_VALUES_FILE)

        # Use the model's scaling factor, unless one was passed.
        config_dea.setdefault(
            "scaling_factor",
            self._config_model.get("scaling_factor", "mean"))

        #-------------------------------------------------------------#

        # Define how a member's differential expression is computed,
        # resuming sample by sample.
        def run_member(name,
                       options):
            """Perform differential expression analysis with a member.

            Parameters
            ----------
            name : :class:`str`
                The member's name.

            options : :class:`dict`
                The member's options.

            Returns
            -------
            outputs : :class:`dict`
                Where the results are, and how many samples were
                written and skipped.

            dfs_stats : :class:`dict`
                Each sample's statistics, if they are to be kept.
            """

            # Get where the member's results go, and make it.
            dea_dir_member = \
                self._get_dea_dir(options = options,
                                  dea_dir = dea_dir)
            os.makedirs(dea_dir_member, exist_ok = True)

            #---------------------------------------------------------#

            # Get the member's predicted means.
            df_pred_means = \
                load_table(os.path.join(options["results_dir"],
                                        pred_means_file),
                           index_col = 0)

            # Get the member's predicted r-values, if it has any.
            path_r_values = \
                os.path.join(options["results_dir"],
                             pred_r_values_file)
            df_pred_r_values = \
                load_table(path_r_values, index_col = 0) \
                if os.path.isfile(path_r_values) else None

            # Get the genes the model predicts for.
            genes = list(df_pred_means.columns)

            # Get the genes the samples have no counts for (often
            # because the counts' gene names carry versions).
            genes_missing = \
                [gene for gene in genes
                 if gene not in df_samples.columns]

            # If the samples are missing any gene
            if genes_missing:

                # Raise an error saying so.
                raise KeyError(
                    f"The samples have no counts for "
                    f"{len(genes_missing)} of the "
                    f"{len(genes)} genes the model predicts for "
                    f"(the first are: "
                    f"{', '.join(genes_missing[:3])}). Check that "
                    "the counts are the ones the model was run on, "
                    "and that their genes are named the same way.")

            #---------------------------------------------------------#

            # Initialize the statistics kept, and the numbers of
            # samples written and skipped.
            dfs_stats = {}
            n_written = 0
            n_skipped = 0

            #---------------------------------------------------------#

            # For each sample
            for sample in df_pred_means.index:

                # If the sample already has results
                if resume \
                and deaio.has_sample(dea_dir_member,
                                     sample,
                                     prefix = prefix):

                    # Count it as skipped, and move on.
                    n_skipped += 1
                    continue

                #-----------------------------------------------------#

                # Get the sample's observed counts and predicted
                # means.
                obs_counts = df_samples.loc[sample, genes]
                pred_means = df_pred_means.loc[sample, genes]

                # Get the sample's predicted r-values.
                r_values = \
                    self._get_r_values(
                        df_pred_r_values = df_pred_r_values,
                        sample = sample,
                        genes = genes)

                #-----------------------------------------------------#

                # Compute the statistics.
                df_stats, _ = \
                    analysis_dea.get_statistics(
                        obs_counts = obs_counts,
                        pred_means = pred_means,
                        r_values = r_values,
                        sample_name = sample,
                        **config_dea)

                # Add the predicted means to the statistics.
                df_stats["dgd_mean"] = pred_means

                # If there are predicted r-values
                if r_values is not None:

                    # Add them to the statistics.
                    df_stats["dgd_r"] = r_values

                #-----------------------------------------------------#

                # Write the sample's statistics.
                save_table(df_stats,
                           os.path.join(dea_dir_member,
                                        f"{prefix}{sample}.parquet"),
                           sep = ",",
                           index = True)

                # Count the sample as written.
                n_written += 1

                # If the statistics are to be kept
                if return_data:

                    # Keep them.
                    dfs_stats[sample] = df_stats

            #---------------------------------------------------------#

            # Inform the user about what was done.
            logger.info(
                f"[dea] '{name}': {n_written} samples were analyzed, "
                f"and {n_skipped} already had results.")

            #---------------------------------------------------------#

            # Start the outputs with where the results are.
            outputs = {"dea_dir" : dea_dir_member,
                       "n_samples_written" : n_written,
                       "n_samples_skipped" : n_skipped}

            # If the results are to be packed
            if zip_results:

                # Pack them.
                outputs["dea_zip"] = \
                    self._zip_dea(dea_dir = dea_dir_member,
                                  prefix = prefix)

            #---------------------------------------------------------#

            # Return the outputs and the data.
            return outputs, dfs_stats

        #-------------------------------------------------------------#

        # Define what it means for a member to be already done (never,
        # the samples being resumed one by one).
        def get_outputs_done(name,
                             options):
            """Get a member's outputs, if it is already done.

            Parameters
            ----------
            name : :class:`str`
                The member's name.

            options : :class:`dict`
                The member's options.

            Returns
            -------
            paths : :obj:`None`
                Always :obj:`None`.
            """

            # Return nothing.
            return None

        #-------------------------------------------------------------#

        # Analyze the samples with each member.
        return self._run_members(
                    config_ensemble = config_ensemble,
                    stage = "dea",
                    get_outputs_done = get_outputs_done,
                    run_member = run_member,
                    resume = resume,
                    return_data = return_data)


    def _get_r_values(self,
                      df_pred_r_values: pd.DataFrame,
                      sample: str,
                      genes: list) -> pd.Series:
        """Get a sample's predicted r-values.

        Parameters
        ----------
        df_pred_r_values : :class:`pandas.DataFrame` or :obj:`None`
            The predicted r-values.

        sample : :class:`str`
            The sample's name.

        genes : :class:`list`
            The genes whose r-values are returned.

        Returns
        -------
        r_values : :class:`pandas.Series` or :obj:`None`
            The sample's predicted r-values.
        """

        # If the output module has no r-values
        if df_pred_r_values is None:

            # Return nothing.
            return None

        #-------------------------------------------------------------#

        # If there is one row per sample (full dispersion)
        if sample in df_pred_r_values.index:

            # Return the sample's row.
            return df_pred_r_values.loc[sample, genes]

        #-------------------------------------------------------------#

        # If there is a single row for every sample (per-gene
        # dispersion)
        if len(df_pred_r_values) == 1:

            # Return the single row.
            return df_pred_r_values.iloc[0][genes]

        #-------------------------------------------------------------#

        # Otherwise, raise an error.
        raise KeyError(
            f"No predicted r-values were found for the sample "
            f"'{sample}'.")


    def _zip_dea(self,
                 dea_dir: str,
                 prefix: str = None) -> str:
        """Pack a member's differential expression into one archive,
        or add the files an existing one is missing, and remove the
        loose files once the archive is verified.

        Parameters
        ----------
        dea_dir : :class:`str`
            The directory containing the results.

        prefix : :class:`str`, optional
            The prefix the per-sample files are named with.

        Returns
        -------
        zip_path : :class:`str`
            The archive.
        """

        # Get the prefix.
        prefix = prefix or deaio.DEA_PREFIX

        # Get where the archive goes, and where it is built.
        zip_path = os.path.join(dea_dir, deaio.DEA_ZIP_NAME)
        zip_path_partial = f"{zip_path}.partial"

        #-------------------------------------------------------------#

        # Get the files to be packed.
        names = sorted(name for name in os.listdir(dea_dir)
                       if name.startswith(prefix)
                       and name.endswith(deaio.READ_EXTENSIONS))

        # Get whether the results are already packed.
        packed = os.path.isfile(zip_path)

        # If the results are already packed
        if packed:

            # Open the archive.
            with zipfile.ZipFile(zip_path) as archive:

                # Get the files already in it.
                names_in_zip = set(archive.namelist())

            # Keep only the files missing from it.
            names = [name for name in names
                     if name not in names_in_zip]

            # If no file is missing from it
            if not names:

                # Inform the user.
                logger.info(
                    f"The results in '{dea_dir}' are already packed.")

                # Return the archive.
                return zip_path

            # Start the new archive from a copy of the existing one.
            shutil.copyfile(zip_path, zip_path_partial)

        #-------------------------------------------------------------#

        # If there is nothing to pack
        if not names:

            # Raise an error.
            errstr = f"There are no results to pack in '{dea_dir}'."
            raise FileNotFoundError(errstr)

        #-------------------------------------------------------------#

        # Build the archive under a temporary name, adding to the copy
        # if the results were already packed.
        with zipfile.ZipFile(zip_path_partial,
                             "a" if packed else "w",
                             compression = zipfile.ZIP_DEFLATED) \
                as archive:

            # For each file
            for name in names:

                # Add it to the archive.
                archive.write(os.path.join(dea_dir, name),
                              arcname = name)

        #-------------------------------------------------------------#

        # Read the archive back.
        with zipfile.ZipFile(zip_path_partial) as archive:

            # If the archive is corrupted
            if archive.testzip() is not None:

                # Raise an error, leaving the loose files alone.
                raise RuntimeError(
                    f"The archive built for '{dea_dir}' is corrupted. "
                    "The loose files were left alone.")

            # Get the files in the archive.
            names_packed = set(archive.namelist())

        # Get the files that did not make it in.
        names_missing = [name for name in names
                         if name not in names_packed]

        # If any file did not make it in
        if names_missing:

            # Raise an error, leaving the loose files alone.
            raise RuntimeError(
                f"{len(names_missing)} of the {len(names)} files in "
                f"'{dea_dir}' are missing from the archive built for "
                "it. The loose files were left alone.")

        #-------------------------------------------------------------#

        # Put the archive in place.
        os.replace(zip_path_partial, zip_path)

        # Drop the archive's stale copy cached for reading, if any.
        deaio.forget_archive(zip_path = zip_path)

        # For each loose file
        for name in names:

            # Remove it.
            os.remove(os.path.join(dea_dir, name))

        #-------------------------------------------------------------#

        # Inform the user about what was packed.
        logger.info(
            f"{len(names)} files in '{dea_dir}' were packed into "
            f"'{deaio.DEA_ZIP_NAME}' and removed.")

        #-------------------------------------------------------------#

        # Return the archive.
        return zip_path


    def gsea(self,
             genes_sets: dict[str, list],
             config_gsea: dict[str, object],
             config_ensemble: dict[str, dict[str, object]] = None,
             resume: bool = True,
             return_data: bool = False) -> dict[str, dict]:
        """Compute the enrichment scores of the genes every member of
        the ensemble calls.

        Parameters
        ----------
        genes_sets : :class:`dict`
            The sets of genes of interest.

        config_gsea : :class:`dict`
            The configuration for the analysis. These keys belong to
            the ensemble:

            * ``"dea_dir"`` (:class:`str`, optional) - where a
              member's differential expression is read from, packed
              or loose. It defaults to ``"dea"``.

            * ``"gsea_dir"`` (:class:`str`, optional) - where a
              member's enrichment scores are written. It defaults to
              ``"gsea"``.

            * ``"genes_all"`` (:class:`list`, optional) - the genes
              the analysis was run on. It defaults to the genes the
              differential expression covers.

            * ``"prefix"`` (:class:`str`, optional) - the prefix the
              per-sample files are named with. It defaults to
              ``"dea_"``.

            Every other key is passed on to
            :func:`bulkdgd.analysis.dea.get_significant_genes`.

        config_ensemble : :class:`dict`, optional
            The configuration to be used. If not passed, the
            ensemble's own is used.

        resume : :class:`bool`, ``True``
            Whether to skip the members that already have the
            enrichment scores.

        return_data : :class:`bool`, ``False``
            Whether to keep each member's enrichment scores. They are
            written to disk either way.

        Returns
        -------
        results : :class:`dict`
            What happened to each member.
        """

        # Get the configuration to be used.
        config_ensemble = \
            self._get_config_ensemble(
                config_ensemble = config_ensemble)

        #-------------------------------------------------------------#

        # Copy the configuration, to take the ensemble's options out
        # of it.
        config_gsea = copy.deepcopy(config_gsea)

        # Get where the differential expression is read from, where
        # the scores go, the genes, and the per-sample files' prefix.
        dea_dir = config_gsea.pop("dea_dir", "dea")
        gsea_dir = config_gsea.pop("gsea_dir", "gsea")
        genes_all = config_gsea.pop("genes_all", None)
        prefix = config_gsea.pop("prefix", deaio.DEA_PREFIX)

        #-------------------------------------------------------------#

        # Define what it means for a member to already have the
        # enrichment scores.
        def get_outputs_done(name,
                             options):
            """Get the file of a member's enrichment scores, if it is
            there.

            Parameters
            ----------
            name : :class:`str`
                The member's name.

            options : :class:`dict`
                The member's options.

            Returns
            -------
            paths : :class:`dict` or :obj:`None`
                The file, or :obj:`None` if it is missing.
            """

            # Get where they would be.
            path = os.path.join(options["results_dir"],
                                gsea_dir,
                                self.E_SCORES_FILE)

            # If they are there
            if os.path.isfile(path):

                # Return the file.
                return {"e_scores" : path}

            # Otherwise, return nothing.
            return None

        #-------------------------------------------------------------#

        # Define how a member's enrichment scores are computed.
        def run_member(name,
                       options):
            """Compute the enrichment scores with a member.

            Parameters
            ----------
            name : :class:`str`
                The member's name.

            options : :class:`dict`
                The member's options.

            Returns
            -------
            outputs : :class:`dict`
                The file that was written.

            df_e_scores_all : :class:`pandas.DataFrame`
                Every sample's enrichment scores.
            """

            # Get where the member's differential expression is.
            dea_dir_member = \
                self._get_dea_dir(options = options,
                                  dea_dir = dea_dir)

            # Get where the enrichment scores go, and make it.
            gsea_dir_member = \
                os.path.join(options["results_dir"], gsea_dir) \
                if not os.path.isabs(gsea_dir) else gsea_dir
            os.makedirs(gsea_dir_member, exist_ok = True)

            #---------------------------------------------------------#

            # Get the samples the member has results for.
            samples = \
                deaio.list_samples(dea_dir_member, prefix = prefix)

            # If it has none
            if not samples:

                # Raise an error.
                raise FileNotFoundError(
                    "There is no differential expression to draw the "
                    f"enrichment scores from in '{dea_dir_member}'.")

            #---------------------------------------------------------#

            # Initialize an empty list to store each sample's scores.
            dfs_e_scores = []

            # Get the genes the analysis was run on, if they were
            # given.
            genes_all_member = genes_all

            #---------------------------------------------------------#

            # For each sample
            for sample in samples:

                # Get the sample's statistics.
                df_stats = \
                    deaio.read_dea(dea_dir_member,
                                   sample,
                                   prefix = prefix,
                                   index_col = 0)

                # If the genes were not given
                if genes_all_member is None:

                    # Take the ones with statistics.
                    genes_all_member = df_stats.index.tolist()

                #-----------------------------------------------------#

                # Get the genes the member calls for the sample.
                df_significant_genes = \
                    analysis_dea.get_significant_genes(
                        df_stats = df_stats,
                        **config_gsea)

                # Compute their enrichment scores.
                df_e_scores = \
                    analysis_dea.get_enrichment_scores(
                        df_significant_genes = df_significant_genes,
                        genes_sets = genes_sets,
                        genes_all = genes_all_member)

                # Add the sample's name to the scores.
                df_e_scores.insert(0,
                                   "sample",
                                   sample)

                # Add them to the list.
                dfs_e_scores.append(df_e_scores)

            #---------------------------------------------------------#

            # Put every sample's scores together.
            df_e_scores_all = \
                pd.concat(dfs_e_scores, ignore_index = True)

            # Get the path to the file.
            path = os.path.join(gsea_dir_member, self.E_SCORES_FILE)

            # Write the scores.
            save_table(df_e_scores_all,
                       path,
                       sep = ",",
                       index = False)

            #---------------------------------------------------------#

            # Inform the user about what was done.
            logger.info(
                f"[gsea] '{name}': the enrichment scores of "
                f"{len(samples)} samples over {len(genes_sets)} sets "
                "of genes were computed.")

            #---------------------------------------------------------#

            # Return the outputs and the data.
            return {"e_scores" : path}, df_e_scores_all

        #-------------------------------------------------------------#

        # Compute the enrichment scores with each member.
        return self._run_members(
                    config_ensemble = config_ensemble,
                    stage = "gsea",
                    get_outputs_done = get_outputs_done,
                    run_member = run_member,
                    resume = resume,
                    return_data = return_data)


    def consensus(
            self,
            df_metadata: pd.DataFrame,
            config_consensus: dict[str, object],
            config_ensemble: dict[str, dict[str, object]] = \
                None) -> dict[str, pd.DataFrame]:
        """Get the tiered list of genes the ensemble agrees on. A
        gene's tier is the number of models calling it in at least a
        given share (recurrence) of a group's samples.

        Parameters
        ----------
        df_metadata : :class:`pandas.DataFrame`
            The samples' metadata, indexed by the samples' names.

            It must contain the column the samples are grouped by, and
            the column they are filtered on, if any.

        config_consensus : :class:`dict`
            The configuration for the consensus. The keys are:

            * ``"group_column"`` (:class:`str`) - the metadata column
              the samples are grouped by, and within which a gene's
              recurrence is computed.

            * ``"group_name"`` (:class:`str`, optional) - the name the
              group is given in the output. It defaults to the name of
              the column the samples are grouped by.

            * ``"filter_column"`` (:class:`str`, optional) - a
              metadata column the samples are filtered on before
              anything else.

            * ``"filter_value"`` (optional) - the value the samples
              must have in ``"filter_column"`` to be kept.

            * ``"dea_dir"`` (:class:`str`, optional) - the directory
              containing a member's differential expression analysis'
              results. If it is a relative path, it is taken relative
              to the member's results' directory. It defaults to
              ``"dea"``.

            * ``"prefix"`` (:class:`str`, optional) - the prefix the
              per-sample files are named with. It defaults to
              ``"dea_"``.

            * ``"q_val"`` (:class:`float`, optional) - the q-value
              below which a gene is called for a sample. It defaults
              to ``0.05``.

            * ``"log2_fold_change"`` (:class:`float`, optional) - the
              absolute log2-fold change above which a gene is called
              for a sample. It defaults to ``1.0``.

            * ``"recurrence"`` (:class:`float`, optional) - the share
              of a group's samples a model must call a gene in for the
              model to consider it. It defaults to ``0.20``.

            * ``"min_tier"`` (:class:`int`, optional) - the tier below
              which a gene is left out of the output. It defaults to
              ``2``.

            * ``"genes_symbols"`` (:class:`dict`, optional) - the
              genes' symbols, mapped from the genes' names.

            * ``"n_processes"`` (:class:`int`, optional) - how many
              processes to score the samples with. It defaults to
              :func:`os.cpu_count`.

        config_ensemble : :class:`dict`, optional
            The configuration to be used. If not passed, the
            ensemble's own is used.

        Returns
        -------
        dfs_consensus : :class:`dict`
            A dictionary mapping each group of samples to a data frame
            with one row per gene, reporting the gene's tier, its
            consensus recurrence (the tier-th largest per-model
            recurrence), and every per-model recurrence, sorted.
        """

        # Get the configuration to be used.
        config_ensemble = \
            self._get_config_ensemble(
                config_ensemble = config_ensemble)

        #-------------------------------------------------------------#

        # Get the column the samples are grouped by.
        group_column = config_consensus["group_column"]

        # Get the name the group is given in the output.
        group_name = \
            config_consensus.get("group_name", group_column)

        # Get the column the samples are filtered on, and the value
        # they must have in it.
        filter_column = config_consensus.get("filter_column")
        filter_value = config_consensus.get("filter_value")

        # Get the directory containing the results, and the prefix the
        # per-sample files are named with.
        dea_dir = config_consensus.get("dea_dir", "dea")
        prefix = config_consensus.get("prefix", deaio.DEA_PREFIX)

        # Get the thresholds a gene must pass to be called.
        q_val = \
            config_consensus.get("q_val", self.DEFAULT_Q_VAL)
        log2_fold_change = \
            config_consensus.get("log2_fold_change",
                                 self.DEFAULT_LOG2_FOLD_CHANGE)

        # Get the share of a group's samples a model must call a gene
        # in to consider it, and the minimum tier.
        recurrence = \
            config_consensus.get("recurrence",
                                 self.DEFAULT_RECURRENCE)
        min_tier = \
            config_consensus.get("min_tier", self.DEFAULT_MIN_TIER)

        # Get the genes' symbols.
        genes_symbols = config_consensus.get("genes_symbols") or {}

        # Get how many processes to score the samples with.
        n_processes = \
            config_consensus.get("n_processes") or os.cpu_count()

        #-------------------------------------------------------------#

        # Get the directories containing the members' results.
        dea_dirs = \
            [self._get_dea_dir(options = config_ensemble[name],
                               dea_dir = dea_dir)
             for name in self.names]

        # Get the members with no results at all.
        dirs_missing = \
            [f"'{name}' ({dea_dir_member})"
             for name, dea_dir_member in zip(self.names, dea_dirs)
             if not deaio.list_samples(dea_dir_member,
                                       prefix = prefix)]

        # If any member's results are missing
        if dirs_missing:

            # Raise an error.
            raise FileNotFoundError(
                "These members have no differential expression "
                f"analysis' results: {'; '.join(dirs_missing)}. A "
                "gene's tier is how many of the ensemble's models "
                "agree on it, so it can only be computed with every "
                "member's results.")

        #-------------------------------------------------------------#

        # Get the samples the first member has results for.
        samples = deaio.list_samples(dea_dirs[0],
                                     prefix = prefix)

        # If the samples are to be filtered
        if filter_column is not None:

            # Get the samples that pass the filter.
            samples_kept = \
                set(df_metadata.index[
                        df_metadata[filter_column].astype(str) \
                            == str(filter_value)])

            # Keep only those.
            samples = [sample for sample in samples
                       if sample in samples_kept]

        #-------------------------------------------------------------#

        # Get the group each sample belongs to, as a dictionary (one
        # group even for a repeated sample).
        groups = df_metadata[group_column].astype(str).to_dict()

        # Pair each sample with its group, leaving out the samples
        # that have no metadata.
        items_samples = \
            [(sample, groups[sample]) for sample in samples
             if sample in groups]

        # If no samples are left
        if not items_samples:

            # Raise an error.
            raise ValueError(
                "No samples are left to draw the consensus from. "
                "Check that the metadata cover the samples the "
                "members have results for, and that the filter, if "
                "any, is not excluding all of them.")

        #-------------------------------------------------------------#

        # Inform the user about what the consensus will be drawn from.
        logger.info(
            f"The consensus will be drawn from {len(items_samples)} "
            f"samples across "
            f"{len(set(group for _, group in items_samples))} groups "
            f"and {self.n_models} models.")

        #-------------------------------------------------------------#

        # Get the positions of the columns to read from the samples'
        # statistics, from the first sample's header.
        usecols = self._get_usecols(dea_dir = dea_dirs[0],
                                    sample = items_samples[0][0],
                                    prefix = prefix)

        # Build the items to be scored.
        items = \
            [(sample, group, dea_dirs, prefix, usecols,
              q_val, log2_fold_change)
             for sample, group in items_samples]

        #-------------------------------------------------------------#

        # Initialize, for each model, the number of a group's samples
        # it calls each gene in.
        counts = \
            [{} for _ in range(self.n_models)]

        # Initialize the number of samples scored in each group.
        n_samples = {}

        # Initialize the number of samples dropped because some model
        # had no results for them.
        n_dropped = 0

        #-------------------------------------------------------------#

        # Score the samples in parallel.
        with mp.Pool(n_processes) as pool:

            # For each sample scored
            for i, result in enumerate(
                    pool.imap_unordered(_get_genes_called,
                                        items,
                                        chunksize = 8)):

                # Every 500 samples
                if i % 500 == 0:

                    # Inform the user about the progress.
                    logger.info(
                        f"{i}/{len(items)} samples were scored.")

                # If some model had no results for the sample
                if result is None:

                    # Count it as dropped, and move on.
                    n_dropped += 1
                    continue

                #-----------------------------------------------------#

                # Unpack the result.
                group, genes_called = result

                # Count the sample in its group.
                n_samples[group] = n_samples.get(group, 0) + 1

                # For each model's called genes
                for i_model, genes in enumerate(genes_called):

                    # Get the counts for the model and the group.
                    counts_group = \
                        counts[i_model].setdefault(group, {})

                    # For each gene the model called
                    for gene in genes:

                        # Count it.
                        counts_group[gene] = \
                            counts_group.get(gene, 0) + 1

        #-------------------------------------------------------------#

        # If any sample was dropped
        if n_dropped:

            # Warn the user that the tiers were drawn from fewer
            # samples.
            logger.warning(
                f"{n_dropped} samples were dropped because at least "
                "one member had no results for them.")

        #-------------------------------------------------------------#

        # Build the consensus.
        return self._get_dfs_consensus(
                    counts = counts,
                    n_samples = n_samples,
                    recurrence = recurrence,
                    min_tier = min_tier,
                    genes_symbols = genes_symbols,
                    group_name = group_name)


    def _get_usecols(self,
                     dea_dir: str,
                     sample: str,
                     prefix: str = deaio.DEA_PREFIX) -> list:
        """Get the positions of the columns to be read from a sample's
        statistics.

        Parameters
        ----------
        dea_dir : :class:`str`
            The directory containing the results.

        sample : :class:`str`
            The sample whose statistics are inspected.

        prefix : :class:`str`, ``"dea_"``
            The prefix the per-sample files are named with.

        Returns
        -------
        usecols : :class:`list`
            The positions of the genes' names, the q-values, and the
            log2-fold changes.
        """

        # Read only the header of the sample's statistics.
        df_head = deaio.read_dea(dea_dir,
                                  sample,
                                  prefix = prefix,
                                  index_col = 0,
                                  header = 0,
                                  nrows = 0)

        # Get the columns the file has, with the genes' names first.
        columns = [""] + list(df_head.columns)

        #-------------------------------------------------------------#

        # Initialize the list to store the positions.
        usecols = [0]

        # For each column that is needed
        for column in ("q_value", "log2_fold_change"):

            # If the file does not have it
            if column not in columns:

                # Raise an error.
                raise KeyError(
                    f"The statistics found in '{dea_dir}' have no "
                    f"'{column}' column. They have: "
                    f"{', '.join(columns[1:])}.")

            # Add its position to the list.
            usecols.append(columns.index(column))

        #-------------------------------------------------------------#

        # Return the positions.
        return usecols


    def _get_dfs_consensus(self,
                           counts: list,
                           n_samples: dict,
                           recurrence: float,
                           min_tier: int,
                           genes_symbols: dict,
                           group_name: str) -> dict[str, pd.DataFrame]:
        """Turn the counts of how many of a group's samples each model
        calls each gene in into the tiered consensus.

        Parameters
        ----------
        counts : :class:`list`
            For each model, the number of a group's samples it calls
            each gene in.

        n_samples : :class:`dict`
            The number of samples scored in each group.

        recurrence : :class:`float`
            The share of a group's samples a model must call a gene in
            to consider it.

        min_tier : :class:`int`
            The tier below which a gene is left out.

        genes_symbols : :class:`dict`
            The genes' symbols, mapped from the genes' names.

        group_name : :class:`str`
            The name the group is given in the output.

        Returns
        -------
        dfs_consensus : :class:`dict`
            A dictionary mapping each group to its consensus.
        """

        # Initialize an empty dictionary to store the consensus for
        # each group.
        dfs_consensus = {}

        # For each group and the number of samples scored in it
        for group, n_samples_group in n_samples.items():

            # Initialize the set of genes any model called in the
            # group.
            genes = set()

            # For each model's counts
            for counts_model in counts:

                # Add the genes the model called in the group.
                genes |= set(counts_model.get(group, {}).keys())

            #---------------------------------------------------------#

            # Initialize an empty list to store the group's genes.
            rows = []

            # For each gene
            for gene in genes:

                # Get the share of the group's samples each model
                # called it in, largest first.
                recurrences = \
                    sorted((counts_model.get(group, {}).get(gene, 0) \
                                / n_samples_group
                            for counts_model in counts),
                           reverse = True)

                # Get the gene's tier, the number of models that
                # considered it.
                tier = \
                    sum(1 for r in recurrences if r >= recurrence)

                # If too few models considered it
                if tier < min_tier:

                    # Move on to the next gene.
                    continue

                #-----------------------------------------------------#

                # Add the gene to the list, with the recurrence at
                # which 'tier' models agree as consensus recurrence.
                rows.append(
                    {group_name : group,
                     "gene" : gene,
                     "symbol" : genes_symbols.get(gene, gene),
                     "tier" : int(tier),
                     "consensus_recurrence" : \
                        round(recurrences[tier - 1], 4),
                     "rec_sorted" : \
                        "|".join(f"{r:.3f}" for r in recurrences)})

            #---------------------------------------------------------#

            # If no gene made it into the group's consensus
            if not rows:

                # Move on to the next group.
                continue

            # Store the group's consensus, with the genes the most
            # models agree on first.
            dfs_consensus[group] = \
                pd.DataFrame(rows).sort_values(
                    ["tier", "consensus_recurrence"],
                    ascending = [False, False]).reset_index(drop = True)

        #-------------------------------------------------------------#

        # Inform the user about the consensus that was drawn.
        logger.info(
            f"The consensus was drawn for {len(dfs_consensus)} "
            "groups "
            f"({sum(len(df) for df in dfs_consensus.values())} genes "
            "in total).")

        #-------------------------------------------------------------#

        # Return the consensus.
        return dfs_consensus
