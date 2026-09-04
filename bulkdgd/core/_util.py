#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-

#    _util.py
#
#    Private model-related utilities.
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


# Import from the standard library.
import copy
import os
import platform
from typing import Optional

# Import from third-party libraries.
import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

# Import from 'bulkdgd'.
from bulkdgd import _internals, defaults
from . import _templates, dataclasses, latents


######################### PRIVATE FUNCTIONS  ##########################


def _get_nested_value(obj: dict, path: str):
    """Get a value from a nested dictionary using dot notation, e.g.
    "field.subfield.key".

    Parameters
    ----------
    obj : :class:`dict`
        The dictionary to traverse.

    path : :class:`str`
        The path to the value using dot notation (e.g.,
        "output_module_name" or "decoder_options.output_module_name").

    Returns
    -------
    value : object or :obj:`None`
        The value at the specified path, or :obj:`None` if the path is
        invalid or does not exist.
    """

    # If the path is empty or the object is not a dictionary
    if not path or not isinstance(obj, dict):

        # Return None.
        return None
    
    # Split the path into keys.
    keys = path.split(".")

    # Set the current level to the input object.
    current = obj
    
    # For each key in the path
    for key in keys:

        # If the current level is a dictionary and contains the key
        if isinstance(current, dict) and key in current:

            # Move down to the next level.
            current = current[key]
        
        # Otherwise
        else:

            # Return None.
            return None
    
    # Return the value found at the end of the path.
    return current


def _check_config_recursive(
        config: dict[str, object],
        template: dict[str, object],
        parent_config: Optional[dict[str, object]] = None,
        path: str = "") -> \
            tuple[dict[str, object],
                  list[str],
                  list[str]]:
    """Recursively validate a configuration against a template,
    filling in defaults for missing leaf options.

    Parameters
    ----------
    config : :class:`dict`
        The configuration (or configuration section) to validate.

    template : :class:`dict`
        The template defining valid options and structure. Each key
        maps to a validation spec, a switch spec, or a nested section.

    parent_config : :class:`dict`, optional
        The root configuration dictionary, used to resolve switch
        references. Defaults to `config` if not given.

    path : :class:`str`, optional
        The current dot-notation path through the configuration
        hierarchy, used in error and warning messages.

    Returns
    -------
    config : :class:`dict`
        The validated configuration with defaults applied where
        applicable.

    errors : :class:`list`
        A list of validation error messages found in the
        configuration.

    warnings : :class:`list`
        A list of validation warning messages regarding the
        configuration.
    """
    
    # Initialize a list to store the errors.
    errors = []

    # Initialize a list to store the warnings.
    warnings = []

    #-----------------------------------------------------------------#
    
    # If there is no configuration
    if config is None:

        # Initialize an empty configuration.
        config = {}

    # Otherwise
    else:

        # Create a deep copy of the configuration to avoid mutating
        # the original during validation and default application.
        config = copy.deepcopy(config)

    #-----------------------------------------------------------------#
    
    # If no parent config is provided
    if parent_config is None:

        # Use the current configuration as parent (for resolving
        # switch references).
        parent_config = config

    #-----------------------------------------------------------------#
    
    # For each specification in the template
    for template_key, template_spec in template.items():
        
        # Build the path for this key.
        current_path = \
            f"{path}.{template_key}" if path else template_key
        
        #-------------------------------------------------------------#
        
        # If we are at a 'switch' block
        if isinstance(template_spec, dict) and \
                "switch" in template_spec:
            
            # Get the configuration of the 'switch' block.
            switch_config = template_spec["switch"]

            # Get the referenced option for the switch.
            option_ref = switch_config.get("option")

            # Get the switch cases.
            cases = switch_config.get("cases", {})
            
            # Resolve the referenced option value. Support:
            # - dotted absolute paths (e.g.
            #   "dec_options.output_module_name")
            # - names relative to the current path
            # - fallback to top-level names
            option_value = None

            #---------------------------------------------------------#

            # If the reference option is a path
            if "." in option_ref:

                # Take the option's value directly using the path.
                option_value = \
                    _get_nested_value(parent_config,
                                      option_ref)

            # Otherwise
            else:
                
                # Assume it it relative to the current path.
                rel_path = \
                    f"{path}.{option_ref}" if path else option_ref
                
                # Take the option's value using the relative path.
                option_value = \
                    _get_nested_value(parent_config,
                                      rel_path)
                
                # If the option's value is 'None'
                if option_value is None:

                    # Try to take the option's value using the
                    # top-level name.
                    option_value = \
                        _get_nested_value(
                            parent_config,
                            option_ref)

            #---------------------------------------------------------#
            
            # Take the corresponding case.
            if option_value in cases:
                case_template = cases[option_value]
                    
                # If the section does not exist in the configuraton
                if template_key not in config:

                    # Initialize it.
                    config[template_key] = {}
                
                # Recursively validate this case's template.
                config[template_key], case_errors, case_warnings = \
                    _check_config_recursive(
                        config[template_key],
                        case_template,
                        parent_config,
                        current_path)
                
                # Add the collected errors to the list of errors.
                errors.extend(case_errors)

                # Add the collected warnings to the list of warnings.
                warnings.extend(case_warnings)

        #-------------------------------------------------------------#
        
        # If we are at a nested 'switch' block
        # (when the template_key itself is 'switch' and the value
        # is a switch definition with 'option' and 'cases')
        elif isinstance(template_spec, dict) and \
                template_key == "switch" and \
                "option" in template_spec and \
                "cases" in template_spec and \
                "switch" not in template_spec:
            
            # Get the switch configuration.
            switch_config = template_spec

            # Get the referenced option for the switch.
            option_ref = switch_config.get("option")

            # Get the switch cases.
            cases = switch_config.get("cases", {})
            
            # Resolve the referenced option value. Support:
            # - dotted absolute paths (e.g.
            #   "dec_options.output_module_name")
            # - names relative to the current path
            # - fallback to top-level names
            option_value = None

            #---------------------------------------------------------#

            # If the reference option is a path
            if "." in option_ref:

                # Take the option's value directly using the path.
                option_value = \
                    _get_nested_value(parent_config,
                                      option_ref)

            # Otherwise
            else:
                
                # Assume it is relative to the current path.
                rel_path = \
                    f"{path}.{option_ref}" if path else option_ref
                
                # Take the option's value using the relative path.
                option_value = \
                    _get_nested_value(parent_config,
                                      rel_path)
                
                # If the option's value is 'None'
                if option_value is None:

                    # Try to take the option's value using the
                    # top-level name.
                    option_value = \
                        _get_nested_value(
                            parent_config,
                            option_ref)

            #---------------------------------------------------------#
            
            # Take the corresponding case.
            case_template = cases[option_value]
            
            # Recursively validate this case's template.
            # Note: for nested switches, we don't wrap in a new config
            # key; we process the case directly into the current config.
            config, case_errors, case_warnings = \
                _check_config_recursive(
                    config,
                    case_template,
                    parent_config,
                    path)
            
            # Add the collected errors to the list of errors.
            errors.extend(case_errors)

            # Add the collected warnings to the list of warnings.
            warnings.extend(case_warnings)

        #-------------------------------------------------------------#
        
        # If it is a regular option specification
        elif isinstance(template_spec, dict) and \
            "type" in template_spec:
            
            # If the option is provided in the configuration
            if template_key in config:
                
                # Validate the option's value.
                error = \
                    _check_value(value = config[template_key],
                                 options = template_spec,
                                 option_name = template_key,
                                 field_name = current_path)
                
                # If there is an error
                if error:

                    # Add it to the list of errors.
                    errors.append(error)
            
            # Otherwise
            else:
                
                # If there is a default option in the template's
                # specification
                if "default" in template_spec:
                    
                    # Get the default value.
                    default_value = template_spec["default"]

                    # Add it to the configuration.
                    config[template_key] = default_value
                    
                    # Generate a warning about using the default
                    warnings.append(
                        f"{current_path}: no '{template_key}' found. "
                        f"The default value '{default_value}' will be "
                        f"used.")
                
                # Otherwise
                else:

                    # Add an error.
                    errors.append(
                        f"{current_path}: missing required option "
                        f"'{template_key}'.")

        #-------------------------------------------------------------#
        
        # If we are at a nested section
        elif isinstance(template_spec, dict) and \
                not ("switch" in template_spec or \
                     "type" in template_spec):

            # If the section does not exist in the configuraton
            if template_key not in config:

                # An absent optional section means "not requested",
                # not "forgotten"; defaulting it in would turn it into
                # a real (defaulted) section, changing the config.
                if template_spec.get("__optional__"):
                    continue

                # Initialize it.
                config[template_key] = {}
            
            # Recursively validate this section's template.
            config[template_key], nested_errors, nested_warnings = \
                _check_config_recursive(
                    config[template_key],
                    template_spec,
                    parent_config,
                    current_path)
            
            # Add the collected errors to the list of errors.
            errors.extend(nested_errors)

            # Add the collected warnings to the list of warnings.
            warnings.extend(nested_warnings)

    #-----------------------------------------------------------------#
    
    # Return the configuration with defaults applied, the errors,
    # and the warnings.
    return config, errors, warnings


def _check_value(value,
                 options,
                 option_name,
                 field_name):
    """Validate a single configuration value against spec.
    
    Parameters
    ----------
    value
        The value to validate.
    
    options : :class:`dict`
        The option specification containing ``type``,
        ``choices``, ``condition``, and ``message``.
    
    option_name : :class:`str`
        The name of the option being validated.
    
    field_name : :class:`str`
        The field path using dot notation (e.g.,
        "latent_options" or "latent_options.decoder_options").
        Used in error messages.
    
    Returns
    -------
    :class:`str`
        An error message if validation fails, or an empty
        string if successful.
    """

    # Initialize the error string.
    err_str = f"{field_name}: '{option_name}' "

    #-----------------------------------------------------------------#

    # Get the type of option.
    option_type = options["type"]

    # Get the choices for the option.
    option_choices = options.get("choices")

    # Get the condition the option must satisfy.
    option_condition = options.get("condition")

    # Get the error message to use if the condition is not satisfied.
    option_message = options.get("message")

    #-----------------------------------------------------------------#

    # If the value is None and None is allowed
    if value is None and type(None) in option_type:
        return ""

    #-----------------------------------------------------------------#

    # If the field type is numeric
    if float in option_type or int in option_type:

        # If there is a condition for the field and the value does not
        # satisfy it
        if option_condition is not None \
            and not option_condition(value):
            
            # Finalize the error string.
            err_str += option_message + "."

    #-----------------------------------------------------------------#

    # If the field type is string
    elif str in option_type:

        # If the value is not a string or, if there are supported
        # choices for the field, the value is not one of the supported
        # choices
        if not isinstance(value, str) or \
            (option_choices is not None \
                and value not in option_choices):

            # If there are supported choices for the field
            if option_choices is not None:

                # Set a string for the supported choices.
                choices_str = \
                    ", ".join([f"'{c}'" for c in option_choices])

                # Add the supported choices to the error string.
                err_str += \
                    "must be a string and one of the supported " \
                    f"values: {choices_str}."
            
            # Otherwise
            else:

                # Add to the error string that the value must be a
                # string.
                err_str += "must be a string."

    #-----------------------------------------------------------------#

    # If the field type is boolean
    elif bool in option_type:

        # If the value is not a boolean
        if not isinstance(value, bool):

            # Add to the error string that the value must be a boolean.
            err_str += "must be a boolean."

    #-----------------------------------------------------------------#

    # If the field type is a list
    elif list in option_type:

        # If there are choices for the field
        if option_choices is not None:

            # If the value is not a list or not all values in the list
            # are among the supported choices
            if not isinstance(value, list) or \
                not all(isinstance(v, str) \
                    and v in option_choices for v in value):

                # Set a string for the supported choices.
                choices_str = \
                    ", ".join([f"'{c}'" for c in option_choices])
                
                # Add the supported choices to the
                # error string.
                err_str += \
                    "must be a list of strings, each being one of " \
                    "the supported values: " \
                    f"{choices_str}."
        
        # If there is a condition for the field
        elif option_condition is not None:

            # If the value is not a list or the list does not
            # satisfy the condition
            if not isinstance(value, list) or \
                not option_condition(value):
                
                # Finalize the error string.
                err_str += option_message + "."

    #-----------------------------------------------------------------#

    # If the error string was updated
    if err_str != f"{field_name}: '{option_name}' ":

        # Store the error.
        return err_str

    # Return an empty string.
    return ""


########################## PUBLIC FUNCTIONS ###########################


def parse_config_model(config: dict[str, object],
                       path: Optional[str] = None) -> \
                        tuple[dict[str, object],
                              list[str],
                              list[str]]:
    """Parse and check the model configuration.

    Parameters
    ----------
    config : :class:`dict`
        The configuration.

    path : :class:`str`, optional
        The path to the configuration file from which the
        configuration was loaded.

    Returns
    -------
    config : :class:`dict`
        The updated configuration.

    errors : :class:`list`
        A list of errors found in the configuration.
    
    warnings : :class:`list`
        A list of warnings regarding the configuration.
    """

    # Initialize an empty list to store the errors.
    errors = []

    # Initialize an empty list to store the warnings.
    warnings = []

    #-----------------------------------------------------------------#

    # If the path is not provided
    if path is None:

        # Set the head of the path to be the current working directory.
        path_head = os.getcwd()
    
    # Otherwise
    else:

        # Set the head of the path to be the directory containing the
        # configuration file.
        path_head = os.path.dirname(path)

    #-----------------------------------------------------------------#

    # Create a deep copy of the configuration.
    config = copy.deepcopy(config)

    #-----------------------------------------------------------------#

    # Use the recursive validator to check
    # and apply defaults to the configuration
    # against the template.
    config_validated, errors_validation, \
        warnings_validation = \
            _check_config_recursive(config = config, 
                                    template = _templates.CONFIG_MODEL)

    # Add validation errors to the error list.
    errors.extend(errors_validation)

    # Add validation warnings to the warning
    # list.
    warnings.extend(warnings_validation)

    #-----------------------------------------------------------------#

    # If there is a 'genes_txt_file' field in the configuration
    if "genes_txt_file" in config_validated:

        # Get the value of the 'genes_txt_file' field.
        genes_txt_file = config_validated["genes_txt_file"]

        # If the 'genes_txt_file' field is not a string
        if not isinstance(genes_txt_file, str):

            # Store the error.
            errors.append(
                "'genes_txt_file' must be a string.")

        # If the default file should be used
        elif genes_txt_file == "default":

            # Get the path to the default file.
            config_validated["genes_txt_file"] = \
                os.path.normpath(\
                    defaults.DATA_FILES_MODEL["genes"])

        # Otherwise
        else:

            # Get the path to the file.
            config_validated["genes_txt_file"] = \
                os.path.normpath(os.path.join(path_head,
                                              genes_txt_file))

    #-----------------------------------------------------------------#

    # If there is a 'latent_pth_file' field in the configuration
    if "latent_pth_file" in \
        config_validated.get("latent_options", {}):

        # Get the value of the 'latent_pth_file' field.
        latent_pth_file = \
            config_validated["latent_options"]["latent_pth_file"]

        # If the 'latent_pth_file' field is not a string
        if not isinstance(latent_pth_file, str):
            
            # Store the error.
            errors.append(
                "latent_options: 'latent_pth_file' must be a "
                "string.")

        # If the default file should be used
        if latent_pth_file == "default":

            # Get the path to the default file.
            config_validated["latent_options"]["latent_pth_file"] = \
                os.path.normpath(\
                    defaults.DATA_FILES_MODEL["gmm"])

        # Otherwise
        else:

            # Get the path to the file.
            config_validated["latent_options"]["latent_pth_file"] = \
                os.path.normpath(os.path.join(path_head,
                                                latent_pth_file))

    #-----------------------------------------------------------------#

    # If there is a 'decoder_pth_file' field in the configuration
    if "decoder_pth_file" in \
        config_validated.get("decoder_options", {}):

        # Get the value of the 'decoder_pth_file' field.
        decoder_pth_file = \
            config_validated["decoder_options"]["decoder_pth_file"]

        # If the 'decoder_pth_file' field is not a string
        if not isinstance(config_validated["decoder_options"][
                            "decoder_pth_file"],
                          str):

            # Store the error.
            errors.append(
                "decoder_options: 'decoder_pth_file' must be a "
                "string.")
        
        # If the default file should be used
        if decoder_pth_file == "default":

            # Get the path to the default file.
            default_decoder_pth_file = \
                os.path.normpath(\
                    defaults.DATA_FILES_MODEL["dec"])

            # If the file is not present locally (it is too large to
            # be distributed with the package on PyPI), download it.
            if not os.path.isfile(default_decoder_pth_file):

                # Download the decoder's weights.
                _internals.download_decoder_pth(\
                    dest_path = default_decoder_pth_file)

            # Point the configuration at the default decoder file.
            config_validated["decoder_options"]["decoder_pth_file"] = \
                default_decoder_pth_file

        # Otherwise
        else:

            # Get the path to the file.
            config_validated["decoder_options"]["decoder_pth_file"] = \
                os.path.normpath(os.path.join(path_head,
                                                decoder_pth_file))

    #-----------------------------------------------------------------#

    # Return the updated configuration, the
    # errors, and the warnings.
    return config_validated, errors, warnings


def parse_config_train(
        config: dict[str, object],
        config_model: Optional[dict[str, object]] = None) -> \
            tuple[dict[str, object],
                  list[str],
                  list[str]]:
    """Parse and check the configuration containing the options for
    training the model.

    Parameters
    ----------
    config : :class:`dict`
        The configuration.
    
    config_model : :class:`dict`, optional
        The configuration of the model. This can be passed to check the
        consistency between the training configuration and the model
        configuration.

    Returns
    -------
    errors : :class:`list`
        A list of errors found in the configuration.
    """

    # Initialize an empty list to store the errors.
    errors = []

    # Initialize an empty list to store the warnings.
    warnings = []

    #-----------------------------------------------------------------#

    # Create a deep copy of the configuration.
    config = copy.deepcopy(config)

    #-----------------------------------------------------------------#

    # Validate against template using recursive checker.
    config_validated, err_val, warn_val = \
        _check_config_recursive(config = config,
                                template = _templates.CONFIG_TRAIN)

    # Add the errors to the list of errors.
    errors.extend(err_val)

    # Add the warnings to the list of warnings.
    warnings.extend(warn_val)

    #-----------------------------------------------------------------#

    # Get the type of latent space in the training configuration.
    latent_type = config_validated.get("latent_type")

    # If the model configuration is provided
    if config_model is not None:

        # Get the type of latent space in the model configuration.
        latent_type_model = config_model.get("latent_type")

        # If the latent types in the training and model configurations
        # do not match
        if latent_type != latent_type_model:

            # Add the error.
            errors.append(
                f"latent_type is different from "
                f"the model configuration: '{latent_type_model}'. "
                f"They must match.")

    #-----------------------------------------------------------------#

    # Return the updated configuration, the errors, and the warnings.
    return config_validated, errors, warnings


def parse_config_rep(config: Optional[dict[str, object]]) -> \
        tuple[dict[str, object],
              list[str],
              list[str]]:
    """Parse and check the configuration for the optimization round(s)
    that find the best representations for a set of samples.

    Parameters
    ----------
    config : :class:`dict`
        The configuration.

    Returns
    -------
    config : :class:`dict`
        The validated configuration.

    errors : :class:`list`
        A list of errors found in the configuration.

    warnings : :class:`list`
        A list of warnings regarding the configuration.
    """

    # Initialize an empty list to store the errors.
    errors = []

    # Initialize an empty list to store the warnings.
    warnings = []

    #-----------------------------------------------------------------#

    # Create a deep copy of the configuration.
    config = copy.deepcopy(config)

    #-----------------------------------------------------------------#

    # Validate against the template.
    config_validated, errors_validation, \
        warnings_validation = \
            _check_config_recursive(config = config,
                                    template = _templates.CONFIG_REP)

    # Add the validation errors to the list of errors.
    errors.extend(errors_validation)

    # Add the validation warnings to the list of warnings.
    warnings.extend(warnings_validation)

    #-----------------------------------------------------------------#

    # Validate the initialization contract separately: 'mode' controls
    # which fields are required, and 'two_opt_multiseed' takes a list
    # of seeds where 'two_opt' takes one, so checks live here to catch
    # a malformed configuration before representations are computed.
    if config_validated.get("latent_type") == "tgmm":

        # Get (or create) the scheme options section.
        scheme_options = config_validated.setdefault(
            "scheme_options", {})

        # Get the initialization section.
        initialization = scheme_options.get("initialization", {})

        # If it is not a dictionary
        if not isinstance(initialization, dict):

            # Add an error.
            errors.append(
                "scheme_options.initialization: must be a dictionary.")

        # Otherwise
        else:

            # Deep-copy it, since we will mutate it below.
            initialization = copy.deepcopy(initialization)

            # Define the options 'initialization' is allowed to have.
            allowed = {
                "mode", "seed", "seeds", "index_file",
                "original_n_samples", "chunk_size"}

            # Get any options that are not allowed.
            unknown = sorted(set(initialization) - allowed)

            # If there are unknown options
            if unknown:

                # Add an error.
                errors.append(
                    "scheme_options.initialization: unsupported "
                    f"option(s): {', '.join(unknown)}.")

            # If 'mode' is missing
            if "mode" not in initialization:

                # Default it to 'sample_keyed'.
                initialization["mode"] = "sample_keyed"

                # Warn that the default was used.
                warnings.append(
                    "scheme_options.initialization.mode: no 'mode' "
                    "found. The default value 'sample_keyed' will be "
                    "used.")

            # Get the (possibly defaulted) mode.
            mode = initialization.get("mode")

            # Define the supported modes.
            modes = {
                "sample_keyed", "legacy_positional", "legacy_indexed"}

            # If the mode is not one of the supported ones
            if mode not in modes:

                # Add an error.
                errors.append(
                    "scheme_options.initialization.mode: 'mode' must "
                    "be one of: legacy_indexed, legacy_positional, "
                    "sample_keyed.")

            # Get the scheme type, which decides whether one seed or a
            # list of seeds is expected.
            scheme_type = config_validated.get("scheme_type")

            # If the scheme uses multiple seeds
            if scheme_type == "two_opt_multiseed":

                # Get the seeds.
                seeds = initialization.get("seeds")

                # If they are missing, empty, or not a list/tuple
                if not isinstance(seeds, (list, tuple)) or not seeds:

                    # Add an error.
                    errors.append(
                        "scheme_options.initialization.seeds: "
                        "'two_opt_multiseed' requires a non-empty list "
                        "of integer seeds.")

                # If any seed is not an integer
                elif any(type(seed) is not int for seed in seeds):

                    # Add an error.
                    errors.append(
                        "scheme_options.initialization.seeds: every "
                        "seed must be an integer.")

                # If the seeds are not all distinct
                elif len(set(seeds)) != len(seeds):

                    # Add an error.
                    errors.append(
                        "scheme_options.initialization.seeds: seeds "
                        "must be distinct.")

                # Otherwise
                else:

                    # Normalize the seeds to a list.
                    initialization["seeds"] = list(seeds)

                # If a single 'seed' was also given
                if "seed" in initialization:

                    # Add an error.
                    errors.append(
                        "scheme_options.initialization.seed: use "
                        "'seeds' with 'two_opt_multiseed', not 'seed'.")

            # Otherwise (the scheme uses a single seed)
            else:

                # Get the seed.
                seed = initialization.get("seed")

                # If the mode requires an integer seed and it is not
                # one
                if mode in {"sample_keyed", "legacy_indexed"} and \
                        type(seed) is not int:

                    # Add an error.
                    errors.append(
                        "scheme_options.initialization.seed: "
                        f"'{mode}' requires one integer seed.")

                # If a seed was given but is not an integer
                elif seed is not None and type(seed) is not int:

                    # Add an error.
                    errors.append(
                        "scheme_options.initialization.seed: 'seed' "
                        "must be an integer when provided.")

                # If a list of 'seeds' was given instead
                if "seeds" in initialization:

                    # Add an error.
                    errors.append(
                        "scheme_options.initialization.seeds: use "
                        "'seed' with 'two_opt', not 'seeds'.")

            # Define the options that only apply to 'legacy_indexed'.
            indexed_options = {
                "index_file", "original_n_samples", "chunk_size"}

            # If the mode is 'legacy_indexed'
            if mode == "legacy_indexed":

                # Get the index file's path.
                index_file = initialization.get("index_file")

                # If it is missing or not a non-empty string
                if not isinstance(index_file, str) or not index_file:

                    # Add an error.
                    errors.append(
                        "scheme_options.initialization.index_file: "
                        "'legacy_indexed' requires a non-empty path.")

                # For each option required alongside the index file
                for option in ("original_n_samples", "chunk_size"):

                    # Get its value.
                    value = initialization.get(option)

                    # If it is not a positive integer
                    if type(value) is not int or value <= 0:

                        # Add an error.
                        errors.append(
                            f"scheme_options.initialization.{option}: "
                            "'legacy_indexed' requires a positive "
                            "integer.")

            # Otherwise (not 'legacy_indexed')
            else:

                # Get any 'legacy_indexed'-only options that were
                # given anyway.
                unexpected = sorted(
                    option for option in indexed_options
                    if option in initialization)

                # If there are any
                if unexpected:

                    # Add an error.
                    errors.append(
                        "scheme_options.initialization: "
                        f"{', '.join(unexpected)} may only be used "
                        "with mode 'legacy_indexed'.")

            # Store the (possibly defaulted/normalized) initialization
            # section back in the configuration.
            scheme_options["initialization"] = initialization

    #-----------------------------------------------------------------#

    # Return the updated configuration, errors, and warnings.
    return config_validated, errors, warnings


def get_data_loader(dataset: dataclasses.GeneExpressionDataset,
                    config: dict[str, object]) -> DataLoader:
    """Get the data loader.

    Parameters
    ----------
    dataset : \
        :class:`bulkdgd.core.dataclasses.GeneExpressionDataset`
        The dataset from which the data loader should be created.

    config : :class:`dict`
        The configuration for the data loader.

    Returns
    -------
    data_loader : :class:`torch.utils.data.DataLoader`
        The data loader.
    """

    # Get the data loader.
    data_loader = DataLoader(dataset = dataset,
                             **config)

    # Return the data loader.
    return data_loader


def get_time_dataframe(time_list: list[tuple[float, float]]) -> \
    pd.DataFrame:
    """Get the data frame containing the information about the
    computing time.

    Parameters
    ----------
    time_list : :class:`list`
        A list of tuples storing, for each epoch, information
        about the CPU and wall clock time used by the entire
        epoch.

    Returns
    -------
    df_time : :class:`pandas.DataFrame`
        A data frame containing data about the computing time.
    """

    # Crate a data frame for the CPU/wall clock time.
    df_time = pd.DataFrame(time_list)

    # Get the platform on which we are running.
    curr_platform = platform.platform()

    # Get the name of the processor.
    curr_processor = platform.processor()

    # Get the number of threads used for running.
    num_threads = torch.get_num_threads()

    # Add a column defining the platform to the data frame.
    df_time.insert(loc = 0,
                   column = "platform",
                   value = curr_platform)

    # Add a column defining the processor to the data frame.
    df_time.insert(loc = 1,
                   column = "processor",
                   value = curr_processor)

    # Add a column defining the number of threads that were used
    # to the data frame.
    df_time.insert(loc = 2,
                   column = "num_threads",
                   value = num_threads)

    # Return the data frame.
    return df_time


def get_final_data_frames_rep(
        rep: torch.Tensor,
        pred_means: torch.Tensor,
        time_opt: list[tuple[float, float]],
        samples_names: list[str],
        genes_names: list[str],
        pred_r_values: Optional[torch.Tensor] = None) -> \
            tuple[pd.DataFrame, pd.DataFrame, \
                  Optional[pd.DataFrame], pd.DataFrame]:
    """Get the final data frames containing the representations, the
    decoder outputs, and the optimization time.

    Parameters
    ----------
    rep : :class:`torch.Tensor`
        The optimized representations (samples x latent dimensions).

    pred_means : :class:`torch.Tensor`
        The predicted means of the distributions modelling the genes'
        counts (samples x genes).

    time_opt : :class:`list`
        Per-epoch CPU/wall clock timing tuples for the optimization.

    samples_names : :class:`list`
        The samples' names.

    genes_names : :class:`list`
        The genes' names.

    pred_r_values : :class:`torch.Tensor` or :obj:`None`
        The predicted r-values of the negative binomial distributions
        (samples x genes), or :obj:`None` for Poisson distributions.

    Returns
    -------
    df_rep : :class:`pandas.DataFrame`
        The representations.

    df_pred_means : :class:`pandas.DataFrame`
        The predicted means of the distributions modelling the genes'
        counts.

    df_pred_r_values : :class:`pandas.DataFrame` or :obj:`None`
        The predicted r-values, or :obj:`None` for Poisson counts.

    df_time_opt : :class:`pandas.DataFrame`
        Data about the optimization time.
    """

    # Convert the tensor containing the predicted scaled means
    # into an array.
    pred_means_array = pred_means.detach().cpu().numpy()

    # Get a data frame containing the predicted scaled means for all
    # samples.
    df_pred_means = pd.DataFrame(pred_means_array.tolist())

    # Set the names of the rows of the data frame to be the names/IDs/
    # indexes of the samples.
    df_pred_means.index = samples_names

    # Set the names of the columns of the data frame to be the names of
    # the genes.
    df_pred_means.columns = genes_names

    #-----------------------------------------------------------------#

    # If the predicted r-values were passed
    if pred_r_values is not None:
        
        # Convert the tensor containing the predicted r-values into an
        # array.
        pred_r_values_array = pred_r_values.detach().cpu().numpy()

        # If the array is one-dimensional (one r-value per gene for all
        # samples)
        if len(pred_r_values_array.shape) == 1:

            # Convert it into a two-dimensional array by repeating the
            # r-values for as many samples we have.
            pred_r_values_array = np.tile(pred_r_values_array,
                                          (len(samples_names), 1))

        # Get a data frame containing the predicted r-values for all
        # samples.
        df_pred_r_values = pd.DataFrame(pred_r_values_array.tolist())

        # Set the names of the rows of the data frame to be the names/
        # IDs/indexes of the samples.
        df_pred_r_values.index = samples_names

        # Set the names of the columns of the data frame to be the
        # names of the genes.
        df_pred_r_values.columns = genes_names

    # Otherwise
    else:

        # The data frame containing the r-values will be None.
        df_pred_r_values = None

    #-----------------------------------------------------------------#

    # Convert the tensor containing the representations into a list.
    rep_list = rep.detach().cpu().numpy().tolist()

    # Create a data frame for the representations.
    df_rep = pd.DataFrame(rep_list)

    # Set the names of the rows of the data frame to be the names/IDs/
    # indexes of the samples.
    df_rep.index = samples_names

    # Name the columns of the data frame as the dimensions of the
    # latent space.
    df_rep.columns = \
        [f"latent_dim_{i}" for i in range(1, df_rep.shape[1]+1)]

    #-----------------------------------------------------------------#

    # Get the data frame containing the information about computing
    # time.
    df_time = get_time_dataframe(time_list = time_opt)

    # Name and sort the columns.
    df_time.columns = \
        ["platform", "processor", "num_threads",
         "opt_round", "epoch",
         "time_tot_epoch_cpu", "time_tot_bw_cpu",
         "time_tot_epoch_wall", "time_tot_bw_wall"]

    #-----------------------------------------------------------------#

    # Return the data frames.
    return df_rep, df_pred_means, df_pred_r_values, df_time


def get_final_data_frames_train(
        reps: tuple[torch.Tensor, torch.Tensor],
        pred_means: tuple[torch.Tensor, torch.Tensor],
        losses_list: list[float],
        time_train: list[tuple[float, float]],
        samples_names_train: list[str],
        samples_names_test: list[str],
        df_other_data_train: pd.DataFrame,
        df_other_data_test: pd.DataFrame,
        genes_names: list[str],
        pred_r_values: Optional[tuple[np.ndarray, np.ndarray]] = None,
        metrics_rows_train: Optional[list[str]] = None,
        metrics_rows_test: Optional[list[str]] = None) -> \
            tuple[tuple[pd.DataFrame, pd.DataFrame],
                  tuple[pd.DataFrame, pd.DataFrame],
                  Optional[tuple[pd.DataFrame, pd.DataFrame]],
                  pd.DataFrame,
                  pd.DataFrame]:
    """Get the final data frames containing the representations, the
    decoder outputs, the training losses, and the training time.

    Parameters
    ----------
    reps : :class:`tuple`
        The optimized representations for the training and test
        samples, as a ``(train, test)`` pair of tensors.

    pred_means : :class:`tuple`
        The predicted means for the training and test samples, as a
        ``(train, test)`` pair of tensors.

    losses_list : :class:`list`
        The losses calculated during each training epoch.

    time_train : :class:`list`
        Per-epoch CPU/wall clock timing tuples for training.

    samples_names_train : :class:`list`
        The training samples' names.

    samples_names_test : :class:`list`
        The test samples' names.

    df_other_data_train : :class:`pandas.DataFrame`
        Additional data about the training samples.

    df_other_data_test : :class:`pandas.DataFrame`
        Additional data about the test samples.

    genes_names : :class:`list`
        The genes' names.

    pred_r_values : :class:`tuple` or :class:`torch.Tensor` or \
        :obj:`None`, optional
        The predicted r-values: :obj:`None` for Poisson output, a
        single tensor shared across samples for feature-dispersion
        negative binomial output, or a ``(train, test)`` tensor pair
        for full-dispersion negative binomial output.

    metrics_rows_train : :class:`list` or :obj:`None`, optional
        Per-epoch clustering metric rows for the training samples.

    metrics_rows_test : :class:`list` or :obj:`None`, optional
        Per-epoch clustering metric rows for the test samples.

    Returns
    -------
    :class:`tuple`
        A ``((df_rep_train, df_rep_test), (df_pred_means_train,
        df_pred_means_test), dfs_pred_r_values, df_loss, dfs_metrics,
        df_time)`` tuple. ``dfs_pred_r_values`` and ``dfs_metrics``
        are :obj:`None` if not applicable, otherwise ``(train,
        test)`` data frame pairs (a one-element tuple for
        feature-dispersion r-values).
    """

    # Get the representations for the training and test samples.
    rep_train, rep_test = reps

    # Get the predicted means for the training and test samples.
    pred_means_train, pred_means_test = pred_means

    #-----------------------------------------------------------------#

    # Create a data frame for the representations.
    df_rep_train = \
        pd.DataFrame(rep_train.detach().cpu().numpy().tolist())

    # Set the names of the rows of the data frame to be the names/IDs/
    # indexes of the samples.
    df_rep_train.index = samples_names_train

    # Name the columns of the data frame as the dimensions of the
    # latent space.
    df_rep_train.columns = \
        [f"latent_dim_{i}" for i in range(1, df_rep_train.shape[1]+1)]

    # Concatenate the data frame with the one containing additional
    # information about the samples.
    df_rep_train = pd.concat([df_rep_train, df_other_data_train],
                             axis = 1)

    #-----------------------------------------------------------------#

    # Create a data frame for the representations.
    df_rep_test = \
        pd.DataFrame(rep_test.detach().cpu().numpy().tolist())

    # Set the names of the rows of the data frame to be the names/IDs/
    # indexes of the samples.
    df_rep_test.index = samples_names_test

    # Name the columns of the data frame as the dimensions of the
    # latent space.
    df_rep_test.columns = \
        [f"latent_dim_{i}" for i in range(1, df_rep_test.shape[1]+1)]

    # Concatenate the data frame with the one containing additional
    # information about the samples.
    df_rep_test = pd.concat([df_rep_test, df_other_data_test],
                            axis = 1)

    #-----------------------------------------------------------------#

    # Get a data frame containing the predicted scaled means for the
    # training samples.
    df_pred_means_train = \
        pd.DataFrame(pred_means_train.detach().cpu().numpy().tolist())

    # Set the names of the rows of the data frame to be the names/IDs/
    # indexes of the training samples.
    df_pred_means_train.index = samples_names_train

    # Set the names of the columns of the data frame to be the names of
    # the genes.
    df_pred_means_train.columns = genes_names

    #-----------------------------------------------------------------#

    # Get a data frame containing the predicted scaled means for the
    # test samples.
    df_pred_means_test = \
        pd.DataFrame(pred_means_test.detach().cpu().numpy())

    # Set the names of the rows of the data frame to be the names/IDs/
    # indexes of the test samples.
    df_pred_means_test.index = samples_names_test

    # Set the names of the columns of the data frame to be the names of
    # the genes.
    df_pred_means_test.columns = genes_names

    #-----------------------------------------------------------------#

    # Initialize the variable to store the data frame containing the
    # predicted r-values to None.
    dfs_pred_r_values = None

    # If the predicted r-values were passed
    if pred_r_values is not None:

        # If there is one r-value per gene for all samples
        if isinstance(pred_r_values, torch.Tensor):

            # Get all the samples' names.
            samples_names = samples_names_train + samples_names_test

            # Convert the tensor containing the predicted r-values into
            # an array.
            pred_r_values_array = pred_r_values.detach().cpu().numpy()

            # If the array is one-dimensional (one r-value per gene for
            # all samples)
            if len(pred_r_values_array.shape) == 1:

                # Convert it into a two-dimensional array by repeating
                # the r-values for as many samples we have.
                pred_r_values_array = np.tile(pred_r_values_array,
                                            (len(samples_names), 1))

            # Get a data frame containing the predicted r-values for
            # all samples.
            df_pred_r_values = \
                pd.DataFrame(pred_r_values_array.tolist())

            # Set the names of the rows of the data frame to be the
            # IDs/indexes of the samples.
            df_pred_r_values.index = samples_names

            # Set the names of the columns of the data frame to be the
            # names of the genes.
            df_pred_r_values.columns = genes_names

            # Overwrite the variable storing the data frame(s)
            # containing the predicted r-values.
            dfs_pred_r_values = (df_pred_r_values,)

        #-------------------------------------------------------------#
        
        # If there are different r-values for the different samples
        elif isinstance(pred_r_values, tuple):

            # Get the predicted r-values for the training and test
            # samples.
            pred_r_values_train, pred_r_values_test = pred_r_values

            # Get a data frame containing the predicted r-values
            # for the training samples.
            df_pred_r_values_train = \
                pd.DataFrame(
                    pred_r_values_train.detach().cpu().numpy(
                        ).tolist())

            # Set the names of the rows of the data frame to be
            # the IDs/indexes of the training samples.
            df_pred_r_values_train.index = samples_names_train

            # Set the names of the columns of the data frame to
            # be the names of the genes.
            df_pred_r_values_train.columns = genes_names
            
            # Get a data frame containing the predicted r-values
            # for the test samples.
            df_pred_r_values_test = \
                pd.DataFrame(
                    pred_r_values_test.detach().cpu().numpy(
                        ).tolist())

            # Set the names of the rows of the data frame to be
            # the IDs/indexes of the test samples.
            df_pred_r_values_test.index = samples_names_test

            # Set the names of the columns of the data frame to
            # be the names of the genes.
            df_pred_r_values_test.columns = genes_names

            # Overwrite the variable storing the data frame(s)
            # containing the predicted r-values.
            dfs_pred_r_values = \
                (df_pred_r_values_train, df_pred_r_values_test)

    #-----------------------------------------------------------------#

    # Create a data frame storing the training losses.
    df_loss = pd.DataFrame(losses_list)

    # Set the data frame's columns.
    df_loss.columns = \
        ["epoch", "-log p_dens(z_train)",
         "-log p_dens(x_train|z_train)", "loss_train",
         "-log p_dens(z_test)", "-log p_dens(x_test|z_test)",
         "loss_test"]

    #-----------------------------------------------------------------#

    # Initialize a variable to store the data frames containing the
    # clustering metrics to None.
    dfs_metrics = None

    # Add per-epoch clustering metrics, if available.
    if metrics_rows_train and metrics_rows_test:

        # Build data frames from per-epoch metric rows.
        df_metrics_train = pd.DataFrame(metrics_rows_train)
        df_metrics_test = pd.DataFrame(metrics_rows_test)

        # Proceed only if both include an epoch column.
        if "epoch" in df_metrics_train.columns \
            and "epoch" in df_metrics_test.columns:

            # Create a tuple with the two data frames.
            dfs_metrics = (df_metrics_train, df_metrics_test)

    #-----------------------------------------------------------------#

    # Get the data frame containing the information about the
    # computing time.
    df_time = get_time_dataframe(time_list = time_train)

    #-----------------------------------------------------------------#

    # Return all the data frames.
    return ((df_rep_train, df_rep_test),
            (df_pred_means_train, df_pred_means_test),
            dfs_pred_r_values,
            df_loss,
            dfs_metrics,
            df_time)


def normalize_loss(loss: torch.Tensor,
                   loss_type: str,
                   loss_norm_type: str,
                   loss_norm_options: dict[str, object]) -> \
                    torch.Tensor:
    """Normalize the loss per epoch.

    Parameters
    ----------
    loss : :class:`torch.Tensor`
        The loss to be normalized.

    loss_type : :class:`str`, \
        {``"latent"``, ``"decoder"``, ``"total"``}
        The type of loss.

    loss_norm_type : :class:`str`
        The name of the normalization method.

    loss_norm_options : :class:`dict`
        A dictionary of options for the normalization method.
    
    Returns
    -------
    loss_normalized: :class:`torch.Tensor`
        The normalized loss.
    """

    # If the normalization method is 'none'
    if loss_norm_type == "none":

        # Simply return the loss.
        return loss

    # If the loss should be normalized by the total number of samples
    elif loss_norm_type == "n_samples":

        # Return the loss normalized by the total number of samples.
        return loss / loss_norm_options["n_samples"]

    # If the loss should be normalized by the number of samples times
    # the total number of genes
    elif loss_norm_type == "n_samples * n_genes":

        # Return the loss normalized by the number of samples times the
        # total number of genes.
        return loss / \
            (loss_norm_options["n_samples"] *
             loss_norm_options["n_genes"])

    # If the loss should be normalized by the number of samples times
    # the latent dimension
    elif loss_norm_type == "n_samples * latent_dim":
        
        # Return the loss normalized by the number of samples times the
        # latent dimension.
        return loss / \
            (loss_norm_options["n_samples"] *
             loss_norm_options["latent_dim"])


def get_pathways_saliency_map(saliency_map: np.ndarray,
                              pathways: dict[str, list[str]],
                              genes_names: list[str]) -> pd.DataFrame:
    """Get the pathway saliency scores by aggregating the gene-level
    saliency scores for the genes in each pathway.
    
    Parameters
    ----------
    saliency_map : :class:`numpy.ndarray`
        A 2D array containing the saliency scores for each gene.
    
    pathways : :class:`dict`
        A dictionary where the keys are pathway names and the values
        are lists of gene IDs belonging to each pathway.
    
    genes_names : :class:`list`
        A list containing the names/IDs of the genes corresponding
        to the rows of the saliency map.

    Returns
    -------
    :class:`pandas.DataFrame`
        A data frame where the rows are pathways and the columns
        are the saliency scores aggregated for each pathway.
    """

    # Aggregate by pathway
    pathway_scores = {}
    
    # For each pathway
    for pathway_name, gene_ids in pathways.items():
        
        # Get the indices of genes in this pathway in the saliency map.
        gene_indices = \
            [i for i, g in enumerate(genes_names) if g in gene_ids]
        
        # Compute the mean saliency across the genes in the pathway.
        pathway_scores[pathway_name] = \
            saliency_map[gene_indices].mean(dim=0)

    # Return a data frame with the pathway scores.
    return pd.DataFrame(pathway_scores).T


#---------------------------------------------------------------------#


def _save_epoch_df(df, path, sep = ","):

    """Write a per-epoch table as Parquet or as text, depending on the
    path's extension.

    Parameters
    ----------
    df : :class:`pandas.DataFrame`
        The table to write.

    path : :class:`str`
        The output file path. A ``.parquet``/``.pq`` extension writes
        Parquet; any other extension writes delimited text.

    sep : :class:`str`, optional
        The field separator used when writing as text.
    """

    # If the path calls for Parquet
    if str(path).lower().endswith((".parquet", ".pq")):

        # Parquet preserves float64 exactly; text round-trips it with
        # up to ~1e-12 error, which matters since these files are the
        # record the run is reconstructed from.
        df.to_parquet(path,
                      engine = "pyarrow",
                      compression = "snappy",
                      index = True)

    # Otherwise
    else:

        # Write as delimited text.
        df.to_csv(path, sep = sep, index = True, header = True)


def save_rep_epoch(epoch: int,
                   prefix: str,
                   latent_dim: int,
                   rep_layer: latents.RepresentationLayer,
                   samples_names: list[str],
                   save_dir: Optional[str] = None) -> None:
    """Save the representations of the samples at the end of a given
    epoch.

    Parameters
    ----------
    epoch : :class:`int`
        The epoch number.

    prefix : :class:`str`
        The prefix to use for the output file name.
    
    latent_dim : :class:`int`
        The dimensionality of the latent space.
    
    rep_layer : :class:`core.latent.RepresentationLayer`
        The layer containing the representations.
    
    samples_names : :class:`list`
        The names of the samples.

    save_dir : :class:`str`
        The directory where to save the representations.
    """

    # Set the names of the columns containing the representations.
    columns_names = \
        [f"latent_dim_{i}" for i in range(1, latent_dim + 1)]
    
    #-----------------------------------------------------------------#
    
    # Get the directory where to save the representations.
    save_dir = \
        save_dir if save_dir is not None else os.getcwd()
    
    # Create the directory if it does not exist.
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    #-----------------------------------------------------------------#

    # Get the current representations.
    rep = rep_layer().detach().cpu().numpy()

    # Create a data frame with the representations.
    df_rep = pd.DataFrame(data = rep,
                          index = samples_names,
                          columns = columns_names)
    
    # Set the path to the file where the representations will be saved.
    rep_out = os.path.join(save_dir, f"{prefix}_rep_{epoch}.parquet")
    
    # Save the representations.
    _save_epoch_df(df_rep, rep_out,
                  sep = ",")


def save_latent_probs_epoch(probs: torch.Tensor,
                            epoch: int,
                            prefix: str,
                            n_components: int,
                            samples_names: list[str],
                            save_dir: Optional[str] = None) -> None:
    """Save the probability densities of the samples at the end of a
    given epoch.

    Parameters
    ----------
    probs : :class:`torch.Tensor`
        A tensor containing the probability densities of the samples.

    epoch : :class:`int`
        The epoch number.
    
    prefix : :class:`str`
        The prefix to use for the output file name.
    
    n_components : :class:`int`
        The number of components of the GMM.
    
    samples_names : :class:`list`
        The names of the samples.

    save_dir : :class:`str`, optional
        The directory where to save the probability densities.
    """

    # Set the names of the columns containing the components.
    columns_names = [f"component_{i}" for i in range(1, n_components + 1)]

    #-----------------------------------------------------------------#

    # Get the directory where to save the probability densities.
    save_dir = \
        save_dir if save_dir is not None \
        else os.getcwd()

    # Create the directory if it does not exist.
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    #-----------------------------------------------------------------#

    # Create a data frame with the probability densities.
    df_probs = \
        pd.DataFrame(data = probs.detach().cpu().numpy(),
                     index = samples_names,
                     columns = columns_names)

    # Set the path to the file where the probability densities will be
    # saved.
    probs_out = \
        os.path.join(save_dir, f"{prefix}_latent_probs_{epoch}.parquet")

    # Save the probability densities for the samples.
    _save_epoch_df(df_probs, probs_out,
                    sep = ",")


def save_latent_means_epoch(epoch: int,
                            means: np.ndarray,
                            latent_dim: int,
                            n_components: int,
                            save_dir: Optional[str] = None) -> None:
    """Save the means of the GMM at the end of a given epoch.

    Parameters
    ----------
    epoch : :class:`int`
        The epoch number.
    
    means : :class:`numpy.ndarray`
        The means of the GMM.
    
    latent_dim : :class:`int`
        The dimensionality of the latent space.
    
    n_components : :class:`int`
        The number of components of the GMM.
    
    save_dir : :class:`str`, optional
        The directory where to save the means.
    """

    # Set the names of the rows.
    index_names = \
        [f"component_{i}" for i in range(1, n_components + 1)]

    # Set the names of the columns.
    columns_names = \
        [f"latent_dim_{i}" for i in range(1, latent_dim + 1)]
    
    #-----------------------------------------------------------------#

    # Get the directory where to save the means.
    save_dir = save_dir if save_dir is not None else os.getcwd()

    # Create the directory if it does not exist.
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    #-----------------------------------------------------------------#

    # Create a data frame with the means.
    df_means = pd.DataFrame(data = means,
                            index = index_names,
                            columns = columns_names)

    # Set the path to the file where the means will be saved.
    means_out = os.path.join(save_dir, f"latent_means_{epoch}.parquet")

    # Save the means for the training samples.
    _save_epoch_df(df_means, means_out,
                    sep = ",")


def save_model_epoch(epoch: int,
                     decoder: "decoders.Decoder",
                     latent: "latents.LatentSpaceBase",
                     save_dir: Optional[str] = None) -> None:
    """Save the decoder's weights and the latent space's parameters at
    the end of a given epoch.

    Parameters
    ----------
    epoch : :class:`int`
        The epoch.

    decoder : :class:`core.decoders.Decoder`
        The decoder, whose weights are saved.

    latent : :class:`core.latents.LatentSpaceBase`
        The latent space, whose parameters are saved.

    save_dir : :class:`str`, optional
        The directory where to save them. If not provided, the current
        working directory is used.
    """

    # Get the directory where to save the model. Saving at every
    # epoch (rather than only at the end) lets a run be resumed if it
    # is interrupted before training finishes.
    save_dir = \
        save_dir if save_dir is not None else os.getcwd()

    # Create the directory if it does not exist.
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    #-----------------------------------------------------------------#

    # Save the decoder's weights, under the same name the end of
    # training uses, with the epoch appended.
    torch.save(decoder.state_dict(),
               os.path.join(save_dir, f"dec_{epoch}.pth"))

    # Save the latent space's parameters.
    torch.save(latent.state_dict(),
               os.path.join(save_dir, f"gmm_{epoch}.pth"))


def save_genes_saliency_maps_epoch(
        epoch: int,
        saliency_map: np.ndarray,
        prefix: str,
        genes_names: list[str],
        save_dir: Optional[str] = None) -> None:
    """Save the saliency map at the end of a given epoch.

    Parameters
    ----------
    epoch : :class:`int`
        The epoch number.
    
    saliency_map : :class:`numpy.ndarray`
        The saliency map.
    
    prefix : :class:`str`
        The prefix to use for the output file name.
    
    genes_names : :class:`list`
        The names of the genes.

    save_dir : :class:`str`, optional
        The directory where to save the saliency map.
    """
    
    # Set the name of the rows.
    index_names = genes_names

    # Set the names of the columns.
    columns_names = \
        [f"latent_dim_{i}" for i \
         in range(1, saliency_map.shape[1] + 1)]

    #-----------------------------------------------------------------#
    
    # Get the directory where to save the saliency map.
    save_dir = save_dir if save_dir is not None else os.getcwd()
    
    # Create the directory if it does not exist.
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    #-----------------------------------------------------------------#
    
    # Create a data frame with the saliency map.
    df_saliency_map = \
        pd.DataFrame(data = saliency_map,
                     index = index_names,
                     columns = columns_names)

    # Set the path to the file where the saliency map will be saved.
    saliency_map_out = \
        os.path.join(save_dir, f"{prefix}_saliency_map_{epoch}.parquet")

    # Save the saliency map.
    _save_epoch_df(df_saliency_map, saliency_map_out,
                           sep = ",")


def save_pathways_saliency_maps_epoch(
        epoch: int,
        saliency_map: np.ndarray,
        prefix: str,
        pathways_names: list[str],
        save_dir: Optional[str] = None) -> None:
    """Save the saliency map at the end of a given epoch.

    Parameters
    ----------
    epoch : :class:`int`
        The epoch number.
    
    saliency_map : :class:`numpy.ndarray`
        The saliency map.
    
    prefix : :class:`str`
        The prefix to use for the output file name.
    
    pathways_names : :class:`list`
        The names of the pathways.

    save_dir : :class:`str`, optional
        The directory where to save the saliency map.
    """
    
    # Set the name of the rows.
    index_names = pathways_names

    # Set the names of the columns.
    columns_names = \
        [f"latent_dim_{i}" for i \
         in range(1, saliency_map.shape[1] + 1)]

    #-----------------------------------------------------------------#
    
    # Get the directory where to save the saliency map.
    save_dir = save_dir if save_dir is not None else os.getcwd()
    
    # Create the directory if it does not exist.
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    #-----------------------------------------------------------------#
    
    # Create a data frame with the saliency map.
    df_saliency_map = \
        pd.DataFrame(data = saliency_map,
                     index = index_names,
                     columns = columns_names)

    # Set the path to the file where the saliency map will be saved.
    saliency_map_out = \
        os.path.join(save_dir, f"{prefix}_saliency_map_{epoch}.parquet")

    # Save the saliency map.
    _save_epoch_df(df_saliency_map, saliency_map_out,
                           sep = ",")


#######################################################################


def load_shipped_model(seed = None):

    """Get the configuration of one member of the ensemble that ships
    with the package.

    Parameters
    ----------
    seed : :class:`int`, optional
        The seed identifying the ensemble member. If not provided,
        the base member's seed is used.

    Returns
    -------
    :class:`dict`
        The keyword arguments ``BulkDGD`` needs to load the trained
        model, with the paths to the fitted parameters filled in.
    """

    # Fall back to the base ensemble member if no seed was given.
    seed = seed if seed is not None else defaults.BASE_SEED

    # If the seed does not identify a shipped ensemble member
    if seed not in defaults.ENSEMBLE_SEEDS:
        errstr = \
            f"'{seed}' is not a member of the shipped ensemble. The " \
            f"members are: {', '.join(defaults.ENSEMBLE_SEEDS)}."
        raise ValueError(errstr)

    # Get the per-seed paths to the model's files.
    files = defaults.data_files_model(seed)

    # Load the member's configuration file.
    with open(files["config"], "r") as f:
        config = yaml.safe_load(f)

    #-----------------------------------------------------------------#

    # Resolve the paths explicitly here rather than leaving "default":
    # that sentinel resolves through 'DATA_FILES_MODEL', which always
    # names the base member, so a non-base seed would silently get the
    # base model's files instead of its own.
    config["latent_options"]["latent_pth_file"] = files["gmm"]

    # The mixture ships with the package; the decoder does not, and is
    # fetched on first use into the same per-seed directory.
    if not os.path.isfile(files["dec"]):
        _internals.util.download_decoder_pth(dest_path = files["dec"],
                                             seed = seed)

    # Point the configuration at the (now guaranteed present) decoder
    # weights.
    config["decoder_options"]["decoder_pth_file"] = files["dec"]

    #-----------------------------------------------------------------#

    # Resolve this sentinel too: "default" is understood by the
    # configuration loader, not by 'BulkDGD.__init__', which passes
    # it straight to 'open'.
    genes = config.get("genes_txt_file", "default")

    # Substitute the shipped gene list's path for the sentinel.
    config["genes_txt_file"] = \
        files["genes"] if genes in (None, "default") else genes

    #-----------------------------------------------------------------#

    # Fill in the defaults 'BulkDGD.__init__' otherwise expects.
    config.setdefault("latent_type", "tgmm")
    config.setdefault("scaling_factor", "mean")
    config.setdefault("dtype", "float32")

    # Return the completed configuration.
    return config
