#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-

#    _util.py
#
#    Private utilities to check the configurations of datasets built
#    from Recount3 studies.
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
    "Private utilities to check the configurations of datasets " \
    "built from Recount3 studies."


#######################################################################


# Import from the standard library.
import copy
import re

# Import from 'bulkdgd'.
from . import _templates, defaults


#######################################################################


# Set how each type is named in error messages.
_TYPES_NAMES = \
    {str : "a string",
     int : "an integer",
     float : "a number",
     bool : "a boolean",
     type(None) : "null",
     list : "a list",
     dict : "a dictionary"}

# Set the templates of the projects whose configuration is a single
# set of options.
_PROJECTS_TEMPLATES = \
    {"GTEx" : _templates.CONFIG_GTEX,
     "TCGA" : _templates.CONFIG_TCGA}


########################## PRIVATE FUNCTIONS ##########################


def _get_types_str(types: tuple[type, ...]) -> str:
    """Get a string naming the given types.

    Parameters
    ----------
    types : :class:`tuple`
        The types.

    Returns
    -------
    types_str : :class:`str`
        The types' names, joined by "or".
    """

    # Return the types' names.
    return " or ".join(_TYPES_NAMES[t] for t in types)


def _as_dict(value: object) -> dict:
    """Get a value if it is a dictionary, or an empty dictionary.

    Parameters
    ----------
    value : :class:`object`
        The value.

    Returns
    -------
    value : :class:`dict`
        The value, or an empty dictionary if it is not one.
    """

    # Return the value if it is a dictionary, or an empty one.
    return value if isinstance(value, dict) else {}


def _as_list(value: object) -> list:
    """Get a value if it is a list, or an empty list.

    Parameters
    ----------
    value : :class:`object`
        The value.

    Returns
    -------
    value : :class:`list`
        The value, or an empty list if it is not one.
    """

    # Return the value if it is a list, or an empty one.
    return value if isinstance(value, list) else []


def _check_value(value: object,
                 spec: dict[str, object],
                 path: str) -> list[str]:
    """Check a value against its specification.

    Parameters
    ----------
    value : :class:`object`
        The value.

    spec : :class:`dict`
        The value's specification: the allowed types (``"type"``)
        and, optionally, a check and its error message
        (``"condition"``, ``"message"``), the template of a
        dictionary (``"section"``), the types of a free dictionary's
        keys and the specification of its values (``"keys"``,
        ``"values"``), and the specification of a list's items
        (``"items"``).

    path : :class:`str`
        The value's path in the configuration, used in error
        messages.

    Returns
    -------
    errors : :class:`list`
        The errors found.
    """

    # Initialize the list of errors.
    errors = []

    #-----------------------------------------------------------------#

    # If the value has a type that is not allowed
    if not isinstance(value, spec["type"]):

        # Add the error.
        errors.append(\
            f"{path}: must be {_get_types_str(spec['type'])}.")

        # Return the errors.
        return errors

    # If the value does not satisfy the condition, if any
    if "condition" in spec and not spec["condition"](value):

        # Add the error.
        errors.append(f"{path}: {spec['message']}.")

    #-----------------------------------------------------------------#

    # If the value is a dictionary with fixed options
    if isinstance(value, dict) and "section" in spec:

        # Check the options.
        errors.extend(\
            _check_section(config = value,
                           template = spec["section"],
                           path = path))

    # If the value is a dictionary with free keys
    elif isinstance(value, dict) and "keys" in spec:

        # For each key and value
        for key, val in value.items():

            # If the key has a type that is not allowed
            if not isinstance(key, spec["keys"]):

                # Add the error.
                errors.append(\
                    f"{path}: the key {key!r} must be " \
                    f"{_get_types_str(spec['keys'])}.")

                # Go to the next key.
                continue

            # Check the value.
            errors.extend(\
                _check_value(value = val,
                             spec = spec["values"],
                             path = f"{path}.{key}"))

    # If the value is a list whose items have a specification
    elif isinstance(value, list) and "items" in spec:

        # For each item
        for num_item, item in enumerate(value):

            # Check the item.
            errors.extend(\
                _check_value(value = item,
                             spec = spec["items"],
                             path = f"{path}[{num_item}]"))

    #-----------------------------------------------------------------#

    # Return the errors.
    return errors


def _check_section(config: dict[str, object],
                   template: dict[str, object],
                   path: str = "") -> list[str]:
    """Check a dictionary of options against a template.

    Parameters
    ----------
    config : :class:`dict`
        The options.

    template : :class:`dict`
        The template, mapping each option to its specification, with
        ``"required"`` set for the options that must be present.

    path : :class:`str`, ``""``
        The options' path in the configuration, used in error
        messages.

    Returns
    -------
    errors : :class:`list`
        The errors found.
    """

    # Initialize the list of errors.
    errors = []

    # Set the head of the error messages.
    head = f"{path}: " if path else ""

    #-----------------------------------------------------------------#

    # Get the options not in the template.
    unknown = [key for key in config if key not in template]

    # If there are any
    if unknown:

        # Get the supported options as a string.
        supported_str = ", ".join(f"'{key}'" for key in template)

        # Add the error.
        errors.append(\
            f"{head}unknown option(s) " \
            f"{', '.join(repr(key) for key in unknown)}. The " \
            f"supported options are: {supported_str}.")

    #-----------------------------------------------------------------#

    # For each option in the template
    for key, spec in template.items():

        # If the option is missing
        if key not in config:

            # If it is required
            if spec.get("required", False):

                # Add the error.
                errors.append(\
                    f"{head}missing required option '{key}'.")

            # Go to the next option.
            continue

        # Check the option's value.
        errors.extend(\
            _check_value(value = config[key],
                         spec = spec,
                         path = f"{path}.{key}" if path else key))

    #-----------------------------------------------------------------#

    # Return the errors.
    return errors


def _get_mapping_warnings(mapping: dict[object, object],
                          path: str) -> list[str]:
    """Get the warnings about the old values of a mapping that are
    not strings.

    Parameters
    ----------
    mapping : :class:`dict`
        The old values mapped to the new ones.

    path : :class:`str`
        The mapping's path in the configuration.

    Returns
    -------
    warnings : :class:`list`
        The warnings.
    """

    # Return a warning for each old value that is not a string.
    return \
        [f"{path}: the key {key!r} is not a string " \
         f"({type(key).__name__}), so it matches no text value. " \
         "Quote it to match text." \
         for key in _as_dict(mapping) if not isinstance(key, str)]


def _get_add_warnings(metadata_to_add: dict[str, dict[str, object]],
                      path: str) -> list[str]:
    """Get the warnings about the columns to add.

    Parameters
    ----------
    metadata_to_add : :class:`dict`
        The columns to add.

    path : :class:`str`
        The columns' path in the configuration.

    Returns
    -------
    warnings : :class:`list`
        The warnings.
    """

    # Return a warning for each column setting '_all_' with other keys.
    return \
        [f"{path}.{col}: '_all_' is set, so the other keys are " \
         "ignored." \
         for col, values in _as_dict(metadata_to_add).items() \
         if "_all_" in _as_dict(values) and len(values) > 1]


def _get_options_warnings(config: dict[str, object],
                          path: str) -> list[str]:
    """Get the warnings about the mappings and the columns to add in
    a set of options.

    Parameters
    ----------
    config : :class:`dict`
        The options.

    path : :class:`str`
        The options' path in the configuration.

    Returns
    -------
    warnings : :class:`list`
        The warnings.
    """

    # Initialize the list of warnings.
    warnings = []

    # Set the head of the paths.
    head = f"{path}." if path else ""

    #-----------------------------------------------------------------#

    # For each option holding mappings
    for option in ["mapping", "mapping_geo"]:

        # For each column and its mapping
        for col, mapping in _as_dict(config.get(option)).items():

            # Get the warnings about the mapping.
            warnings.extend(\
                _get_mapping_warnings(\
                    mapping = mapping,
                    path = f"{head}{option}.{col}"))

    #-----------------------------------------------------------------#

    # For each option holding columns to add
    for option in ["metadata_to_add", "metadata_to_add_geo"]:

        # Get the warnings about the columns.
        warnings.extend(\
            _get_add_warnings(\
                metadata_to_add = config.get(option),
                path = f"{head}{option}"))

    #-----------------------------------------------------------------#

    # Return the warnings.
    return warnings


def _check_config_tcga(config: dict[str, object]) -> \
        tuple[list[str], list[str]]:
    """Check the options of a TCGA dataset's configuration that the
    template cannot check.

    Parameters
    ----------
    config : :class:`dict`
        The configuration.

    Returns
    -------
    errors : :class:`list`
        The errors found.

    warnings : :class:`list`
        The warnings.
    """

    # Initialize the list of errors.
    errors = []

    # Initialize the list of warnings.
    warnings = []

    #-----------------------------------------------------------------#

    # For each column split at separators and its options
    for col, options in \
        _as_dict(config.get("metadata_to_process")).items():

        # Get the separators' names.
        names = \
            [get_separator_name(sep) for sep \
             in _as_list(_as_dict(options).get("separators")) \
             if isinstance(sep, str)]

        # For each name given to a new column
        for name in _as_dict(_as_dict(options).get("columns")):

            # If no separator has it
            if name not in names:

                # Add the warning.
                warnings.append(\
                    f"metadata_to_process.{col}.columns: no " \
                    f"separator is named {name!r}, so its column " \
                    "is empty. The separators' names are: " \
                    f"{', '.join(repr(n) for n in names)}.")

    #-----------------------------------------------------------------#

    # For each new column built from other columns and its options
    for col, options in \
        _as_dict(config.get("metadata_to_merge")).items():

        # If the options are not a dictionary
        if not isinstance(options, dict):

            # Go to the next column.
            continue

        # Set the path to the column's options.
        path = f"metadata_to_merge.{col}"

        # Get whether the columns are joined or merged.
        is_join = "columns_to_join" in options
        is_merge = "columns_to_merge" in options

        # If both or neither are given
        if is_join == is_merge:

            # Add the error.
            errors.append(\
                f"{path}: exactly one of 'columns_to_join' and " \
                "'columns_to_merge' must be given.")

            # Go to the next column.
            continue

        # Get the option giving the columns.
        given = "columns_to_join" if is_join else "columns_to_merge"

        # Set the options it ignores.
        ignored = \
            ["mapping", "serial"] if is_join else ["separator"]

        # For each ignored option given
        for option in [o for o in ignored if o in options]:

            # Add the warning.
            warnings.append(\
                f"{path}: '{option}' is ignored with '{given}'.")

        # Get the warnings about the merged values' mapping.
        warnings.extend(\
            _get_mapping_warnings(\
                mapping = options.get("mapping"),
                path = f"{path}.mapping"))

        # For each column joined
        for num_item, item in \
            enumerate(_as_list(options.get("columns_to_join"))):

            # Get the warnings about the column's mapping.
            warnings.extend(\
                _get_mapping_warnings(\
                    mapping = _as_dict(item).get("mapping"),
                    path = f"{path}.columns_to_join[{num_item}]" \
                           ".mapping"))

    #-----------------------------------------------------------------#

    # Return the errors and the warnings.
    return errors, warnings


def _check_config_sra_study(config: dict[str, object],
                            path: str) -> \
        tuple[list[str], list[str]]:
    """Check the options of an SRA study that the template cannot
    check.

    Parameters
    ----------
    config : :class:`dict`
        The study's options.

    path : :class:`str`
        The study's path in the configuration.

    Returns
    -------
    errors : :class:`list`
        The errors found.

    warnings : :class:`list`
        The warnings.
    """

    # Initialize the list of errors.
    errors = []

    # Initialize the list of warnings.
    warnings = []

    #-----------------------------------------------------------------#

    # Get the GEO code of the study.
    geo_code = \
        _as_dict(config.get("metadata_study")).get("study_geo_code")

    # If the GEO code is given and is not a string
    if geo_code is not None and not isinstance(geo_code, str):

        # Add the error.
        errors.append(\
            f"{path}.metadata_study.study_geo_code: must be a " \
            "string.")

    #-----------------------------------------------------------------#

    # Set the options using the GEO metadata.
    options_geo = \
        ["query_string_geo",
         "metadata_to_keep_geo",
         "metadata_to_add_geo",
         "mapping_geo"]

    # If there is no GEO code
    if geo_code is None:

        # For each option using the GEO metadata given
        for option in [o for o in options_geo if o in config]:

            # Add the warning.
            warnings.append(\
                f"{path}: '{option}' is ignored without " \
                "'metadata_study.study_geo_code'.")

    # If the GEO metadata's values are mapped but none of its columns
    # are kept
    elif "mapping_geo" in config \
    and "metadata_to_keep_geo" not in config:

        # Add the warning.
        warnings.append(\
            f"{path}: 'mapping_geo' is ignored without " \
            "'metadata_to_keep_geo'.")

    #-----------------------------------------------------------------#

    # Get the warnings about the mappings and the columns to add.
    warnings.extend(\
        _get_options_warnings(config = config,
                              path = path))

    #-----------------------------------------------------------------#

    # Return the errors and the warnings.
    return errors, warnings


########################## PUBLIC FUNCTIONS ###########################


def lowercase_capitalized_words(text: str) -> str:
    """Lowercase a text's capitalized words, keeping all-caps words.

    Parameters
    ----------
    text : :class:`str`
        The text.

    Returns
    -------
    text : :class:`str`
        The converted text.
    """

    # Split the text at underscores, spaces and hyphens, keeping them.
    parts = re.split(r"(_|\s|-)",
                     text)

    # Initialize the list of converted parts.
    converted = []

    # For each part
    for part in parts:

        # If it is a delimiter, an all-caps word or an all-caps word
        # ending in a lowercase letter
        if part in ("_", " ", "-") \
        or (part.isupper() and len(part) > 1) \
        or (part[:-1].isupper() and part[-1:].islower()):

            # Keep it.
            converted.append(part)

        # If it is a capitalized word
        elif part and part[0].isupper():

            # Lowercase it.
            converted.append(part.lower())

        # Otherwise
        else:

            # Keep it.
            converted.append(part)

    # Return the converted text.
    return "".join(converted)


def get_separator_name(sep: str) -> str:
    """Get the name a separator gives to the text following it.

    Parameters
    ----------
    sep : :class:`str`
        The separator.

    Returns
    -------
    name : :class:`str`
        The separator's name: the separator without a leading digit,
        with its capitalized words lowercased.
    """

    # Return the name.
    return lowercase_capitalized_words(\
        sep[1:] if sep[:1].isdigit() else sep)


def parse_config_recount3(config: object,
                          project_name: str) -> \
        tuple[dict[str, object],
              list[str],
              list[str]]:
    """Parse and check the configuration of a dataset built from
    Recount3 studies.

    Parameters
    ----------
    config : :class:`dict`
        The configuration.

    project_name : :class:`str`, {``"GTEx"``, ``"TCGA"``, ``"SRA"``}
        The Recount3 project the studies belong to.

    Returns
    -------
    config : :class:`dict`
        The checked configuration.

    errors : :class:`list`
        The errors found.

    warnings : :class:`list`
        The warnings.
    """

    # Initialize the list of errors.
    errors = []

    # Initialize the list of warnings.
    warnings = []

    #-----------------------------------------------------------------#

    # If the project is not supported
    if project_name not in defaults.RECOUNT3_PROJECTS:

        # Get the supported projects as a string.
        projects_str = \
            ", ".join(repr(p) for p in defaults.RECOUNT3_PROJECTS)

        # Add the error.
        errors.append(\
            f"Unknown project '{project_name}'. The supported " \
            f"projects are: {projects_str}.")

        # Return the configuration, the errors and the warnings.
        return config, errors, warnings

    # If the configuration is not a dictionary
    if not isinstance(config, dict):

        # Add the error.
        errors.append(\
            "The configuration must be a dictionary of options." \
            if project_name in _PROJECTS_TEMPLATES else \
            "The configuration must be a dictionary mapping the " \
            "SRA studies' codes to their options.")

        # Return the configuration, the errors and the warnings.
        return config, errors, warnings

    # Create a deep copy of the configuration.
    config = copy.deepcopy(config)

    #-----------------------------------------------------------------#

    # If the configuration is a single set of options
    if project_name in _PROJECTS_TEMPLATES:

        # Check it against the template.
        errors.extend(\
            _check_section(\
                config = config,
                template = _PROJECTS_TEMPLATES[project_name]))

        # Get the warnings about the mappings and the columns to add.
        warnings.extend(\
            _get_options_warnings(config = config,
                                  path = ""))

        # If it is a TCGA configuration
        if project_name == "TCGA":

            # Check the options the template cannot check.
            errors_tcga, warnings_tcga = \
                _check_config_tcga(config = config)

            # Add the errors.
            errors.extend(errors_tcga)

            # Add the warnings.
            warnings.extend(warnings_tcga)

    #-----------------------------------------------------------------#

    # If the configuration maps the SRA studies to their options
    else:

        # If there are no studies
        if not config:

            # Add the error.
            errors.append("The configuration has no studies.")

        # For each study and its options
        for sra_code, options in config.items():

            # Set the study's path.
            path = str(sra_code)

            # Check the study's options against the template.
            errors.extend(\
                _check_value(value = options,
                             spec = {"type" : (dict,),
                                     "section" : \
                                        _templates.CONFIG_SRA_STUDY},
                             path = path))

            # If the study's options are a dictionary
            if isinstance(options, dict):

                # Check the options the template cannot check.
                errors_sra, warnings_sra = \
                    _check_config_sra_study(config = options,
                                            path = path)

                # Add the errors.
                errors.extend(errors_sra)

                # Add the warnings.
                warnings.extend(warnings_sra)

    #-----------------------------------------------------------------#

    # Return the configuration, the errors and the warnings.
    return config, errors, warnings
