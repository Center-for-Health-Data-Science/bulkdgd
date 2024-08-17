#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-

#    util.py
#
#    General utilities.
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
__doc__ = "General utilities."


#######################################################################


# Import from the standard library.
import copy
import os
import re
# Import from bulkDGD.
from . import defaults


#######################################################################


def uniquify_file_path(file_path):
    """If ``file_path`` exists, number it uniquely.

    Parameters
    ----------
    file_path : :class:`str`
        The file path.

    Returns
    -------
    unique_file_path : :class:`str`
        A unique file path generated from the original file path.
    """
    
    # Get the file's name and extension.
    file_name, file_ext = os.path.splitext(file_path)

    # Set the counter to 1.
    counter = 1

    # If the file already exists
    while os.path.exists(file_path):

        # Set the path to the new unique file.
        file_path = file_name + "_" + str(counter) + file_ext

        # Update the counter.
        counter += 1

    # Return the new path.
    return file_path


def check_config_against_template(config,
                                  template,
                                  ignore_if_missing,
                                  ignore_if_varying):
    """Check a configuration against the configuration's template.

    Parameters
    ----------
    config : :class:`dict`
        The configuration.

    template : :class:`dict`
        A template describing how the configuration should be
        structured.

    ignore_if_missing : :class:`set`
        A set containing the fields that can be missing from the
        configuration.

    ignore_if_varying : :class:`set`
        A set containing the fields that can vary in the
        configuration.

    Returns
    -------
    config : :class:`dict`
        The configuration loaded from the file provided by
        the user, if all checks were successful.
    """

    # Define the recursion.
    def recurse(config,
                template,
                key):

        # If both the current configuration dictionary and the template
        # dictionary are dictionaries (they are either the full
        # dictionaries or sub-dictionaries of the original ones in the
        # recursive calls)
        if (isinstance(config, dict) \
        and isinstance(template, dict)):

            # Get the fields (= keys) in the configuration.
            config_fields = set(config.keys())
            
            # Get the fields (= keys) in the template.
            template_fields = set(template.keys())
            
            # Get the fields (= keys) unique to the configuration
            # (= not found in the template).
            unique_config_fields = \
                config_fields - template_fields

            # Get the fields (= keys) unique to the template
            # (= not found in the configuration).
            unique_template_fields = \
                template_fields - config_fields

            #---------------------------------------------------------#

            # If any unique field was found in the configuration
            if len(unique_config_fields) != 0:

                # Raise an exception.
                fields = \
                    ", ".join([f"'{f}'" for f \
                               in unique_config_fields])
                errstr = \
                    f"Unrecognized field(s) in configuration: " \
                    f"{fields}."
                raise KeyError(errstr)

            #---------------------------------------------------------#

            # If any field was found in the template but not in the
            # configuration.
            if len(unique_template_fields) != 0:

                # Get the fields that need to be present.
                needed_fields = \
                    ignore_if_missing.union(unique_template_fields) \
                    - ignore_if_missing

                # If some fields that are needed are missing.
                if needed_fields:

                    # Raise an exception.
                    fields = \
                        ", ".join([f"'{f}'" for f in needed_fields])
                    errstr = \
                        f"Missing field(s) in configuration: " \
                        f"{fields}."
                    raise KeyError(errstr)

            #---------------------------------------------------------#

            # For each key, value pair in the configuration
            for k, val_config in config.items():

                # If the element corresponds to a dictionary of
                # options that can vary
                if k in ignore_if_varying:

                    # Just ignore it.
                    continue

                # Get the corresponding value in the template
                # (we are sure that there will be a value for
                # the given key, because we checked the equality
                # of all keys between configuration and template
                # earlier).
                val_template = template[k]

                # Recursively check the values.
                recurse(config = val_config,
                        template = val_template,
                        key = k)

        #-------------------------------------------------------------#

        # If the user-supplied configuration is a dictionary, but the
        # template is not
        elif (isinstance(config, dict) \
        and not isinstance(template, dict)) and template is not None:

            # Raise an exception.
            errstr = \
                f"The configuration contains sub-fields " \
                f"for the field '{str(template)}', which is " \
                f"not supposed to have sub-fields."
            raise TypeError(errstr)

        #-------------------------------------------------------------#

        # If the template is a dictionary, but the configuration is
        # not
        elif (not isinstance(config, dict) \
        and isinstance(template, dict)):

            # Raise an exception.
            errstr = \
                f"The configuration does not contain " \
                f"sub-fields for the field '{str(config)}', " \
                f"while it is supposed to have the following " \
                f"sub-fields: {str(template)}."
            raise TypeError(errstr)

        #-------------------------------------------------------------#

        # If both the configuration and the template are not
        # dictionaries (we are in a "leaf" value, not a key of a
        # a nested dictionary)
        elif (not isinstance(config, dict) \
        and not isinstance(template, dict)):

            # If the type of value found in the configuration does not
            # match the type set in the template
            if not isinstance(config, template):

                # Raise an exception.
                errstr = \
                    f"'{key}' must be of type '{str(template)}', "\
                    f"not {str(type(config))} ('{config}')."
                raise TypeError(errstr)

        #-------------------------------------------------------------#

        # Return the dictionary.
        return config

    #-----------------------------------------------------------------#

    # Recurse and return the result.
    return recurse(config = config,
                   template = template,
                   key = None)


def recursive_map_dict(d,
                       func,
                       keys = None):
    """Recursively traverse a dictionary mapping a function to the
    dictionary's leaf values (the function substitutes the values with
    the return value of the function applied to those values).

    Parameters
    ----------
    d : :class:`dict`
        The input dictionary.

    func : any callable
        A callable taking as keyword arguments the values of a
        dictionary and returning a single value.

    keys : :class:`list` or :class:`set`, optional
        A list of specific keys on whose items the mapping should be
        performed.

        This means that all values associated with keys different
        from those in the list will not be affected.

        If :const:`None`, all keys and associated values will be
        considered.
    
    Returns
    -------
    new_d : :class:`dict`
        The new dictionary.
    """

    # Define the recursion.
    def recurse(d,
                func,
                keys):

        # If the current object is a dictionary
        if isinstance(d, dict):
            
            # Get the keys of the items on which the mapping will be
            # performed. If no keys are passed, all keys in the
            # dictionary will be considered.
            sel_keys = keys if keys else d.keys()

            # For each key, value pair in the dictionary
            for k, v in list(d.items()):

                # If the value is a dictionary
                if isinstance(v, dict):

                    # If the key is in the selected keys
                    if k in sel_keys:

                        # Substitute the value with the return value
                        # of 'func' applied to it.
                        d[k] = func(**v)
                    
                    # Otherwise
                    else:

                        # Recursively check the sub-dictionaries
                        # in the current dictionary.
                        recurse(d = v,
                                func = func,
                                keys = sel_keys)

    #-----------------------------------------------------------------#

    # Create a copy of the input dictionary.
    new_d = copy.deepcopy(d)

    #-----------------------------------------------------------------#

    # Recurse through the new dictionary.
    recurse(d = new_d,
            func = func,
            keys = keys)

    #-----------------------------------------------------------------#

    # Return the new dictionary.
    return new_d


def recursive_add(d,
                  d2,
                  keys):
    """Recursively add all elements from a dictionary to another
    dictionary in specific places.

    Parameters
    ----------
    d : :class:`dict`
        The input dictionary.

    d2 : :class:`dict`
        The dictionary whose elements should be added to the input
        dictionary.

    keys : :class:`list` or :class:`set` or :class:`tuple`
        The keys corresponding to the places where the key, value
        pairs contained in ``d2`` will be added to the input
        dictionary.

    Returns
    -------
    new_d : :class:`dict`
        The updated dictionary.
    """

    # Define the recursion.
    def recurse(d,
                d2,
                keys):

        # If first dictionary is in fact a dictionary
        if isinstance(d, dict):

            # For each key in the first dictionary
            for key in d:
                
                # If the key is among the selected keys
                if key in keys:
                    
                    # If the associated value is a dictionary and the
                    # second dictionary is in fact a dictionary
                    if isinstance(d[key], dict) \
                    and isinstance(d2, dict):

                        # For each key, value pair in the second
                        # dictionary
                        for k, v in d2.items():

                            # If the key is not among the keys in the
                            # value associated with 'key' in the
                            # fist dictionary
                            if k not in d[key]:

                                # Add the key and associated value to
                                # the first dictionary.
                                d[key][k] = v
                    

                # If the value associated with they key is a dictionary
                if isinstance(d[key], dict):

                    # Recurse through the dictionary.
                    recurse(d = d[key],
                            d2 = d2,
                            keys = keys)

    #-----------------------------------------------------------------#

    # Create a copy of the input dictionary.
    new_d = copy.deepcopy(d)

    #-----------------------------------------------------------------#

    # Recurse through the copy of the dictionary.
    recurse(d = new_d,
            d2 = d2,
            keys = keys)

    #-----------------------------------------------------------------#

    # Return the modified dictionary.
    return new_d


def recursive_merge_dicts(*dicts):
    """Recursively merge multiple dictionaries.

    Parameters
    ----------
    *dicts : multiple :class:`dict`
        The dictionaries to be merged.

    Returns
    -------
    merged : :class:`dict`
        A dictionary representing the result of the merging.
    """

    # Define a recursive function to merge two dictionaries at a time.
    def merge_two_dicts(d1, d2):

        # For each key, value pair in the second dictionary
        for k, v in d2.items():

            # If the key is in the first dictionary and the associated
            # value in both dictionaries is another dictionary
            if k in d1 \
            and isinstance(d1[k], dict) and isinstance(v, dict):

                # Recursively merge the associated values.
                d1[k] = merge_two_dicts(d1[k], v)

            # Otherwise
            else:

                # The value associated to the key in the first
                # dictionary will be the value associated to the key
                # in the second dictionary.
                d1[k] = v

        # Return the updated first dictionary.
        return d1

    #-----------------------------------------------------------------#

    # Initialize an empty dictionary to store the result of the
    # merging.
    merged = {}

    #-----------------------------------------------------------------#

    # For each provided dictionary
    for d in dicts:

        # Recursively merge it with the current result.
        merged = merge_two_dicts(merged, d)

    #-----------------------------------------------------------------#

    # Return the result of the merging.
    return merged


def recursive_get(d,
                  key_path):
    """Recursively get an item from a nested dictionary given the
    item's "key path".

    Parameters
    ----------
    d : :class:`dict`
        The input dictionary.

    key_path : :class:`list`
        The "key path" leading to the item of interest.

    Returns
    -------
    item : any object
        The item of interest.
    """

    # Define the recursion.
    def recurse(d,
                key_path):

        # If the key  path is empty
        if not key_path:

            # Return None
            return None
        
        # Get the first key in the path.
        key = key_path[0]

        # If the key is in the dictionary
        if key in d:

            # If this is the last key in the path
            if len(key_path) == 1:

                # Return the value.
                return d[key]

            # Otherwise, if the value associated with the current key is
            # a dictionary
            elif isinstance(d[key], dict):

                # Continue the recursion with the next level of the
                # dictionary.
                return recurse(d = d[key],
                               key_path = key_path[1:])

        # If the key is not found, return None.
        return None

    #-----------------------------------------------------------------#

    # Return the result of the recursion.
    return recurse(d = d,
                   key_path = key_path)



#######################################################################


def load_list(list_file):
    """Load a list of newline-separated entities from a plain text
    file.

    Parameters
    ----------
    list_file : :class:`str`
        The plain text file containing the entities of interest.

    Returns
    -------
    list_entities : :class:`list`
        The list of entities.
    """

    # Return the list of entities from the file (exclude blank
    # and comment lines).
    return \
        [line.rstrip("\n") for line in open(list_file, "r") \
         if (not line.startswith("#") \
             and not re.match(r"^\s*$", line))]


#######################################################################


def check_config_model(config):
    """Check the model's configuration.

    Parameters
    ----------
    config : :class:`dict`
        The configuration.

    Returns
    -------
    config : :class:`dict`
        The configuration.
    """

    # Set the fields that can be missing from the configuration,
    # so that they do not raise an exception if they are not found.
    ignore_if_missing = {"gmm_pth_file", "dec_pth_file"}

    # Set the fields that can vary in the configuration, so that they
    # do not raise an exception if they are different than expected.
    ignore_if_varying = {"gmm_options", "dec_options"}

    #-----------------------------------------------------------------#

    # Check the configuration against the template.
    config = check_config_against_template(\
                config = config,
                template = defaults.CONFIG_MODEL_TEMPLATE,
                ignore_if_missing = ignore_if_missing,
                ignore_if_varying = ignore_if_varying)

    #-----------------------------------------------------------------#

    # Return the configuration.
    return config


def check_config_rep(config):
    """Check the configuration containing the options for the
    optimization round(s) to find the best representations for a
    set of samples.

    Parameters
    ----------
    config : :class:`dict`
        The configuration.

    Returns
    -------
    config : :class:`dict`
        The configuration.
    """

    # Set the names of the supported optimization schemes.
    optimization_schemes = ["one_opt", "two_opt"]

    #-----------------------------------------------------------------#

    # Get the optimization scheme from the configuration.
    opt_scheme = config.get("scheme")

    # If no scheme was defined
    if not opt_scheme:

        # Raise an error.
        errstr = \
            "The configuration must include a 'scheme' key " \
            "associated with the name of the optimization " \
            "scheme the configuration refers to."
        raise KeyError(errstr)

    # If the optimization scheme is not supported
    if opt_scheme not in optimization_schemes:

        # Raise an error.
        supported_schemes = \
            ", ".join([f"'{s}'" for s in optimization_schemes])
        errstr = \
            f"Unsupported optimization scheme '{opt_scheme}'. " \
            "Supported optimization schemes are: " \
            f"'{supported_schemes}."
        raise ValueError(errstr)

    #-----------------------------------------------------------------#

    # Check the configuration against the template.
    config = check_config_against_template(\
                config = config,
                template = defaults.CONFIG_REP_TEMPLATE[opt_scheme],
                ignore_if_missing = set(),
                ignore_if_varying = set())

    #-----------------------------------------------------------------#

    # Return the configuration.
    return config


def check_config_train(config):
    """Check the configuration containing the options for training the
    model.

    Parameters
    ----------
    config : :class:`dict`
        The configuration.

    Returns
    -------
    config : :class:`dict`
        The configuration.
    """

    # Check the configuration against the template.
    config = check_config_against_template(\
                config = config,
                template = defaults.CONFIG_TRAIN_TEMPLATE,
                ignore_if_missing = set(),
                ignore_if_varying = set())

    #-----------------------------------------------------------------#

    # Return the configuration.
    return config


def check_config_plot(config):
    """Check the configuration for a plot.

    Parameters
    ----------
    config : :class:`dict`
        The configuration.

    Returns
    -------
    config : :class:`dict`
        The configuration.
    """

    # Return the configuration.
    return config


def check_config_genes(config):
    """Check the configuration for creating a new list of genes.

    Parameters
    ----------
    config : :class:`dict`
        The configuration.

    Returns
    -------
    config : :class:`dict`
        The configuration.
    """

    # Return the configuration.
    return config


def check_config_dim_red(config):
    """Check the configuration for performing a dimensionality
    reduction analysis.

    Parameters
    ----------
    config : :class:`dict`
        The configuration.

    Returns
    -------
    config : :class:`dict`
        The configuration.
    """

    # Return the configuration.
    return config
