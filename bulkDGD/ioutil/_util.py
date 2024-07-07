#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-

#    _util.py
#
#    Private utilities to load and save files.   
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
__doc__ = "Private utilities to load and save files."


#######################################################################


# Import from the standard library.
import copy
import logging as log


#######################################################################


# Get the module's logger.
logger = log.getLogger(__name__)


#######################################################################


def check_config_against_template(config,
                                  template,
                                  ignore_if_missing,
                                  ignore_if_varying):
    """Check a configuration against the configuration's template.

    Parameters
    ----------
    config : ``dict``
        The configuration.

    template : ``dict``
        A template describing how the configuration should be
        structured.

    ignore_if_missing : ``set``
        A set containing the fields that can be missing from the
        configuration.

    ignore_if_varying : ``set``
        A set containing the fields that can vary in the
        configuration.

    Returns
    -------
    config : ``dict``
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
    d : ``dict``
        The input dictionary.

    func : any callable
        A callable taking as keyword arguments the values of a
        dictionary and returning a single value.

    keys : ``list``, ``set``, optional
        A list of specific keys on whose items the mapping should be
        performed.

        This means that all values associated with keys different
        from those in the list will not be affected.

        If ``None``, all keys and associated values will be considered.
    
    Returns
    -------
    new_d : ``dict``
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
