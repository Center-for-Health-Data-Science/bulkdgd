# Configuration file for the Sphinx documentation builder.
# For the full list of built-in configuration values, see the
# documentation: https://www.sphinx-doc.org/en/master/usage/
# configuration.html.


# Standard library
import os
import sys


#------------------------ Project information ------------------------#


# Get the absolute path to the package
path = os.path.abspath("../../BulkDGD")

# Insert it into the PATH
sys.path.insert(0, path)


# https://www.sphinx-doc.org/en/master/usage/configuration.html
# #project-information


# Name of the project
project = "BulkDGD"

# Copyright of the project
copyright = "2023, Valentina Sora"

# Name of the project's author(s)
author = "Valentina Sora"


#----------------------- General configuration -----------------------#


# https://www.sphinx-doc.org/en/master/usage/configuration.html
# #general-configuration

# A list of strings that are module names of extensions:
# - 'sphinx.ext.autodoc' is needed to automatically generate
#   documentation from source code files, and it is installed
#   together with Sphinx.
# - 'numpydoc' is needed to parse NumPy-style docstrings, and it
#   needs to be installed separately (it can be done with
#   'pip install numpydoc').
# - 'myst_parser' is needed to parse Markwodn files, and it needs
#   to be installed separately (it can be done with 'pip install
#   myst-parser').
extensions = \
   ["sphinx.ext.autodoc",
    "numpydoc",
    "myst_parser"]


autodoc_member_order = "groupwise"
numpydoc_show_class_members = False 

# The file extensions of source files. Sphinx considers the files
# with this suffix as sources. The value can be a dictionary mapping
# file extensions to file types.
source_suffix = \
	{".rst" : "restructuredtext",
	 ".txt" : "restructuredtext",
	 ".md" : "markdown"}

# A list of paths that contain extra templates (or templates that
# overwrite builtin/theme-specific templates). Relative paths are
# taken as relative to the configuration directory.
# As these files are not meant to be built, they are automatically
# added to exclude_patterns.
templates_path = ["_templates"]


#--------------------- Markdown parsing options ----------------------#


# Which 'myst-parser' extensions should be anabled
myst_enable_extensions = ["amsmath", "dollarmath"]


#---------------------- Options for HTML output ----------------------#


# https://www.sphinx-doc.org/en/master/usage/configuration.html
# #options-for-html-output

# A list of paths that contain custom static files (such as style
# sheets or script files). Relative paths are taken as relative to
# the configuration directory. They are copied to the output’s
# _static directory after the theme’s static files, so a file
# named default.css will overwrite the theme’s default.css.
html_static_path = ["_static"]

# HTML theme to be used.
# NOTE: 'pydata_sphinx_theme' needs to be installed separately.
# This can be done with 'pip install pydata-sphinx-theme'
html_theme = "pydata_sphinx_theme"

# A dictionary of values to pass into the template engine’s context
# for all pages
html_context = { 
   # ...
   "default_mode": "light"
}