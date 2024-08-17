# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values for Sphinx, see
# the documentation here:
#
# https://www.sphinx-doc.org/en/master/usage/configuration.html.


#######################################################################


# Import from the standard library
import os
import sys


#######################################################################


# Get the absolute path to the package.
path = os.path.abspath("../../bulkDGD")

# Insert the path into the PATH.
sys.path.insert(0, path)

#---------------------------------------------------------------------#

# https://www.sphinx-doc.org/en/master/usage/configuration.html
# #project-information

# Set the name of the project.
project = "bulkDGD"

# Set the copyright of the project.
copyright = "2024, Valentina Sora"

# Set the name(s) of the project's author(s).
author = \
   "Valentina Sora, Viktoria Schuster, Inigo Prada Luengo, " \
   "Anders Lykkebo-Valløe, Andreas Bjerregaard,, Anders Krogh"

#---------------------------------------------------------------------#

# https://www.sphinx-doc.org/en/master/usage/configuration.html
# #general-configuration

# Set a list of strings that are module names of extensions needed:
#
# - 'myst_parser' is needed to parse Markdown files.
#   Install it with 'pip install myst-parser'.
#
# - 'sphinx.ext.autodoc' is needed to automatically generate
#   documentation from source code files, and it is installed
#   together with Sphinx.
#
# - 'numpydoc' is needed to parse NumPy-style docstrings.
#   Install it with 'pip install numpydoc'.
#
# - 'sphinx_design' is needed for some graphical features.
#   Install it with 'pip install sphinx_design'.
#
# - 'sphinxcontrib.bibtex' is needed for the bibliography.
#   Install it with 'pip install sphinxcontrib-bibtex'.
#
# - 'sphinx.ext.intersphinx' is needed for cross-referencing with the
#   documentations of other packages and is installed together with
#   Sphinx.
extensions = \
   ["myst_parser",
    "sphinx.ext.autodoc",
    "numpydoc",
    "sphinx_design",
    "sphinxcontrib.bibtex",
    "sphinx.ext.intersphinx"]

# Set the mapping so that intersphinx can work out the cross-references
# to other packages.
intersphinx_mapping = \
   {"python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "seaborn": ("https://seaborn.pydata.org/", None)}

# Set the file extensions of source files. Sphinx considers the files
# with this suffix as sources. The value can be a dictionary mapping
# file extensions to file types.
source_suffix = \
   {".rst" : "restructuredtext",
    ".txt" : "restructuredtext",
    ".md" : "markdown"}

# Set a list of paths that contain extra templates (or templates that
# overwrite builtin/theme-specific templates). Relative paths are
# taken as relative to the configuration directory.
# As these files are not meant to be built, they are automatically
# added to exclude_patterns.
templates_path = ["_templates"]

#---------------------------------------------------------------------#

# https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html

# Set how to order the functions/classes in the modules' documentation:
#
# - 'alphabetical' sorts the functions/classes alphabetically.
#
# - 'groupwise' sorts the functions/classes by member type.
#
# - 'bysource' sorts the functions/classes by source order.
autodoc_member_order = "groupwise"

#---------------------------------------------------------------------#

# https://numpydoc.readthedocs.io/en/latest/install.html

# Set whether to suppress showing all class members by default
numpydoc_show_class_members = False

#---------------------------------------------------------------------#

# https://www.sphinx-doc.org/en/master/usage/extensions/
# autosummary.html

# Set whether to overwrite .rst files when creating autosummaries.
autosummary_generate_overwrite = False

#---------------------------------------------------------------------#

# Set a list of .bib files to be used for the bibliography.
bibtex_bibfiles = ["./bib_files/references.bib"]

#---------------------------------------------------------------------#

# https://myst-parser.readthedocs.io/en/latest/configuration.html

# Set which 'myst-parser' extensions should be enabled.
myst_enable_extensions = ["amsmath", "dollarmath"]

# Set the lowest level at which to create automatic heading anchors.
myst_heading_anchors = 3

# Set whether to show the warning regarding having the document ending
# with footnotes preceded by a heading.
myst_footnote_transition = False

#---------------------------------------------------------------------#

# https://www.sphinx-doc.org/en/master/usage/configuration.html
# #options-for-html-output

# Set a list of paths that contain custom static files (such as style
# sheets or script files).
# 
# Relative paths are taken as relative to the directory where the
# configuration file is. They are copied to the output’s
# '_static' directory after the theme’s static files. Therefore, a file
# named 'default.css' will overwrite the theme’s 'default.css'.
html_static_path = ["_static"]

# Set the HTML theme to be used.
# - If 'sphinx_rtd_theme', install it first by running
#   'pip install sphinx-rtd-theme'.
# - If 'pydata_sphinx_theme', install it first by running
#   'pip install pydata_sphinx_theme'.
html_theme = "sphinx_rtd_theme"

# Set a dictionary of values to pass into the template engine’s context
# for all pages.
html_context = { 
   # ...
   "default_mode": "light"
}
