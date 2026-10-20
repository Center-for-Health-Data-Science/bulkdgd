Installing bulkdgd
==================

.. toctree::
   :maxdepth: 1
   :hidden:

   installation_venv
   installation_conda


bulkdgd is a Python package requiring **Python 3.10 or higher** and several open-source Python packages, and is published on `PyPI <https://pypi.org/project/bulkdgd/>`_.

Make sure Python is installed before you start.

The required Python packages are installed automatically.

The bulkdgd package has been tested only on Unix-based systems.

The instructions use Python 3.11. The same steps apply to the other supported versions.

Once you have activated a Python environment, install bulkdgd by running:

.. code-block:: shell

   pip install bulkdgd

The sections below describe this in more detail for two ways of managing Python environments.

.. important::

   The trained decoders' parameters are too large to be distributed with the package on PyPI: there is one decoder per member of the shipped ensemble, each 1.79 GiB in ``float64``. Instead, ``bulkdgd`` downloads the one it needs the first time you use that member (for instance, when running :doc:`bulkdgd_find_representations <command_line_interface>` for the first time). No manual step is needed. Each member is downloaded once, and later runs reuse the file. If the machine you are running ``bulkdgd`` on has no internet access, see the note on manual download in the installation methods below.

Instructions are provided for installing bulkdgd in:

*  A :doc:`"virtualenv" environment <installation_venv>`.
*  An :doc:`Anaconda ("conda") environment <installation_conda>`.
