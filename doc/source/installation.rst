Installing bulkdgd
==================

.. toctree::
   :maxdepth: 1
   :hidden:

   installation_venv
   installation_conda


bulkdgd is a Python package requiring **Python 3.11 or higher** and several open-source Python packages, and is published on `PyPI <https://pypi.org/project/bulkdgd/>`_.

Please ensure that you have Python installed before proceeding with the installation.

The required Python packages will be installed automatically during the installation process.

The bulkdgd package has been tested only on Unix-based systems.

We will show the installation using Python 3.11, but the same steps remain valid for later versions.

In short, once you have activated the Python environment of your choice, installing bulkdgd amounts to running:

.. code-block:: shell

   pip install bulkdgd

The sections below walk through this in more detail, for two common ways of managing Python environments.

.. important::

   The trained decoder's parameters (``dec.pth``) are too large to be distributed with the package on PyPI. Instead, ``bulkdgd`` downloads this file automatically the first time you use the pre-trained model (for instance, when running :doc:`bulkdgd_find_representations <command_line_interface>` for the first time) - no manual step is needed. This is a one-off download of about 900 MB; subsequent runs reuse the downloaded file. If the machine you are running ``bulkdgd`` on has no internet access, see the note on manual download in the installation methods below.

Here, we provide instructions for installing bulkdgd in:

*  A simple :doc:`"virtualenv" environment <installation_venv>`.
*  An :doc:`Anaconda ("conda") environment <installation_conda>` .
