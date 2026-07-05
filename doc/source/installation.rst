Installing bulkdgd
==================

.. toctree::
   :maxdepth: 1
   :hidden:

   installation_venv
   installation_conda


bulkdgd is a Python package requiring **Python 3.11 or higher** and several open-source Python packages.

Please ensure that you have Python installed before proceeding with the installation.

The required Python packages will be installed automatically during the installation process.

The bulkdgd package has been tested only on Unix-based systems.

We will show the installation using Python 3.11, but the same steps remain valid for later versions.

.. important::

   Before installing, you will need to separately download the trained decoder's parameters (``dec.pth``), since the file is too large to be hosted on GitHub. Both installation methods below include the download link and tell you where to place the file - make sure to do this **before** running the installation command, so that the file is picked up and included in the installed package.

Here, we provide instructions for installing bulkdgd in:

*  A simple :doc:`"virtualenv" environment <installation_venv>`.
*  An :doc:`Anaconda ("conda") environment <installation_conda>` .
