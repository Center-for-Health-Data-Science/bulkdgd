Installing with ``conda``
=========================

Step 1 - Install ``conda``
--------------------------

Go `to the Miniconda documentation <https://docs.conda.io/en/latest/miniconda.html>`_ for detailed instructions on how to install ``conda``.

Installing ``miniconda`` rather than the full ``anaconda`` package is advised.

Once ``conda`` is installed on your system, you can create a virtual environment.

Step 2 - Get bulkDGD
--------------------

Download the latest version of bulkDGD from `its GitHub repository <https://github.com/Center-for-Health-Data-Science/bulkDGD/releases/latest>`_.

Step 3 - Create the ``conda`` environment
-----------------------------------------

You can create your ``conda`` environment:

.. code-block:: shell
    
    conda create -n bulkdgd-env python=3.11

Step 4 - Activate the environment
---------------------------------

You can activate the ``conda`` environment by running:

.. code-block:: shell
    
    conda activate bulkdgd-env

Step 5 - Get the ``dec.pth`` file
---------------------------------

You must download the ``dec.pth`` file containing the trained decoder's parameters before installing bulkDGD, so that the file is copied to the installation directory. The file cannot be shipped together with the GitHub package because of its size, but can be downloaded `here <https://drive.google.com/file/d/1GKMkVmmcEH8glNrQ4092VWYQgq6maYW1/view?usp=sharing>`_.

Once downloaded, place the file into the ``bulkDGD/ioutil/data`` folder before performing the installation.

Step 6 - Install bulkDGD
------------------------

You can now install bulkDGD using ``pip``.

.. code-block:: shell
    
    pip install ./bulkDGD

bulkDGD should now be installed.

Every time you need to run bulkDGD after opening a new shell, just run step 4 beforehand.
