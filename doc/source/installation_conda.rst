Installing with ``conda``
=========================

Step 1 - Install ``conda``
--------------------------

Go `here <https://docs.conda.io/en/latest/miniconda.html>`_ for detailed instructions on how to install ``conda``.

Installing ``miniconda`` rather than the full ``anaconda`` package is advised.

Once ``conda`` is installed on your system, you can create a virtual environment.

Step 2 - Get bulkDGD
--------------------

Clone the bulkDGD source code from its GitHub repository within a directory of your choice and enter the local copy of the repository.

.. code-block:: shell

    git clone https://github.com/Center-for-Health-Data-Science/bulkDGD.git

If the ``git`` command is unavailable, you can download the repository content as a ZIP file from the bulkDGD GitHub repository web page and uzip it.

Step 3 - Create the ``conda`` environment
-----------------------------------------

You can create your ``conda`` environment from the provided environment file:

.. code-block:: shell
    
    conda env create --prefix ./bulkdgd-env python=3.11

In this case, we ask ``conda`` to create the environment locally (``--prefix``), which is optional.

If your home directory is size-constrained, the installation might fail for lack of space.

In that case, select another directory with more space available in which conda will download packages:

.. code-block:: shell
    
    conda config --add pkgs_dirs /my/other/directory/conda/pkgs

Step 4 - Activate the environment
---------------------------------

You can activate the ``conda`` environment by running the command line that ``conda`` suggests at the end of the previous step.

It is usually something like this:

.. code-block:: shell
    
    conda activate ./bulkdgd-env

Step 5 - Get the ``dec.pth`` file
---------------------------------

You must download the ``dec.pth`` file containing the trained decoder's parameters before installing bulkDGD, so that the file is copied to the installation directory. The file cannot be shipped together with the GitHub package because of its size, but can be downloaded `here <https://drive.google.com/file/d/1SZaoazkvqZ6DBF-adMQ3KRcy4Itxsz77/view?usp=sharing>`_.

Once downloaded, place the file into the ``bulkDGD/ioutil/data`` folder before performing the installation.

Step 6 - Install bulkDGD
------------------------

You can now install bulkDGD using ``pip``.

.. code-block:: shell
    
    pip install ./bulkDGD

bulkDGD should now be installed.

Every time you need to run bulkDGD after opening a new shell, just run step 4 beforehand.
