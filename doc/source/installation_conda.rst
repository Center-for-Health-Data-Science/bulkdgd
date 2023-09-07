Installing with ``conda``
=========================

Step 1 - Install ``conda``
--------------------------

Go `here <https://docs.conda.io/en/latest/miniconda.html>` for detailed instructions on how to install ``conda``.

Installing ``miniconda`` rather than the full ``anaconda`` package is advised.

We have provided an ``environment.yaml`` file that can be used together with ``conda`` to automatically create an environment containing all required Python packages.

Once ``conda`` is installed on your system, you can create a virtual environment.

Step 2 - Create the ``conda`` environment
-----------------------------------------

You can create your ``conda`` environment from the provided environment file:

.. code-block:: shell
    
    conda env create --file bulkDGD/environment.yaml --prefix ./bulkDGD-env

In this case, we ask ``conda`` to create the environment locally (``--prefix``), which is optional.

If your home directory is size-constrained, the installation might fail for lack of space.

In that case, select another directory with more space available in which conda will download packages:

.. code-block:: shell
    
    conda config --add pkgs_dirs /my/other/directory/conda/pkgs

Step 3 - Activate the environment
---------------------------------

You can activate the `conda` environment by running the command line that ``conda`` suggests at the end of the previous step.

It is usually something like this:

.. code-block:: shell
    
    conda activate ./bulkDGD-env

Step 4 - Get bulkDGD
------------------------

Clone the bulkDGD source code from its GitHub repository within a directory of your choice and enter the local copy of the repository.

.. code-block:: shell

    git clone https://github.com/Center-for-Health-Data-Science/bulkDGD.git
    cd bulkDGD

If the ``git`` command is unavailable, you can download the repository content as a ZIP file from the bulkDGD GitHub repository web page, unzip it, and enter it.

Step 5 - Install bulkDGD
----------------------------

You can now install bulkDGD:

.. code-block:: shell

    python setup.py install

bulkDGD should now be installed.

Every time you need to run bulkDGD after opening a new shell, just run step 3 beforehand.