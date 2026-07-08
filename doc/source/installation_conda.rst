Installing with ``conda``
=========================

Step 1 - Install ``conda``
--------------------------

Go `to the Miniconda documentation <https://docs.conda.io/en/latest/miniconda.html>`_ for detailed instructions on how to install ``conda``.

Installing ``miniconda`` rather than the full ``anaconda`` package is advised.

Once ``conda`` is installed on your system, you can create a virtual environment.

Step 2 - Create the ``conda`` environment
-----------------------------------------

You can create your ``conda`` environment:

.. code-block:: shell

    conda create -n bulkdgd-env python=3.11

Step 3 - Activate the environment
---------------------------------

You can activate the ``conda`` environment by running:

.. code-block:: shell

    conda activate bulkdgd-env

Step 4 - Install bulkdgd
------------------------

You can now install ``bulkdgd`` from `PyPI <https://pypi.org/project/bulkdgd/>`_ using ``pip``:

.. code-block:: shell

    pip install bulkdgd

``bulkdgd`` should now be installed.

The trained decoder's parameters (``dec.pth``) are not distributed with the package because of their size. You do not need to do anything about this now: the first time you use the pre-trained model (for instance, the first time you run ``bulkdgd_find_representations``), ``bulkdgd`` will automatically download ``dec.pth`` (about 900 MB) and cache it inside the installed package for subsequent runs.

.. note::

   If the machine you are running ``bulkdgd`` on has no internet access, download ``dec.pth`` manually on another machine from `this URL <https://github.com/Center-for-Health-Data-Science/bulkdgd/releases/download/v2.0.1/dec.pth>`_ (replace ``v2.0.1`` with your installed ``bulkdgd`` version - run ``python -c "import bulkdgd; print(bulkdgd.__version__)"`` to check), transfer it to the offline machine, find where ``bulkdgd`` was installed with ``python -c "import bulkdgd, os; print(os.path.dirname(bulkdgd.__file__))"``, and place the file in the ``data/model/dec`` sub-folder of that directory (creating it if needed).

Every time you need to run ``bulkdgd`` after opening a new shell, just run step 3 beforehand.
