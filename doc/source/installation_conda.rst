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

The trained decoders' parameters are not distributed with the package because of their size: there is one per member of the shipped ensemble, each 1.79 GiB in ``float64``. No action is needed: the first time you use a member (for instance, the first time you run ``bulkdgd_find_representations``), ``bulkdgd`` downloads that member's decoder and caches it inside the installed package for later runs.

.. note::

   If the machine you are running ``bulkdgd`` on has no internet access, download the decoder manually on another machine from `this URL <https://github.com/Center-for-Health-Data-Science/bulkdgd/releases/download/v2.3.0/dec_seed37.pth>`_ (replace ``v2.3.0`` with your installed ``bulkdgd`` version, which ``python -c "import bulkdgd; print(bulkdgd.__version__)"`` prints, and ``seed37`` with the member you want, if it is not the default one), transfer it to the offline machine, find where ``bulkdgd`` was installed with ``python -c "import bulkdgd, os; print(os.path.dirname(bulkdgd.__file__))"``, and place the file in the ``data/model/seed37`` sub-folder of that directory, under the name ``dec.pth``. The release asset carries the seed in its name, the file on disk does not: each member's directory holds its own ``dec.pth``.

Each time you open a new shell, run step 3 before running ``bulkdgd``.
