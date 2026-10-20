Installing with ``virtualenv``
==============================

This section shows how to install the ``bulkdgd`` package in a virtual environment, that is, an instance of Python isolated from your system.

A virtual environment is not required: to install ``bulkdgd`` system-wide, follow step 4 only.

Step 1 - Install ``virtualenv``
-------------------------------

First, check whether the ``virtualenv`` Python package is installed on your system, that is, whether the ``virtualenv`` command is available.

If you need to install it, your distribution usually provides it as a package. For instance, on Debian-based systems (such as Debian or Ubuntu), install the ``python-virtualenv`` package.

We recommend installing the ``virtualenv`` package for your local user using ``pip``:

.. code-block:: shell

    pip install --user virtualenv

If the installation is successful, the ``virtualenv`` command will be available.

Step 2 - Create the virtual environment
---------------------------------------

Create your virtual environment in a directory of your choice (in this case, it will be ``./bulkdgd-env``):

.. code-block:: shell

    virtualenv -p /usr/bin/python3.11 bulkdgd-env

Replace the argument of option ``-p`` with the location of the Python interpreter you want to use inside the virtual environment.

Step 3 - Activate the environment
---------------------------------

Activate the environment:

.. code-block:: shell

    source bulkdgd-env/bin/activate

Step 4 - Install bulkdgd
----------------------------

You can now install ``bulkdgd`` from `PyPI <https://pypi.org/project/bulkdgd/>`_ using ``pip``:

.. code-block:: shell

    pip install bulkdgd

``bulkdgd`` should now be installed.

The trained decoders' parameters are not distributed with the package because of their size: there is one per member of the shipped ensemble, each 1.79 GiB in ``float64``. No action is needed: the first time you use a member (for instance, the first time you run ``bulkdgd_find_representations``), ``bulkdgd`` downloads that member's decoder and caches it inside the installed package for later runs.

.. note::

   If the machine you are running ``bulkdgd`` on has no internet access, download the decoder manually on another machine from `this URL <https://github.com/Center-for-Health-Data-Science/bulkdgd/releases/download/v2.3.0/dec_seed37.pth>`_ (replace ``v2.3.0`` with your installed ``bulkdgd`` version, which ``python -c "import bulkdgd; print(bulkdgd.__version__)"`` prints, and ``seed37`` with the member you want, if it is not the default one), transfer it to the offline machine, find where ``bulkdgd`` was installed with ``python -c "import bulkdgd, os; print(os.path.dirname(bulkdgd.__file__))"``, and place the file in the ``data/model/seed37`` sub-folder of that directory, under the name ``dec.pth``. The release asset carries the seed in its name, the file on disk does not: each member's directory holds its own ``dec.pth``.

Each time you open a new shell, run step 3 before running ``bulkdgd``.
