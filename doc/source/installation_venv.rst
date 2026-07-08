Installing with ``virtualenv``
==============================

This section guides you in installing the ``bulkdgd`` package in a virtual environment, meaning an instance of Python that is isolated from your system.

This is not strictly necessary, and ``bulkdgd`` may be installed system-wide similarly, following steps 4 to 6.

Step 1 - Install ``virtualenv``
-------------------------------

First, check if the ``virtualenv`` Python package is installed in your system. This can be done by verifying whether the ``virtualenv`` command is available.

It is usually available as a package in your distribution if you need to install it. For instance, on Debian-based systems (such as Debian or Ubuntu), it is sufficient to install the ``python-virtualenv`` package.

We recommend installing the ``virtualenv`` package for your local user using ``pip``:

.. code-block:: shell

    pip install --user virtualenv

If the installation is successful, the ``virtualenv`` command will be available.

Step 2 - Create the virtual environment
---------------------------------------

Create your virtual environment in a directory of your choice (in this case, it will be ``./bulkdgd-env``):

.. code-block:: shell

    virtualenv -p /usr/bin/python3.11 bulkdgd-env

You should replace the argument of option ``-p`` according to the location of the Python interpreter you want to use inside the virtual environment.

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

The trained decoder's parameters (``dec.pth``) are not distributed with the package because of their size. You do not need to do anything about this now: the first time you use the pre-trained model (for instance, the first time you run ``bulkdgd_find_representations``), ``bulkdgd`` will automatically download ``dec.pth`` (about 900 MB) and cache it inside the installed package for subsequent runs.

.. note::

   If the machine you are running ``bulkdgd`` on has no internet access, download ``dec.pth`` manually on another machine from `this URL <https://github.com/Center-for-Health-Data-Science/bulkdgd/releases/download/v2.0.1/dec.pth>`_ (replace ``v2.0.1`` with your installed ``bulkdgd`` version - run ``python -c "import bulkdgd; print(bulkdgd.__version__)"`` to check), transfer it to the offline machine, find where ``bulkdgd`` was installed with ``python -c "import bulkdgd, os; print(os.path.dirname(bulkdgd.__file__))"``, and place the file in the ``data/model/dec`` sub-folder of that directory (creating it if needed).

Every time you need to run ``bulkdgd`` after opening a new shell, just run step 3 beforehand.
