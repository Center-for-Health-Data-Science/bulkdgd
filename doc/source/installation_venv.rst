Installing in a Python virtual environment
==========================================

This section guides you in installing the bulkDGD package in a virtual environment, meaning an instance of Python that is isolated from your system.

This is not strictly necessary, and bulkDGD may be installed system-wide similarly, following steps 4 to 6.

Step 1 - Install ``virtualenv``
-------------------------------

First, check if the ``virtualenv`` Python package is installed in your system. This can be done by verifying whether the ``virtualenv`` command is available.

It is usually available as a package in your distribution if you need to install it. For instance, on Debian-based systems (such as Debian or Ubuntu), it is sufficient to install the ``python-virtualenv`` package.

If you want to install ``virtualenv`` system-wide, run the following command:

.. code-block:: shell

    sudo apt install python-virtualenv

If this is not possible for you, you may still install the ``virtualenv`` package for just your local user using ``pip``:

.. code-block:: shell

    pip install --user virtualenv

If the installation is successful, the ``virtualenv`` command will be available.

Step 2 - Create the virtual environment
---------------------------------------

Create your virtual environment in a directory of your choice (in this case, it will be ``./bulkDGD-env``):

.. code-block:: shell

    virtualenv -p /usr/bin/python3.11 bulkDGD-env

You should replace the argument of option ``-p`` according to the location of the Python interpreter you want to use inside the virtual environment.

Step 3 - Activate the environment
---------------------------------

Activate the environment:

.. code-block:: shell

    source bulkDGD-env/bin/activate

Step 4 - Get bulkDGD
------------------------

Clone the bulkDGD source code from its GitHub repository within a directory of your choice and enter the local copy of the repository.

.. code-block:: shell

    git clone https://github.com/Center-for-Health-Data-Science/bulkDGD.git
    cd bulkDGD

If the ``git`` command is unavailable, you can download the repository content as a ZIP file from the bulkDGD GitHub repository web page, unzip it, and enter it.

Step 5 - Install the required Python packages
---------------------------------------------

You can install all the required Python packages specified in the ``requirements.txt`` file using ``pip``:

.. code-block:: shell

    pip install -r requirements.txt

Step 6 - Get the ``dec.pth`` file
---------------------------------

You must download the ``dec.pth`` file containing the trained decoder's parameters before installing bulkDGD, so that the file is copied to the installation directory. The file cannot be shipped together with the GitHub package because of its size, but can be downloaded `here <https://drive.google.com/file/d/1SZaoazkvqZ6DBF-adMQ3KRcy4Itxsz77/view?usp=sharing>`_.

Once downloaded, place the file into the ``bulkDGD/ioutil/data`` folder before starting the installation.

Step 7 - Install bulkDGD
----------------------------

You can now install bulkDGD:

.. code-block:: shell

    python setup.py install

bulkDGD should now be installed.

Every time you need to run bulkDGD after opening a new shell, just run step 3 beforehand.