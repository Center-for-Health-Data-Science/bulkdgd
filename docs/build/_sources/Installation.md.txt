## Installing bulkDGD

`bulkDGD` is a Python package requiring **Python 3.11 or higher** and several open-source Python packages.

Please ensure that you have Python installed before proceeding with the installation. The required Python packages will be installed automatically during the installation process. You can inspect the list of packages by opening either the `requirements.txt` or the `environment.yaml` file.

The `bulkDGD` package has been tested only on Unix-based systems.

Here, we provide instructions for installing `bulkDGD` in a simple Python virtual environment or a `conda` environment.

We will show the installation using Python 3.11, but the same steps remain valid for later versions.

### Installing in a Python virtual environment

This section guides you in installing the `bulkDGD` package in a virtual environment, meaning an instance of Python that is isolated from your system.

This is not strictly necessary, and `bulkDGD` may be installed system-wide similarly, following steps 4 to 6.

#### Step 1 - Install `virtualenv`

First, check if the `virtualenv` Python package is installed in your system. This can be done by verifying whether the `virtualenv` command is available.

It is usually available as a package in your distribution if you need to install it. For instance, on Debian-based systems (such as Debian or Ubuntu), it is sufficient to install the `python-virtualenv` package.

If you want to install `virtualenv` system-wide, run the following command:

```shell
sudo apt install python-virtualenv
```

If this is not possible for you, you may still install the `virtualenv` package for just your local user using `pip`:

```shell
pip install --user virtualenv
```

If the installation is successful, the `virtualenv` command will be available.

#### Step 2 - Create the virtual environment

Create your virtual environment in a directory of your choice (in this case, it will be `./bulkDGD-env`):

```shell
virtualenv -p /usr/bin/python3.11 bulkDGD-env
```

You should replace the argument of option `-p` according to the location of the Python interpreter you want to use inside the virtual environment.

#### Step 3 - Activate the environment

Activate the environment:

```shell
source bulkDGD-env/bin/activate
```

#### Step 4 - Get `bulkDGD`

Clone the `bulkDGD` source code from its GitHub repository within a directory of your choice and enter the local copy of the repository.

```shell
git clone https://github.com/Center-for-Health-Data-Science/bulkDGD.git
cd bulkDGD
```

If the `git` command is unavailable, you can download the repository content as a ZIP file from the `bulkDGD` GitHub repository web page, unzip it, and enter it.

#### Step 5 - Install the required Python packages

You can install all the required Python packages specified in the `requirements.txt` file via `pip`:

```shell
pip install -r requirements.txt
```

#### Step 6 - Install `bulkDGD`

You can now install `bulkDGD`:

```shell
python setup.py install
```

`bulkDGD` should now be installed.

Every time you need to run `bulkDGD` after opening a new shell, just run step 3 beforehand.

### Installing with `conda`

#### Step 1 - Install `conda`

Go [here](https://docs.conda.io/en/latest/miniconda.html) for detailed instructions on how to install `conda`.

Installing `miniconda` rather than the full `anaconda` package is advised.

We have provided an `environment.yaml` file that can be used together with `conda` to automatically create an environment containing all required Python packages.

Once `conda` is installed on your system, you can create a virtual environment, similar to what you would do using the `virtualenv` package, as previously detailed.

#### Step 2 - Create the `conda` environment

You can create your `conda` environment from the provided environment file:

```shell
conda env create --file bulkDGD/environment.yaml --prefix ./bulkDGD-env
```

In this case, we ask `conda` to create the environment locally (`--prefix`), which is optional.

If your home directory is size-constrained, the installation might fail for lack of space.

In that case, select another directory with more space available in which conda will download packages:

```shell
conda config --add pkgs_dirs /my/other/directory/conda/pkgs
```

#### Step 3 - Activate the environment

You can activate the `conda` environment by running the command line that `conda` suggests at the end of the previous step.

It is usually something like this:

```shell
conda activate ./bulkDGD-env
```

#### Step 4 - Get `bulkDGD`

Clone the `bulkDGD` source code from its GitHub repository within a directory of your choice and enter the local copy of the repository.

```shell
git clone https://github.com/Center-for-Health-Data-Science/bulkDGD.git
cd bulkDGD
```

If the `git` command is unavailable, you can download the repository content as a ZIP file from the `bulkDGD` GitHub repository web page, unzip it, and enter it.

#### Step 5 - Install `bulkDGD`

You can now install `bulkDGD`:

```shell
python setup.py install
```

`bulkDGD` should now be installed.

Every time you need to run `bulkDGD` after opening a new shell, just run step 3 beforehand.