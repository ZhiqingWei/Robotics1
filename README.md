# COMP0037-CW1

## Development Guide

### First time setup

#### Basic Python Environment Setup

Install `anaconda 4.9.2`, the latest version of `Anaconda`.

Then, we use `conda` to create a distinct development environment to ensure compatible Python and
third-party dependencies versions. We use `conda-forge` to get more recent versions of point
releases of Python `3.8.*` series and third party dependencies not included in the base `conda`
registry.

```bash
# Create a new environment
conda create -n comp0037 -c conda-forge python=3.8.6 
# Activate the new environment
conda activate comp0037
# Functionality dependencies
conda install numpy
```

Note that `conda` typically uses `base` as the default Python environment when a new shell instance
is launched. To activate the previously created project environment, use

```bash
conda activate comp0037
```

To deactivate the project environment, use

```bash
conda deactivate
```

#### Linters and Formatters

In the `comp0037` project environment, we now add some development conveniences to help reduce
formatting and logical issues. We use [`autopep8`]  to format our code, then use [`flake8`] to check
for conformance with PEP8 formatting standard and identify other code quality issues.

```bash
# Install pre-commit hook to check for stylistic and logical issues
conda install -c conda-forge pre-commit
# Install rope for refactoring
conda install -c conda-forge rope
# Install autopep8 and flake8
conda install -c conda-forge autopep8 flake8
```

Add `.pre-commit-config.yaml` with relevant git hooks ([`autopep8`] and [`flake8`]) to configure the
pre-commit hooks. Now, run

```bash
pre-commit install
```

to install the git pre-commit hooks to local `.git`.

[`flake8`]: https://github.com/PyCQA/flake8
[`autopep8`]: https://github.com/hhatto/autopep8