
# Mortality
Mortality python package

## Overview

**XAct**

**📖 Description:**

XAct is a python package that provides experience study tools for actuaries with the 
goal of being *exact*. There are tools for reviewing mortality as well as tools for 
reviewing experience.

**🔬 Jupyter Notebook:**

- [XAct Example](https://nbviewer.jupyter.org/github/jkoestner/xact/blob/main/notebooks/mortality.ipynb)

## Installation

### Local Install
To install, this repository can be installed by running the following command in 
the environment of choice.

The following command can be run to install the packages in the pyproject.toml file.

```
pip install -e .
```

## Other Tools
### Jupyter Lab Usage

To have conda environments work with Jupyter Notebooks a kernel needs to be defined. This can be done defining a kernel, shown below when
in the conda environment.

```
python -m ipykernel install --user --name=xact
```
### Logging

If wanting to get more detail in output of messages the logging can increased
```python
from xact.utils import helpers
helpers.set_log_level("DEBUG")
```