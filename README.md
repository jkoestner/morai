
## Overview

**Morai**

**📖 Description:**

[Moirai](https://en.wikipedia.org/wiki/Moirai#:~:text=In%20ancient%20Greek%20religion%20and,Moirai) 
in greek mythology are known as the fates. They are personifications of destiny.
The name Morai was chosen as the package is designed to help actuaries review 
mortality and experience data.

**🔬 Jupyter Notebook:**

- [Morai Example](https://nbviewer.jupyter.org/github/jkoestner/morai/blob/main/notebooks/mortality.ipynb)

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
python -m ipykernel install --user --name=morai
```
### Logging

If wanting to get more detail in output of messages the logging can increased
```python
from morai.utils import helpers
helpers.set_log_level("DEBUG")
```