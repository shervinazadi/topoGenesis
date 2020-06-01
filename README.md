# VolPy

VolPy is a python package for scientific computing with volumetric data.

## Module Installation

1. create a virtual environment with Python 3.7.4 using conda

```Shell Script
conda create --name genesis python=3.7.4
```

2. activate the environment

```Shell Script
conda activate genesis
```

3. install all dependencies:["compas", "pandas"]

```Shell Script
conda install compas
conda install pandas
```

4. go to the source directory of volpy and install it locally within your environment with symlink:

```Shell Script
cd GSS_PYHOU_SETUP/src
python -m pip install -e .
```

Ensure that the pip install is run from the python of the same conda environment (genesis): thus, first navigate to that environment and type "python -m pip install -e". The addition -e uses the "symlink" module from pip ecosystem to ensure that a work-in-progress library is always updated from the latest source.

### VS Code Setup

to make sure that the terminal inside VS Code is using the same python as your conda environment (before step 4), make sure to copy the following in the settings.json

```JSON
"terminal.integrated.env.osx": {
            "PATH": ""
    }
```

## GSS-Lab

This project is part of the [Generative Sciences & Systems Lab Setup](https://github.com/shervinazadi/GSS_PyHou_Setup). It is currently being developed by [Shervin Azadi](https://github.com/shervinazadi) and [Pirouz Nouria](https://github.com/Pirouz-Nourian) at GSS-Lab, Department of Architectural Engineering and Technology, at TU Delft.
