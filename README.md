# GSS-Lab Python-Houdini Setup

This is a setup to convert python geometrical-numerical result into Houdini geometry and visualize it in Houdini.


## Installation

1. create a virtual environment with python 3.7.4
2. install all dependencies : (compas, pandas)
3. install the volpy module within the environment with symlink:
``` shell script
python -m pip install -e .
```
### VS Code Setup

to make sure that the terminal inside VS Code is using the same python as your conda environment (before step 3), make sure to copy the following in the settings.json

``` json
"terminal.integrated.env.osx": {
            "PATH": ""
    }
```

