# GSS-Lab Python Modules + Python-Houdini Setup

This is the draft of the python modules that are currently being developed at GSS-Lab, Department of Architectural Engineering and Technology, at TU Delft. These modules are mainly working with volumetric data-sets and fields. We also include an example folder that contains example algorithms that utilizes the modules and a standard setup for using Houdini as the Visualizer for algorithms.

## Module Installation

1. create a virtual environment with Python 3.7.4 using conda
``` Shell Script
conda create --name genesis python=3.7.4
```

2. activate the environment
``` Shell Script
conda activate genesis
```

3. install all dependencies:["compas", "pandas"]
``` Shell Script
conda install compas
conda install pandas
```

4. go to the source directory of volpy and install it locally within your environment with symlink:
``` Shell Script
cd GSS_PYHOU_SETUP/src
python -m pip install -e .
```

### VS Code Setup

to make sure that the terminal inside VS Code is using the same python as your conda environment (before step 4), make sure to copy the following in the settings.json

``` JSON
"terminal.integrated.env.osx": {
            "PATH": ""
    }
```

