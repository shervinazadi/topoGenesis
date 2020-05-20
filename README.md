# GSS-Lab Python Modules + Python-Houdini Setup

This is the draft of the python modules that are currently being developed by [Shervin Azadi](https://github.com/shervinazadi) and [Pirouz Nouria](https://github.com/Pirouz-Nourian) at GSS-Lab, Department of Architectural Engineering and Technology, at TU Delft. These modules are mainly working with volumetric data-sets and fields. We also include an example folder that contains example algorithms that utilizes the modules and a standard setup for using [Houdini](https://www.sidefx.com/) as the Visualizer for algorithms.

---

## Houdni Assets

### 1. GSS Call Python

This asset allows the user to choose a python file and choose a specfic python environment and run the python script with the selected environment. This allows for separation of environments through a workflow and also solves the problem of importing libraries in houdini completely. This asset is compatible with conda environments as well.

This module uses the `sqlite3` standard library to replicate the houdini's geo database and passes the address to the subprocess. The sub process will read the database, execute the computational process and saves the data into the database. You can decide to keep or delete the database for trouble shooting distances. The input data is saved to tables named `"POINTS", "PRIMITIVE", "DETAIL"` and the output data needs to be saved to tables named `"POINTS_NEW", "PRIMITIVE_NEW", "DETAIL_NEW"`.

You can find an example python script along with the asset.

Notice:

- currently you need to have at least one attribute on prim and detail level for the node to work
- attribute names should not include the charachter "\_"
- currently vertex table is not passed on
- currently all attributes are passed (before) and updated (after) by defualt
- currently there is no access to intrinsic attributes
- currently list attributes are not supported

Upcoming Features:

- Support of vertex table
- Support of selective attribute transfer to resduce the overhead
- Support of list attributes
- Support of intrinsic attributes
- Support of writing python code inside houdini interface
- Saving the databse file in the temp folder of the computer instead of the working directory
- Ability to create new attributes
- Ability to create and delete points
- Ability to create and delete primitives

---

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

---

## To Do

Checkout & Compare:

- https://wiki.python.org/moin/NumericAndScientific
- [numdifftools](https://github.com/pbrod/numdifftools)
- [fipy](https://www.ctcms.nist.gov/fipy/)
- http://hplgit.github.io/pyhpc/doc/pub/._project001.html
- https://www.math.ubc.ca/~pwalls/math-python/differentiation/differentiation/
- [scipy laplacian](https://mail.python.org/pipermail/scipy-user/2013-April/034452.html)

Coding Style:

- [PEP8](https://www.python.org/dev/peps/pep-0008/)
