.. _citing:

************
Citing
************

.. describe cross-platform ness like on homepage

.. Installing Editable Version
.. ================

.. 1. create a virtual environment using conda

.. .. code-block:: bash

..     conda create --name genesis

.. 2. activate the environment

.. .. code-block:: bash

..     conda activate genesis

.. 3. install all dependencies:[`numpy`, `pandas`, `pyvista`]

.. .. code-block:: bash

..     conda install numpy pandas pyvista

.. 4. install an editable version of topogenesis within your environment with a symlink

.. .. code-block:: bash

..     python -m pip install -e .

.. Ensure that the pip install is run from the python of the same conda environment (genesis): thus, first, navigate to that environment and type "python -m pip install -e". The addition -e uses the "symlink" module from pip ecosystem to ensure that a work-in-progress library is always updated from the latest source.
