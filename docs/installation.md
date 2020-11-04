# Installation

1. create a virtual environment using conda

```bash
conda env create -f environment.yml
```

2. activate the environment

```bash
conda activate genesis
```

3. install topogenesis locally within your environment with a symlink:

```bash
python -m pip install -e .
```

Ensure that the pip install is run from the python of the same conda environment (genesis): thus, first, navigate to that environment and type "python -m pip install -e". The addition -e uses the "symlink" module from pip ecosystem to ensure that a work-in-progress library is always updated from the latest source.
