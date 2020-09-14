.. topoGenesis documentation master file, created by
   sphinx-quickstart on Sun Sep 13 23:05:59 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Introduction to topoGenesis
===========================


topoGenesis is an open-source python package that provides topological structures and functions for Generative Systems and Sciences for various application areas such as:

- generative design in architecture and built environment
- generative spatial simulations
- 3D image processing
- topological data analysis
- machine learning

Vision
------

topoGenesis aims to utilize the vast functionalities of fields (mathematical objects) in generative systems and sciences. Therefore it seeks to:

1. offer basic mathematical functionalities on field data models
2. offer functionalities of computational topology on top of the field structures
3. facilitate the conversion between mesh-based data models and field data models.
4. facilitate field simulations, whether governed by differential equations, spectral models or based on computational models (ABM)
5. construct a bridge between spatial data models and tensor data structures to facilitate the utilization of the latest artificial intelligence models

Core functionalities
--------------------

- Mesh to Field: Rasterization
   - Point Cloud Regularization
   - Line Network Voxelation
   - Mesh Surface Voxelation
   - Signed Distance Field
- Field to Mesh: Isosurface
   - Boolean Marching Cubes
   - Marching Cubes
   - Surface Nets
- Local Computation
   - Stencil / Kernels
      - von Neumann neighbourhood
      - Moore neighbourhood
      - Cube neighbourhood
      - Custom neighbourhoods
   - Universal Functions & Mathematical Operators (Numerical)
- Field Simulations (Vectorized)
   - Dynamic Systems (based on Differential Equations)
   - Agent-Based Modeling
   - Cellular Automata

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   Introduction <self>
   installation
   tutorial
   api
   license
   citing


Genesis Lab: Laboratory of Generative Systems and Sciences
----------------------------------------------------------

This project is currently being developed by `Shervin Azadi <https://github.com/shervinazadi>`_ and `Pirouz Nourian <https://github.com/Pirouz-Nourian>`_ at `Genesis Lab: Laboratory of Generative Systems and Sciences <https://www.researchgate.net/lab/Genesis-Laboratory-of-Generative-Systems-and-Sciences-Pirouz-Nourian>`_ , Department of Architectural Engineering and Technology, at TU Delft.

.. Indices and tables
.. ==================

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`