<!-- # Welcome to MkDocs

For full documentation visit [mkdocs.org](https://www.mkdocs.org).

## Commands

* `mkdocs new [dir-name]` - Create a new project.
* `mkdocs serve` - Start the live-reloading docs server.
* `mkdocs build` - Build the documentation site.
* `mkdocs -h` - Print help message and exit.

## Project layout

    mkdocs.yml    # The configuration file.
    docs/
        index.md  # The documentation homepage.
        ...       # Other markdown pages, images and other files. 

# MathJax examples 
The homomorphism $f$ is injective if and only if its kernel is only the 
singleton set $e_G$, because otherwise $\exists a,b\in G$ with $a\neq b$ such 
that $f(a)=f(b)$.

$$
\operatorname{ker} f=\{g\in G:f(g)=e_{H}\}{\mbox{.}}
$$

-->

# topoGenesis

[![Build Status](https://travis-ci.org/shervinazadi/topoGenesis.svg?branch=master)](https://travis-ci.org/shervinazadi/topoGenesis)
[![codecov](https://codecov.io/gh/shervinazadi/topoGenesis/branch/master/graph/badge.svg)](https://codecov.io/gh/shervinazadi/topoGenesis)
[![GitHub](https://img.shields.io/github/license/shervinazadi/topogenesis)](https://github.com/shervinazadi/topogenesis)
[![GitHub - Source Code](https://img.shields.io/badge/-source%20code-grey?logo=github)](https://github.com/shervinazadi/topogenesis)
[![DOI](https://zenodo.org/badge/268286072.svg)](https://zenodo.org/badge/latestdoi/268286072)
[![Documentation Status](https://readthedocs.org/projects/topogenesis/badge/?version=latest)](https://topogenesis.readthedocs.io/?badge=latest)

[topoGenesis](https://github.com/shervinazadi/topoGenesis) is an open-source python package that provides topological structures and functions for Generative Systems and Sciences for various application areas such as:

- generative design in architecture and built environment
- generative spatial simulations
- 3D image processing
- topological data analysis
- machine learning

## Vision

topoGenesis aims to utilize the vast functionalities of fields (mathematical objects) in generative systems and sciences. Therefore it seeks to:

1. offer basic mathematical functionalities on field data models
2. offer functionalities of computational topology on top of the field structures
3. facilitate the conversion between mesh-based data models and field data models.
4. facilitate field simulations, whether governed by differential equations, spectral models or based on computational models (ABM)
5. construct a bridge between spatial data models and tensor data structures to facilitate the utilization of the latest artificial intelligence models

## Structure

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

## Genesis Lab

This project is currently being developed by [Shervin Azadi](https://github.com/shervinazadi) and [Pirouz Nourian](https://github.com/Pirouz-Nourian) at [Genesis Lab: Laboratory of Generative Systems and Sciences](https://www.researchgate.net/lab/Genesis-Laboratory-of-Generative-Systems-and-Sciences-Pirouz-Nourian), Department of Architectural Engineering and Technology, at TU Delft.
