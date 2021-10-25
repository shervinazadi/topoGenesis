# topoGenesis

[![Build Status](https://travis-ci.org/shervinazadi/topoGenesis.svg?branch=master)](https://travis-ci.org/shervinazadi/topoGenesis)
[![codecov](https://codecov.io/gh/shervinazadi/topoGenesis/branch/master/graph/badge.svg)](https://codecov.io/gh/shervinazadi/topoGenesis)
[![GitHub](https://img.shields.io/github/license/shervinazadi/topogenesis)](https://github.com/shervinazadi/topogenesis)
[![Documentation Status](https://readthedocs.org/projects/topogenesis/badge/?version=latest)](https://topogenesis.readthedocs.io/?badge=latest)
[![DOI](https://zenodo.org/badge/268286072.svg)](https://zenodo.org/badge/latestdoi/268286072)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4006514.svg)](https://doi.org/10.5281/zenodo.4006514)



[topoGenesis](https://topogenesis.readthedocs.io/) is an open-source python package providing topological data structures and functions for working with voxel data in Generative Systems and Sciences. The envisaged application areas are:

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
  - Point Cloud Voxelation
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

## Installation

for installation check [this tutorial](https://topogenesis.readthedocs.io/installation/)

## Genesis Lab: Laboratory of Generative Systems and Sciences

This project is currently being developed by [Shervin Azadi](https://github.com/shervinazadi) and [Pirouz Nourian](https://github.com/Pirouz-Nourian) at [Genesis Lab: Laboratory of Generative Systems and Sciences](https://www.tudelft.nl/en/architecture-and-the-built-environment/research/research-facilities/genesis-lab), Department of Architectural Engineering and Technology, at TU Delft.
