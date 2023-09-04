# Adaptnet

Adaptnet is a Python pipeline to predict mesh adaptation to a physical problem, given an initial CAD and boundary conditions. 

This pipeline relies on:
- [Meshnet](https://github.com/UgoPelissier/meshnet) network for predicting mesh parameters given a CAD model.
- [Graphnet](https://github.com/UgoPelissier/graphnet) network for predicting the solution of a PDE given a mesh.

Both models are based on the [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/) library. And both networks are based on the [MeshGraphNets](https://arxiv.org/abs/2010.03409) architecture.

## Setup @Safran™

### Conda environment
```bash
module load conda
module load artifactory
mamba env create -f envs/adaptnet_no_builds.yml
conda activate adaptnet
```
