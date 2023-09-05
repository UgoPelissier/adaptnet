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
mamba env create -f src/envs/adaptnet_no_builds.yml
conda activate adaptnet
```

### Install Meshnet and Graphnet
Download the Meshnet and Graphnet repositories in the same folder as Adaptnet:
```bash
git clone https://github.com/UgoPelissier/meshnet.git
git clone https://github.com/UgoPelissier/graphnet.git
```

Your folder should look like this:
```
├── .vscode
├── graphnet
├── meshnet
├── scripts
├── src
├── .gitignore
└── README.md
```

### Train Meshnet and Graphnet
Follow the instructions in the README.md files of Meshnet and Graphnet to download the data and train the models. This will generate checkpoint files in the `meshnet/logs/version_$VERSION_NUMBER/checkpoints` and `graphnet/logs/version_$VERSION_NUMBER/checkpoints` folders.

Alternatively, if you already have the datasets, you can use the scripts in `scripts` to train and test the models:
```bash
bash ./scripts/meshnet/train.sh env=$ENV_NAME (mines or safran)
bash ./scripts/meshnet/test.sh env=$ENV_NAME (mines or safran)
```

```bash
bash ./scripts/graphnet/train.sh env=$ENV_NAME (mines or safran)
bash ./scripts/graphnet/test.sh env=$ENV_NAME (mines or safran)
```

### Run Adaptnet
Create a data folder in `src`:
```bash
mkdir src/data
```	
and put your CAD file inside it. For instance, `src` folder could look like this:
```
└── src
    ├── configs
    └── data
        └── cad_500.geo
    ├── envs
    ├── utils
    ├── __init__.py
    ├── main.py
    └── dataset.py
```

Don't forget to open the `src/configs/{$ENV_NAME}.yaml` file and change the paths according to your setup.

Then, run Adaptnet:
```bash
bash ./scripts/predict.sh env=$ENV_NAME (mines or safran)
```