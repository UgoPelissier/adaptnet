#!/usr/bin/env bash
ckpt_path=/home/eleve05/adaptnet/graphnet/logs/version_14/checkpoints/epoch=3803-step=357576.ckpt # TODO: change this to the path of the checkpoint you want to test

clear

cd $PYTHONPATH
python graphnet/main.py test -c graphnet/configs/config.yaml --ckpt_path $ckpt_path