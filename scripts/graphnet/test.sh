#!/usr/bin/env bash
ckpt_path=/home/eleve05/adaptnet/graphnet/logs/version_4/checkpoints/epoch=9999-step=940000.ckpt # TODO: change this to the path of the checkpoint you want to test

clear

cd $PYTHONPATH
python graphnet/main.py test -c graphnet/configs/config.yaml --ckpt_path $ckpt_path