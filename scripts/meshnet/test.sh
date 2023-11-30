#!/usr/bin/env bash
ckpt_path=/home/eleve05/adaptnet/meshnet/logs/version_3/checkpoints/epoch=999-step=188000.ckpt # TODO: change this to the path of the checkpoint you want to test

clear

cd $PYTHONPATH
python meshnet/main.py test -c meshnet/configs/config.yaml --ckpt_path $ckpt_path