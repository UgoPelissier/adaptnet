#!/usr/bin/env bash
ckpt_path=/data/users/upelissier/30-Code/graphnet/logs/version_0/checkpoints/epoch=900-step=169388.ckpt

clear
cd $PYTHONPATH
python graphnet/main.py test --ckpt_path $ckpt_path