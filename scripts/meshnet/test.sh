#!/usr/bin/env bash
ckpt_path=/data/users/upelissier/30-Code/meshnet/logs/version_7/checkpoints/epoch=99-step=1000.ckpt

clear
cd $PYTHONPATH
python meshnet/main.py test --ckpt_path $ckpt_path