#!/usr/bin/env bash
ckpt_path=/home/eleve05/adaptnet/graphnet/logs/version_11/checkpoints/epoch=999-step=188000.ckpt

clear

if [ "$1" == "-h" ]; then
    echo "Usage: ./stokes2.sh -c <c>"
    echo "c: Configuration file"
    exit 0
fi

while getopts c: flag
do
    case "${flag}" in
        c) c=${OPTARG};;
    esac
done

cd $PYTHONPATH
python graphnet/main.py test -c graphnet/configs/${c}.yaml --ckpt_path $ckpt_path