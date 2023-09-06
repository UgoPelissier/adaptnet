#!/usr/bin/env bash
ckpt_path=/home/eleve05/adaptnet/meshnet/logs/version_4/checkpoints/epoch=999-step=188000.ckpt

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
python meshnet/main.py test -c meshnet/configs/${c}.yaml --ckpt_path $ckpt_path