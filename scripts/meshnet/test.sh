#!/usr/bin/env bash
ckpt_path=/home/eleve05/adaptnet/meshnet/logs/version_6/checkpoints/epoch=115-step=2784.ckpt

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