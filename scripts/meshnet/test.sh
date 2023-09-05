#!/usr/bin/env bash
ckpt_path=/home/eleve05/adaptnet/meshnet/logs/version_4/checkpoints/epoch=999-step=188000.ckpt

clear

for arg in "$@"
do
   key=$(echo $arg | cut -f1 -d=)

   key_length=${#key}
   value="${arg:$key_length+1}"

   export "$key"="$value"
done

cd $PYTHONPATH
python meshnet/main.py test -c meshnet/configs/${env}.yaml --ckpt_path $ckpt_path