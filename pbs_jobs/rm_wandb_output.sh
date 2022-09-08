#!/bin/bash
find  ./wandb/ -name "run-2022*" -mmin +$1 -fprintf del_wandb.sh 'rm -rf ./wandb/%P \n' 