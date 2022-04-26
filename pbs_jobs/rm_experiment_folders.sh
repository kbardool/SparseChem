#!/bin/bash
find  ~/WSL-projs/experiments/mini-SparseChem -type d  -mtime +$1 fprintf del_exp_folders.sh 'rm -rf %P \n'