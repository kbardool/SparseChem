#!/bin/bash
find  ~/WSL-projs/experiments/SparseChem-mini -type d  -mmin +$1 -fprintf del_exp_folders.sh 'rm -rf %P \n'