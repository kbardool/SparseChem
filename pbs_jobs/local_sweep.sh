#!/bin/bash
# config=$1
# if [ -z "${config}" ]; then
#     exit 1
# fi
export epochs=100
# for dropout in   0.6 0.8 0.9 0.95 1.0; do
# for layer in 800 ; do                   
# for lr in 1e-05; do
# for lr in  0.001 0.0001 1e-06; do
export outdir="../../experiments/mini-SparseChem"
# export config="../yamls/chembl_synt_train_1task.yaml"
# export config="../yamls/chembl_synt_train_3task.yaml"
# export config="../yamls/chembl_synt_train_5task.yaml"

for layer in  400; do                   
    for lr in  0.001 ; do
        # for dropout in  0.0  ; do
        # for dropout in  0.0 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5; do
        # for dropout in  0.0 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5; do
        for dropout in  0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95 1.0; do
        # for dropout in  0.55 0.6 0.65 0.7 0.75; do
        # for dropout in  0.8 0.85 0.9 0.95 1.0; do
        # for dropout in  0.0 0.05 0.1 0.15 0.2 0.25; do
        # for dropout in  0.3 0.35 0.4 0.45 0.5 ; do
        # for dropout in  0.25  0.5  0.75  0.9  1.0; do
        # for dropout in  0.55  0.75  0.85  0.95  1.0; do
            echo "Layer size: $layer   Task LR: $lr  Dropout: $dropout     outdir: $outdir"
            . pbs_train.sh  
        done
    done
done