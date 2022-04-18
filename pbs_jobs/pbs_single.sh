#!/bin/bash
# config=$1
# if [ -z "${config}" ]; then
#     exit 1
# fi
# for dropout in   0.6 0.8 0.9 0.95 1.0; do
# for layer in 800 ; do                   
# for lr in 1e-05; do
# for lr in  0.001 0.0001 1e-06; do
# export config="../yamls/chembl_synt_train_1task.yaml"
# export config="../yamls/chembl_synt_train_3task.yaml"
# export config="../yamls/chembl_synt_train_5task.yaml"
# for dropout in  0.25  0.5  0.75  0.9  1.0; do
# for dropout in  0.55  0.75  0.85  0.95  1.0; do
export epochs=100
export datadir="../../MLDatasets/chembl23_mini"
export  outdir="../../experiments/mini-SparseChem"

# for layer in 50 100 200; do                   
for layer in 50  ; do                   
    for lr in  0.001 ; do
        for dropout in  0.1 ; do
            echo "Layer size: $layer   Task LR: $lr  Dropout: $dropout  datadir: $datadir outdir: $outdir"
            qsub pbs_train_gpu.sh -v epochs=${epochs},layer=${layer},dropout=${dropout},lr=${lr},datadir=${datadir},outdir=${outdir} 
        done
    done
done