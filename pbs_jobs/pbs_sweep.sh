#!/bin/bash
# config=$1
# if [ -z "${config}" ]; then
#     exit 1
# fi
export pbs_cpu="-l nodes=1:ppn=9,walltime=06:00:00"
export pbs_gpu="-l nodes=1:ppn=9:gpus=1,partition=gpu,walltime=06:00:00"
export epochs=100
export datadir="../../MLDatasets/chembl23_mini"
export  outdir="../../experiments/mini-SparseChem"

# for layer in 50 100 200; do                   
for layer in  300; do                   
    for lr in  0.001 ; do
        # for dropout in  0.0 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95; do
        for dropout in  0.95 ; do
        # for dropout in  0.0 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5; do
        # for dropout in  0.3 0.35 0.4 0.45 0.5 ; do
            echo "Layer size: $layer   Task LR: $lr  Dropout: $dropout  datadir: $datadir outdir: $outdir"
            qsub SC_train.sh -N SCtrain  -A lp_symbiosys \
                                  $pbs_gpu \
                                  -e ../pbs_output/  -o ../pbs_output/    \
                                  -v epochs=${epochs},layer=${layer},dropout=${dropout},lr=${lr},datadir=${datadir},outdir=${outdir} 
        done
    done
done