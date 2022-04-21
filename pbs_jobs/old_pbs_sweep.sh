#!/bin/bash
# config=$1
# if [ -z "${config}" ]; then
#     exit 1
# fi
pbs_cpu="-l nodes=1:ppn=9,walltime=06:00:00"
pbs_gpu="-l nodes=1:ppn=9:gpus=1,partition=gpu,walltime=06:00:00"
epochs=100
datadir="../../MLDatasets/chembl23_mini"
outdir="../../experiments/mini-SparseChem"

# for layer in 50 100 200; do                   
# for layer in  200 300 400 500 600 700 800; do                   
echo  datadir: $datadir outdir: $outdir

num_layers=1

for lyr in  400 ; do                   
    for lr in  0.001 ; do
        # for do in  0.0 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95; do
        for do in  0.75 ; do
        # for dropout in  0.0 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5; do
        #   -v epochs=${epochs},layer=${layers},dropout=${dropouts},lr=${lr},datadir=${datadir},outdir=${outdir} 
        # for dropout in  0.3 0.35 0.4 0.45 0.5 ; do
            layers=""
            dropouts=""
            for ((i=0 ; i < $num_layers ; i +=1)); do
                layers+=" $lyr "
                dropouts+=" $do "
            done
            echo "Layers : $layers  Dropout: $dropouts   Task LR: $lr   datadir: $datadir   outdir: $outdir"
            qsub pbs_train_gpu.sh -N SCtrain  -A lp_symbiosys \
                                  $pbs_gpu \
                                  -e ../pbs_output/  -o ../pbs_output/    \
                                  -v epochs,layers,dropouts,datadir,outdir,lr=$lr
        done
    done
done