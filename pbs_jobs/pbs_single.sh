#!/bin/bash
# config=$1
pbs_account="-A lp_symbiosys "
PBS_SCRIPT=SC_train.sh
pbs_folders="-e ../pbs_output/  -o ../pbs_output/ "
# pbs_allocate="-l nodes=1:ppn=9,walltime=06:00:00 "
pbs_allocate="-l nodes=1:ppn=9:gpus=1,partition=gpu,walltime=06:00:00 "
config="../yamls/chembl_mini_train.yaml"
datadir="../../MLDatasets/chembl23_mini"
outdir="../../experiments/mini-SparseChem"
# echo  " DATADIR: $datadir    OUTDIR: $outdir    CONFIG FILE: $config"
# echo  " $pbs_account"
# echo  " $pbs_folders"
echo  " $pbs_allocate"

dev="cuda:0"
epochs=100

num_layers=3
layer=100  
dropout=0.95 

lr=0.001  
pbs_name="-N SC-${layer}x${num_layers}-${dropout}"

printf " $pbs_name  Epochs: %s    Lyrs: %d   Lyr sz: %4d   Dropout: %.2f  Task LR: %.3f  device: %s \n"  $epochs $num_layers $layer $dropout $lr $dev

qsub $PBS_SCRIPT \
     $pbs_name   \
     $pbs_account \
     $pbs_allocate\
     $pbs_folders \
     -v epochs=$epochs,num_layers=$num_layers,layer=$layer,dropout=$dropout,datadir=$datadir,outdir=$outdir,config=$config,lr=$lr,dev=$dev            
             