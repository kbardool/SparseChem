#!/bin/bash
pbs_account="-A lp_symbiosys "
pbs_folders="-e ../pbs_output/  -o ../pbs_output/ "
pbs_allocate="-l nodes=1:ppn=9:gpus=1,partition=gpu,walltime=06:00:00 "
# pbs_allocate="-l nodes=1:ppn=9,walltime=06:00:00 "
# echo  " $pbs_account"
# echo  " $pbs_folders"
# echo  " $pbs_allocate"


RUN_SCRIPT=SC_train_pbs.sh
config="../yamls/chembl_mini_train.yaml"
datadir="../../MLDatasets/chembl23_mini"
outdir="../../experiments/mini-SparseChem"
project_name="SparseChem-Mini"
# echo  " DATADIR: $datadir    OUTDIR: $outdir    CONFIG FILE: $config"

dev="cuda:0"
epochs=100
lr=0.001  

layer=4000
num_layers=2
dropout=0.40
job_name="SC-${layer}x${num_layers}-${dropout}" 

printf " $job_name  Epochs: $epochs   Task LR: $lr  ---> "
qsub $RUN_SCRIPT  -N $job_name  $pbs_account  $pbs_allocate  $pbs_folders \
     -v epochs=$epochs,num_layers=$num_layers,layer=$layer,dropout=$dropout,datadir=$datadir,outdir=$outdir,config=$config,lr=$lr,dev=$dev            
             