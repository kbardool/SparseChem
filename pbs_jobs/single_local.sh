#!/bin/bash 
# pbs_account="-A lp_symbiosys "
# pbs_folders="-e ../pbs_output/  -o ../pbs_output/ "
# pbs_allocate="-l nodes=1:ppn=9,walltime=06:00:00 "
# pbs_allocate="-l nodes=1:ppn=9:gpus=1,partition=gpu,walltime=06:00:00 "
# echo  " $pbs_account"
# echo  " $pbs_folders"
# echo  " $pbs_allocate"


RUN_SCRIPT=SC_train_local.sh
config="../yamls/chembl_mini_train.yaml"
datadir="../../MLDatasets/chembl23_mini"
outdir="../../experiments/mini-SparseChem"
project_name="SparseChem-Mini"
x_file="chembl_23mini_x.npy"
y_file="chembl_23mini_y.npy"
fold_file="chembl_23mini_folds.npy"
# echo  " DATADIR: $datadir    OUTDIR: $outdir    CONFIG FILE: $config"

dev="cuda:0"
epochs=100
lr=0.001

layer=50
num_layers=4
dropout=0.65
job_name="SC-${layer}x${num_layers}-${dropout}"
output_file="../pbs_output/${job_name}.out"

echo " Epochs: $epochs    Lyrs: $num_layers  Lyr sz: $layer   Dropout: $dropout  Task LR: $lr  device: $dev  output: $output_file "   
. $RUN_SCRIPT > $output_file 2>&1 &