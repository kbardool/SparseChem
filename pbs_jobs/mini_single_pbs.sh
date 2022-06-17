#!/bin/bash
dev="cuda:0"
epochs=100
lr=0.00001  

layer=4000
num_layers=2
dropout=0.30
batch_size=128




# project_name="SparseChem-Synt"
# datadir="../../MLDatasets/chembl29"
# outdir="../../experiments/SparseChem-cb29"
# x_file="chembl_29_x.npy"
# y_file="chembl_29_thresh_y.npy"
# fold_file="chembl_29_folding.npy"

project_name="SparseChem-Mini"
datadir="../../MLDatasets/chembl23_mini"
outdir="../../experiments/SparseChem-mini"
x_file="chembl_23mini_x.npy"
y_file="chembl_23mini_y.npy"
fold_file="chembl_23mini_folds.npy"

# echo  " DATADIR: $datadir    OUTDIR: $outdir    "

RUN_SCRIPT=SC_train_pbs.sh
# PBS -M kevin.bardool@kuleuven.be
# PBS -l pmem=5gb
# PBS -l qos=debugging
pbs_account="-A lp_symbiosys "
pbs_folders="-e ../pbs_output/  -o ../pbs_output/ "
pbs_allocate="-l nodes=1:ppn=9:gpus=1,partition=gpu,walltime=06:00:00 "
# pbs_allocate="-l nodes=1:ppn=9,walltime=06:00:00 "
# echo  " $pbs_account"
# echo  " $pbs_folders"
# echo  " $pbs_allocate"

# echo  " DATADIR: $datadir    OUTDIR: $outdir    CONFIG FILE: $config"


job_name="SC-MN-${layer}x${num_layers}-${dropout}" 

printf " $job_name  Epochs: $epochs   Task LR: $lr  ---> "
qsub $RUN_SCRIPT  -N $job_name  $pbs_account  $pbs_allocate  $pbs_folders \
-v epochs=$epochs,batch_size=$batch_size,num_layers=$num_layers,layer=$layer,dropout=$dropout,datadir=$datadir,outdir=$outdir,lr=$lr,dev=$dev,\
project_name=$project_name,x_file=$x_file,y_file=$y_file,fold_file=$fold_file