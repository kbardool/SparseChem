#!/bin/bash
# PBS -M kevin.bardool@kuleuven.be
# PBS -l pmem=5gb
# PBS -l qos=debugging
pbs_account="-A lp_symbiosys "
pbs_folders="-e ../pbs_output/  -o ../pbs_output/ "
# pbs_allocate="-l nodes=1:ppn=9,walltime=24:00:00 "
pbs_allocate="-l nodes=1:ppn=9:gpus=1,partition=gpu,walltime=24:00:00 "
project_name="SparseChem-cb29"
datadir="../../MLDatasets/chembl29"
outdir="../../experiments/SparseChem-cb29"
RUN_SCRIPT=SC_train_pbs.sh
x_file="chembl_29_x.npy"
y_file="chembl_29_thresh_y.npy"
fold_file="chembl_29_folding.npy"
#=========================================================================
dev="cuda:0"
epochs=200
lr_list=(0.001)
batch_size=4096

num_layers_list=( 5 )
layer_size_list=( 4000 )
# dropout_list=( 0.20 0.30 0.40) 
dropout_list=(0.10 0.20  0.30  0.40  0.50  0.60  0.70  0.80  0.90)
# dropout_list=( 0.70  0.80  0.90)
# dropout_list=(0.10 0.20 0.30 0.40 0.50 0.60 )
#=========================================================================
# dropout_list=(0.00 0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95)
# layer_size_list=(100 200 300 400 500 600 700 800 )
# echo  " pbs_account:   $pbs_account"
# echo  " pbs_folders:   $pbs_folders"
# echo  " pbs_allocat:   $pbs_allocate"
# echo  " DATADIR    :   $datadir      OUTDIR: $outdir    CONFIG FILE: $config"

submit_list(){ 
    for num_layers in ${num_layers_list[@]}; do                 
        for layer in ${layer_size_list[@]}; do                   
            for lr in ${lr_list[@]} ; do
                for dropout in  ${dropout_list[@]}; do
                    # echo " Epochs: $epochs    | Batch Size: $batch_size   Num Lyrs: $num_layers   Lyr sz: $layer   Dropout: $dropout  Task LR: $lr device: $dev"
                    # echo " datadir: $datadir  | outdir: $outdir   Project name: $project_name"
                    job_name="SC-29-${layer}x${num_layers}-${dropout}"
                    printf " $job_name  Epochs: $epochs   Task LR: $lr  dev: $dev ---> "
                    qsub $1   -N $job_name  $pbs_account  $pbs_allocate  $pbs_folders\
                    -v epochs=$epochs,batch_size=$batch_size,num_layers=$num_layers,layer=$layer,dropout=$dropout,datadir=$datadir,outdir=$outdir,lr=$lr,dev=$dev,project_name=$project_name,x_file=$x_file,y_file=$y_file,fold_file=$fold_file
                done
            done
        done
    done
}
 

submit_list "SC_train_pbs.sh" 
 
 
