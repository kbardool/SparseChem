#!/bin/bash
dev = "cuda:0"
epochs = 100
lr_list=(0.001 )

# layer_size_list=(100 200 300 400 500 600 700 800 )
# dropout_list=(0.9 )
# dropout_list=(0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95 )
# dropout_list=(0.0 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 )
# dropout_list=(0.0 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95)
num_layers_list=(4)
layer_size_list=(800 )
dropout_list=(0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95)
# dropout_list=(0.0 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95)


config="../yamls/chembl_mini_train.yaml"
datadir="../../MLDatasets/chembl23_mini"
outdir="../../experiments/mini-SparseChem"
program="../SparseChem_Train_mini.py"
# echo  " DATADIR: $datadir    OUTDIR: $outdir    CONFIG FILE: $config"

# PBS -M kevin.bardool@kuleuven.be
# PBS -l pmem=5gb
# PBS -l qos=debugging
pbs_account="-A lp_symbiosys "
pbs_folders="-e ../pbs_output/  -o ../pbs_output/ "
pbs_allocate="-l nodes=1:ppn=9:gpus=1,partition=gpu,walltime=06:00:00 "
# pbs_allocate="-l nodes=1:ppn=9,walltime=06:00:00 "
# echo  $pbs_account
# echo  $pbs_folders
# echo  $pbs_allocate

submit_list(){ 
    for num_layers in ${num_layers_list[@]}; do                 
        for layer in ${layer_size_list[@]}; do                   
            for lr in ${lr_list[@]} ; do
                for dropout in  ${dropout_list[@]}; do
                    # echo " Epochs: $epochs    Lyrs: $num_layers   Lyr sz: $layer   Dropout: $dropout  Task LR: $lr device: $dev \n"
                    # printf " Epochs: %s    Lyrs: %d   Lyr sz: %4d   Dropout: %.2f  Task LR: %.3f  device: %s --> "  $epochs $num_layers $layer $dropout $lr $dev
                    job_name="SC-${layer}x${num_layers}-${dropout}"
                    printf " $job_name  Epochs: $epochs   Task LR: $lr  ---> "
                    qsub $1 -N $job_name  $pbs_account   $pbs_allocate   $pbs_folders \
                        -v epochs=$epochs,num_layers=$num_layers,layer=$layer,dropout=$dropout,datadir=$datadir,outdir=$outdir,config=$config,lr=$lr,dev=$dev
                done
            done
        done
    done
}
 

submit_list "SC_train_pbs.sh" 
 
 
