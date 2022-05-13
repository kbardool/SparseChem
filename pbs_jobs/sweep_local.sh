#!/bin/bash
dev="cuda:0"
epochs=100
lr_list=(0.001 )

# num_layers_list=(3)
# layer_size_list=( 800 )
# dropout_list=(0.9 )
# dropout_list=(0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95)
# layer_size_list=(100 200 300 400 500 600 700 800 )
# dropout_list=(0.0 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95)
# dropout_list=(0.0 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 )
# dropout_list=(0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95 )

submit_list "SC_train_local.sh"
config="../yamls/chembl_mini_train.yaml"
datadir="../../MLDatasets/chembl23_mini"
outdir="../../experiments/mini-SparseChem"
program="../SparseChem_Train_mini.py"
x_file="chembl_23mini_x.npy"
y_file="chembl_23mini_y.npy"
fold_file="chembl_23mini_folds.npy"
# echo  " DATADIR: $datadir    OUTDIR: $outdir    CONFIG FILE: $config"
 
# PBS -M kevin.bardool@kuleuven.be
# PBS -l pmem=5gb
# PBS -l qos=debugging
# pbs_account="-A lp_symbiosys "
# pbs_folders="-e ../pbs_output/  -o ../pbs_output/ "
# pbs_allocate="-l nodes=1:ppn=9,walltime=06:00:00 "
# pbs_allocate="-l nodes=1:ppn=9:gpus=1,partition=gpu,walltime=06:00:00 "
# echo  $pbs_account
# echo  $pbs_folders
# echo  $pbs_allocate

submit_list(){ 
    for num_layers in ${num_layers_list[@]}; do                 
        for layer in ${layer_size_list[@]}; do                   
            for lr in ${lr_list[@]} ; do
                for dropout in  ${dropout_list[@]}; do
                    job_name="SC-${layer}x${num_layers}-${dropout}"
                    output_file="../pbs_outputs/${job_name}.out"
                    echo " Epochs: $epochs    Lyrs: $num_layers  Lyr sz: $layer   Dropout: $dropout  Task LR: $lr  device: $dev  output: $output_file\n"   
                    . $1 > $output_file 2>&1 &
                done
            done
        done
    done
}
 