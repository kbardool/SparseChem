#!/bin/bash 
#PBS -l nodes=1:ppn=9:gpus=1
#PBS -l partition=gpu
#PBS -l walltime=01:00:00
#PBS -A lp_symbiosys

# module purge
source /user/leuven/326/vsc32647/.initconda
cd $PBS_O_WORKDIR # cd to the directory from which qsub is run
echo PBS VERSION is $PBS_VERSION
echo config file is $config
which conda
echo switch to pyt-gpu 
conda activate pyt-gpu
python -V
export program="../SparseChem_Train_mini.py"

python                           ${program} \
   --data_dir                    ${datadir} \
   --output_dir                   ${outdir} \
   --x                  chembl_23mini_x.npy \
   --y_class            chembl_23mini_y.npy \
   --folding        chembl_23mini_folds.npy \
   --dev                cuda:0 \
   --fold_va                 0 \
   --fold_inputs         32000 \
   --batch_size          128   \   
   --hidden_sizes       ${layer} \
   --dropouts_trunk   ${dropout} \
   --weight_decay         1e-4 \
   --epochs          ${epochs} \
   --lr                  ${lr} \
   --lr_steps               10 \
   --lr_alpha              0.3 \
   --prefix                 sc \
   --min_samples_class       1 

echo Job Finished 

#    --batch_ratio          0.01 \
# echo PTH is $PATH
# source $PBS_O_HOME/.bashrc
# cd /data/leuven/326/vsc32647/projs/pbs
# echo PBS HOMEDIR is $PBS_O_HOMEDIR
# echo PBS WORKDIR is $PBS_O_WORKDIR