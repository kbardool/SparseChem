#!/bin/bash 
# -l nodes=1:ppn=9,walltime=01:00:00
# -A lp_symbiosys

# module purge
export JOBID=${PBS_JOBID:0:8}
echo Job $JOBID start : $(date)
source /user/leuven/326/vsc32647/.initconda
cd $PBS_O_WORKDIR # cd to the directory from which qsub is run
echo PBS VERSION is $PBS_VERSION
echo config file is $config
echo switch to pyt-gpu 
conda activate pyt-gpu
python -V
export program="../SparseChem_Train_mini.py"

python                           ${program} \
   --data_dir                    ${datadir} \
   --output_dir                   ${outdir} \
   --exp_desc            ${JOBID} - SparseChem Train  \
   --x                  chembl_23mini_x.npy \
   --y_class            chembl_23mini_y.npy \
   --folding        chembl_23mini_folds.npy \
   --dev                   cpu \
   --fold_va                 0 \
   --fold_inputs         32000 \
   --batch_size          128   \
   --hidden_sizes       ${layer}   ${layer}    ${layer}   ${layer}\
   --dropouts_trunk   ${dropout} ${dropout}  ${dropout} ${dropout}\
   --weight_decay         1e-4 \
   --epochs          ${epochs} \
   --lr                  ${lr} \
   --lr_steps               10 \
   --lr_alpha              0.3 \
   --prefix                 sc \
   --min_samples_class       1 

echo Job $JOBID finished : $(date)