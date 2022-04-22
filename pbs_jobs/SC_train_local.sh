#!/bin/bash 
# module purge
export JOBID=00000001
echo Job $JOBID start : $(date)
# source /user/leuven/326/vsc32647/.initconda
# cd $PBS_O_WORKDIR # cd to the directory from which qsub is run
# echo PBS VERSION is $PBS_VERSION
echo config file is $config
echo switch to pyt-gpu 
conda activate pyt-gpu
python -V

echo program excution start: $(date)

layers=""
dropouts=""
echo "Number Layers: $num_layers  of Layer size: $layer   Dropout: $dropout  Task LR: $lr "

## SparseChem needs a list of Layers + 1 , eg. 100 x 2 hidden layers : [100 100 100] 
for ((i=0 ; i <= $num_layers ; i +=1)); do
    layers+=" $layer "
    dropouts+=" $dropout "
done
echo "Number Layers: $num_layers   Layer size: $layers   Dropout: $dropouts  Task LR: $lr  device: $dev"


python                               ${program} \
   --data_dir                        ${datadir} \
   --output_dir                       ${outdir} \
   --exp_desc       ${JOBID} - SparseChem Train \
   --x                      chembl_23mini_x.npy \
   --y_class                chembl_23mini_y.npy \
   --folding            chembl_23mini_folds.npy \
   --dev                                 ${dev} \
   --fold_va                                  0 \
   --fold_inputs                          32000 \
   --batch_size                             128 \
   --hidden_sizes                     ${layers} \
   --dropouts_trunk                 ${dropouts} \
   --weight_decay                          1e-4 \
   --epochs                           ${epochs} \
   --lr                                   ${lr} \
   --lr_steps                                10 \
   --lr_alpha                               0.3 \
   --prefix                                  sc \
   --save_model                               0 \
   --min_samples_class                        2 

echo Job $JOBID finished : $(date)