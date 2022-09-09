#!/bin/bash 
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

echo program excution start: $(date)
layers=""
dropouts=""
program="../SparseChem_Train.py"

echo "Num Layers: $num_layers   Layer size: $layers   Dropout: $dropouts  Task LR: $lr  device: $dev"

echo "Project name: $project_name" 
echo "batch size  : $batch_size"
echo "x file      : $x_file"
echo "y file      : $y_file"
echo "folding file: $fold_file"
echo "config      : $config"
echo "data dir    : $datadir"
echo "out  dir    : $outdir"
## SparseChem needs a list of Layers + 1 , eg. 100 x 2 hidden layers : [100 100 100] 
for ((i=0 ; i <= $num_layers ; i +=1)); do
    layers+=" $layer "
    dropouts+=" $dropout "
done

echo "Num Layers: $num_layers   Layer size: $layers   Dropout: $dropouts  Task LR: $lr  device: $dev"


python                               ${program} \
   --data_dir                        ${datadir} \
   --output_dir                       ${outdir} \
   --project_name               ${project_name} \
   --exp_desc       ${JOBID} - SparseChem Train \
   --x                                ${x_file} \
   --y_class                          ${y_file} \
   --folding                       ${fold_file} \
   --dev                                 ${dev} \
   --fold_va                                  0 \
   --fold_te                                  1 \
   --fold_inputs                          32000 \
   --batch_size                   ${batch_size} \
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