#!/usr/bin/env python
# coding: utf-8
# Copyright (c) 2020 KU Leuven

# ## Train SparseChem on Chembl_mini 
# Output to `experiments/SparseChem`

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

import argparse
import sys
import os.path
import time
import json
import functools
import types
import wandb
import pprint
import csv
import copy 
import warnings
import sparsechem as sc
from  datetime import datetime
from  contextlib import redirect_stdout
from  sparsechem import Nothing
from  sparsechem.notebook_modules import init_wandb,check_for_improvement, initialize

import scipy.io
import scipy.sparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
# from torch.serialization import SourceChangeWarning 
from pytorch_memlab import MemReporter

from pynvml import *

pp = pprint.PrettyPrinter(indent=4)
np.set_printoptions(edgeitems=3, infstr='inf', linewidth=150, nanstr='nan')
torch.set_printoptions( linewidth=132)

# os.environ['WANDB_NOTEBOOK_NAME'] = 'SparseChem_Train_mini'
#warnings.filterwarnings("ignore", category=UserWarning)    
warnings.filterwarnings("ignore", category=UserWarning)
    
if torch.cuda.is_available():
    nvmlInit()

# import multiprocessing
# multiprocessing.set_start_method('fork', force=True)

#------------------------------------------------------------------
# ### Initialization
#------------------------------------------------------------------

args = initialize() 
pp.pprint(vars(args))


#------------------------------------------------------------------
# ### Assertions
#------------------------------------------------------------------
 
if (args.last_hidden_sizes is not None) and ((args.last_hidden_sizes_class is not None) or (args.last_hidden_sizes_reg is not None)):
    raise ValueError("Head specific and general last_hidden_sizes argument were both specified!")
if (args.last_hidden_sizes is not None):
    args.last_hidden_sizes_class = args.last_hidden_sizes
    args.last_hidden_sizes_reg   = args.last_hidden_sizes

if args.last_hidden_sizes_reg is not None:
    assert len(args.last_hidden_sizes_reg) == len(args.dropouts_reg), "Number of hiddens and number of dropout values specified must be equal in the regression head!"
if args.last_hidden_sizes_class is not None:
    assert len(args.last_hidden_sizes_class) == len(args.dropouts_class), "Number of hiddens and number of dropout values specified must be equal in the classification head!"
if args.hidden_sizes is not None:
    assert len(args.hidden_sizes) == len(args.dropouts_trunk), "Number of hiddens and number of dropout values specified must be equal in the trunk!"

if args.class_feature_size == -1:
    args.class_feature_size = args.hidden_sizes[-1]
if args.regression_feature_size == -1:
    args.regression_feature_size = args.hidden_sizes[-1]

assert args.regression_feature_size <= args.hidden_sizes[-1], "Regression feature size cannot be larger than the trunk output"
assert args.class_feature_size <= args.hidden_sizes[-1], "Classification feature size cannot be larger than the trunk output"
assert args.regression_feature_size + args.class_feature_size >= args.hidden_sizes[-1], "Unused features in the trunk! Set regression_feature_size + class_feature_size >= trunk output!"
#if args.regression_feature_size != args.hidden_sizes[-1] or args.class_feature_size != args.hidden_sizes[-1]:
#    raise ValueError("Hidden spliting not implemented yet!")

assert args.input_size_freq is None, "Using tail compression not yet supported."

if (args.y_class is None) and (args.y_regr is None):
    raise ValueError("No label data specified, please add --y_class and/or --y_regr.")

#------------------------------------------------------------------
# ### Summary writer
#------------------------------------------------------------------
if args.profile == 1:
    assert (args.save_board==1), "Tensorboard should be enabled to be able to profile memory usage."
if args.save_board:
    # tb_name = os.path.join(args.output_dir, "", args.name)
    writer  = SummaryWriter(args.output_dir)
else:
    writer = Nothing()


#------------------------------------------------------------------
# ### Load datasets
#------------------------------------------------------------------
ecfp     = sc.load_sparse(args.x)
y_class  = sc.load_sparse(args.y_class)
y_regr   = sc.load_sparse(args.y_regr)
y_censor = sc.load_sparse(args.y_censor)

if (y_regr is None) and (y_censor is not None):
    raise ValueError("y_censor provided please also provide --y_regr.")
if y_class is None:
    y_class = scipy.sparse.csr_matrix((ecfp.shape[0], 0))
if y_regr is None:
    y_regr  = scipy.sparse.csr_matrix((ecfp.shape[0], 0))
if y_censor is None:
    y_censor = scipy.sparse.csr_matrix(y_regr.shape)

folding = np.load(args.folding)
assert ecfp.shape[0] == folding.shape[0], "x and folding must have same number of rows"

## Loading task weights
tasks_class = sc.load_task_weights(args.weights_class, y=y_class, label="y_class")
tasks_regr  = sc.load_task_weights(args.weights_regr, y=y_regr, label="y_regr")

#------------------------------------------------------------------
## Input transformation
#------------------------------------------------------------------
ecfp = sc.fold_transform_inputs(ecfp, folding_size=args.fold_inputs, transform=args.input_transform)
print(f"count non zero:{ecfp[0].count_nonzero()}")
num_pos    = np.array((y_class == +1).sum(0)).flatten()
num_neg    = np.array((y_class == -1).sum(0)).flatten()
num_class  = np.array((y_class != 0).sum(0)).flatten()
if (num_class != num_pos + num_neg).any():
    raise ValueError("For classification all y values (--y_class/--y) must be 1 or -1.")

num_regr   = np.bincount(y_regr.indices, minlength=y_regr.shape[1])

assert args.min_samples_auc is None, "Parameter 'min_samples_auc' is obsolete. Use '--min_samples_class' that specifies how many samples a task needs per FOLD and per CLASS to be aggregated."

if tasks_class.aggregation_weight is None:
    ## using min_samples rule
    fold_pos, fold_neg = sc.class_fold_counts(y_class, folding)
    n = args.min_samples_class
    tasks_class.aggregation_weight = ((fold_pos >= n).all(0) & (fold_neg >= n)).all(0).astype(np.float64)

if tasks_regr.aggregation_weight is None:
    if y_censor.nnz == 0:
        y_regr2 = y_regr.copy()
        y_regr2.data[:] = 1
    else:
        ## only counting uncensored data
        y_regr2      = y_censor.copy()
        y_regr2.data = (y_regr2.data == 0).astype(np.int32)
    fold_regr, _ = sc.class_fold_counts(y_regr2, folding)
    del y_regr2
    tasks_regr.aggregation_weight = (fold_regr >= args.min_samples_regr).all(0).astype(np.float64)

print(f"Input dimension: {ecfp.shape[1]}")
print(f"#samples:        {ecfp.shape[0]}")
print(f"#classification tasks:  {y_class.shape[1]}")
print(f"#regression tasks:      {y_regr.shape[1]}")
print(f"Using {(tasks_class.aggregation_weight > 0).sum()} classification tasks for calculating aggregated metrics (AUCROC, F1_max, etc).")
print(f"Using {(tasks_regr.aggregation_weight > 0).sum()} regression tasks for calculating metrics (RMSE, Rsquared, correlation).")



if args.fold_te is not None and args.fold_te >= 0:
    ## removing test data
    assert args.fold_te != args.fold_va, "fold_va and fold_te must not be equal."
    keep    = folding != args.fold_te
    ecfp    = ecfp[keep]
    y_class = y_class[keep]
    y_regr  = y_regr[keep]
    y_censor = y_censor[keep]
    folding = folding[keep]

normalize_inv = None
if args.normalize_regression == 1 and args.normalize_regr_va == 1:
   y_regr, mean_save, var_save = sc.normalize_regr(y_regr)
fold_va = args.fold_va
idx_tr  = np.where(folding != fold_va)[0]
idx_va  = np.where(folding == fold_va)[0]

y_class_tr = y_class[idx_tr]
y_class_va = y_class[idx_va]
y_regr_tr  = y_regr[idx_tr]
y_regr_va  = y_regr[idx_va]
y_censor_tr = y_censor[idx_tr]
y_censor_va = y_censor[idx_va]

if args.normalize_regression == 1 and args.normalize_regr_va == 0:
   y_regr_tr, mean_save, var_save = sc.normalize_regr(y_regr_tr) 
   if args.inverse_normalization == 1:
      normalize_inv = {}
      normalize_inv["mean"] = mean_save
      normalize_inv["var"]  = var_save
num_pos_va  = np.array((y_class_va == +1).sum(0)).flatten()
num_neg_va  = np.array((y_class_va == -1).sum(0)).flatten()
num_regr_va = np.bincount(y_regr_va.indices, minlength=y_regr.shape[1])
pos_rate = num_pos_va/(num_pos_va+num_neg_va)
pos_rate_ref = args.pi_zero
pos_rate = np.clip(pos_rate, 0, 0.99)
cal_fact_aucpr = pos_rate*(1-pos_rate_ref)/(pos_rate_ref*(1-pos_rate))

print(f"Input dimension   : {ecfp.shape[1]}")
print(f"Input dimension   : {ecfp.shape[1]}")
print(f"Training dataset  : {ecfp[idx_tr].shape}")
print(f"Validation dataset: {ecfp[idx_va].shape}")
print()
print(f"#classification tasks:  {y_class.shape[1]}")
print(f"#regression tasks    :      {y_regr.shape[1]}")
print(f"Using {(tasks_class.aggregation_weight > 0).sum():3d} classification tasks for calculating aggregated metrics (AUCROC, F1_max, etc).")
print(f"Using {(tasks_regr.aggregation_weight > 0).sum():3d} regression tasks for calculating metrics (RMSE, Rsquared, correlation).")

num_int_batches = 1
if args.batch_size is not None:
    batch_size = args.batch_size
else:
    batch_size = int(np.ceil(args.batch_ratio * idx_tr.shape[0]))

print(f"orig batch size:   {batch_size}")
print(f"orig num int batches:   {num_int_batches}")

if args.internal_batch_max is not None:
    if args.internal_batch_max < batch_size:
        num_int_batches = int(np.ceil(batch_size / args.internal_batch_max))
        batch_size      = int(np.ceil(batch_size / num_int_batches))
print(f"batch size:   {batch_size}")
print(f"num_int_batches:   {num_int_batches}")


tasks_cat_id_list = None
select_cat_ids = None
if tasks_class.cat_id is not None:
    tasks_cat_id_list = [[x,i] for i,x in enumerate(tasks_class.cat_id) if str(x) != 'nan']
    tasks_cat_ids = [i for i,x in enumerate(tasks_class.cat_id) if str(x) != 'nan']
    select_cat_ids = np.array(tasks_cat_ids)
    cat_id_size = len(tasks_cat_id_list)
else:
    cat_id_size = 0


#------------------------------------------------------------------
# ### Dataloaders
#------------------------------------------------------------------
dataset_tr = sc.ClassRegrSparseDataset(x=ecfp[idx_tr], y_class=y_class_tr, y_regr=y_regr_tr, y_censor=y_censor_tr, y_cat_columns=select_cat_ids)
dataset_va = sc.ClassRegrSparseDataset(x=ecfp[idx_va], y_class=y_class_va, y_regr=y_regr_va, y_censor=y_censor_va, y_cat_columns=select_cat_ids)

loader_tr = DataLoader(dataset_tr, batch_size=batch_size, num_workers = 8, pin_memory=True, collate_fn=dataset_tr.collate, shuffle=True)
loader_va = DataLoader(dataset_va, batch_size=batch_size, num_workers = 4, pin_memory=True, collate_fn=dataset_va.collate, shuffle=False)

args.input_size  = dataset_tr.input_size
args.output_size = dataset_tr.output_size

args.class_output_size = dataset_tr.class_output_size
args.regr_output_size  = dataset_tr.regr_output_size
args.cat_id_size = cat_id_size


#------------------------------------------------------------------
# ### WandB setup
#------------------------------------------------------------------
ns = types.SimpleNamespace()
ns.current_epoch  = 0
ns.current_iter   = 0
ns.best_results   = {}
ns.best_metrics   = None
ns.best_value     = 0 
ns.best_iter      = 0
ns.best_epoch     = 0
ns.p_epoch        = 0
ns.num_prints     = 0

init_wandb(ns, args)
wandb.define_metric("best_accuracy", summary="last")
wandb.define_metric("best_epoch", summary="last")

#------------------------------------------------------------------
# ### Network
#------------------------------------------------------------------
dev  = torch.device(args.dev)

net  = sc.SparseFFN(args).to(dev)
loss_class = torch.nn.BCEWithLogitsLoss(reduction="none")
loss_regr  = sc.censored_mse_loss

if not args.censored_loss:
    loss_regr = functools.partial(loss_regr, censored_enabled=False)

tasks_class.training_weight = tasks_class.training_weight.to(dev)
tasks_regr.training_weight  = tasks_regr.training_weight.to(dev)
tasks_regr.censored_weight  = tasks_regr.censored_weight.to(dev)

#------------------------------------------------------------------
# ###  Optimizer, Scheduler, GradScaler
#------------------------------------------------------------------
optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = MultiStepLR(optimizer, milestones=args.lr_steps, gamma=args.lr_alpha)
scaler = torch.cuda.amp.GradScaler()

wandb.watch(net, log='all', log_freq= 10)     ###  Weights and Biases Initialization 
reporter = None
h = None


#------------------------------------------------------------------
# ### setup memory profiling reporter
#------------------------------------------------------------------
if args.profile == 1:
   torch_gpu_id = torch.cuda.current_device()
   if "CUDA_VISIBLE_DEVICES" in os.environ:
      ids = list(map(int, os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")))
      nvml_gpu_id = ids[torch_gpu_id] # remap
   else:
      nvml_gpu_id = torch_gpu_id
   h = nvmlDeviceGetHandleByIndex(nvml_gpu_id)

if args.profile == 1:
   #####   output saving   #####
   if not os.path.exists(args.output_dir):
       os.makedirs(args.output_dir)

   reporter = MemReporter(net)

   with open(f"{args.output_dir}/memprofile.txt", "w+") as profile_file:
        with redirect_stdout(profile_file):
             profile_file.write(f"\nInitial model detailed report:\n\n")
             reporter.report()


#------------------------------------------------------------------
# ### Display network and other values
#------------------------------------------------------------------
print("Network:")
print(net)
print(optimizer)
print(f"dev                  :    {dev}")
print(f"args.lr              :    {args.lr}")
print(f"args.weight_decay    :    {args.weight_decay}")
print(f"args.lr_steps        :    {args.lr_steps}")
print(f"args.lr_steps        :    {args.lr_steps}")
print(f"num_int_batches      :    {num_int_batches}")
print(f"batch_size           :    {batch_size}")
print(f"EPOCHS               :    {args.epochs}")
print(f"scaler               :    {scaler}")
print(f"args.normalize_loss  :    {args.normalize_loss}")
print(f"loss_class           :    {loss_class}")
print(f"mixed precision      :    {args.mixed_precision}")
print(f"args.eval_train      :    {args.eval_train}")
 


#------------------------------------------------------------------
# ##  Training Loop
#------------------------------------------------------------------
ns.end_epoch = ns.current_epoch + args.epochs

for ns.current_epoch in range(ns.current_epoch, ns.end_epoch, 1):
    t0 = time.time()
    sc.train_class_regr(
        net, optimizer,
        loader          = loader_tr,
        loss_class      = loss_class,
        loss_regr       = loss_regr,
        dev             = dev,
        weights_class   = tasks_class.training_weight * (1-args.regression_weight) * 2,
        weights_regr    = tasks_regr.training_weight * args.regression_weight * 2,
        censored_weight = tasks_regr.censored_weight,
        normalize_loss  = args.normalize_loss,
        num_int_batches = num_int_batches,
        progress        = False,
        writer          = writer,
        epoch           = ns.current_epoch,
        args            = args,
        scaler          = scaler,
        nvml_handle     = h)

    if args.profile == 1:
       with open(f"{args.output_dir}/memprofile.txt", "a+") as profile_file:
            profile_file.write(f"\nAfter epoch {epoch} model detailed report:\n\n")
            with redirect_stdout(profile_file):
                 reporter.report()

    t1 = time.time()
    eval_round = (args.eval_frequency > 0) and ((ns.current_epoch + 1) % args.eval_frequency == 0)
    last_round = ns.current_epoch == args.epochs - 1

    if eval_round or last_round:

        results_va = sc.evaluate_class_regr(net, loader_va, loss_class, loss_regr, 
                                            tasks_class= tasks_class, 
                                            tasks_regr = tasks_regr, 
                                            dev        = dev, 
                                            progress   = False, 
                                            normalize_inv=normalize_inv, 
                                            cal_fact_aucpr=cal_fact_aucpr)
        
        for key, val in results_va["classification_agg"].items():
            writer.add_scalar("val_metrics:aggregated/"+key, val, ns.current_epoch * batch_size)


        if args.eval_train:
            results_tr = sc.evaluate_class_regr(net, loader_tr, loss_class, loss_regr, 
                                                tasks_class = tasks_class, 
                                                tasks_regr  = tasks_regr, 
                                                dev         = dev, 
                                                progress    = args.verbose >= 2)
            for key, val in results_tr["classification_agg"].items():
                writer.add_scalar("trn_metrics:aggregated/"+key, val, ns.current_epoch * batch_size)

        else:
            results_tr = None

        if args.verbose:
            ## printing a new header every 20 lines
            header = ns.num_prints % 20 == 0
            ns.num_prints += 1
            sc.print_metrics_cr(ns.current_epoch, t1 - t0, results_tr, results_va, header)
        wandb.log(results_va["classification_agg"].to_dict())
        check_for_improvement(ns, results_va)
    
    scheduler.step()

#print("DEBUG data for hidden spliting")
#print (f"Classification mask: Sum = {net.classmask.sum()}\t Uniques: {np.unique(net.classmask)}")
#print (f"Regression mask:     Sum = {net.regmask.sum()}\t Uniques: {np.unique(net.regmask)}")
#print (f"overlap: {(net.regmask * net.classmask).sum()}")


print(f"Best Epoch :       {ns.best_epoch}\n"
      f"Best Iteration :   {ns.best_iter} \n"
      f"Best Precision :   {ns.best_value:.5f}\n")

pp.pprint(results_va['classification_agg'].to_dict())


#------------------------------------------------------------------
# ## Post Training 
#------------------------------------------------------------------
writer.close()
print()
if args.profile == 1:
   multiplexer = sc.create_multiplexer(tb_name)
#   sc.export_scalars(multiplexer, '.', "GPUmem", "testcsv.csv")
   data = sc.extract_scalars(multiplexer, '.', "GPUmem")
   print(f"Peak GPU memory used: {sc.return_max_val(data)}MB")
print("Saving performance metrics (AUCs) and model.")

#####   model saving   #####
if not os.path.exists(args.output_dir):
   os.makedirs(args.output_dir)

model_file = f"{args.output_dir}/{args.name}.pt"
out_file   = f"{args.output_dir}/{args.name}.json"

if args.save_model:
   torch.save(net.state_dict(), model_file)
   print(f"Saved model weights into '{model_file}'.")

results_va["classification"]["num_pos"] = num_pos_va
results_va["classification"]["num_neg"] = num_neg_va
results_va["regression"]["num_samples"] = num_regr_va

if results_tr is not None:
    results_tr["classification"]["num_pos"] = num_pos - num_pos_va
    results_tr["classification"]["num_neg"] = num_neg - num_neg_va
    results_tr["regression"]["num_samples"] = num_regr - num_regr_va

stats=None
if args.normalize_regression == 1 :
   stats={}
   stats["mean"] = mean_save
   stats["var"]  = np.array(var_save)[0]
sc.save_results(out_file, args, validation=results_va, training=results_tr, stats=stats)

print(f"Saved config and results into '{out_file}'.\nYou can load the results by:\n  import sparsechem as sc\n  res = sc.load_results('{out_file}')")
print()
print(results_va['classification'][0:20])
pp.pprint(results_va)
print()
pp.pprint(results_va['classification_agg'])
print()
df = results_va['classification']
print(df[pd.notna(df.roc_auc_score)])

ns.wandb_run.finish()
 