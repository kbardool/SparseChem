import os
import argparse
import sys
import os.path
import time
import json
import functools
# import multiprocessing
import types
import wandb
from datetime import datetime
import pprint
import numpy as np
# import csv
# import copy 
# from contextlib import redirect_stdout
# import warnings
# import sparsechem as sc
# from sparsechem import Nothing

def vprint(s="", verbose = False):
    if verbose:
        print(s)


def print_separator(text, total_len=50):
    print('#' * total_len)
    text = f" {text} "
    text_len = len(text)
    left_width = (total_len - text_len )//2
    right_width = total_len - text_len - left_width
    print("#" * left_width + text + "#" * right_width)
    print('#' * total_len)

def print_dbg(text, verbose = False):
    if verbose:
        print(text)

# @debug_off
def print_heading(text,  verbose = False, force = False, out=[sys.stdout]):
    len_ttl = max(len(text)+4, 50)
    len_ttl = min(len_ttl, 120)
    if verbose or force: 
        for file in out:
            print('-' * len_ttl, file = file)
            print(f"{text}", file = file)
            # left_width = (total_len - len(text))//2
            # right_width = total_len - len(text) - left_width
            # print("#" * left_width + text + "#" * right_width)
            print('-' * len_ttl, '\n', file=file)



def print_underline(text,  verbose = False, out=[sys.stdout]):
    len_ttl = len(text)+2
    if verbose:
        for file in out:
            print(f"\n {text}", file=file)
            print('-' * len_ttl, file=file)

def init_wandb(ns, args, resume = "allow" ):

    # opt['exp_id'] = wandb.util.generate_id()
    print(args.exp_id, args.exp_name, args.project_name) # , opt['exp_instance'])
    ns.wandb_run = wandb.init(project = args.project_name, 
                              entity  = "kbardool", 
                              id      = args.exp_id, 
                              name    = args.exp_name,
                              notes   = args.exp_desc,                                     
                              resume  = resume )
    wandb.config.update(args)

    print(f" PROJECT NAME: {ns.wandb_run.project}\n"
          f" RUN ID      : {ns.wandb_run.id} \n"
          f" RUN NAME    : {ns.wandb_run.name}")     
    return 



def check_for_improvement(ns, metrics):
    #------------------------------------------------------------------------ 
    #  Save Best Checkpoint Code (saved below and in sparsechem_env_dev.py)
    #----------------------------------------------------------------------- 
    ## ns.curriculum_epochs = (environ.num_layers * opt['curriculum_speed']) 

    if  metrics["classification_agg"]['avg_prec_score'] > ns.best_value:
        print('Previous best_epoch: %5d   best iter: %5d,   best_value: %.5f' % (ns.best_epoch, ns.best_iter, ns.best_value))        
        ns.best_value   = metrics['classification_agg']['avg_prec_score']
        ns.best_metrics = metrics
        ns.best_epoch   = ns.current_epoch
        print('New      best_epoch: %5d   best iter: %5d,   best_value: %.5f' % (ns.best_epoch, ns.best_iter, ns.best_value))  
        wandb.log({"best_accuracy": ns.best_value,
                   "best_epoch"   : ns.best_epoch})               
#         model_label     = 'model_best_seed_%04d' % (opt['random_seed'])
#         environ.save_checkpoint(model_label, ns.current_iter, ns.current_epoch) 
#         metrics_label = 'metrics_best_seed_%04d.pickle' % (opt['random_seed'])
#         save_to_pickle(environ.val_metrics, environ.opt['paths']['checkpoint_dir'], metrics_label)    
    return

def get_command_line_args(input = None, display = True):
    parser = argparse.ArgumentParser(description="Training a multi-task model.")
    parser.add_argument("--data_dir", help="Input data directory", type=str, default=None, required = True)
    parser.add_argument("--output_dir", help="Output directory, including boards (default 'models')", type=str, default=None, required=True)
    parser.add_argument("--x", help="Descriptor file (matrix market, .npy or .npz)", type=str, default=None)
    parser.add_argument("--y_class", "--y"   , type=str,   help="Activity file (matrix market, .npy or .npz)", default=None)
    parser.add_argument("--project_name"     , type=str,   help="Project name used by wandb ", required=True)

    parser.add_argument("--exp_id"           , type=str,   help="experiment unqiue id, used by wandb - defaults to wandb.util.generate_id()")
    parser.add_argument("--exp_name"         , type=str,   help="experiment name, used as folder prefix and wandb name, defaults to mmdd_hhmm")
    parser.add_argument("--exp_desc"         , type=str,   nargs='+', default=[] , help="experiment description")
    parser.add_argument("--folder_sfx"       , type=str,   help="experiment folder suffix, defaults to None")

    parser.add_argument("--hidden_sizes", nargs="+", help="Hidden sizes of trunk", default=[], type=int, required=True)
    parser.add_argument("--dropouts_trunk", nargs="+", help="List of dropout values used in the trunk", default=[], type=float)    
    parser.add_argument("--class_feature_size", help="Number of leftmost features used from the output of the trunk (default: use all)", type=int, default=-1)
    parser.add_argument("--last_hidden_sizes", nargs="+", help="Hidden sizes in the head (if specified , class and reg heads have this dimension)", default=None, type=int)

    parser.add_argument("--epochs", help="Number of epochs", type=int, default=20)
    parser.add_argument("--batch_size",        help="Batchsize - default read from config file", type=int, default=None)

    parser.add_argument("--weight_decay", help="Weight decay", type=float, default=0.0)
    parser.add_argument("--last_non_linearity", help="Last layer non-linearity (depecrated)", type=str, default="relu", choices=["relu", "tanh"])
    parser.add_argument("--middle_non_linearity", "--non_linearity", help="Before last layer non-linearity", type=str, default="relu", choices=["relu", "tanh"])
    parser.add_argument("--input_transform", help="Transformation to apply to inputs", type=str, default="none", choices=["binarize", "none", "tanh", "log1p"])
    parser.add_argument("--lr", help="Learning rate", type=float, default=1e-3)
    parser.add_argument("--lr_alpha", help="Learning rate decay multiplier", type=float, default=0.3)
    parser.add_argument("--lr_steps", nargs="+", help="Learning rate decay steps", type=int, default=[10])

    parser.add_argument("--weights_class", "--task_weights", "--weights_classification", help="CSV file with columns task_id, training_weight, aggregation_weight, task_type (for classification tasks)", type=str, default=None)
    parser.add_argument("--weights_regr", "--weights_regression", help="CSV file with columns task_id, training_weight, censored_weight, aggregation_weight, aggregation_weight, task_type (for regression tasks)", type=str, default=None)
    parser.add_argument("--fold_va", help="Validation fold number", type=int, default=0)
    parser.add_argument("--fold_te", help="Test fold number (removed from data, type=strset)", type=int, default=None)
    parser.add_argument("--batch_ratio", help="Batch ratio", type=float, default=0.02)
    parser.add_argument("--internal_batch_max", help="Maximum size of the internal batch", type=int, default=None)

    parser.add_argument("--censored_loss", help="Whether censored loss is used for training (default 1)", type=int, default=1)
    parser.add_argument("--folding", help="Folding file (npy)", type=str, required=True)
    parser.add_argument("--y_regr", "--y_regression", type=str   ,   help="Activity file (matrix market, .npy or .npz)", default=None)
    parser.add_argument("--y_censor"         , type=str,   help="Censor mask for regression (matrix market, .npy or .npz)", default=None)

    parser.add_argument("--normalize_loss", help="Normalization constant to divide the loss (default uses batch size)", type=float, default=None)
    parser.add_argument("--normalize_regression", help="Set this to 1 if the regression tasks should be normalized", type=int, default=0)
    parser.add_argument("--normalize_regr_va", help="Set this to 1 if the regression tasks in validation fold should be normalized together with training folds", type=int, default=0)
    parser.add_argument("--inverse_normalization", help="Set this to 1 if the regression tasks in validation fold should be inverse normalized at validation time", type=int, default=0)

    #parser.add_argument("--middle_dropout", help="Dropout for layers before the last", type=float, default=0.0)
    #parser.add_argument("--last_dropout", help="Last dropout", type=float, default=0.2)
    parser.add_argument("--input_size_freq",   help="Number of high importance features", type=int, default=None)
    parser.add_argument("--fold_inputs",       help="Fold input to a fixed set (default no folding)", type=int, default=None)
    parser.add_argument("--pi_zero",           help="Reference class ratio to be used for calibrated aucpr", type=float, default=0.1)
    parser.add_argument("--min_samples_class", help="Minimum number samples in each class and in each fold for AUC calculation (only used if aggregation_weight is not provided in --weights_class)", type=int, default=5)
    parser.add_argument("--min_samples_auc",   help="Obsolete: use 'min_samples_class'", type=int, default=None)
    parser.add_argument("--min_samples_regr",  help="Minimum number of uncensored samples in each fold for regression metric calculation (only used if aggregation_weight is not provided in --weights_regr)", type=int, default=10)
    parser.add_argument("--dev",               help="Device to use", type=str, default="cuda:0")
    parser.add_argument("--run_name",          help="Run name for results", type=str, default=None)
    parser.add_argument("--prefix",            help="Prefix for run name (default 'run')", type=str, default='run')
    parser.add_argument("--verbose",           help="Verbosity level: 2 = full; 1 = no progress; 0 = no output", type=int, default=2, choices=[0, 1, 2])
    parser.add_argument("--save_model",        help="Set this to 0 if the model should not be saved", type=int, default=1)
    parser.add_argument("--save_board",        help="Set this to 0 if the TensorBoard should not be saved", type=int, default=1)
    parser.add_argument("--profile",           help="Set this to 1 to output memory profile information", type=int, default=0)
    parser.add_argument("--mixed_precision",   help="Set this to 1 to run in mixed precision mode (vs single precision)", type=int, default=0)
    parser.add_argument("--eval_train",        help="Set this to 1 to calculate AUCs for train data", type=int, default=0)
    parser.add_argument("--enable_cat_fusion", help="Set this to 1 to enable catalogue fusion", type=int, default=0)
    parser.add_argument("--eval_frequency",    help="The gap between AUC eval (in epochs), -1 means to do an eval at the end.", type=int, default=1)
    parser.add_argument("--regression_weight", help="between 0 and 1 relative weight of regression loss vs classification loss", type=float, default=0.5)
    parser.add_argument("--scaling_regularizer", help="L2 regularizer of the scaling layer, if inf scaling layer is switched off", type=float, default=np.inf)
    #hybrid model features
    parser.add_argument("--regression_feature_size", help="Number of rightmost features used from the output of the trunk (default: use all)", type=int, default=-1)
    parser.add_argument("--last_hidden_sizes_reg", nargs="+", help="Hidden sizes in the regression head (overwritten by last_hidden_sizes)", default=None, type=int)
    parser.add_argument("--last_hidden_sizes_class", nargs="+", help="Hidden sizes in the classification head (overwritten by last_hidden_sizes)", default=None, type=int)
    parser.add_argument("--dropouts_reg"  , nargs="+", help="List of dropout values used in the regression head (needs one per last hidden in reg head, ignored if last_hidden_sizes_reg not specified)", default=[], type=float)
    parser.add_argument("--dropouts_class", nargs="+", help="List of dropout values used in the classification head (needs one per last hidden in class head, ignored if no last_hidden_sizes_class not specified)", default=[], type=float)

    if input is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(input)
    
    args.exp_desc = ' '.join(str(e) for e in args.exp_desc)
    
    if display:
        print_underline(' command line parms : ', True)
        for key, val in vars(args).items():
            print(f" {key:.<25s}  {val}")
        print('\n\n')
    return args 

def initialize(input_args = None):

    input_args = input_args.split() if input_args is not None else input_args
    args = get_command_line_args(input_args, display = True)
    # args = parser.parse_args()

    rstr = datetime.now().strftime("%m%d_%H%M")
    args.x       = os.path.join(args.data_dir, args.x)
    args.y_class = os.path.join(args.data_dir, args.y_class)
    args.folding = os.path.join(args.data_dir, args.folding)
    args.num_hdn_layers = len(args.hidden_sizes) -1
    args.hdn_layer_size = args.hidden_sizes[0]
    dir_name      = f"{args.hdn_layer_size}x{args.num_hdn_layers}_{rstr}_lr{args.lr}_do{args.dropouts_trunk[0]}"
    args.output_dir = os.path.join(args.output_dir, dir_name)
    print(args.output_dir)
    print(args.x)
    print(args.y_class)
    print(args.folding)
    print(args.num_hdn_layers, args.hdn_layer_size)


    # dev = "gpu" 
    # rstr = datetime.now().strftime("%m%d_%H%M")
    # data_dir = "../MLDatasets/chembl23_mini"
    # output_dir = "../experiments/mini-SparseChem"
    # print(output_dir)
    # rm_output=False

    if args.exp_id is None:
        args.exp_id = wandb.util.generate_id()

    if args.exp_name is None:
        args.exp_name = rstr 

    if args.folder_sfx is not None:
        args.exp_name  += f"_{args.folder_sfx}"
        
    if args.run_name is not None:
        args.name = args.run_name
    else:
        args.name  = f"{args.prefix}"
        args.name += f"_{'.'.join([str(h) for h in args.hidden_sizes])}"
    #     name += f"_do{'.'.join([str(d) for d in args.dropouts_trunk])}"
        args.name += f"_lr{args.lr}"
        args.name += f"_do{args.dropouts_trunk[0]}"
    #     name += f"_wd{args.weight_decay}"
    #     name += f"_hs{'.'.join([str(h) for h in args.hidden_sizes])}"
        
    #     name += f"_lrsteps{'.'.join([str(s) for s in args.lr_steps])}_ep{args.epochs}"
    #     name += f"_fva{args.fold_va}_fte{args.fold_te}"
        if args.mixed_precision == 1:
            args.name += f"_mixed_precision"
    print(f"Run name is '{args.name}'.")

    # if args.run_name is not None:
    #     name = args.run_name
    # else:
    #     name  = f"sc_{args.prefix}_h{'.'.join([str(h) for h in args.hidden_sizes])}_ldo_r{'.'.join([str(d) for d in args.dropouts_reg])}_wd{args.weight_decay}"
    #     name += f"_lr{args.lr}_lrsteps{'.'.join([str(s) for s in args.lr_steps])}_ep{args.epochs}"
    #     name += f"_fva{args.fold_va}_fte{args.fold_te}"
    #     if args.mixed_precision == 1:
    #         name += f"_mixed_precision"
    # vprint(f"Run name is '{name}'.")
    return args


def assertions(args):

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

    print("All assertions passed successfully")