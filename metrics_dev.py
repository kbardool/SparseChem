
import sklearn.metrics
# from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
import scipy.sparse
import scipy.io
import scipy.special
# import types
# import json
# import warnings
# import math
import torch.nn.functional as F
# import csv
# from pynvml import *
# from contextlib import redirect_stdout
# from sparsechem import censored_mse_loss_numpy
# from collections import namedtuple
# from scipy.sparse import csr_matrix


def compute_metrics(cols, y_true, y_score, num_tasks, cal_fact_aucpr):
    if len(cols) < 1:
        return pd.DataFrame({
            "roc_auc_score": np.nan,
            "auc_pr": np.nan,
            "avg_prec_score": np.nan,
            "f1_max": np.nan,
            "p_f1_max": np.nan,
            "kappa": np.nan,
            "kappa_max": np.nan,
            "p_kappa_max": np.nan,
            "bceloss": np.nan}, index=np.arange(num_tasks))
    df   = pd.DataFrame({"task": cols, "y_true": y_true, "y_score": y_score})
    if hasattr(cal_fact_aucpr, "__len__"):
        metrics = df.groupby("task", sort=True).apply(lambda g:
              all_metrics(
                  y_true  = g.y_true.values,
                  y_score = g.y_score.values,
                  cal_fact_aucpr_task = cal_fact_aucpr[g['task'].values[0]]))
    else:
        metrics = df.groupby("task", sort=True).apply(lambda g:
              all_metrics(
                  y_true  = g.y_true.values,
                  y_score = g.y_score.values,
                  cal_fact_aucpr_task = 1.0))
    metrics.reset_index(level=-1, drop=True, inplace=True)
    return metrics.reindex(np.arange(num_tasks))


def all_metrics(y_true, y_score, cal_fact_aucpr_task):
    """Compute classification metrics.
    Args:
        y_true     true labels (0 / 1)
        y_score    logit values
    """
    if len(y_true) <= 1 or (y_true[0] == y_true).all():
        df = pd.DataFrame({"roc_auc_score": [np.nan], "auc_pr": [np.nan], "avg_prec_score": [np.nan], "f1_max": [np.nan], "p_f1_max": [np.nan], "kappa": [np.nan], "kappa_max": [np.nan], "p_kappa_max": [np.nan], "bceloss": [np.nan], "auc_pr_cal": [np.nan]})
        return df

    fpr, tpr, tpr_thresholds = sklearn.metrics.roc_curve(y_true=y_true, y_score=y_score)

    roc_auc_score = sklearn.metrics.auc(x=fpr, y=tpr)

    precision, recall, pr_thresholds = sklearn.metrics.precision_recall_curve(y_true = y_true, probas_pred = y_score)

    with np.errstate(divide='ignore'):
         #precision can be zero but can be ignored so disable warnings (divide by 0)
         precision_cal = 1/(((1/precision - 1)*cal_fact_aucpr_task)+1)

    ## BCE Loss computation
    bceloss = F.binary_cross_entropy_with_logits(input  = torch.FloatTensor(y_score),
                                                 target = torch.FloatTensor(y_true),
                                                 reduction="none").mean().item()

    ## calculating F1 for all cutoffs
    F1_score       = np.zeros(len(precision))
    mask           = precision > 0
    F1_score[mask] = 2 * (precision[mask] * recall[mask]) / (precision[mask] + recall[mask])
    f1_max_idx     = F1_score.argmax()
    f1_max         = F1_score[f1_max_idx]
    p_f1_max       = scipy.special.expit(pr_thresholds[f1_max_idx])

    auc_pr = sklearn.metrics.auc(x = recall, y = precision)
    auc_pr_cal = sklearn.metrics.auc(x = recall, y = precision_cal)

    avg_prec_score = sklearn.metrics.average_precision_score(y_true  = y_true,
                                                            y_score = y_score)
    y_classes = np.where(y_score >= 0.0, 1, 0)

    ## accuracy for all thresholds
    acc, kappas   = calc_acc_kappa(recall=tpr, fpr=fpr, num_pos=(y_true==1).sum(), num_neg=(y_true==0).sum())
    kappa_max_idx = kappas.argmax()
    kappa_max     = kappas[kappa_max_idx]
    p_kappa_max   = scipy.special.expit(tpr_thresholds[kappa_max_idx])

    kappa = sklearn.metrics.cohen_kappa_score(y_true, y_classes)
    df = pd.DataFrame({"roc_auc_score": [roc_auc_score], "auc_pr": [auc_pr], "avg_prec_score": [avg_prec_score], "f1_max": [f1_max], "p_f1_max": [p_f1_max], "kappa": [kappa], "kappa_max": [kappa_max], "p_kappa_max": p_kappa_max, "bceloss": bceloss, "auc_pr_cal": [auc_pr_cal]})
    return df



def calc_acc_kappa(recall, fpr, num_pos, num_neg):
    """Calculates accuracy from recall and precision."""
    num_all = num_neg + num_pos
    tp = np.round(recall * num_pos).astype(np.int)
    fn = num_pos - tp
    fp = np.round(fpr * num_neg).astype(np.int)
    tn = num_neg - fp
    acc   = (tp + tn) / num_all
    pexp  = num_pos / num_all * (tp + fp) / num_all + num_neg / num_all * (tn + fn) / num_all
    kappa = (acc - pexp) / (1 - pexp)
    return acc, kappa
