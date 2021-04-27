#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_resnet.py: This file contains function to score 
a trained resnet on various trials.
"""

__author__ = "Duret Jarod, Brignatz Vincent"
__license__ = "MIT"

import torch
import numpy as np

from collections import OrderedDict
from sklearn.metrics import roc_curve
from sklearn.metrics.pairwise import paired_distances
from sklearn.preprocessing import normalize
from tqdm import tqdm
from math import log10, floor
from pathlib import Path
from loguru import logger

import dataset
import data_io
from models import resnet34

# @logger.catch
def compute_unique_utt_xvec(generator, trial_ds, device=torch.device("cuda")):
    """
        TODO Extract the x-vectors only for sessions required by trial.
    """
    # set the model in eval mode
    generator.eval()

    all_xv = {}

    with torch.no_grad():
        for utt, feats in trial_ds.unique_feats.items():
            feats = feats.unsqueeze(0).unsqueeze(1)
            xv = generator(feats.to(device)).cpu().numpy()
            all_xv[utt] = xv
    
    # set the model in train mode
    generator.train()

    return all_xv

def eer_from_ers(fpr, tpr):
    fnr = 1 - tpr
    eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    return eer

def compute_min_dcf(fpr, tpr, thresholds, p_target=0.01, c_miss=1, c_fa=1):
    # adapted from compute_min_dcf.py in kaldi sid
    # thresholds, fpr, tpr = list(zip(*sorted(zip(thresholds, fpr, tpr))))
    incr_score_indices = np.argsort(thresholds, kind="mergesort")
    thresholds = thresholds[incr_score_indices]
    fpr = fpr[incr_score_indices]
    tpr = tpr[incr_score_indices]

    fnr = 1. - tpr
    min_c_det = float("inf")
    for i in range(0, len(fnr)):
        c_det = c_miss * fnr[i] * p_target + c_fa * fpr[i] * (1 - p_target)
        if c_det < min_c_det:
            min_c_det = c_det

    c_def = min(c_miss * p_target, c_fa * (1 - p_target))
    min_dcf = min_c_det / c_def
    return min_dcf

def score_utt_utt(generator, ds_test, mindcf=False):
    """ 
        Score the model on the trials of type :
        <spk> <utt> 0/1
    """

    utt2xv = compute_unique_utt_xvec(generator, ds_test)

    targets = np.asarray(ds_test.tar_l, dtype=int)

    xv = np.vstack(list(utt2xv.values()))
    xv = normalize(xv, axis=1)

    utt2xv_norm = {k:v for k, v in zip(utt2xv.keys(), xv)}

    emb0 = np.array([utt2xv_norm[k] for k in ds_test.utt1_l])
    emb1 = np.array([utt2xv_norm[k] for k in ds_test.utt2_l])

    scores = paired_distances(emb0, emb1, metric='cosine')
    fpr, tpr, thresholds = roc_curve(1 - targets, scores, pos_label=1, drop_intermediate=False)
    eer = eer_from_ers(fpr, tpr)*100

    if mindcf:
        mindcf1 = compute_min_dcf(fpr, tpr, thresholds, p_target=0.01)
        mindcf2 = compute_min_dcf(fpr, tpr, thresholds, p_target=0.001)
        print(f'EER :{eer:.4f}%  minDFC p=0.01 :{mindcf1}  minDFC p=0.001 :{mindcf2}  ')
        res = {"eer":eer, "mindcf1":mindcf1, "mindcf2":mindcf2}
    else:
        print(f'EER :{eer:.4f}%')
        res = {"eer":eer}
    return res
