#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
xvector_scoring.py: This file is used to score xvectors according to a trial file.
Exemple
python xvector_scoring.py my_trial xvectors1.scp xvectors2.scp
"""

__author__ = "Duret Jarod, Brignatz Vincent"
__license__ = "MIT"


import numpy as np
import argparse
from kaldi_io import read_vec_flt
from sklearn.preprocessing import normalize

from data_io import load_n_col
from test_resnet import extract_and_score, score_xvectors

if __name__ == "__main__":

    # ARGUMENTS PARSING
    parser = argparse.ArgumentParser(description='Test xvectors according to trial')

    parser.add_argument('trial', type=str, help="the trial")
    parser.add_argument("xvectors", type=str, nargs="+", help="the scp xvectors files")

    args = parser.parse_args()

    # load xvectors
    utt2xv = {}
    for scp in args.xvectors:
        scp = load_n_col(scp)
        for utt, path in zip(scp[0], scp[1]):
            utt2xv[utt] = read_vec_flt(path)

    xv = np.vstack(list(utt2xv.values()))
    xv = normalize(xv, axis=1)
    utt2xv_norm = {k:v for k, v in zip(utt2xv.keys(), xv)}

    # load targets
    trial = load_n_col(args.trial)
    targets = np.array(trial[0])
    enrollments = np.array([utt2xv_norm[utt] for utt in trial[1]])
    tests       = np.array([utt2xv_norm[utt] for utt in trial[2]])

    score_xvectors(enrollments, tests, targets)
