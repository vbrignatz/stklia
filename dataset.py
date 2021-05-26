#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
dataset.py: This file contains funtions and class of our speaker dataset.
"""

__author__ = "Duret Jarod, Brignatz Vincent"
__license__ = "MIT"

from pathlib import Path

import data_io

import torch
import numpy as np
from kaldi_io import read_mat
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder

class Trial:
    def __init__(self, name, targets, enrolls, tests):
        self.name = name
        self.targets = targets
        self.enrolls = enrolls
        self.tests = tests

        self.uniq_utts = set(enrolls).union(set(tests))

    def __getitem__(self, idx):
        return self.targets[idx], self.enrolls[idx], self.tests[idx]

    def __len__(self):
        return len(self.targets)

class TrialDataset(Dataset):
    def __init__(self, utt2path, loading_method, trial_files):
        self.loading_method = loading_method
        self.utt2path = utt2path
        if not isinstance(trial_files, list):
            trial_files = [trial_files]

        self.trials = []
        for tf in trial_files:
            tar, enroll, test = data_io.load_n_col(tf)
            self.trials.append(Trial(tf.stem, tar, enroll, test))

        # print("Loading Trials ...")
        self.unique_feats = self.load_unique_feats()

    def __len__(self):
        return len(self.trials)

    def __getitem__(self, idx):
        return self.trials[idx]

    # def iterate_over_trial(self, idx):
    #     for i in range(len(self.trials[idx])):
    #         tar, enroll, test = self.trials[idx][i]
    #         yield tar, self.unique_feats[enroll], self.unique_feats[test]

    def load_unique_feats(self):
        unique_utt = set()
        for tr in self.trials:
            unique_utt.update(tr.uniq_utts)

        unique_feats = {}
        for utt in unique_utt:
            unique_feats[utt] = torch.FloatTensor(read_mat(self.utt2path[utt]))
        return unique_feats

# Dataset class
class SpeakerDataset(Dataset):
    """ Characterizes a dataset for Pytorch """
    def __init__(self, utt2path, utt2spk, spk2utt, loading_method, seq_len=None):

        # is_noised = lambda x : x != utt2uniq(x)
        # spk2utt = {k:[utt for utt in v if not is_noised(utt)] for k, v in spk2utt.items()}

        self.utt2path = utt2path
        self.loading_method = loading_method
        self.seq_len = seq_len

        speakers = list(spk2utt.keys())
        spkrs_utt_sorted = list(utt2spk.values())

        label_enc = LabelEncoder()
        speakers = label_enc.fit_transform(speakers)
        spkrs_utt_sorted = label_enc.transform(spkrs_utt_sorted)

        self.spk2utt = {k: v for k, v in zip(speakers, spk2utt.values())}
        self.utt2spk = {k: v for k, v in zip(utt2spk.keys(), spkrs_utt_sorted)}

        self.num_classes = len(label_enc.classes_)

    def __repr__(self):
        return f"SpeakerDataset w/ {len(self.spk2utt)} speakers and {len(self.utt2spk)} sessions."

    def __len__(self):
        return len(self.spk2utt)

    def __getitem__(self, idx):
        """ Returns one random utt of selected speaker """
        
        utt = np.random.choice(self.spk2utt[idx])

        spk = self.utt2spk[utt]
        feats = self.loading_method(self.utt2path[utt])

        if self.seq_len:
            feats = data_io.train_transform(feats, self.seq_len)

        return feats, spk, utt
    
    def get_utt_feats(self):
        for utt, path in self.utt2path.items():

            feats = self.loading_method(path)
            if self.seq_len:
                feats = data_io.train_transform(feats, self.seq_len)
            yield feats, utt

# Recettes :
def load_multiple_kaldi_metadata(ds_path):
    if not isinstance(ds_path, list):
        ds_path = [ds_path]
    
    utt2spk, spk2utt, utt2path = {}, {}, {}
    for path in ds_path:
        utt2path.update(data_io.read_scp(path / 'feats.scp'))
        utt2spk.update(data_io.read_scp(path / 'utt2spk'))
        # can't do spk2utt.update(t_spk2utt) as update is not additive
        t_spk2utt = data_io.load_one_tomany(path / 'spk2utt')
        for spk, utts in t_spk2utt.items():
            try:
                spk2utt[spk] += utts
            except KeyError:
                spk2utt[spk] = utts
    
    return utt2spk, spk2utt, utt2path

def make_kaldi_train_ds(ds_path, seq_len=400):
    """ 
    Make a SpeakerDataset from only the path of the kaldi dataset.
    This function will use the files 'feats.scp', 'utt2spk' 'spk2utt'
    present in ds_path to create the SpeakerDataset.
    """
    utt2spk, spk2utt, utt2path = load_multiple_kaldi_metadata(ds_path)

    ds = SpeakerDataset(
        utt2path = utt2path,
        utt2spk  = utt2spk,
        spk2utt  = spk2utt,
        loading_method = lambda path: torch.FloatTensor(read_mat(path)),
        seq_len  = seq_len,
    )
    return ds

def make_kaldi_trial_ds(ds_path, trials):
    """ 
    Make a SpeakerDataset from only the path of the kaldi dataset.
    This function will use the files 'feats.scp', 'utt2spk' 'spk2utt'
    present in ds_path to create the SpeakerDataset.
    """
    utt2spk, spk2utt, utt2path = load_multiple_kaldi_metadata(ds_path)

    ds = TrialDataset(
        utt2path = utt2path,
        loading_method = lambda path: torch.FloatTensor(read_mat(path)),
        trial_files  = trials,
    )
    return ds

if __name__ == "__main__":
    ds = make_kaldi_train_ds(Path("exemples/metadata_no_sil/"))
    print(ds)
    for i in range(len(ds)):
        print(ds[i])