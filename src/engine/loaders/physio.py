import sys,json
# PATHS = json.load(open("./paths.json"))
# for k in PATHS:
#     sys.path.append(PATHS[k])
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from helper_code import (load_header, get_labels)
import pandas as pd
import torch
import time
import pickle as pkl
import numpy as np
import os
import sys
import json
import joblib

class HRVLoader(Dataset):
    def __init__(self, data, leads, folds=None, N_folds=5, mode='train', random_state=100):
        super(HRVLoader, self).__init__()

        # Initialization
        self.data = data
        self.folds = folds
        self.N_folds = N_folds
        self.mode = mode
        self.random_state = random_state
        self.leads = leads

        # Set Reproducibility
        np.random.seed(random_state)
        torch.manual_seed(random_state)
        if not folds is None: # train and validation state
            mask_folds = self.data.k.isin(folds)
            self.data = self.data[mask_folds].reset_index(drop=True)
        self.CLASSES = joblib.load('./datasets/disease_classes.pkl')
        # self._get_num_labels()
    
    def _get_folds(self):
        available_folds = np.arange(self.N_folds)

        curr_folds = np.array(self.folds)
        comp_folds = np.array([f for f in available_folds if f not in self.folds])

        if self.mode == "train": return curr_folds, comp_folds
        else: return comp_folds, curr_folds

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        sample_data = self.data.loc[index]
        hrv_path = sample_data.hrv_path
        hrv_sample = joblib.load(hrv_path)
        lead_stacked_feature = {}
        for lead_type, lead_feat in hrv_sample['fea_ECG_multi_lead'].items():
            try:
                features = []
                for feat_type, feature in hrv_sample['fea_ECG_multi_lead'][lead_type].items():
                    features+=list(feature.values[0])
                features = np.array(features)
                features[np.isnan(features)] = 0
                features[np.isinf(features)] = 0
            except: features = np.zeros(92)
            lead_stacked_feature[lead_type] = features
        lead_feature = np.hstack([lead_stacked_feature[f"{sample_data.ID}-{l}"] for l in self.leads])
        # Label
        labels = np.zeros(len(self.CLASSES))
        header = load_header(sample_data.hea_path)
        current_labels = get_labels(header)
        for label in current_labels:
            if label in self.CLASSES:
                j = self.CLASSES.index(label)
                labels[j] = 1

        # labels = np.zeros((1, self.num_classes), dtype=int) # One-hot encoding of classes
        # header = load_header(sample_data.hea_path)
        # current_labels = get_labels(header)
        # for label in current_labels:
        #     if label in self.classes:
        #         j = self.classes.index(label)
        #         labels[0, j] = 1

        return lead_feature.astype("float32"), labels.astype("float32")