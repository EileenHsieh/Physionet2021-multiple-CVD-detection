#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 12 17:35:05 2021

@author: chadyang
"""
#%%
import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import joblib
import pandas as pd
import numpy as np
import json
import os
import multiprocessing as mp

from pathlib import Path
import sys

from lib.evaluation_2021.evaluate_model import *
from src.engine.utils import NORMAL_CLASS, CLASSES, WEIGHTS, SCORED_CLASSES
from skmultilearn.problem_transform import LabelPowerset
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE

import multiprocessing as mp
from multiprocessing import Pool
from src.engine.utils import istarmap
from functools import partial
from tqdm import tqdm
from sklearn.model_selection import KFold


import coloredlogs, logging
coloredlogs.install()
logger = logging.getLogger(__name__)


class HrvDataModule(pl.LightningDataModule):
    def __init__(self, config, fea_lead_all_cv, label_bin_all_cv, fea_lead_all_test=None):
        super().__init__()
        self.config = config
        self.leads = self.config.param_loader.leads
        self.fea_lead_all_cv = fea_lead_all_cv
        self.label_bin_all_cv = label_bin_all_cv
        self.fea_lead_all_test = fea_lead_all_test

    def setup(self):
        # =============================================================================
        # 1. cross validation data
        # =============================================================================
        # concat selected leads
        if self.config.param_feature.hrv_nk2.concat:
            if len(self.leads)>1:
                fea_lead_select = [fea_lead for lead, fea_lead in self.fea_lead_all_cv.items() if lead in self.leads]
                fea_lead_select = pd.concat(fea_lead_select, axis=1)
                fea_lead_select = fea_lead_select.loc[:,~fea_lead_select.columns.duplicated()] # drop potential duplicated columns (like age/sex)

        # prepare k-fold data
        subjects = np.unique(self.label_bin_all_cv.index.values)
        idxs = np.arange(subjects.shape[0])
        kfold = KFold(n_splits=self.config.exp.N_fold, shuffle=True, random_state=self.config.exp.random_state)

        fold_data_all = []
        for k, (trainIdx, testIdx) in enumerate(kfold.split(idxs)):
            sampleIdx = self.label_bin_all_cv.index[testIdx]
            fold_fea = fea_lead_select.loc[sampleIdx,:]
            fold_lab = self.label_bin_all_cv.loc[sampleIdx,:]
            fold_data = [fold_fea, fold_lab]
            fold_data_all.append(fold_data)
        self.fold_data_all = fold_data_all
        self.config.param_model.dim_in = self.fold_data_all[0][0].shape[1]
        self.config.param_model.dim_out = self.label_bin_all_cv.shape[1]

        # define model input size according to the feature dimension
        if self.config.param_feature.hrv_nk2.concat:
            self.config.param_model.input_size = fold_fea.shape[1]
        else:
            logger.error("Wrong Feature Type. Currentlyly only support Hrv Concat")

        # =============================================================================
        # 2. test data
        # =============================================================================

    def setup_kfold(self, folds_train, folds_val, folds_test):
        self.folds_train = folds_train
        self.folds_val = folds_val
        self.folds_test = folds_test

    def _get_loader(self, folds, mode):
        num_workers = min(mp.cpu_count()-1, self.config.param_loader.num_workers)
        dataset = HrvDataset(self.config,
                            self.fold_data_all.copy(),
                            folds = folds,
                            mode=mode)
        return DataLoader(dataset=dataset, batch_size=self.config.param_model.batch_size, num_workers=num_workers, shuffle=(mode=="train")) 

    def train_dataloader(self):
        return self._get_loader(self.folds_train, "train")

    def val_dataloader(self):
        return self._get_loader(self.folds_val, "val")

    def test_dataloader(self):
        return self._get_loader(self.folds_test, "test")

#%%
class HrvDataset(Dataset):
    def __init__(self, config, fold_data_all, folds=None,  mode='train'):
        super(HrvDataset, self).__init__()

        # Initialization
        self.config = config
        self.folds = folds
        self.N_folds = self.config.exp.N_fold
        self.mode = mode
        self.random_state = self.config.exp.random_state
        resample = self.config.param_loader.resample

        # resample methods:
        if resample=="LPROS": # oversample
            resample_tf = RandomOverSampler(random_state=self.random_state)
        elif resample=="LPRUS": # undersample
            resample_tf = RandomUnderSampler(random_state=self.random_state)
        elif resample=="SMOTE": # oversample
            resample_tf = SMOTE(random_state=self.random_state)

        # Set Reproducibility
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        if not folds is None: # train and validation state
            data = [[],[]]
            for fold in folds:
                fea, lab = fold_data_all[fold]
                data[0].append(fea.values)
                data[1].append(lab.values)
            # self.data = [pd.concat(d, axis=0) for d in data]
            self.data = [np.concatenate(d, axis=0) for d in data]
    
        # Resample
        if (resample!='None') and (self.mode=='train'):
            num_smaples_ori = len(self.data[1])
            lp = LabelPowerset()
            yt = lp.transform(self.data[1])
            fea, lab = resample_tf.fit_resample(self.data[0], yt)
            lab = lp.inverse_transform(lab).toarray()
            num_smaples_new = len(lab)
            logger.info(f"Resample with {resample_tf}: Ori:{num_smaples_ori} -> Sampled:{num_smaples_new}")
            self.data = [fea, lab]

    def _get_folds(self):
        available_folds = np.arange(self.N_folds)
        curr_folds = np.array(self.folds)
        comp_folds = np.array([f for f in available_folds if f not in self.folds])
        if self.mode == "train": return curr_folds, comp_folds
        else: return comp_folds, curr_folds

    def __len__(self):
        return len(self.data[0])
    
    def __getitem__(self, idx):
        # fea, lab = self.data[0].iloc[idx].values, self.data[1].iloc[idx].values
        fea, lab = self.data[0][idx], self.data[1][idx]
        return fea.astype("float32"), lab.astype("float32")


#%%
def load_hrv_pkls(config, hrv_pkl_paths, leads):
    fea_lead_all, label_all = get_data_hrv_mp(hrv_pkl_paths, targets=SCORED_CLASSES, leads=leads, 
                                        fea_types=config.param_feature.hrv_nk2.fea_types)
    return fea_lead_all, label_all


#%%
def load_hrv_data(config, leads):
    logger.info('-'*60)
    logger.info("Prepare Data ...")

    # perpare
    data_directory_abs = Path('.').absolute()/config.data_directory
    
    # load pkl
    if os.listdir(data_directory_abs)[0].startswith('WFDB'): 
        sites = os.listdir(data_directory_abs)
        hrv_pkl_paths = []
        for site in sites:
            hrv_root = Path('.').absolute()/f"datasets/processed/nk2/hrv/{site}"
            hrv_pkl_paths_site = sorted(list(hrv_root.glob("*.pkl")))
            hrv_pkl_paths.extend(hrv_pkl_paths_site)

    # Only input a site
    elif os.listdir(data_directory_abs)[0].endswith('.hea') or os.listdir(data_directory_abs)[0].endswith('.mat'):
        site = data_directory_abs.name
        hrv_root = Path('.').absolute()/f"datasets/processed/nk2/hrv/{site}"
        hrv_pkl_paths = sorted(list(hrv_root.glob("*.pkl")))

    # load all pkls -> fea_lead_all (n*d dataframe), label_bin_all (n*24 dataframe)
    fea_lead_all, label_all = get_data_hrv_mp(hrv_pkl_paths, targets=SCORED_CLASSES, leads=leads, 
                                        fea_types=config.hrv_nk2.fea_types)
    # fea_lead_all = pd.concat(fea_lead_all, axis=0)
    return fea_lead_all, label_all



#%%
def get_data_hrv_mp(hrv_pkl_paths, targets, leads, fea_types):
    partial_func = partial(parse_data_hrv, targets=targets, leads=leads, fea_types=fea_types)
    fea_lead_all_tmp, label_all = [], {}
    with Pool(processes=mp.cpu_count()-1) as pool:
        for fea_lead, label in tqdm(pool.istarmap(partial_func, zip(hrv_pkl_paths)), total=len(hrv_pkl_paths), disable=False):
            fea_lead_all_tmp.append(fea_lead)
            label_all.update(label)

    fea_lead_all = {}
    for lead in leads:
        fea_lead_all[lead] = pd.concat([f[lead] for f in fea_lead_all_tmp], axis=0)
    # label_bin_all = pd.concat(label_bin_all, axis=0)
    return fea_lead_all, label_all


#%%
def parse_data_hrv(hrv_pkl_path, targets=["270492004"], leads=("I"), fea_types=["fea_demo"]):
    GET_FEA_DIMS = {'fea_demo':2, 'hrv_time':14, 'hrv_nonlin':29, 'hrv_freq':9, 'ecg_rate_func10':10, 'ecg_quality_func10':10, 'ecg_phase_comp_vent_func10':10, 'ecg_phase_comp_atrial_func10':10}
    # FEA_NAMES = json.load(open(os.path.join("/HDD/HDD2/Projects/Challenges/Physionet/cinc/scripts/PhysioNet-CinC-Challenges-2021/src/engine/loaders", "./FEA_NAMES_HRV.json")))
    FEA_NAMES = json.load(open(Path(__file__).parent.absolute()/"./FEA_NAMES_HRV.json"))

    data = joblib.load(hrv_pkl_path)

    # features
    sampleId = data['sampleId']
    fea_lead_all = {}
    for lead in leads:
        fea_multi_lead = data['fea_ECG_multi_lead'][f'{sampleId}-{lead}']
        if fea_multi_lead is not None:
            fea_lead = [f for f_name, f in fea_multi_lead.items() if f_name in fea_types]
            fea_lead = pd.concat(fea_lead,  axis=1)
            fea_lead.columns = [f'{lead}-{f_col_name}' for f_col_name in fea_lead.columns]
            fea_lead.index = [sampleId]
        
        else: # insert nan  to empty lead features
            fea_lead = []
            for fea_type in fea_types:
                if fea_type != 'fea_demo':
                    fake_col_name = [f'{lead}-{f_col_name}' for f_col_name in FEA_NAMES[fea_type]]
                    f = pd.DataFrame(np.asarray([np.nan]*GET_FEA_DIMS[fea_type]).reshape(1,-1), index=[sampleId], columns=fake_col_name)
                    fea_lead.append(f)
            fea_lead = pd.concat(fea_lead, axis=1)

        if 'fea_demo' in fea_types:
            fea_demo = data['fea_demo']
            fea_demo.index =[sampleId]
            fea_lead = pd.concat([fea_demo, fea_lead], axis=1)
        fea_lead_all.update({lead:fea_lead})

    # labels
    labels = {sampleId:data['labels']}
    return fea_lead_all, labels


