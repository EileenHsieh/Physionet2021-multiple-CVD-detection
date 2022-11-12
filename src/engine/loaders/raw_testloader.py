#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 12 17:35:05 2021

@author: chadyang
"""
#%%
from helper_code import get_age, get_frequency, get_sex, load_recording, get_adc_gains, get_baselines
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import json, random
import copy
from pathlib import Path
from lib.evaluation_2021.evaluate_model import *
from src.engine.utils import (map_sex,SCORED_CLASSES,remove_nan)
from src.engine.loaders.transforms import *
from tqdm import tqdm
from neurokit2.signal import signal_resample
from torch.nn import functional as F
from torchvision import transforms
# from torchaudio.transforms import Spectrogram
import coloredlogs, logging
coloredlogs.install()
logger = logging.getLogger(__name__)


#%%
class TestDataset(Dataset):
    def __init__(self, config, header, recording, folds=None, transform=None, mode='test'):
        super(TestDataset, self).__init__()

        # Initialization
        self.config = config
        self.ECG_THRE = config.param_aug.ECG_THRE
        self.header = header
        self.recording = recording
        # self.recording = remove_nan(recording)
        self.param_aug = config.param_aug
        # self.num_leads = config.param_loader.num_leads
        self.param_loader = self.config.param_loader
        self.max_length = int(config.param_feature.raw.sr*config.param_feature.raw.window)
        if self.config.param_loader.get("spectrogram"): self.max_length = int(np.ceil(self.max_length/self.param_loader.spectrogram.HOP_LEN))
        self.mode = mode
        self.random_state = config.exp.random_state
        self.transform = transform if transform else self._get_transform(mode)
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)

        # process signal and to patches 
        self._get_patch()

    def __len__(self):
        return len(self.all_patch)

    def _get_transform(self, mode="train"):
        # input dimension: num_lead*len_signal
        tfm = transforms.Compose([])
        
        # time series raw signal in numpy
        if self.param_aug.get("RemoveBaselineWander"): tfm.transforms.append(RemoveBaselineWander(cutoff=self.param_aug.RemoveBaselineWander.cutoff))
        if self.param_aug.get("BandPass"): tfm.transforms.append(BandPass(sr=self.config.param_feature.raw.sr, cutoff=self.param_aug.BandPass.cutoff))
        if self.param_aug.get("filter"):
            for k,v in dict(self.param_aug.filter).items():
                tfm.transforms.append(FilterSignal(**dict(v)))

        # times sereis
        if self.param_aug.get("Rescale"): 
            if self.param_aug.Rescale=="zscore": tfm.transforms.append(Zscore())
            elif self.param_aug.Rescale=="minmax": tfm.transforms.append(MinMaxScaler())
        tfm.transforms.append(transforms.ToTensor())
        # if mode=="train" and self.param_aug.get("RandomCrop"): tfm.transforms.append(transforms.RandomCrop(size=[self.num_leads, self.max_length], padding=0, pad_if_needed=True))
        # else:
        #     print("hererere") 
        #     tfm.transforms.append(transforms.CenterCrop(size=[len(self.recording), self.max_length]))

        return tfm

    def _get_patch(self):
        # =============================================================================
        # naive preprocess
        # =============================================================================
        # sampling rate
        sr = get_frequency(self.header)
        if sr!=self.config.param_feature.raw.sr:
            # recording = filter_signal(self.recording, sample_rate=sr, cutoff=[3,45], filtertype="bandpass") # bandpass before resample (prevent high-freq noise warping)
            self.recording = signal_resample(list(self.recording.T), sampling_rate=sr, desired_sampling_rate=self.config.param_feature.raw.sr, method="FFT").T
        
        # =============================================================================
        # check number of patches
        # =============================================================================
        # segment test recording to several patches
        num_leads, fea_dim = self.recording.shape[0], self.recording.shape[1]
        win_size = self.max_length
        overlap = self.config.exp.overlap_size
        num_patch = ((fea_dim-win_size)//(win_size-overlap)) + 1
        # if recording is too short, return only one patch
        if num_patch<=0:  
            patch = copy.deepcopy(self.recording)
            all_is_garbage = []
            all_valid_leads = []
            # =============================================================================
            # check valid lead
            # =============================================================================
            is_garbage = False
            num_leads = patch.shape[0]
            
            valid_leads = np.asarray([True for _ in range(num_leads)])
            valid_leads = np.logical_and((np.abs(patch).max(axis=1))<self.ECG_THRE, valid_leads)
            patch[~valid_leads,:] = 0 # insert 0 to invalid leads
            

            if self.transform:
                patch = self.transform(patch)
                if self.config.param_loader.get("spectrogram") is None:
                    patch = patch.squeeze(0)
            patch = patch.float()

            tr = transforms.CenterCrop(size=[len(patch), self.max_length])
            all_patch = [tr(patch)]

            # check valid
            for leadIdx in list(set(np.where(np.isnan(patch.numpy()))[0])):
                valid_leads[leadIdx] = False
            valid_leads = torch.tensor(valid_leads).int()
            if sum(valid_leads)<(num_leads/2): is_garbage = True

            all_is_garbage.append(is_garbage)
            all_valid_leads.append(valid_leads)
        else:
            all_patch = []
            all_is_garbage = []
            all_valid_leads = []
            for p in range(num_patch):
                patch = self.recording[:, p*win_size:(p+1)*win_size]
                if patch.shape[1]<self.max_length:  continue
                
                # =============================================================================
                # check valid lead
                # =============================================================================
                is_garbage = False
                num_leads = patch.shape[0]
                valid_leads = np.asarray([True for _ in range(num_leads)])
                valid_leads = np.logical_and((np.abs(patch).max(axis=1))<self.ECG_THRE, valid_leads)
                patch[~valid_leads,:] = 0 # insert 0 to invalid leads
                

                if self.transform:
                    patch = self.transform(patch)
                    if self.config.param_loader.get("spectrogram") is None:
                        patch = patch.squeeze(0)
                patch = patch.float()


                # collect patches
                all_patch.append(patch)

                # check valid
                for leadIdx in list(set(np.where(np.isnan(patch.numpy()))[0])):
                    valid_leads[leadIdx] = False
                valid_leads = torch.tensor(valid_leads).int()
                if sum(valid_leads)<(num_leads/2): is_garbage = True

                all_is_garbage.append(is_garbage)
                all_valid_leads.append(valid_leads)

        self.all_patch = torch.stack(all_patch)
        self.all_is_garbage = all_is_garbage
        self.all_valid_leads = torch.stack(all_valid_leads)

    def __getitem__(self, idx):
        
        # get demo
        # age, sex = np.asarray(get_age(self.header)).astype("float32"), get_sex(self.header)
        # sex = map_sex(sex, impute_nan=1)
    
        # collect
        x = {"signal":self.all_patch[idx],
             "valid_leads":self.all_valid_leads[idx], 
             "is_garbage":self.all_is_garbage[idx]}

        if np.isnan(self.all_patch[idx].numpy()).any():
            return None
        # =============================================================================
        # label
        # =============================================================================
        dummy = np.tile(0, len(SCORED_CLASSES))
        return x, dummy

from torch.utils.data._utils.collate import default_collate
def my_collate_fn(batch):
    batch = [b for b in batch if b is not None]
    return default_collate(batch)


#%% test data loader
if __name__=="__main__":
    from omegaconf import OmegaConf
    from src.engine.utils import wfdb_process
    config = OmegaConf.load("./config/hydra/attn_rsn_classic_12leads.yaml")
    data_root = '../../cinc2021/datasets/raw/'
    record_name = 'I0068'#'E04603'
    header = load_header(f'{data_root}/{record_name}.hea')
    recording = load_recording(f'{data_root}/{record_name}.mat')
    recording = wfdb_process(header, recording, config.param_loader.leads)
    # recording = np.concatenate([np.full([12,5000], np.nan), rec],axis=1)
    
    testset = TestDataset(config, header, recording)
    # print("is NAN:", torch.isnan(testset[0][0]["signal"]).all())
    testLoader = DataLoader(dataset=testset, batch_size=len(testset), shuffle=False, collate_fn=my_collate_fn) 
    testData = next(iter(testLoader))[0]
    # DataLoader(dataset=dataset, batch_size=self.config.param_model.batch_size, num_workers=self.config.param_loader.num_workers, shuffle=(mode=="train"), pin_memory=True, collate_fn=my_collate_fn)
