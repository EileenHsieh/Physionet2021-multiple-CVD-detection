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
import wfdb
from copy import deepcopy

from pathlib import Path
from lib.evaluation_2021.evaluate_model import *
from src.engine.utils import map_sex, replace_equivalent_classes, SCORED_CLASSES, EQ_CLASSES, ALL_TRAIN_SITES, get_wafdb_age
from src.engine.loaders.transforms import *
from tqdm import tqdm
from neurokit2.signal import signal_resample
from torchvision import transforms
# from torchaudio.transforms import Spectrogram
import coloredlogs, logging
coloredlogs.install()
logger = logging.getLogger(__name__)


#%%
class RawDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.param_aug = config.param_aug
        self.leads = self.config.param_loader.leads
        self.train_sites = ALL_TRAIN_SITES if config.exp.train_sites=="all" else config.exp.train_sites
        self.eval_sites = ALL_TRAIN_SITES if config.exp.eval_sites=="all" else config.exp.eval_sites
        self.N_fold = config.exp.N_fold
        self.N_fold_Order = config.exp.N_fold_Order
        self.fold_data_scored_all, self.fold_data_unscored_all = None, None

    def setup(self):
        # parse scored fold jsons
        fold_data_scored_all = {mode:[[] for _ in range(self.N_fold)] for mode in ["train","eval"]}
        fold_data_unscored_all = {mode:[[] for _ in range(self.N_fold)] for mode in ["train","eval"]}

        # train
        for site in self.train_sites:
            for foldIdx in range(self.N_fold):
                # add scored samples
                fold_data_json = Path(self.config.path.data_directory)/"CVFolds"/f"cv-{self.N_fold}_order-{self.N_fold_Order}"/site/f"fold_{foldIdx}_scored.json"
                with open(fold_data_json, "r") as f:
                    sampleIds = json.load(f)
                    fold_data = [f"{site}/{sampleId}" for sampleId in sampleIds]
                    fold_data_scored_all["train"][foldIdx]+=fold_data
                
                # add unscored smaples
                fold_data_json = Path(self.config.path.data_directory)/"CVFolds"/f"cv-{self.N_fold}_order-{self.N_fold_Order}"/site/f"fold_{foldIdx}_non_scored.json"
                with open(fold_data_json, "r") as f:
                    sampleIds = json.load(f)
                    fold_data = [f"{site}/{sampleId}" for sampleId in sampleIds]
                    fold_data_unscored_all["train"][foldIdx]+=fold_data

        # eval
        for site in self.eval_sites:
            for foldIdx in range(self.N_fold):
                # add scored samples
                fold_data_json = Path(self.config.path.data_directory)/"CVFolds"/f"cv-{self.N_fold}_order-{self.N_fold_Order}"/site/f"fold_{foldIdx}_scored.json"
                with open(fold_data_json, "r") as f:
                    sampleIds = json.load(f)
                    fold_data = [f"{site}/{sampleId}" for sampleId in sampleIds]
                    fold_data_scored_all["eval"][foldIdx]+=fold_data
                
                # add unscored smaples
                fold_data_json = Path(self.config.path.data_directory)/"CVFolds"/f"cv-{self.N_fold}_order-{self.N_fold_Order}"/site/f"fold_{foldIdx}_non_scored.json"
                with open(fold_data_json, "r") as f:
                    sampleIds = json.load(f)
                    fold_data = [f"{site}/{sampleId}" for sampleId in sampleIds]
                    fold_data_unscored_all["eval"][foldIdx]+=fold_data

        self.fold_data_scored_all = fold_data_scored_all
        self.fold_data_unscored_all = fold_data_unscored_all


    def setup_kfold(self, folds_train, folds_val, folds_test):
        self.folds_train = folds_train
        self.folds_val = folds_val
        self.folds_test = folds_test

    def _get_loader(self, folds, mode):
        dataset = RawDataset(self.config,
                            self.fold_data_scored_all.copy(),
                            self.fold_data_unscored_all.copy(),
                            folds = folds,
                            transform=None,
                            mode=mode)
        return DataLoader(dataset=dataset, batch_size=self.config.param_model.batch_size, num_workers=self.config.param_loader.num_workers, shuffle=(mode=="train"), pin_memory=False, collate_fn=my_collate_fn) 

    def train_dataloader(self):
        return self._get_loader(self.folds_train, "train")

    def val_dataloader(self):
        return self._get_loader(self.folds_val, "val")

    def test_dataloader(self):
        return self._get_loader(self.folds_test, "test")

    def _cal_train_class_weight(self):
        logger.info("Calculate class weights ....")
        loader = self.train_dataloader()
        labels = []
        for _,label in loader:
            labels.extend(label.numpy())
        labels = np.asarray(labels)
#%%
        zeros = np.sum(labels==0, axis=0)
        ones = np.sum(labels==1, axis=0)
        pos_weight = torch.tensor(zeros/ones)
        logger.info(f"Positive Weight: {pos_weight}")
        self.pos_weight = pos_weight
        return pos_weight

#%%
class RawDataset(Dataset):
    def __init__(self, config, fold_data_scored_all, fold_data_unscored_all, folds=None, transform=None, mode='train'):
        super(RawDataset, self).__init__()

        # Initialization
        self.config = config
        self.ECG_THRE = config.param_aug.ECG_THRE if config.param_aug.get('ECG_THRE') else np.inf
        self.ECG_LEAD_DROP = config.param_aug.get("ECG_LEAD_DROP")
        self.param_aug = config.param_aug
        self.folds = folds
        self.N_folds = config.exp.N_fold
        self.num_leads = config.param_loader.num_leads
        self.param_loader = self.config.param_loader
        self.max_length = int(config.param_feature.raw.sr*config.param_feature.raw.window)
        if self.config.param_loader.get("spectrogram"): self.max_length = int(np.ceil(self.max_length/self.param_loader.spectrogram.HOP_LEN))
        self.mode = mode
        self.random_state = config.exp.random_state
        self.transform = transform if transform else self._get_transform(mode)
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)

        # prepare fold
        if not folds is None: # train and validation state
            self.fold_data = []
            for fold in folds:
                if mode=="train":
                    self.fold_data.extend(fold_data_scored_all["train"][fold])
                    if config.exp.N_fold_Use_Unscored:
                        self.fold_data.extend(fold_data_unscored_all["train"][fold])
                else:
                    self.fold_data.extend(fold_data_scored_all["eval"][fold])
                    self.fold_data.extend(fold_data_unscored_all["eval"][fold])

    def __len__(self):
        return len(self.fold_data)

    def _get_transform(self, mode="train"):
        # input dimension: num_lead*len_signal
        tfm = transforms.Compose([])
        
        # time series raw signal in numpy
        if self.param_aug.get("nk2multilead"):
            tfm.transforms.append(NK2_ECG_Process(sr=self.config.param_feature.raw.sr, method="neurokit", concat_lead=True))
        else:
            if self.param_aug.get("RemoveBaselineWander"): tfm.transforms.append(RemoveBaselineWander(cutoff=self.param_aug.RemoveBaselineWander.cutoff))
            if self.param_aug.get("BandPass"): tfm.transforms.append(BandPass(sr=self.config.param_feature.raw.sr, cutoff=self.param_aug.BandPass.cutoff))
            if self.param_aug.get("filter"):
                for k,v in dict(self.param_aug.filter).items():
                    tfm.transforms.append(FilterSignal(**dict(v)))

        # # spectrogram
        # if self.config.param_loader.get("spectrogram"):
        #     tfm.transforms.append(MinMaxScaler())
        #     tfm.transforms.append(AsTensor())
        #     tfm.transforms.append(Spectrogram(n_fft=self.param_loader.spectrogram.NFFT, hop_length=self.param_loader.spectrogram.HOP_LEN))
        #     if mode=="train" and self.param_aug.get("RandomCrop"): 
        #         tfm.transforms.append(transforms.RandomCrop(
        #             size=[int(self.param_loader.spectrogram.NFFT/2)+1, self.max_length], padding=0, pad_if_needed=True))
        #     else: tfm.transforms.append(transforms.CenterCrop(size=[int(self.param_loader.spectrogram.NFFT/2)+1, self.max_length]))

        # times sereis
        # else:
        if self.param_aug.get("Rescale"): 
            if self.param_aug.Rescale=="zscore": tfm.transforms.append(Zscore())
            elif self.param_aug.Rescale=="minmax": tfm.transforms.append(MinMaxScaler())
        tfm.transforms.append(transforms.ToTensor())

        if mode=="train":
            if self.param_aug.get("RandomShuflleLead"): tfm.transforms.append(RandomShuflleLead(p=self.param_aug.RandomShuflleLead))
            if self.param_aug.get("RandomLeadMask"): tfm.transforms.append(RandomLeadMask(p=self.param_aug.RandomLeadMask))
            if self.param_aug.get("AddGaussianNoise"): tfm.transforms.append(AddGaussianNoise(self.param_aug.AddGaussianNoise.mean, self.param_aug.AddGaussianNoise.mean))

        if self.param_aug.get("nk2multilead"):
            if mode=="train" and self.param_aug.get("RandomCrop"): tfm.transforms.append(transforms.RandomCrop(size=[self.num_leads*5, self.max_length], padding=0, pad_if_needed=True))
            else: tfm.transforms.append(transforms.CenterCrop(size=[self.num_leads*5, self.max_length]))
        else:
            if mode=="train" and self.param_aug.get("RandomCrop"): tfm.transforms.append(transforms.RandomCrop(size=[self.num_leads, self.max_length], padding=0, pad_if_needed=True))
            else: tfm.transforms.append(transforms.CenterCrop(size=[self.num_leads, self.max_length]))
        return tfm

    def __getitem__(self, idx):
        smpale_info = self.fold_data[idx]
        site, sampleId = smpale_info.split("/")
        # site = "WFDB_StPetersburg"
        # sampleId = "I0023"
        # site = "WFDB_ChapmanShaoxing"
        # sampleId = "JS09864"
        

        # =============================================================================
        # wfdb loading method
        # =============================================================================
        data_path = str(Path(self.config.path.data_directory)/'raw'/site/sampleId)
        # data_path = str(Path(self.config.path.official_data_directory)/sampleId)
        record_wfdb = wfdb.rdrecord(data_path, channel_names=self.config.param_loader.leads, physical=True)
        recording = record_wfdb.p_signal.T # channel x length
        sr = float(record_wfdb.fs)

        header_info = {r.split(':')[0]:r.split(':')[1].strip() for r in record_wfdb.comments}
        age = get_wafdb_age(header_info=header_info)
        sex = header_info['Sex']
        sex = map_sex(sex, impute_nan=1)
        label = [l.strip() for l in header_info['Dx'].split(',')]


        # =============================================================================
        # check valid lead
        # =============================================================================
        is_garbage = False
        num_leads = recording.shape[0]
        valid_leads = np.asarray([True for _ in range(num_leads)])
        valid_leads = np.logical_and((np.abs(recording).max(axis=1))<self.ECG_THRE, valid_leads)
        recording[~valid_leads,:] = 0 # insert 0 to invalid leads


        # =============================================================================
        # naive preprocess
        # =============================================================================
        # sampling rate
        if sr!=self.config.param_feature.raw.sr:
            # recording = filter_signal(recording, sample_rate=sr, cutoff=[3,45], filtertype="bandpass") # bandpass before resample (prevent high-freq noise warping)
            recording = signal_resample(list(recording.T), sampling_rate=sr, desired_sampling_rate=self.config.param_feature.raw.sr, method="FFT").T
        
        # augmentation
        if self.transform:
            recording = self.transform(recording)
            if self.config.param_loader.get("spectrogram") is None:
                recording = recording.squeeze(0)
        recording = recording.float()

        # check valid
        for leadIdx in list(set(np.where(np.isnan(recording.numpy()))[0])):
            valid_leads[leadIdx] = False


        # =============================================================================
        # random drop lead
        # =============================================================================
        if self.ECG_LEAD_DROP and self.mode=="train":
            rand_num = np.random.uniform(low=0.0, high=1.0, size=num_leads)
            drop_leads = rand_num<self.ECG_LEAD_DROP
            recording[drop_leads,:] = 0
            valid_leads = np.logical_and(valid_leads, ~drop_leads)

        valid_leads = torch.tensor(valid_leads).int()
        if sum(valid_leads)<(num_leads/2): is_garbage = True


        # =============================================================================
        # collect
        # =============================================================================
        # x = {"signal":recording, "age":age, "sex":sex}
        x = {"signal":recording, "info":f"{site}/{sampleId}", "valid_leads":valid_leads, "is_garbage":is_garbage}


        # =============================================================================
        # label
        # =============================================================================
        label_bin = pd.DataFrame(np.zeros((1,len(SCORED_CLASSES))), columns=SCORED_CLASSES, index=[sampleId])
        labels = [l for l in label if l in SCORED_CLASSES]
        label_bin[labels] = 1
        if self.config.param_loader.get('label_target'):
            label_bin = label_bin[self.config.param_loader.label_target]
        y = label_bin.values.flatten().astype("float32")

        # if np.isnan(recording.numpy()).any():
        #     return None
        return x, y

from torch.utils.data._utils.collate import default_collate
def my_collate_fn(batch):
    batch = [b for b in batch if b is not None]
    return default_collate(batch)
