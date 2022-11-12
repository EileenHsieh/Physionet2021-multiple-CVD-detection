#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 27 11:33:56 2021

@author: chadyang
"""
#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger
from torchmetrics import Metric

from src.engine.utils import logit_To_Scalar_Binary, NORMAL_CLASS, CLASSES, WEIGHTS
from src.engine.losses import *
from lib.evaluation_2021.evaluate_model import *

import numpy as np
from pathlib import Path
from copy import deepcopy

import multiprocessing as mp
from multiprocessing import Pool
from functools import partial
from copy import deepcopy
from itertools import product
from time import time

import coloredlogs, logging
coloredlogs.install()
logger = logging.getLogger(__name__)  


#%%
# NUMCPU = mp.cpu_count()
NUMCPU = 8

class Classifier(pl.LightningModule):
    def __init__(self, param_model, random_state=0, pos_weight=None):
        super().__init__()
        # log hyperparameters
        self.param_model = param_model
        self.save_hyperparameters()

        # define model architecture
        if self.param_model.get("dim_in") and self.param_model.get("dim_out"):
            self.model = nn.Linear(self.param_model.dim_in, self.param_model.dim_out)

        # loss function
        if param_model.get('loss'):
            logger.warning(f"Using Objective Function: {self.param_model.loss}")
            param_loss = dict(deepcopy(param_model.loss))
            del param_loss['name']
            self.criterion = eval(param_model.loss.name)(**dict(param_loss))
        else:
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        # self.criterion = nn.BCEWithLogitsLoss()

        # load weights for metric calculation
        self.normal_class = NORMAL_CLASS
        self.classes, self.weights = CLASSES, WEIGHTS
        
        # optimized binary boundary
        self.register_buffer("bin_thre_arr", torch.ones(len(self.classes))*0.5)
        self.update_bin_thre_arr = False # flag to control whetehr update the binary threshold



    # =============================================================================
    # train / val / test
    # =============================================================================
    def forward(self, x):
       x = self.model(x)
       return x

    def _shared_step(self, batch):
        x, label = batch
        logit = self.model(x)
        loss = self.criterion(logit, label)
        return loss, logit, label

    def training_step(self, batch, batch_idx):
        loss, logit, label = self._shared_step(batch)
        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True)
        return {"loss":loss, "logit":logit, "label":label}
    
    def training_epoch_end(self, train_step_outputs):
        logit = torch.cat([v["logit"] for v in train_step_outputs], dim=0)
        label = torch.cat([v["label"] for v in train_step_outputs], dim=0)
        metrics = self._cal_metric(logit.detach(), label.detach())
        self._log_metric(metrics, mode="train")

    def validation_step(self, batch, batch_idx):
        loss, logit, label = self._shared_step(batch)
        self.log('val_loss', loss, prog_bar=True, on_epoch=True)
        return {"loss":loss, "logit":logit, "label":label}

    def validation_epoch_end(self, val_step_end_out):
        logit = torch.cat([v["logit"] for v in val_step_end_out], dim=0)
        label = torch.cat([v["label"] for v in val_step_end_out], dim=0)
        if self.param_model.binary_thre_opt and self.update_bin_thre_arr and self.current_epoch>0:
            scalar_output = F.sigmoid(logit)
            self._find_opt_thre(scalar_output, label)
        metrics = self._cal_metric(logit.detach(), label.detach())
        self._log_metric(metrics, mode="val")
        return val_step_end_out

    def test_step(self, batch, batch_idx):
        loss, logit, label = self._shared_step(batch)
        self.log('test_loss', loss, prog_bar=True)
        return {"loss":loss, "logit":logit, "label":label}

    def test_epoch_end(self, test_step_end_out):
        logit = torch.cat([v["logit"] for v in test_step_end_out], dim=0)
        label = torch.cat([v["label"] for v in test_step_end_out], dim=0)
        metrics = self._cal_metric(logit.detach(), label.detach())
        self._log_metric(metrics, mode="test")
        return test_step_end_out


    # =============================================================================
    # optimizer
    # =============================================================================
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.param_model.lr)
        if self.param_model.get("scheduler_ReduceLROnPlateau"):
            lr_schedulers = {"scheduler":ReduceLROnPlateau(optimizer,**(self.param_model.scheduler_ReduceLROnPlateau)), "monitor":"val_loss"}
            return [optimizer], lr_schedulers
        if self.param_model.get("scheduler_WarmUp"):
            lr_schedulers = {"scheduler":ReduceLROnPlateau(optimizer,**(self.param_model.scheduler_WarmUp)), "monitor":"val_loss"}
            return [optimizer], lr_schedulers
        return optimizer
    

    # =============================================================================
    # inference
    # =============================================================================
    def get_logits(self, loader:torch.utils.data.dataloader.DataLoader):
        self.model.eval()
        labels, logits = [], []
        with torch.no_grad():
            for batch in loader:
                x, label = batch
                logit = self.model(x)
                labels.append(label.detach().cpu())
                logits.append(logit.detach().cpu())
        logits = torch.cat(logits, dim=0)
        labels = torch.cat(labels, dim=0)
        scalar_output, binary_output = logit_To_Scalar_Binary(logits, bin_thre_arr=self.bin_thre_arr.cpu())
        return logits, labels, scalar_output, binary_output


    # =============================================================================
    # utils
    # =============================================================================
    def _cal_metric(self, logit:torch.tensor, label:torch.tensor):
        scalar_output, binary_output = logit_To_Scalar_Binary(logit, bin_thre_arr=self.bin_thre_arr)
        auroc, auprc, auroc_classes, auprc_classes = compute_auc(label.cpu().numpy(), scalar_output.cpu().numpy())
        cm = compute_challenge_metric(self.weights, label.cpu().numpy(), binary_output.cpu().numpy(), self.classes, self.normal_class)
        return{"auroc":auroc, "auprc":auprc, "cm":cm}    
    def _log_metric(self, metrics, mode):
        for k,v in metrics.items():
            self.log(f"{mode}_{k}", v, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            # self.log(f"{mode}_{k}", v, on_step=False, on_epoch=True)
    
    def _find_opt_thre(self, scalar_output:torch.tensor, label:torch.tensor):
        num_classes = label.shape[1]

        # =============================================================================
        # method of CINC2020 1st paper
        # =============================================================================
        if self.param_model.binary_thre_opt.method==1:
  
            # first round: 0.1~1 search for all classes
            if self.param_model.binary_thre_opt.get("mp") == True:

                #  layer 1
                thre_results = []
                partial_func = partial(find_thre_method1_level1, weights=self.weights, classes=self.classes, normal_class=self.normal_class, label=deepcopy(label.cpu().numpy()), scalar_output=deepcopy(scalar_output.cpu().numpy()))
                with Pool(processes=NUMCPU) as pool:
                    for thre, cm in pool.istarmap(partial_func, zip(np.arange(0,1.1,0.1))):
                        thre_results.append([thre, cm])
                optm_thre_base, optm_cm = thre_results[np.argmax([r[1] for r in thre_results])]
                bin_thre_arr = nn.Parameter(torch.tensor([optm_thre_base]*num_classes, dtype=torch.float32), requires_grad=False).to(self.device)
                self.bin_thre_arr = bin_thre_arr

                #  layer 2
                for classIdx in range(num_classes):
                    thre_class_results = []
                    partial_func = partial(find_thre_method1_level2, bin_thre_arr=deepcopy(bin_thre_arr.cpu().numpy()), weights=self.weights, classes=self.classes, normal_class=self.normal_class, label=deepcopy(label.cpu().numpy()), scalar_output=deepcopy(scalar_output.cpu().numpy()))
                    with Pool(processes=NUMCPU) as pool:
                        for classIdx, thre_class, cm in pool.istarmap(partial_func, product([classIdx], np.arange(-0.3,0.3,0.1))):
                            thre_class_results.append([thre_class, cm])
                    optm_thre_class, optm_cm = thre_class_results[np.argmax([r[1] for r in thre_class_results])]
                    bin_thre_arr[classIdx] +=optm_thre_class
                self.bin_thre_arr = bin_thre_arr

                #  layer 3
                for classIdx in range(num_classes):
                    thre_class_results = []
                    partial_func = partial(find_thre_method1_level2, bin_thre_arr=deepcopy(bin_thre_arr.cpu().numpy()), weights=self.weights, classes=self.classes, normal_class=self.normal_class, label=deepcopy(label.cpu().numpy()), scalar_output=deepcopy(scalar_output.cpu().numpy()))
                    with Pool(processes=NUMCPU) as pool:
                        for classIdx, thre_class, cm in pool.istarmap(partial_func, product([classIdx], np.arange(-0.05,0.06,0.01))):
                            thre_class_results.append([thre_class, cm])
                    optm_thre_class, optm_cm = thre_class_results[np.argmax([r[1] for r in thre_class_results])]
                    bin_thre_arr[classIdx] +=optm_thre_class
                self.bin_thre_arr = bin_thre_arr

            else:
                # first round: 0.1~1 search for all classes
                thre_results = []
                for thre in np.arange(0,1.1,0.1):
                    thre = round(thre,2)
                    bin_thre_arr = torch.tensor([thre]*num_classes, dtype=torch.float32)
                    binary_output = torch.zeros_like(scalar_output, dtype=torch.float32)
                    binary_output[scalar_output>=bin_thre_arr] = 1
                    binary_output[scalar_output<bin_thre_arr] = 0
                    cm = compute_challenge_metric(self.weights, label.cpu().numpy(), binary_output.numpy(), self.classes, self.normal_class)
                    # cm = compute_challenge_metric(weights, labels, binary_output, classes, normal_class)
                    thre_results.append([thre, cm])
                optm_thre_base = thre_results[np.argmax([r[1] for r in thre_results])][0]
                bin_thre_arr = nn.Parameter(torch.tensor([optm_thre_base]*num_classes, dtype=torch.float32), requires_grad=False)
                self.bin_thre_arr = bin_thre_arr
    
                for classIdx in range(num_classes):
                    thre_class_results = []
                    for thre_class in np.arange(-0.05,0.06,0.01):
                        thre_class = round(thre_class, 2)
                        bin_thre_arr_tmp = deepcopy(bin_thre_arr)
                        bin_thre_arr_tmp[classIdx] += thre_class
                        binary_output = torch.zeros_like(scalar_output, dtype=torch.float32)
                        binary_output[scalar_output>=bin_thre_arr_tmp] = 1
                        binary_output[scalar_output<bin_thre_arr_tmp] = 0
                        cm = compute_challenge_metric(self.weights, label.cpu().numpy(), binary_output.numpy(), self.classes, self.normal_class)
                        thre_class_results.append([thre_class, cm])
                    optm_thre_class = thre_class_results[np.argmax([r[1] for r in thre_class_results])][0]
                    bin_thre_arr[classIdx] = optm_thre_base+optm_thre_class
                    # logger.info(classIdx, cm)
                self.bin_thre_arr = bin_thre_arr


#%%
def add_weight_decay(model, weight_decay=1e-4, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]


#%%
def find_thre_method1_level1(thre, weights, classes, normal_class, label, scalar_output):
    thre = round(thre,2)
    num_classes = scalar_output.shape[1]
    bin_thre_arr = np.asarray([thre]*num_classes).astype('float')
    binary_output = np.zeros_like(scalar_output).astype('float')
    scalar_output = scalar_output
    binary_output[scalar_output>=bin_thre_arr] = 1
    binary_output[scalar_output<bin_thre_arr] = 0
    cm = compute_challenge_metric(weights, label, binary_output, classes, normal_class)
    return thre, cm

def find_thre_method1_level2(classIdx, thre_class, bin_thre_arr, weights, classes, normal_class, label, scalar_output):
    thre_class = round(thre_class, 2)
    bin_thre_arr_tmp = deepcopy(bin_thre_arr)
    bin_thre_arr_tmp[classIdx] += thre_class
    binary_output = np.zeros_like(scalar_output).astype('float')
    binary_output[scalar_output>=bin_thre_arr_tmp] = 1
    binary_output[scalar_output<bin_thre_arr_tmp] = 0
    cm = compute_challenge_metric(weights, label, binary_output, classes, normal_class)
    return classIdx, thre_class, cm
