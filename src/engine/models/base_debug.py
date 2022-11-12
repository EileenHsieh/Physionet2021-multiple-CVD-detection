#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 23 18:27:01 2021

@author: chadyang
"""

import sys
# sys.path.append("/workspace/PhysioNet-CinC-Challenges-2021")
from sklearn.impute import SimpleImputer
from helper_code import *
import copy
import numpy as np
import joblib


import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from tqdm import tqdm

import coloredlogs, logging
coloredlogs.install()
logger = logging.getLogger(__name__)  




def to_tensor(x):
    if not torch.is_tensor(x): x = torch.tensor(x)
    if torch.cuda.is_available(): x = x.cuda()
    return x

def to_numpy(x):
    if x.is_cuda: return x.detach().cpu().data.numpy()
    return x.detach().data.numpy()

class Dropout(nn.Module):
    def __init__(self, p=0.5):
        super(Dropout, self).__init__()
        self.p = p
        
    def forward(self, x):
        if self.training:
            prob_mask = torch.FloatTensor(x.shape).uniform_(0,1).to(x.device) <= self.p
            return x * prob_mask
        return x * self.p

class Classifier:
    DEFAULTS = {}   
    _engine = None
    _optimizer = None
    def __init__(self, param_model, random_state):
        self.param_model = param_model
        # Ensure reproducibility
        np.random.seed(random_state)
        torch.manual_seed(random_state)

        # set device
        if torch.cuda.is_available():  self.DEVICE = torch.device("cuda:0")
        else:  self.DEVICE = torch.device("cpu")

    def fit(self, loader):
        if self.param_model.use_scheduler:
            logger.info("Running experiment usign scheduler")
            trainer = torch.optim.lr_scheduler.OneCycleLR(self._optimizer, max_lr=self.param_model.lr, steps_per_epoch=len(loader), epochs=self.param_model.N_epoch)
        pos_weight = torch.ones([self.param_model.output_size]).cuda()  # All weights are equal to 1
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

#%%
        pos_weight = torch.ones([24]).cuda()  # All weights are equal to 1
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        # model = nn.Linear(998, 24).cuda()
        model = general_model._engine
        # optimizer = optim.SGD(model.parameters(), lr=1e-1)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        # for epoch in tqdm(range(self.param_model.N_epoch)):
        for epoch in tqdm(range(1000)):
            epoch_loss = []

            for itr, (x_data, y) in enumerate(loader):
                
                # Convert the input as tensor
                # model._optimizer.zero_grad()
                x_data = to_tensor(x_data)
                y = to_tensor(y)

                # self._engine.to(self.DEVICE)
                # pred = self._engine(x_data)  
                # pred = model._engine(x_data)  
                optimizer.zero_grad()
                pred = model(x_data)
                # assert pred.shape == y.shape
                total_loss = criterion(pred, y)  # -log(sigmoid(1.5))

                # self._optimizer.zero_grad()
                total_loss.backward()
                # self._optimizer.step()
                # model._optimizer.step()
                optimizer.step()
            print(epoch, total_loss)
#%%
                epoch_loss.append(total_loss.item())
                
                if self.param_model.use_scheduler:
                    trainer.step()
                    
            # Log diagnostics
            if (epoch+1) % self.param_model.sample_step == 0:
                print("\r[{}/{}] Loss: {:.8f}".format(epoch+1, self.param_model.N_epoch, np.mean(epoch_loss)), end="\r")        
                
    def calibrate(self, loader, keep_dropout=True):
        # Initialize the layers
        engine_ft = copy.deepcopy(self._engine)            
        for item in engine_ft._modules:
            # Skip non-main layers
            if "main" not in item: continue
                
            # Remove dropout
            if keep_dropout == False:
                if item == "main":
                    engine_ft.main = nn.Sequential(*[x for x in engine_ft._modules[item] if isinstance(x, Dropout) == False])
                if item == "main_mlp":
                    engine_ft.main_mlp = nn.Sequential(*[x for x in engine_ft.main_mlp if isinstance(x, Dropout) == False])
                if item == "main_cnn":
                    engine_ft.main_cnn = nn.Sequential(*[x for x in engine_ft.main_cnn if isinstance(x, Dropout) == False])

            # Freeze layers
            for p in engine_ft._modules[item].parameters():
                p.requires_grad = False
                    
        # Initialize optimizer
        optimizer_ft = torch.optim.AdamW(engine_ft.parameters(), lr=self.lr_cal)                                
        engine_ft.train()
    
        # Run the calibration epochs                
        for epoch in range(self.N_epoch_calibration):
            for (x_data, x_cat, x_signal, y) in loader:
                # Convert the input as tensor
                x_data = to_tensor(x_data)
                x_cat = to_tensor(x_cat)
                x_signal = to_tensor(x_signal)
                y = to_tensor(y)
                
                # Compute loss
                pred = engine_ft(x_data, x_cat, x_signal)  
                assert pred.shape == y.shape
                loss_regression = torch.mean((pred - y)**2)

                # Regularization
                loss_l2 = 0
                for p1, p2 in zip(engine_ft.clf.parameters(), self._engine.clf.parameters()):
                    loss_l2 = torch.mean((p1-p2)**2)
              
                total_loss = loss_regression + self.config.param_model.lambda_cal_l2 * loss_l2
                
                optimizer_ft.zero_grad()
                total_loss.backward()
                optimizer_ft.step()
                            
        for item in engine_ft._modules:
            # Skip non-main layers
            if "main" not in item: continue
                
            # Freeze layers
            for p in engine_ft._modules[item].parameters():
                p.requires_grad = True
                
        self._engine = copy.deepcopy(engine_ft)
    
    def predict(self, loader, return_label=False):
        # Output buffer
        out_pred = []
        out_label = []

        # Inference
        with torch.no_grad():
            self._engine.eval() 
            self._engine.to(self.DEVICE)   
            for (x_data, y) in loader:

                x_data = to_tensor(x_data)

                y = to_tensor(y)
                pred = self._engine(x_data)

                # Make sure the prediction and the label has the same shape
                assert pred.shape == y.shape
                
                out_pred.append(to_numpy(pred))
                out_label.append(to_numpy(y))

            self._engine.train()

        # Stacking
        out_pred = np.concatenate(out_pred, axis=0)
        out_label = np.concatenate(out_label, axis=0)

        if return_label: return out_pred, out_label
        
    def save(self, model_path):
        save_dict = {
            "model_state_dict": self._engine.state_dict(),
            "optimizer_state_dict": self._optimizer.state_dict()
        }
        torch.save(save_dict, model_path)
        
    def load(self, model_path):
        checkpoint = torch.load(model_path)
        self._engine.load_state_dict(checkpoint["model_state_dict"])
        self._optimizer.load_state_dict(checkpoint["optimizer_state_dict"])


class SimpleClassifier:
    DEFAULTS = {}   
    _engine = None
    _optimizer = None
    _classes = None
    _leads = None
    def __init__(self, config, random_state):
        self.config = config
        # Ensure reproducibility
        np.random.seed(random_state)

    def fit(self, loader):
        data, labels, classes = loader
        logger.info("Batch data size: %s" %str(data.shape))
        leads = twelve_leads
        feature_indices = [twelve_leads.index(lead) for lead in leads] + [12, 13]
        features = data[:, feature_indices]
        imputer = SimpleImputer().fit(features)
        features = imputer.transform(features)

        self._classes = classes
        self._leads = leads
        self._engine.fit(features, labels) 
                
    def predict(self, loader, return_label=False):
        pass

    def save(self):
        # Construct a data structure for the model and save it.
        d = {'classes': self._classes, 'leads': self._leads, 'classifier': self._engine}
        joblib.dump(d, './testtoymodel.pkl', protocol=0)

    def load(self, model_path):
        pass
