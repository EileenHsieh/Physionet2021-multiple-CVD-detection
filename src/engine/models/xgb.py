#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 19 17:48:40 2021

@author: chadyang
"""

from sklearn.multiclass import OneVsRestClassifier
from xgboost import XGBClassifier
from models.base import Classifier
import joblib
from sklearn.preprocessing import MultiLabelBinarizer


class Xgb(Classifier):
    def __init__(self, param_model, random_state):
        super(Xgb, self).__init__(param_model, random_state)
        self.model = OneVsRestClassifier(XGBClassifier(**param_model), n_jobs=param_model.n_jobs)
        
    def fit(self, loader):
        feas, labels = loader.dataset[:]
        self.model.fit(feas, labels)

    def predict(self, loader, return_label=False):
        feas, labels = loader.dataset[:]
        out_pred = self.model.predict(feas)
        out_label = labels
        if return_label: return out_pred, out_label

    def save(self, model_path):
        save_dict = {
            "model": self.model,
        }
        joblib.dump(save_dict, model_path)

    def load(self, model_path):
        checkpoint = joblib.load(model_path)
        self.model = checkpoint["modle"]

    
        
        
        
