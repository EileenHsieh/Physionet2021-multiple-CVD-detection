#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 20 10:32:02 2021

@author: chadyang
"""


from models.base import Classifier
from skmultilearn.problem_transform import BinaryRelevance
from xgboost import XGBClassifier


        
class Xgb(Classifier):
    def __init__(self, param_model, random_state):
        super(Xgb, self).__init__(param_model, random_state)
        self.model = BinaryRelevance(XGBClassifier(**param_model), require_dense=[False, True])
        self.
        
    def fit(loader):
        
        pass
    
    def predict():
        pass
    
    def save():
        pass
    
    def load():
        pass
    
    
        
        
        
