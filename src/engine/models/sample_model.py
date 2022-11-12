#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 19 18:11:22 2021

@author: chadyang
"""


from models.base import Classifier

        
class SampleModel(Classifier):
    def __init__(self, param_model, random_state):
        super(SampleModel, self).__init__(param_model, random_state)
        self.model = 'a'
    
    def fit():
        pass
    
    def predict():
        pass
    
    def save():
        pass
    
    def load():
        pass
    
    
        
        
        
