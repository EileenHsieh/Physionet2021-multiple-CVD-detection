#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 23 18:35:30 2021

@author: chadyang
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base import Classifier, Dropout

class MLPToyArch(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, n_layers, dropout_prob):
        super(MLPToyArch, self).__init__()
        # Regressor
        self.clf = nn.Linear(input_size, output_size)

    def forward(self, x_data):
        # Use only the tabular data
        x = x_data

        # Rank for X should be 2: (N, F)
        if len(x.shape) !=2:
            raise RuntimeError("Tensor rank ({}) is not supported!".format(len(x.shape)))
            
        y = self.clf(x)
        return  y # output logits

class MLPToy(Classifier):
    def __init__(self, param_model, random_state):
        super(MLPToy, self).__init__(param_model, random_state)
        self._engine = MLPToyArch(param_model.input_size, param_model.output_size, param_model.hidden_size, 
                                        param_model.n_layers, param_model.dropout_prob)
        self._optimizer = torch.optim.AdamW(self._engine.parameters(), lr=param_model.lr)  # Optimizer
        if torch.cuda.is_available():
            self._engine.cuda()