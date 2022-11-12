#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 28 10:52:20 2021

@author: chadyang
"""

import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
from .base_pl import Classifier

#%%
class MLPVanilla(Classifier):
    def __init__(self, param_model, random_state=0, class_weight=None):
        super(MLPVanilla, self).__init__(param_model, random_state, class_weight)
    # def __init__(self, random_state=0, class_weight=None, **kwargs):
    #     super(MLPVanilla, self).__init__(random_state=random_state, class_weight=class_weight, **kwargs)
        dim_hids = [param_model.input_size]+list(param_model.dim_hids)
        self.model = MLPVanillaArch(dim_hids, param_model.output_size, param_model.dropout_prob)
        # dim_hids = [input_size]+list(dim_hids)
        # self.model = MLPVanillaArch(dim_hids, output_size, dropout_prob)


class MLPVanillaArch(nn.Module):
    def __init__(self, dim_hids, dim_out, drop_p=0.5):
        super(MLPVanillaArch, self).__init__()
        dim_hids = deepcopy(dim_hids)
        self.MLP = nn.Sequential()
        for i, (in_size, out_size) in enumerate( zip(dim_hids[:-1], dim_hids[1:]) ):
            self.MLP.add_module(name="L%i"%(i), module=nn.Linear(in_size, out_size)),
            self.MLP.add_module(name="B%i"%(i), module=nn.BatchNorm1d(out_size)),
            self.MLP.add_module(name="A%i"%(i), module=nn.LeakyReLU()),
            self.MLP.add_module(name="D%i"%(i), module=nn.Dropout(p=drop_p))
        self.out = nn.Linear(dim_hids[-1], dim_out)
        self.apply(init_weights)

    def forward(self, x):
        x = self.MLP(x)
        x = self.out(x)
        return x


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight.data, gain=1.414)