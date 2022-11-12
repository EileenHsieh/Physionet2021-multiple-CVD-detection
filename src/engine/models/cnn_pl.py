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
import omegaconf
from .base_pl import Classifier

#%%
class CnnVanilla(Classifier):
    def __init__(self, param_model, random_state=0, class_weight=None):
        with omegaconf.open_dict(param_model):
            param_model.dim_in=1
            param_model.dim_out=1
        super(CnnVanilla, self).__init__(param_model, random_state, class_weight)
        self.model = CnnVanillaArch(param_model.dim_hids, param_model.output_size, param_model.dropout_prob)


class CnnVanillaArch(nn.Module):
    def __init__(self, dim_hids, dim_out, drop_p=0.5):
        super(CnnVanillaArch, self).__init__()
        self.conv1 = nn.Conv1d(12, 1, 3, stride=2)
        self.d1 = nn.Linear(2499,26)
        self.apply(init_weights)

    def forward(self, x):
        hid = F.relu(self.conv1(x["signal"])).squeeze(1)
        out = self.d1(hid)
        return out


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