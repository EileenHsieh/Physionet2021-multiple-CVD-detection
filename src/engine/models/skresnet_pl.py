import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
from .base_pl import Classifier
from .utils import CosineWarmupScheduler
from .loss import bce_logit_loss, cmloss
from .skresnet import SKRsnClf

#%%
class SkResnet(Classifier):
    def __init__(self, param_model, random_state=0, class_weight=None):
        # param_model.dim_in=1
        # param_model.dim_out=1
        super(SkResnet, self).__init__(param_model, random_state, class_weight)
        # dim_hids = [param_model.input_size]+list(param_model.dim_hids)
        self.model = SKRsnClf(param_model.output_size, param_model.n_demo,
                              in_channels=param_model.in_channel, 
                              base_filters=param_model.base_filters,
                              first_kernel_size=param_model.first_kernel_size, 
                              kernel_size=param_model.kernel_size, 
                              stride=param_model.stride, 
                              groups=param_model.groups, 
                              n_block=param_model.n_block,
                              is_se=param_model.is_se)
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.param_model.lr)
        if self.param_model.scheduler_WarmUp:
            print("!!!!!!!! is using warm up !!!!!!!!")
            self.lr_scheduler = {"scheduler":CosineWarmupScheduler(optimizer,**(self.param_model.scheduler_WarmUp)), "monitor":"val_loss"}
            return [optimizer], self.lr_scheduler
        return optimizer

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        self.lr_scheduler["scheduler"].step() # Step per iteration
    
    def _shared_step(self, batch):
        x, label = batch
        logit = self.model(x)
        if self.param_model.is_cm_loss:
            loss = cmloss(logit, label)
        else:
            # loss = self.criterion(logit, label)
            loss = bce_logit_loss(logit, label, self.param_model.bce_loss_type)
            
        return loss, logit, label


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