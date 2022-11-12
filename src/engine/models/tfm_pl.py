import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
from .base_pl import Classifier
from .transformer import TransformerEncoder, CNNFeatExtractor, CNN2dFeatExtractor
from .utils import CosineWarmupScheduler
from .loss import bce_logit_loss, cmloss

#%%
class TfmVanilla(Classifier):
    def __init__(self, param_model, random_state=0, class_weight=None):
        # param_model.dim_in=1
        # param_model.dim_out=1
        super(TfmVanilla, self).__init__(param_model, random_state, class_weight)
        # dim_hids = [param_model.input_size]+list(param_model.dim_hids)
        self.model = CnnTfmArch(param_model.in_channel, param_model.out_channel,
                                    param_model.num_layers, param_model.embed_dim,
                                    param_model.dim_feedforward, param_model.num_heads,
                                    param_model.dropout, param_model.out_dim,
                                    param_model.n_demo, param_model.output_size)

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

class CnnTfmArch(nn.Module):
    def __init__(self, in_channel, out_channel, num_layers, embed_dim,
                    dim_feedforward, num_heads, dropout, 
                    out_dim, n_demo, output_size):
        super().__init__()               
        '''
        in_channel, out_channel: params of CNN extractor
        num_layers, embed_dim, dim_feedforward, num_heads, dropout, out_dim: params of transformer
        out_dim: input of fully connected layer
        output_size: nb of target classes 
        '''
        self.n_demo = n_demo
        # Feature extractor
        # self.main_extract = CNNFeatExtractor(in_channel, out_channel)
        self.main_extract = CNN2dFeatExtractor(in_channel, out_channel)
        # Main model
        self.main =  TransformerEncoder(num_layers=num_layers,
                                input_dim=out_channel, 
                                dim_feedforward=dim_feedforward, 
                                num_heads=num_heads,
                                dropout=dropout)
        # FC layer
        self.main_fc = nn.Linear(embed_dim, out_dim)

        # classifier
        if not self.n_demo:
            print(self.n_demo)
            self.main_clf = nn.Linear(out_dim + self.n_demo, output_size)
        else: 
            self.main_clf = nn.Linear(out_dim, output_size)
        self.apply(init_weights)
    
    def forward(self, x, mask=None):
        '''
        x_demo: (n_batch, 2) # age, gender
        x_sig: (n_batch, n_lead, x_dim)
        out: (n_batch, x_dim)
        '''
        # ===== Feature extraction
        h = self.main_extract(x["signal"]) # (n_batch, n_channel, seq_len)
        h = torch.mean(h, dim=-2) # (n_batch, n_channel, f_domain, t_domain)

        # ===== Transformer 
        h = h.permute(0, 2, 1) # (n_batch, seq_len, n_channel)
        h = self.main(h, mask=mask)
        atten_maps = self.main.get_attention_maps(h)

        # ===== Global Average Pooling
        h = h.mean(dim=1)
        h = self.main_fc(h)

        # ===== Concat x_demo
        if not self.n_demo:
            h = torch.cat((h, x["age"].unsqueeze(-1), x["sex"]), dim=-1)
        out = self.main_clf(h)
        return out#, atten_maps


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