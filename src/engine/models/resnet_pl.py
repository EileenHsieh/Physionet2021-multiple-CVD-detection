import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
from .base_pl import Classifier, add_weight_decay
from .resnet import MyConv1dPadSame, MyMaxPool1dPadSame, BasicBlock
from .utils import CosineWarmupScheduler
from .loss import bce_logit_loss, cmloss
from .ema import ModelEma
from .mixstyle import MixStyle

import coloredlogs, logging
coloredlogs.install()
logger = logging.getLogger(__name__)  

#%%
class RsnVanilla(Classifier):
    def __init__(self, param_model, random_state=0, pos_weight=None):
        # param_model.dim_in=1
        # param_model.dim_out=1
        super(RsnVanilla, self).__init__(param_model, random_state, pos_weight)
        # dim_hids = [param_model.input_size]+list(param_model.dim_hids)
        if param_model.get("n_mix_block") and (param_model.get("mix_alpha")):
            self.model = ResNet1D(param_model.in_channel, param_model.base_filters,
                                    param_model.first_kernel_size, param_model.kernel_size, 
                                    param_model.stride, param_model.groups, param_model.n_block,
                                    param_model.output_size, param_model.n_demo, 
                                    param_model.is_se, param_model.n_mix_block, param_model.mix_alpha)
       
        else:
            self.model = ResNet1D(param_model.in_channel, param_model.base_filters,
                                        param_model.first_kernel_size, param_model.kernel_size, 
                                        param_model.stride, param_model.groups, param_model.n_block,
                                        param_model.output_size, param_model.n_demo, param_model.is_se)
        # exponential moving average (add model here instead of in base_pl to log correct model)
        if param_model.get("use_ema"):
            self.ema = ModelEma(self.model, param_model.use_ema)  # 0.9997^641=0.82


    def configure_optimizers(self):
        if self.param_model.get("true_weight_decay"):
            logger.warning("!!!!!!!! Using True Weight Decay !!!!!!!!")
            parameters = add_weight_decay(self.model, self.param_model.true_weight_decay)
            optimizer = torch.optim.Adam(params=parameters, lr=self.param_model.lr, weight_decay=0)
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.param_model.lr)
        if self.param_model.scheduler_WarmUp:
            logger.info("!!!!!!!! is using warm up !!!!!!!!")
            self.lr_scheduler = {"scheduler":CosineWarmupScheduler(optimizer,**(self.param_model.scheduler_WarmUp)), "monitor":"val_loss"}
            return [optimizer], self.lr_scheduler
        return optimizer

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        self.lr_scheduler["scheduler"].step() # Step per iteration
        if self.param_model.get("use_ema"):
            self.ema.update(self.model)


    def _shared_train_step(self, batch):
        x, label = batch
        if self.param_model.use_ema and (not self.training):
            logit = self.ema(x)
        else:
            logit,shallow_intance = self.model(x)
        loss = self.criterion(logit, label)
        return loss, logit, label


class ResNet1D(nn.Module):
    """
    
    Input:
        X: (n_samples, n_channel, n_length)
        Y: (n_samples)
        
    Output:
        out: (n_samples)
        
    Pararmetes:
        in_channels: dim of input, the same as n_channel
        base_filters: number of filters in the first several Conv layer, it will double at every 4 layers
        kernel_size: width of kernel
        stride: stride of kernel moving
        groups: set larget to 1 as ResNeXt
        n_block: number of blocks
        n_classes: number of classes
        
    """

    def __init__(self, in_channels, base_filters, first_kernel_size, kernel_size, stride, 
                        groups, n_block, output_size, n_demo, is_se=False, n_mix_block=None, 
                        alpha=None, downsample_gap=2, 
                        increasefilter_gap=2, use_bn=True, use_do=True, verbose=False):
        super(ResNet1D, self).__init__()
        
        self.verbose = verbose
        self.n_block = n_block
        self.first_kernel_size = first_kernel_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.use_bn = use_bn
        self.use_do = use_do
        self.n_demo = n_demo
        self.is_se = is_se
        self.n_mix_block = n_mix_block

        self.downsample_gap = downsample_gap # 2 for base model
        self.increasefilter_gap = increasefilter_gap # 4 for base model

        if not n_mix_block is None:
            self.mixstyle = MixStyle(p=0.5, alpha=alpha)

        # first block
        self.first_block_conv = MyConv1dPadSame(in_channels=in_channels, out_channels=base_filters, kernel_size=self.first_kernel_size, stride=1)
        self.first_block_bn = nn.BatchNorm1d(base_filters)
        self.first_block_relu = nn.ReLU()
        self.first_block_maxpool = MyMaxPool1dPadSame(kernel_size=self.stride)
        out_channels = base_filters
                
        # residual blocks
        self.basicblock_list = nn.ModuleList()
        for i_block in range(self.n_block):
            # is_first_block
            if i_block == 0:
                is_first_block = True
            else:
                is_first_block = False
            # downsample at every self.downsample_gap blocks
            if i_block % self.downsample_gap == 1:
                downsample = True
            else:
                downsample = False
            # in_channels and out_channels
            if is_first_block:
                in_channels = base_filters
                out_channels = in_channels
            else:
                # increase filters at every self.increasefilter_gap blocks
                in_channels = int(base_filters*2**((i_block-1)//self.increasefilter_gap))
                if (i_block % self.increasefilter_gap == 0) and (i_block != 0):
                    out_channels = in_channels * 2
                else:
                    out_channels = in_channels
            
            tmp_block = BasicBlock(
                in_channels=in_channels, 
                out_channels=out_channels, 
                kernel_size=self.kernel_size, 
                stride = self.stride, 
                groups = self.groups, 
                downsample=downsample, 
                use_bn = self.use_bn, 
                use_do = self.use_do, 
                is_first_block=is_first_block,
                is_se=self.is_se)
            self.basicblock_list.append(tmp_block)

        # final prediction
        self.final_bn = nn.BatchNorm1d(out_channels)
        self.final_relu = nn.ReLU(inplace=True)

        # Classifier
        if not self.n_demo:
            self.main_clf = nn.Linear(out_channels+n_demo, output_size)
        else:
            self.main_clf = nn.Linear(out_channels, output_size)

    # def forward(self, x):
    def forward(self, x):
        '''
        x_demo: (n_batch, 2) # age, gender
        x_sig: (n_batch, n_lead, x_dim)
        out: (n_batch, x_dim)
        '''
        assert len(x["signal"].shape) == 3

        # first conv
        if self.verbose:
            logger.info('input shape', x["signal"].shape)
        out = self.first_block_conv(x["signal"])
        if self.verbose:
            logger.info('after first conv', out.shape)
        if self.use_bn:
            out = self.first_block_bn(out)
        out = self.first_block_relu(out)
        out = self.first_block_maxpool(out)
        
        # residual blocks, every block has two conv
        for i_block in range(self.n_block):
            net = self.basicblock_list[i_block]
            if self.verbose:
                logger.info('i_block: {0}, in_channels: {1}, out_channels: {2}, downsample: {3}'.format(i_block, net.in_channels, net.out_channels, net.downsample))
            out = net(out)
            if self.verbose:
                logger.info(out.shape)
            if (not self.n_mix_block is None) and (i_block<self.n_mix_block):
                out = self.mixstyle(out)
                shallow_instance = out

        # final prediction
        if self.use_bn:
            out = self.final_bn(out)
        h = self.final_relu(out)
        h = h.mean(-1) # (n_batch, out_channels)
        # logger.info('final pooling', h.shape)

        # ===== Concat x_demo
        if not self.n_demo:
            # Concat demo data:(nb_batch, nb_ppg_segment, 116)
            h = torch.cat((h, x["age"].unsqueeze(-1), x["sex"]), dim=-1)
        out = self.main_clf(h)
        return out#, shallow_instance  

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
# %%
