import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
from .base_pl import Classifier, add_weight_decay
from .resnet import MyConv1dPadSame, MyMaxPool1dPadSame, BasicBlock
from .utils import CosineWarmupScheduler#, disable_bn, enable_bn
from .loss import bce_logit_loss, cmloss
from .ema import ModelEma
from .mixstyle import MixStyle
from .resnet import SE_Block, MyConv1dPadSame
# from src.engine.optimizers.sam import SAM

import coloredlogs, logging
coloredlogs.install()
logger = logging.getLogger(__name__)  

#%%
class RsnClassic(Classifier):
    def __init__(self, param_model, random_state=0, pos_weight=None):
        # param_model.dim_in=1
        # param_model.dim_out=1
        super(RsnClassic, self).__init__(param_model, random_state, pos_weight)
        # dim_hids = [param_model.input_size]+list(param_model.dim_hids)
        if param_model.get("n_mix_block") and (param_model.get("mix_alpha")):
            self.model = ResNet50(input_channel=param_model.in_channel, 
                                  num_classes=param_model.output_size, 
                                  first_ksize=param_model.first_kernel_size, 
                                  kernel_size=param_model.kernel_size,  
                                  base_filters=param_model.base_filters,
                                  is_se=param_model.is_se, 
                                  groups=param_model.groups)
                                #   n_mix_block=param_model.n_mix_block, 
                                #   mix_alpha=param_model.mix_alpha)
            
        else:
            self.model = ResNet50(input_channel=param_model.in_channel, 
                                  num_classes=param_model.output_size, 
                                  first_ksize=param_model.first_kernel_size,
                                  kernel_size=param_model.kernel_size,  
                                  base_filters=param_model.base_filters,
                                  is_se=param_model.is_se, 
                                  groups=param_model.groups)
        # exponential moving average (add model here instead of in base_pl to log correct model)
        if param_model.get("use_ema"):
            self.ema = ModelEma(self.model, param_model.use_ema)  # 0.9997^641=0.82
        
        # custom optimizer
        if param_model.get("custom_optimizer"):
            logger.info("!!!!!!!! is using custom optimization !!!!!!!!")
            self.automatic_optimization=False


    def configure_optimizers(self):
        # if self.param_model.get("custom_optimizer"):
        #     if self.param_model.custom_optimizer=='sam':
        #         logger.info("!!!!!!!! is using SAM !!!!!!!!")
        #         base_optimizer = torch.optim.Adam
        #         # optimizer = SAM(self.parameters(), base_optimizer, rho=0.05, adaptive=True, lr=0.1, momentum=0.9, weight_decay=0.0005)
        #         optimizer = SAM(self.parameters(), base_optimizer, rho=self.param_model.rho, lr=self.param_model.lr)
        #     elif self.param_model.custom_optimizer=='cdr':
        #         optimizer = torch.optim.Adam(self.parameters(), lr=self.param_model.lr)
        # else:
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


    def _shared_step(self, batch):
        x, label = batch
        if self.param_model.get("use_ema"): 
            if (self.param_model.use_ema) and (not self.training):
                logit = self.ema(x)
            else:
                logit = self.model(x)
        else: logit = self.model(x)
        loss = self.criterion(logit, label)

        # # custom optimizer
        # if self.param_model.get("custom_optimizer"):
        #     if self.param_model.custom_optimizer=='sam':
        #         optimizer = self.optimizers()

        #         # first forward-backward pass
        #         enable_bn(self.model)
        #         self.manual_backward(loss, optimizer)
        #         optimizer.first_step(zero_grad=True)

        #         # second forward-backward pass
        #         disable_bn(self.model)
        #         loss_2 = self.criterion(self.model(x), label) # make sure to do a full forward pass
        #         self.manual_backward(loss_2, optimizer)
        #         optimizer.second_step(zero_grad=True)

        #     elif self.param_model.custom_optimizer=='cdr':
        #         optimizer = self.optimizers()
        #         optimizer.zero_grad()
        #         self.manual_backward(loss)

        #         to_concat_g = []
        #         to_concat_v = []
        #         for name, param in self.model.named_parameters():
        #             if param.dim() in [2, 3]:
        #                 to_concat_g.append(param.grad.data.view(-1))
        #                 to_concat_v.append(param.data.view(-1))
        #         all_g = torch.cat(to_concat_g)
        #         all_v = torch.cat(to_concat_v)
        #         metric = torch.abs(all_g * all_v)
        #         num_params = all_v.size(0)
        #         nz = int(self.param_model.noise_label_rate * num_params)
        #         top_values, _ = torch.topk(metric, nz)
        #         thresh = top_values[-1]

        #         for name, param in self.model.named_parameters():
        #             if param.dim() in [2, 4]:
        #                 mask = (torch.abs(param.data * param.grad.data) >= thresh).type(torch.cuda.FloatTensor)
        #                 mask = mask * self.param_model.noise_label_rate
        #                 param.grad.data = mask * param.grad.data

        #         optimizer.step()
        return loss, logit.detach(), label.detach()

#%%
class Linear(nn.Module):
    def __init__(self, in_features, out_features):

        super(Linear, self).__init__()
        self.w = nn.Parameter(torch.randn(in_features, out_features))
         
    def forward(self, x):
        x = x.mm(self.w)
        return x 
    
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, is_se=False, groups=1):
        super(BasicBlock, self).__init__()
        self.is_se = is_se
        
        # self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, groups=groups)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
        self.bn2 = nn.BatchNorm1d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False, groups=groups),
                nn.BatchNorm1d(self.expansion*planes)
            )

        # Squeeze and excitation layer
        if self.is_se:  self.se = SE_Block(planes, 16)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # Squeeze and excitation layer
        if self.is_se: 
            out = self.se(out)

        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_planes, planes, kernel_size=3, stride=1, is_se=False, groups=1):
        super(Bottleneck, self).__init__()
        self.is_se = is_se  
        # self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=1, bias=False, groups=groups)
        self.conv1 = MyConv1dPadSame(
            in_channels=in_planes, 
            out_channels=planes, 
            kernel_size=1, 
            stride=1,
            groups=groups)
        self.bn1 = nn.BatchNorm1d(planes)
        # self.conv2 = nn.Conv1d(planes, planes, kernel_size=kernel_size, stride=stride, padding=1, bias=False, groups=groups)
        self.conv2 = MyConv1dPadSame(
            in_channels=planes, 
            out_channels=planes, 
            kernel_size=kernel_size, 
            stride=stride,
            groups=groups)
        self.bn2 = nn.BatchNorm1d(planes)
        # self.conv3 = nn.Conv1d(planes, self.expansion*planes, kernel_size=1, bias=False, groups=groups)
        self.conv3 = MyConv1dPadSame(
            in_channels=planes, 
            out_channels=self.expansion*planes, 
            kernel_size=1, 
            stride=1,
            groups=groups)
        self.bn3 = nn.BatchNorm1d(self.expansion*planes)

        # self.max_pool = MyMaxPool1dPadSame(kernel_size=stride)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False, groups=groups),
                nn.BatchNorm1d(self.expansion*planes)
            )

        # Squeeze and excitation layer
        if self.is_se:  self.se = SE_Block(self.expansion*planes, 16)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        # Squeeze and excitation layer
        if self.is_se: 
            out = self.se(out)

        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, input_channel, num_classes, first_ksize, kernel_size, base_filters, groups=1, is_se=False,
                 n_mix_block=None, alpha=None):
        super(ResNet, self).__init__()

        self.is_se = is_se
        self.in_planes = base_filters
        self.conv1 = nn.Conv1d(input_channel, base_filters, kernel_size=first_ksize, stride=1, padding=0, bias=False, groups=groups)
        self.bn1 = nn.BatchNorm1d(base_filters)
        self.layer1 = self._make_layer(block, base_filters, num_blocks[0], kernel_size=kernel_size, stride=1, is_se=self.is_se, groups=groups)
        self.layer2 = self._make_layer(block, base_filters*2, num_blocks[1], kernel_size=kernel_size, stride=3, is_se=self.is_se, groups=groups)
        self.layer3 = self._make_layer(block, base_filters*4, num_blocks[2], kernel_size=kernel_size, stride=3, is_se=self.is_se, groups=groups)
        self.layer4 = self._make_layer(block, base_filters*8, num_blocks[3], kernel_size=kernel_size, stride=3, is_se=self.is_se, groups=groups)
        self.linear = nn.Linear(base_filters*8*block.expansion, num_classes)
        self.avgpool = nn.AdaptiveAvgPool1d((1))

        
        self.n_mix_block = n_mix_block

        if not n_mix_block is None:
            self.mixstyle = MixStyle(p=0.5, alpha=alpha)
        
        
           
    def _make_layer(self, block, planes, num_blocks, kernel_size, stride, is_se, groups=1):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, kernel_size, stride, is_se=is_se, groups=groups))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x["signal"]
        x = x.float()
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.squeeze(-1)
        out = self.linear(out)
        return out
        
def ResNet18(input_channel, num_classes, first_ksize, kernel_size, base_filters, is_se, groups):
    return ResNet(BasicBlock, [2,2,2,2], input_channel, num_classes, first_ksize, kernel_size, base_filters, is_se=is_se, groups=groups)
    
def ResNet34(input_channel, num_classes, first_ksize, kernel_size, base_filters, is_se, groups):
    return ResNet(BasicBlock, [3,4,6,3], input_channel, num_classes, first_ksize, kernel_size, base_filters, is_se=is_se, groups=groups)

def ResNet50(input_channel, num_classes, first_ksize, kernel_size, base_filters, is_se, groups):
    return ResNet(Bottleneck, [2,2,2,2], input_channel, num_classes, first_ksize, kernel_size, base_filters, is_se=is_se, groups=groups)

def ResNet101(input_channel, num_classes, first_ksize, kernel_size, base_filters, is_se, groups):
    return ResNet(Bottleneck, [3,4,23,3], input_channel, num_classes, first_ksize, kernel_size, base_filters, is_se=is_se, groups=groups)

def ResNet152(input_channel, num_classes, first_ksize, kernel_size, base_filters, is_se, groups):
    return ResNet(Bottleneck, [3,8,36,3], input_channel, num_classes, first_ksize, kernel_size, base_filters, is_se=is_se, groups=groups)

#%%
if __name__=='__main__':
    import torch
    x = {"signal": torch.randn(4,12,5000)}
    model = ResNet50(12, 26, 15, 7, 64, True, 1)
    pred = model(x)