import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
from copy import deepcopy
from .base_pl import Classifier, add_weight_decay
from .resnet import MyConv1dPadSame, MyMaxPool1dPadSame, BasicBlock
from .utils import CosineWarmupScheduler#, disable_bn, enable_bn
from .loss import bce_logit_loss, cmloss
from .ema import ModelEma
from .mixstyle import MixStyle
# from src.engine.losses.warp import WARPLoss
# from src.engine.optimizers.sam import SAM

import coloredlogs, logging
coloredlogs.install()
logger = logging.getLogger(__name__)  

#%%
class AttnRsn(Classifier):
    def __init__(self, param_model, random_state=0, pos_weight=None):
        # param_model.dim_in=1
        # param_model.dim_out=1
        super(AttnRsn, self).__init__(param_model, random_state, pos_weight)
        # dim_hids = [param_model.input_size]+list(param_model.dim_hids)
        if param_model.get("n_mix_block") and (param_model.get("mix_alpha")):
            self.model = AttnResNet(param_model.in_channel, param_model.base_filters,
                                    param_model.first_kernel_size, param_model.kernel_size, 
                                    param_model.stride, param_model.groups, param_model.n_block,
                                    param_model.output_size, param_model.n_demo, 
                                    param_model.is_se, param_model.n_mix_block, param_model.mix_alpha,
                                    num_heads=param_model.num_heads, emb_dim=param_model.emb_dim)
       
        else:
            self.model = AttnResNet(param_model.in_channel, param_model.base_filters,
                                        param_model.first_kernel_size, param_model.kernel_size, 
                                        param_model.stride, param_model.groups, param_model.n_block,
                                        param_model.output_size, param_model.n_demo, param_model.is_se,
                                        num_heads=param_model.num_heads, emb_dim=param_model.emb_dim)
        # exponential moving average (add model here instead of in base_pl to log correct model)
        if param_model.get("use_ema"):
            self.ema = ModelEma(self.model, param_model.use_ema)  # 0.9997^641=0.82
        
        if param_model.get("warp_loss") and (param_model.warp_loss):
            logger.info("!!!!!!!! is using warp loss !!!!!!!!")
            self.criterion = WARPLoss(max_num_trials=None)

        # custom optimizer
        if param_model.get("custom_optimizer"):
            logger.info("!!!!!!!! is using custom optimization !!!!!!!!")
            self.automatic_optimization=False


    def configure_optimizers(self):
        if self.param_model.get("custom_optimizer"):
            if self.param_model.custom_optimizer=='sam':
                logger.info("!!!!!!!! is using SAM !!!!!!!!")
                base_optimizer = torch.optim.Adam
                # optimizer = SAM(self.parameters(), base_optimizer, rho=0.05, adaptive=True, lr=0.1, momentum=0.9, weight_decay=0.0005)
                optimizer = SAM(self.parameters(), base_optimizer, rho=self.param_model.rho, lr=self.param_model.lr)
                self.lr_scheduler = {"scheduler":MultiStepLR(optimizer, milestones=[10, 20], gamma=0.1), "monitor":"val_loss"}
        else:        
            if self.param_model.get("true_weight_decay"):
                logger.warning("!!!!!!!! Using True Weight Decay !!!!!!!!")
                parameters = add_weight_decay(self.model, self.param_model.true_weight_decay)
                optimizer = torch.optim.Adam(params=parameters, lr=self.param_model.lr, weight_decay=0)
            else:
                optimizer = torch.optim.Adam(self.parameters(), lr=self.param_model.lr)
        if self.param_model.get("scheduler_WarmUp"):
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
        # exponential moving average (EMA)
        if self.param_model.get("use_ema") and (not self.training):
            logit = self.ema(x)
        else:
            logit = self.model(x)
        loss = self.criterion(logit, label)

        # custom optimizer
        if self.param_model.get("custom_optimizer") and self.training:
            optimizer = self.optimizers()

            # first forward-backward pass
            enable_bn(self.model)
            self.manual_backward(loss, optimizer)
            optimizer.first_step(zero_grad=True)

            # second forward-backward pass
            disable_bn(self.model)
            loss_2 = self.criterion(self.model(x), label) # make sure to do a full forward pass
            self.manual_backward(loss_2, optimizer)
            optimizer.second_step(zero_grad=True)

        return loss, logit.detach(), label.detach()

#%%

class AttnResNet(nn.Module):
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
                        groups, n_block, output_size, n_demo=None, is_se=False, n_mix_block=None, 
                        alpha=None, num_heads=8, emb_dim=320, downsample_gap=2, 
                        increasefilter_gap=3, use_bn=True, use_do=True, verbose=False):
        super(AttnResNet, self).__init__()
        
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
        self.n_leads = in_channels
        self.num_heads = num_heads
        self.emb_dim = emb_dim
        self.output_size = output_size

        self.downsample_gap = downsample_gap # 2 for base model
        self.increasefilter_gap = increasefilter_gap # 4 for base model

        if not n_mix_block is None:
            self.mixstyle = MixStyle(p=0.5, alpha=alpha)

        # first block
        self.first_block_conv = MyConv1dPadSame(in_channels=in_channels, out_channels=base_filters, 
                                                kernel_size=self.first_kernel_size, 
                                                stride=1, groups=self.groups)
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

        # Multihead ATtention
        # self.multihead_attn = nn.MultiheadAttention(256, self.num_heads)
        self.multihead_attn = MultiheadAttention(out_channels//self.n_leads, emb_dim, num_heads)

        # Classifier
        # if not self.n_demo is None:
        #     self.main_clf = nn.Linear(out_channels+n_demo, output_size)
        # else:
        self.main_clf = nn.Linear(out_channels, output_size)

        # multilable clf
        self.out_w = nn.Parameter(torch.zeros([self.output_size,emb_dim//output_size,1]))
        self.out_b = nn.Parameter(torch.zeros([self.output_size]))
        nn.init.xavier_uniform_(self.out_w)

    # def forward(self, x):
    def forward(self, x):
        '''
        x_demo: (n_batch, 2) # age, gender
        x_sig: (n_batch, n_lead, x_dim)
        out: (n_batch, x_dim)
        '''
        valid_leads = x["valid_leads"]

        assert len(x["signal"].shape) == 3
        assert len(valid_leads.shape) == 2
        
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
        h = self.final_relu(out) # (n_batch, out_channels, fea_dim)
        h = h.mean(-1) # (n_batch, out_channels)
        out = self.main_clf(h).squeeze(-1)




        # h = h.view(h.shape[0], self.n_leads, -1) # (n_batch, n_leads, lead_channels)
        # # print('final pooling', h.shape)

        # # Multihead Attention
        # mask = gen_mask(valid_leads, self.num_heads).to(h.device)
        # attn_out = self.multihead_attn(h, mask=mask) #  (n_batch, emb_dim)
        # h = attn_out.view(attn_out.shape[0], self.output_size, -1) #  (n_batch, output_size, embed_dim//output_size)
        # # print('Attention output', h.shape)

        # # ===== Concat x_demo
        # # if not self.n_demo is None:
        # #     # Concat demo data:(nb_batch, nb_ppg_segment, 116)
        # #     h = torch.cat((h, x["age"].unsqueeze(-1), x["sex"]), dim=-1)
        # # out = self.main_clf(h).squeeze(-1)
        
        # # multi label classify
        # out = torch.matmul(h.unsqueeze(-2), self.out_w).squeeze() + self.out_b

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
import math
def gen_mask(valid_leads, num_heads):
    valid_leads = valid_leads.unsqueeze(-1).float()
    mask = torch.matmul(valid_leads, valid_leads.transpose(-2, -1))
    mask = mask.unsqueeze(1)
    mask = mask.repeat(1,num_heads,1,1)
    return mask

def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1]
    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)
    # print("attn_logits",attn_logits.shape)
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
    attention = F.softmax(attn_logits, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention

class MultiheadAttention(nn.Module):

    def __init__(self, input_dim, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Stack all weight matrices 1...h together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        self.qkv_proj = nn.Linear(input_dim, 3*embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()


    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)


    def forward(self, x, mask=None, return_attention=False):
        batch_size, seq_length, _ = x.size()
        qkv = self.qkv_proj(x)

        # Separate Q, K, V from linear output
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3*self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3) # [Batch, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=-1)

        # Determine value outputs
        values, attention = scaled_dot_product(q, k, v, mask=mask)
        values = values.permute(0, 2, 1, 3) # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, self.embed_dim)
        o = self.o_proj(values)
        o = o.mean(dim=1)

        if return_attention:
            return o, attention
        else:
            return o


#%%
if __name__ == '__main__':
    def gen_mask(valid_leads, num_heads):
        valid_leads = valid_leads.unsqueeze(-1)
        mask = torch.matmul(valid_leads, valid_leads.transpose(-2, -1))
        mask = mask.unsqueeze(1)
        mask = mask.repeat(1,num_heads,1,1)
        return mask

    batch_size = 4
    seq_length = 12
    input_dim = 160
    embed_dim = 260
    num_heads = 26

    valid_leads = torch.ones([batch_size, seq_length])
    valid_leads[0, 3] = 0
    valid_leads[0, 5] = 0

    valid_leads = valid_leads.unsqueeze(-1)
    
    mask = torch.matmul(valid_leads, valid_leads.transpose(-2, -1))
    mask = mask.unsqueeze(1)
    mask = mask.repeat(1,num_heads,1,1)

    x = torch.randn(batch_size, seq_length, input_dim)
    model = MultiheadAttention(input_dim, embed_dim, num_heads)
    self = model
    o = model(x, mask, return_attention=False)


#%%
    model = AttnResNet(12, 120,
                15, 7, 
                3, 12, 10,
                26, None, True,
                num_heads=26, emb_dim=260)
    x = {}
    x["signal"] = torch.randn(4, 12, 5000)    
    pred = model(x)