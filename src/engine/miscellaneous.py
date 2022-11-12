#%%
import os 
os.chdir('/homes/GPU0/chadyang/Projects/Challenges/Physionet/cinc/scripts/PhysioNet-CinC-Challenges-2021')
from omegaconf import OmegaConf
import copy

EXPNAME = 'rsn_raw_bce'
config_template = OmegaConf.load(f'./config/hydra/{EXPNAME}.yaml')

# all sites
sites = {'6':['I', 'II', 'III', 'aVR', 'aVL', 'aVF'],
         '4':['I', 'II', 'III', 'V2'], 
         '3':['I', 'II', 'V2'], 
         '2':['I', 'II']}

# os.makedirs(save_root, exist_ok=True)
randi = 0
for num_lead, val in sites.items():
    config = copy.deepcopy(config_template)
    config.param_loader.num_leads = int(num_lead)
    config.param_loader.leads = val
    config.param_model.in_channel = int(num_lead)
    config.exp.exp_name = f'{EXPNAME}_l{num_lead}'
    randi +=1
    with open(f'./config/hydra/{EXPNAME}_l{num_lead}.yaml', 'w') as file:
        OmegaConf.save(config=config, f=file)


#%%
from glob import glob
all_configs = sorted(glob(f'config/hyperparameter/{model_name}/*yaml'))
print(len(all_configs))
base = f'./run_{model_name}_hyperparameter.sh'
with open(f'run_{model_name}_hyperparameter.sh', 'a+') as file:
    for conf in all_configs:
        file.write(f"python team_code_raw.py {conf}\n")


#%%
import torch.nn as nn
import torch
from functools import reduce

class SKConv(nn.Module):
    def __init__(self,in_channels,out_channels,stride=1,M=2,r=16,L=32):
        '''
        :param in_channels:  输入通道维度
        :param out_channels: 输出通道维度   原论文中 输入输出通道维度相同
        :param stride:  步长，默认为1
        :param M:  分支数
        :param r: 特征Z的长度，计算其维度d 时所需的比率（论文中 特征S->Z 是降维，故需要规定 降维的下界）
        :param L:  论文中规定特征Z的下界，默认为32
        '''
        super(SKConv,self).__init__()
        d=max(in_channels//r,L)   # 计算向量Z 的长度d
        self.M=M
        self.out_channels=out_channels
        self.conv=nn.ModuleList()  # 根据分支数量 添加 不同核的卷积操作
        for i in range(M):
            # 为提高效率，原论文中 扩张卷积5x5为 （3X3，dilation=2）来代替。 且论文中建议组卷积G=32
            self.conv.append(nn.Sequential(nn.Conv1d(in_channels,out_channels,3,stride,padding=1+i,dilation=1+i,groups=32,bias=False),
                                           nn.BatchNorm1d(out_channels),
                                           nn.ReLU(inplace=True)))
        self.global_pool=nn.AdaptiveAvgPool1d(1) # 自适应pool到指定维度    这里指定为1，实现 GAP
        self.fc1=nn.Sequential(nn.Conv1d(out_channels,d,1,bias=False),
                               nn.BatchNorm1d(d),
                               nn.ReLU(inplace=True))   # 降维
        self.fc2=nn.Conv1d(d,out_channels*M,1,1,bias=False)  # 升维
        self.softmax=nn.Softmax(dim=1) # 指定dim=1  使得两个全连接层对应位置进行softmax,保证 对应位置a+b+..=1
    def forward(self, input):
        batch_size=input.size(0)
        output=[]
        #the part of split
        for i,conv in enumerate(self.conv):
            #print(i,conv(input).size())
            output.append(conv(input))
        #the part of fusion
        U=reduce(lambda x,y:x+y,output) # 逐元素相加生成 混合特征U
        s=self.global_pool(U)
        z=self.fc1(s)  # S->Z降维
        a_b=self.fc2(z) # Z->a，b 升维  论文使用conv 1x1表示全连接。结果中前一半通道值为a,后一半为b
        a_b=a_b.reshape(batch_size,self.M,self.out_channels,-1) #调整形状，变为 两个全连接层的值
        a_b=self.softmax(a_b) # 使得两个全连接层对应位置进行softmax
        #the part of selection
        a_b=list(a_b.chunk(self.M,dim=1))#split to a and b   chunk为pytorch方法，将tensor按照指定维度切分成 几个tensor块
        a_b=list(map(lambda x:x.reshape(batch_size,self.out_channels,1),a_b)) # 将所有分块  调整形状，即扩展两维
        V=list(map(lambda x,y:x*y,output,a_b)) # 权重与对应  不同卷积核输出的U 逐元素相乘
        V=reduce(lambda x,y:x+y,V) # 两个加权后的特征 逐元素相加
        return V
class SKBlock(nn.Module):
    '''
    基于Res Block构造的SK Block
    ResNeXt有  1x1Conv（通道数：x） +  SKConv（通道数：x）  + 1x1Conv（通道数：2x） 构成
    '''
    expansion=2 #指 每个block中 通道数增大指定倍数
    def __init__(self,inplanes,planes,stride=1,downsample=None):
        super(SKBlock,self).__init__()
        self.conv1=nn.Sequential(nn.Conv1d(inplanes,planes,1,1,0,bias=False),
                                 nn.BatchNorm1d(planes),
                                 nn.ReLU(inplace=True))
        self.conv2=SKConv(planes,planes,stride)
        self.conv3=nn.Sequential(nn.Conv1d(planes,planes*self.expansion,1,1,0,bias=False),
                                 nn.BatchNorm1d(planes*self.expansion))
        self.relu=nn.ReLU(inplace=True)
        self.downsample=downsample
    def forward(self, input):
        shortcut=input
        output=self.conv1(input)
        output=self.conv2(output)
        output=self.conv3(output)
        if self.downsample is not None:
            shortcut=self.downsample(input)
        output+=shortcut
        return self.relu(output)
class SKNet(nn.Module):
    '''
    参考 论文Table.1 进行构造
    '''
    def __init__(self,in_channels, base_filters, first_kernel_size,
                 output_size, block=SKBlock,nums_block_list=[3, 4, 6, 3]):
        super(SKNet,self).__init__()
        self.base_filters=base_filters
        # in_channel=3  out_channel=64  kernel=7x7 stride=2 padding=3
        self.conv=nn.Sequential(nn.Conv1d(in_channels,self.base_filters,first_kernel_size,2,3,bias=False),
                                nn.BatchNorm1d(self.base_filters),
                                nn.ReLU(inplace=True))
        self.maxpool=nn.MaxPool1d(3,2,1) # kernel=3x3 stride=2 padding=1
        self.layer1=self._make_layer(block,self.base_filters*2,nums_block_list[0],stride=4) # 构建表中 每个[] 的部分
        self.layer2=self._make_layer(block,self.base_filters*4,nums_block_list[1],stride=4)
        # self.layer3=self._make_layer(block,512//2,nums_block_list[2],stride=2)
        # self.layer4=self._make_layer(block,1024//2,nums_block_list[3],stride=2)
        self.avgpool=nn.AdaptiveAvgPool1d(1) # GAP全局平均池化
        self.fc=nn.Linear(self.base_filters*4*block.expansion,output_size) # 通道 2048 -> 1000
        self.softmax=nn.Softmax(-1) # 对最后一维进行softmax
    def forward(self, x):
        input = x["signal"]
        output=self.conv(input)
        output=self.maxpool(output)
        output=self.layer1(output)
        output=self.layer2(output)
        # output=self.layer3(output)
        # output=self.layer4(output)
        output=self.avgpool(output)
        output=output.squeeze(-1).squeeze(-1)
        output=self.fc(output)
        output=self.softmax(output)
        return output
    def _make_layer(self,block,planes,nums_block,stride=1):
        downsample=None
        if stride!=1 or self.base_filters!=planes*block.expansion:
            downsample=nn.Sequential(nn.Conv1d(self.base_filters,planes*block.expansion,1,stride,bias=False),
                                     nn.BatchNorm1d(planes*block.expansion))
        layers=[]
        layers.append(block(self.base_filters,planes,stride,downsample))
        self.base_filters=planes*block.expansion
        for _ in range(1,nums_block):
            layers.append(block(self.base_filters,planes))
        return nn.Sequential(*layers)
def SKNet50(nums_class=1000):
    return SKNet(nums_class,SKBlock,[3, 4, 6, 3]) # 论文通过[3, 4, 6, 3]搭配出SKNet50
def SKNet18(in_channels, base_filters, first_kernel_size, output_size):
    return SKNet(in_channels, base_filters, first_kernel_size, output_size,SKBlock,[2,2,2,2])
#%%
if __name__=='__main__':
    x = torch.rand(2, 12, 500)
    model=SKNet18(12,256,15,26)
    y=model(x)
    print(y.shape) # shape [2,1000]

#%% Produce fake data
from helper_code import *

from scipy.io import loadmat, savemat
recording_file = './test_data/I0068.mat'
recording = loadmat(recording_file)['val']

recording = load_recording('./test_data/I0068.mat')
savemat(recording_file, {'val':recording[:6,:]})