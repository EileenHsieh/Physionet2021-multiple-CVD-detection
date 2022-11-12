#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce
from .resnet import MyConv1dPadSame, MyMaxPool1dPadSame, SE_Block

#%%
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
        # self.fc1=nn.Sequential(nn.Conv1d(out_channels,d,1,bias=False),
        #                        nn.BatchNorm1d(d),
        #                        nn.ReLU(inplace=True))   # 降维
        self.fc1=nn.Sequential(MyConv1dPadSame(in_channels=out_channels,
                                               out_channels=d,
                                               kernel_size=1,
                                               stride=1),
                               nn.BatchNorm1d(d),
                               nn.ReLU(inplace=True))   # 降维
        # self.fc2=nn.Conv1d(d,out_channels*M,1,1,bias=False)  # 升维
        self.fc2=MyConv1dPadSame(in_channels=d,
                                 out_channels=out_channels*M,
                                 kernel_size=1,
                                 stride=1)  # 升维
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

class SKBasicBlock(nn.Module):
    """
    ResNet Basic Block
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, 
                groups, downsample, use_bn, use_do, is_first_block=False, is_se=False):
        super(SKBasicBlock, self).__init__()
        
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.stride = stride
        self.groups = groups
        self.downsample = downsample
        if self.downsample:
            self.stride = stride
        else:
            self.stride = 1
        self.is_first_block = is_first_block
        self.is_se = is_se
        self.use_bn = use_bn
        self.use_do = use_do

        # the first conv
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.relu1 = nn.ReLU()
        self.do1 = nn.Dropout(p=0.5)
        self.conv1 = MyConv1dPadSame(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=1, 
            stride=self.stride,
            groups=1)

        # the second conv
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.do2 = nn.Dropout(p=0.5)
        self.conv2 = MyConv1dPadSame(
            in_channels=out_channels, 
            out_channels=out_channels, 
            kernel_size=1, 
            stride=1,
            groups=1)

        # SK layer
        self.sk=SKConv(out_channels,out_channels,stride=1)

        self.max_pool = MyMaxPool1dPadSame(kernel_size=self.stride)

        # Squeeze and excitation layer
        if self.is_se:  self.se = SE_Block(out_channels, 16)

    def forward(self, x):
        identity = x
        
        # the first conv
        out = x
        if not self.is_first_block:
            if self.use_bn:
                out = self.bn1(out)
            out = self.relu1(out)
            if self.use_do:
                out = self.do1(out)
        out = self.conv1(out)

        # SK layer
        out = self.sk(out)
        
        # the second conv
        if self.use_bn:
            out = self.bn2(out)
        out = self.relu2(out)
        if self.use_do:
            out = self.do2(out)
        out = self.conv2(out)
        # if downsample, also downsample identity
        if self.downsample:
            identity = self.max_pool(identity)
            # print("skblock downsample identity", identity.shape)
            
        # if expand channel, also pad zeros to identity
        if self.out_channels != self.in_channels:
            identity = identity.transpose(-1,-2)
            ch1 = (self.out_channels-self.in_channels)//2
            ch2 = self.out_channels-self.in_channels-ch1
            identity = F.pad(identity, (ch1, ch2), "constant", 0)
            identity = identity.transpose(-1,-2)
            # print("skblock expand channel identity", identity.shape)
        
        # Squeeze and excitation layer
        if self.is_se: 
            out = self.se(out)
        # shortcut
        out += identity

        return out

class SKResNetArch(nn.Module):
    """
    
    Input:
        X: (n_samples, n_channel, n_length)
        Y: (n_samples)
        
    Output:
        out: (n_samples, n_channel, fea_dim)
        
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
                        groups, n_block, is_se=False, downsample_gap=2, 
                        increasefilter_gap=2, use_bn=True, use_do=True, verbose=False):
        super(SKResNetArch, self).__init__()
        
        self.verbose = verbose
        self.n_block = n_block
        self.first_kernel_size = first_kernel_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.use_bn = use_bn
        self.use_do = use_do
        self.is_se = is_se

        self.downsample_gap = downsample_gap # 2 for base model
        self.increasefilter_gap = increasefilter_gap # 4 for base model

        # first block
        self.first_block_conv = MyConv1dPadSame(in_channels=in_channels, out_channels=base_filters, kernel_size=self.first_kernel_size, stride=1)
        self.first_block_bn = nn.BatchNorm1d(base_filters)
        self.first_block_relu = nn.ReLU()
        self.first_block_maxpool = MyMaxPool1dPadSame(kernel_size=self.stride)
        self.out_channels = base_filters
                
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
                self.out_channels = in_channels
            else:
                # increase filters at every self.increasefilter_gap blocks
                in_channels = int(base_filters*2**((i_block-1)//self.increasefilter_gap))
                if (i_block % self.increasefilter_gap == 0) and (i_block != 0):
                    self.out_channels = in_channels * 2
                else:
                    self.out_channels = in_channels
            
            tmp_block = SKBasicBlock(
                in_channels=in_channels, 
                out_channels=self.out_channels, 
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
        self.final_bn = nn.BatchNorm1d(self.out_channels)
        self.final_relu = nn.ReLU(inplace=True)

    # def forward(self, x):
    def forward(self, x):
        '''
        x["age"]: (n_batch, 1) # age, gender
        x["sex"]: (n_batch, 1) # gender
        x["signal"]: (n_batch, n_lead, x_dim)
        out: (n_batch, n_channel, feat_dim)
        '''
        assert len(x["signal"].shape) == 3

        # first conv
        if self.verbose:
            print('input shape', x["signal"].shape)
        out = self.first_block_conv(x["signal"])
        if self.verbose:
            print('after first conv', out.shape)
        if self.use_bn:
            out = self.first_block_bn(out)
        out = self.first_block_relu(out)
        out = self.first_block_maxpool(out)
        
        # residual blocks, every block has two conv
        for i_block in range(self.n_block):
            net = self.basicblock_list[i_block]
            if self.verbose:
                print('i_block: {0}, in_channels: {1}, out_channels: {2}, downsample: {3}'.format(i_block, net.in_channels, net.out_channels, net.downsample))
            out = net(out)
            if self.verbose:
                print(out.shape)

        # final prediction
        if self.use_bn:
            out = self.final_bn(out)
        h = self.final_relu(out)

        return h  

class SKRsnClf(nn.Module):
    def __init__(self, output_size, n_demo, **rsn_args):
        super().__init__()
        self.main_extract = SKResNetArch(**rsn_args)
        out_channels = self.main_extract.out_channels
        self.n_demo = n_demo
        #------ Classifier
        if self.n_demo!='None':
            self.main_clf = nn.Linear(out_channels+self.n_demo, output_size)
        else:
            self.main_clf = nn.Linear(out_channels, output_size)
    
    def forward(self, x):
        '''
        x["age"]: (n_batch, 1) # age, gender
        x["sex"]: (n_batch, 1) # gender
        x["signal"]: (n_batch, n_lead, x_dim)
        out: (n_batch, n_channel, feat_dim)
        '''
        h = self.main_extract(x)
        h = h.mean(-1)
        if self.n_demo!='None':
            # Concat demo data:(nb_batch, nb_ppg_segment, 116)
            h = torch.cat((h, x["age"].unsqueeze(-1), x["sex"]), dim=-1)
        out = self.main_clf(h)
        return out 

#%%
if __name__=='__main__':
#%% Run ResNetArch
    resnet = SKResNetArch(12, 256, 15, 7, 3, 2, 1)
    x_demo = torch.randn(4, 2)
    x_sig = torch.randn(4, 12, 5000)
    x = {"signal": x_sig}
    output = resnet(x)
    print("output shape: ", output.shape)

#%% Run RsnClf
    resnet = SKRsnClf(26, None, in_channels=12, base_filters=256, first_kernel_size=15, 
            kernel_size=7, stride=3, groups=2, n_block=1, is_se=True)
    x_demo = torch.randn(4, 2)
    x_sig = torch.randn(4, 12, 5000)
    x = {"signal": x_sig}
    output = resnet(x)
    print("output shape: ", output.shape)





#%%
