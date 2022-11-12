#%%
import math
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

#%%
def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1]
    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)
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
        batch_size, seq_length, embed_dim = x.size()
        qkv = self.qkv_proj(x)

        # Separate Q, K, V from linear output
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3*self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3) # [Batch, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=-1)

        # Determine value outputs
        values, attention = scaled_dot_product(q, k, v, mask=mask)
        values = values.permute(0, 2, 1, 3) # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, embed_dim)
        o = self.o_proj(values)

        if return_attention:
            return o, attention
        else:
            return o

class EncoderBlock(nn.Module):

    def __init__(self, input_dim, num_heads, dim_feedforward, dropout=0.0):
        """
        Inputs:
            input_dim - Dimensionality of the input
            num_heads - Number of heads to use in the attention block
            dim_feedforward - Dimensionality of the hidden layer in the MLP
            dropout - Dropout probability to use in the dropout layers
        """
        super().__init__()

        # Attention layer
        self.self_attn = MultiheadAttention(input_dim, input_dim, num_heads)

        # Two-layer MLP
        self.linear_net = nn.Sequential(
            nn.Linear(input_dim, dim_feedforward),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(dim_feedforward, input_dim)
        )

        # Layers to apply in between the main layers
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x, mask=None):
        # Attention part
        attn_out = self.self_attn(x, mask=mask)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        # MLP part
        linear_out = self.linear_net(x)
        x = x + self.dropout(linear_out)
        x = self.norm2(x)

        return x

class TransformerEncoder(nn.Module):

    def __init__(self, num_layers, **block_args):
        super().__init__()
        self.layers = nn.ModuleList([EncoderBlock(**block_args) for _ in range(num_layers)])


    def forward(self, x, mask=None):
        for l in self.layers:
            x = l(x, mask=mask)
        return x


    def get_attention_maps(self, x, mask=None):
        attention_maps = []
        for l in self.layers:
            _, attn_map = l.self_attn(x, mask=mask, return_attention=True)
            attention_maps.append(attn_map)
            x = l(x)
        return attention_maps

class CNN2dFeatExtractor(nn.Module):
    def __init__(self, in_channel, out_channel, dropout=None):
        super().__init__()
        # =====
        # Feature extractor
        # =====
        layers = []
        # in_channels, out_channels, kernel_size, stride = 1, padding = 0,
        layers.append(nn.Conv2d(in_channel, out_channel//2,
                                kernel_size=5, stride=1, padding=2))
        layers.append(nn.LeakyReLU(0.02, inplace=True))
        layers.append(nn.BatchNorm2d(out_channel//2))
        # layers.append(nn.Dropout(dropout))

        layers.append(nn.Conv2d(out_channel//2, out_channel,
                                kernel_size=5, stride=1, padding=2))
        layers.append(nn.LeakyReLU(0.02, inplace=True))
        layers.append(nn.BatchNorm2d(out_channel))
        # layers.append(nn.Dropout(dropout))

        layers.append(nn.Conv2d(out_channel, out_channel,
                                kernel_size=3, stride=1, padding=1))
        layers.append(nn.LeakyReLU(0.02, inplace=True))
        layers.append(nn.BatchNorm2d(out_channel))
        # layers.append(nn.Dropout(dropout))

        layers.append(nn.Conv2d(out_channel, out_channel,
                                kernel_size=3, stride=1, padding=1))
        layers.append(nn.LeakyReLU(0.02, inplace=True))
        layers.append(nn.BatchNorm2d(out_channel))
        # layers.append(nn.Dropout(dropout))


        layers.append(nn.Conv2d(out_channel, out_channel,
                                kernel_size=3, stride=1, padding=1))
        layers.append(nn.LeakyReLU(0.02, inplace=True))
        layers.append(nn.BatchNorm2d(out_channel))
        # layers.append(nn.Dropout(dropout))


        layers.append(nn.Conv2d(out_channel, out_channel,
                                kernel_size=3, stride=1, padding=1))
        layers.append(nn.LeakyReLU(0.02, inplace=True))
        layers.append(nn.BatchNorm2d(out_channel))
        # layers.append(nn.Dropout(dropout))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        '''
        x: (n_batch, n_lead, x_dim)
        h: (n_batch, n_channel, seq_len)
        '''
        
        h = self.main(x)
        return h

class CNNFeatExtractor(nn.Module):
    def __init__(self, in_channel, out_channel, dropout=None):
        super().__init__()
        # =====
        # Feature extractor
        # =====
        layers = []
        # in_channels, out_channels, kernel_size, stride = 1, padding = 0,
        layers.append(nn.Conv1d(in_channel, out_channel//2,
                                kernel_size=14, stride=3, padding=2))
        layers.append(nn.LeakyReLU(0.02, inplace=True))
        layers.append(nn.BatchNorm1d(out_channel//2))
        # layers.append(nn.Dropout(dropout))

        layers.append(nn.Conv1d(out_channel//2, out_channel,
                                kernel_size=14, stride=3, padding=0))
        layers.append(nn.LeakyReLU(0.02, inplace=True))
        layers.append(nn.BatchNorm1d(out_channel))
        # layers.append(nn.Dropout(dropout))

        layers.append(nn.Conv1d(out_channel, out_channel,
                                kernel_size=10, stride=2, padding=0))
        layers.append(nn.LeakyReLU(0.02, inplace=True))
        layers.append(nn.BatchNorm1d(out_channel))
        # layers.append(nn.Dropout(dropout))

        layers.append(nn.Conv1d(out_channel, out_channel,
                                kernel_size=10, stride=2, padding=0))
        layers.append(nn.LeakyReLU(0.02, inplace=True))
        layers.append(nn.BatchNorm1d(out_channel))
        # layers.append(nn.Dropout(dropout))

        layers.append(nn.Conv1d(out_channel, out_channel,
                                kernel_size=10, stride=1, padding=0))
        layers.append(nn.LeakyReLU(0.02, inplace=True))
        layers.append(nn.BatchNorm1d(out_channel))
        # layers.append(nn.Dropout(dropout))

        layers.append(nn.Conv1d(out_channel, out_channel,
                                kernel_size=10, stride=1, padding=0))
        layers.append(nn.LeakyReLU(0.02, inplace=True))
        layers.append(nn.BatchNorm1d(out_channel))
        # layers.append(nn.Dropout(dropout))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        '''
        x: (n_batch, n_lead, x_dim)
        h: (n_batch, n_channel, seq_len)
        '''
        h = self.main(x)
        return h

class CNN2dFeatExtractor(nn.Module):
    def __init__(self, in_channel, out_channel, dropout=None):
        super().__init__()
        # =====
        # Feature extractor
        # =====
        layers = []
        # in_channels, out_channels, kernel_size, stride = 1, padding = 0,
        layers.append(nn.Conv2d(in_channel, out_channel//2,
                                kernel_size=5, stride=1, padding=2))
        layers.append(nn.LeakyReLU(0.02, inplace=True))
        layers.append(nn.BatchNorm2d(out_channel//2))
        # layers.append(nn.Dropout(dropout))
        self.main1 = nn.Sequential(*layers)

        layers = []
        layers.append(nn.Conv2d(out_channel//2, out_channel,
                                kernel_size=5, stride=1, padding=2))
        layers.append(nn.LeakyReLU(0.02, inplace=True))
        layers.append(nn.BatchNorm2d(out_channel))
        # layers.append(nn.Dropout(dropout))
        self.main2 = nn.Sequential(*layers)

        layers = []
        layers.append(nn.Conv2d(out_channel, out_channel,
                                kernel_size=3, stride=1, padding=1))
        layers.append(nn.LeakyReLU(0.02, inplace=True))
        layers.append(nn.BatchNorm2d(out_channel))
        # layers.append(nn.Dropout(dropout))
        self.main3 = nn.Sequential(*layers)

        layers = []
        layers.append(nn.Conv2d(out_channel, out_channel,
                                kernel_size=3, stride=1, padding=1))
        layers.append(nn.LeakyReLU(0.02, inplace=True))
        layers.append(nn.BatchNorm2d(out_channel))
        # layers.append(nn.Dropout(dropout))
        self.main4 = nn.Sequential(*layers)


        layers = []
        layers.append(nn.Conv2d(out_channel, out_channel,
                                kernel_size=3, stride=1, padding=1))
        layers.append(nn.LeakyReLU(0.02, inplace=True))
        layers.append(nn.BatchNorm2d(out_channel))
        # layers.append(nn.Dropout(dropout))
        self.main5 = nn.Sequential(*layers)


        layers = []
        layers.append(nn.Conv2d(out_channel, out_channel,
                                kernel_size=3, stride=1, padding=1))
        layers.append(nn.LeakyReLU(0.02, inplace=True))
        layers.append(nn.BatchNorm2d(out_channel))
        # layers.append(nn.Dropout(dropout))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        '''
        x: (n_batch, n_lead, x_dim)
        h: (n_batch, n_channel, seq_len)
        '''
        
        h = self.main(x)
        return h

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
        self.main_clf = nn.Linear(out_dim + n_demo, output_size)
    
    def forward(self, x_demo, x_sig, mask=None):
        '''
        x_demo: (n_batch, 2) # age, gender
        x_sig: (n_batch, n_lead, x_dim)
        out: (n_batch, x_dim)
        '''
        # ===== Feature extraction
        h = self.main_extract(x_sig) # (n_batch, n_channel, seq_len)
        h = torch.mean(h, dim=-2) # (n_batch, n_channel, f_domain, t_domain)
        # ===== Transformer 
        h = h.permute(0, 2, 1) # (n_batch, seq_len, n_channel)
        h = self.main(h, mask=mask)
        atten_maps = self.main.get_attention_maps(h)

        # ===== Global Average Pooling
        h = h.mean(dim=1)
        h = self.main_fc(h)

        # ===== Concat x_demo
        h = torch.cat((h, x_demo), dim=-1)

        return self.main_clf(h), atten_maps
#%%
# class CnnTransformer(Classifier):
#     def __init__(self, config, random_state):
#         super(CnnTransformer, self).__init__(config, random_state)

#         self._engine = CnnTfmArch(self.input_channel, self.output_channel, self.num_layers,
#                                 self.embed_dim, self.dim_feedforward, self.num_heads,
#                                 self.dropout, self.out_dim, self.n_demo, self.output_size)
#         self._optimizer = torch.optim.AdamW(self._engine.parameters(), lr=self.lr)  # Optimizer
        
#         if torch.cuda.is_available():
#             self._engine.cuda()

def plot_attention_maps(input_data, attn_maps, idx=0):
    if input_data is not None:
        input_data = input_data[idx].detach().cpu().numpy()
    else:
        input_data = np.arange(attn_maps[0][idx].shape[-1])
    attn_maps = [m[idx].detach().cpu().numpy() for m in attn_maps]

    num_heads = attn_maps[0].shape[0]
    num_layers = len(attn_maps)
    seq_len = input_data.shape[0]
    fig_size = 4 if num_heads == 1 else 3
    fig, ax = plt.subplots(num_layers, num_heads, figsize=(num_heads*fig_size, num_layers*fig_size))
    if num_layers == 1:
        ax = [ax]
    if num_heads == 1:
        ax = [[a] for a in ax]
    for row in range(num_layers):
        for column in range(num_heads):
            ax[row][column].imshow(attn_maps[row][column], origin='lower', vmin=0)
            ax[row][column].set_xticks(list(range(seq_len)))
            ax[row][column].set_xticklabels(input_data.tolist())
            ax[row][column].set_yticks(list(range(seq_len)))
            ax[row][column].set_yticklabels(input_data.tolist())
            ax[row][column].set_title("Layer %i, Head %i" % (row+1, column+1))
    fig.subplots_adjust(hspace=0.5)
    plt.show()

def generateMask(x):
    '''
    x: (fea_dim, seq_len)
    mask: (seq_len, seq_len)
    '''
    mask = torch.matmul(x.T, x)
    return mask



#%%

if __name__=='__main__':
#%% Run scaled_dot_product
    seq_len, d_k = 3, 2
    q = torch.randn(seq_len, d_k)
    k = torch.randn(seq_len, d_k)
    v = torch.randn(seq_len, d_k)
    values, attention = scaled_dot_product(q, k, v)
    print("Q\n", q)
    print("K\n", k)
    print("V\n", v)
    print("Values\n", values)
    print("Attention\n", attention)

#%% Run tranformer
    transformer = TransformerEncoder(num_layers=8,
                                    input_dim=256, #self.hparams.model_dim,
                                    dim_feedforward=2048, #2*self.hparams.model_dim,
                                    num_heads=8,
                                    dropout=0.1)
    x = torch.randn(4,100,256)
    mask = torch.randn(100, 100).uniform_() > 0.8
    x = transformer(x, mask=mask)

#%% Run CNNTransformer
    cnn_tfm = CnnTfmArch(12, 256, 8, 256, 2048, 8, 0.1, 64, 2, 26)
    x_demo = torch.randn(4, 2)
    x_sig = torch.randn(4, 12, 5000)
    out, atten_maps = cnn_tfm(x_demo, x_sig, mask=None)

#%% Run CNN2dTransformer
    cnn_tfm = CnnTfmArch(12, 256, 8, 256, 2048, 8, 0.1, 64, 2, 26)
    x_demo = torch.randn(4, 2)
    x_sig = torch.randn(4, 12, 256, 20)
    out, atten_maps = cnn_tfm(x_demo, x_sig, mask=None)

#%% plot attention map
    atten_map = transformer.get_attention_maps(x)
    plot_attention_maps(input_data=None, attn_maps=atten_map, idx=0)

#%% generate mask if there's need
    x = torch.tensor([[1, 1,  0],[0, 1, 0]])
    mask = generateMask(x)