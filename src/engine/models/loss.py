#%%
import torch
from src.engine.utils import NORMAL_CLASS, CLASSES, WEIGHTS
from copy import deepcopy

def sigmoid(x):
    return 1.0 / (1.0 + torch.exp(-x))

def norm_partial_label(label, alpha=1, gamma=-1): 
    alpha, beta, gamma = alpha, 0, gamma
    py = torch.sum(label, dim=1) 
    g_py = alpha * (py**gamma) + beta
    g_py[py==0] = 1
    return g_py.unsqueeze(1)

def confusion_weighted(y):
    tf_y = deepcopy(y)
    tf_y[tf_y==0] = -1
    # return (n_batch, n_class)
    weights = torch.FloatTensor(1 - WEIGHTS).to(tf_y.device)
    cow = torch.matmul(tf_y, weights)
    cow = cow / torch.sum(tf_y, dim=1).unsqueeze(1)

    return cow

def bce_logit_loss(x, y, alpha=1, gamma=0, type=None):
    n_batch, n_class = x.shape[0], x.shape[1]

    x_sigmoid = sigmoid(x) 
    x_sigmoid[y==0] = torch.stack([1 - sig_x for sig_x in x_sigmoid[y==0]])
    if type=="cow": #confusion weighted based on given evaluation weight
        cow = confusion_weighted(y)

        x_sigmoid = x_sigmoid*cow
    x_sigmoid = -torch.log(x_sigmoid)
    if type=="sign":
        sign = abs(x-y)
        sign[sign<0.5] = torch.stack([s**2 for s in sign[sign<0.5]])
        sign[sign>=0.5] = 1
        x_sigmoid = torch.mul(sign, x_sigmoid)
    if type=="norm":
        g_py = norm_partial_label(y, alpha=alpha, gamma=gamma)
        x_sigmoid = x_sigmoid*g_py
    
    batch_loss = torch.sum(x_sigmoid, dim=1) / n_class

    return torch.mean(batch_loss)

def cmloss(x, y):
    n_batch = x.shape[0]
    x_sigmoid = sigmoid(x) 
    y_sqz = y.unsqueeze(-1)
    sum_x_sigmoid = torch.sum(x_sigmoid, dim=1)
    x_sigmoid = x_sigmoid.unsqueeze(1)
    
    conf = torch.matmul(y_sqz, x_sigmoid)
    norm_conf = conf / sum_x_sigmoid.unsqueeze(1).unsqueeze(1)
    weight = torch.FloatTensor(1 - WEIGHTS).to(y.device)

    loss = torch.stack([torch.sum(norm_conf[i, :]*weight) for i in range(n_batch)])
    return torch.mean(loss)

#%%
if __name__=='__main__':
    import torch.nn as nn

    x = torch.randn(64,26)
    y = torch.ones(64,26)
    y[2,3] = 0
    y[50,15] = 0
    y[25, 19] = 0

    criterion = nn.BCEWithLogitsLoss()
    print("From function: ", criterion(x, y))
    print("From scratch vanilla: ", bce_logit_loss(x, y))
    print("From scratch with norm: ", bce_logit_loss(x, y, "norm"))
    print("From scratch with sign: ", bce_logit_loss(x, y, "sign"))
    print("From scratch with cow: ", bce_logit_loss(x, y, "cow"))
    print("cm loss: ", cmloss(x, y))