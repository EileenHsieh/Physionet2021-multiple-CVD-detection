#%%
# https://github.com/Alibaba-MIIL/ASL

'''
[Training Loss][https://github.com/Alibaba-MIIL/ASL/issues/22]
our default params for ASL are for highly imbalanced multi label datasets.
i suggest you try gradually variants of ASL, and make sure results are logical and consistent

(1)
start with simple CE, and make sure you reproduce your BCEloss results:
loss_function=AsymmetricLoss(gamma_neg=0, gamma_pos=0, clip=0)

(2) than try simple focal loss:
loss_function=AsymmetricLoss(gamma_neg=2, gamma_pos=2, clip=0)

(3) try now ASL:
loss_function=AsymmetricLoss(gamma_neg=2, gamma_pos=1, clip=0)
loss_function=AsymmetricLoss(gamma_neg=4, gamma_pos=1, clip=0.05)

(4) also try the 'disable_torch_grad_focal_loss' mode, it can stabilize results:
loss_function=AsymmetricLoss(gamma_neg=4, gamma_pos=1, clip=0.05,disable_torch_grad_focal_loss=True)

---
[Training Tricks] [https://github.com/Alibaba-MIIL/ASL/issues/30#issuecomment-750780576]
We honestly haven't encountered any case where ASL has not outperformed easily cross entropy.

here are some training tricks we used (they are quite standard and can be found also in public repositories like this), see if something resonant differently from your framework:

    for learning rate, we use one cycle policy (warmup + cosine decay) with Adam optimizer and max learning rate of ~2e-4 to 4e-4
    very important to use also EMA
    true weight decay of 1e-4 ("true" == no wd for batch norm and bias)
    we have our own augmentation package, but important to use at least standard AutoAugment.
    cutout of 0.5 (very important)
    squish resizing, not crop (important)
try replacing resnet with TResNet. it will give you the same GPU speed, with higher accuracy

'''

import torch
import torch.nn as nn


class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True):
        super(AsymmetricLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.sum()


class AsymmetricLossOptimized(nn.Module):
    ''' Notice - optimized version, minimizes memory allocation and gpu uploading,
    favors inplace operations'''

    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=False):
        super(AsymmetricLossOptimized, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

        # prevent memory allocation and gpu uploading every iteration, and encourages inplace operations
        self.targets = self.anti_targets = self.xs_pos = self.xs_neg = self.asymmetric_w = self.loss = None

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        self.targets = y
        self.anti_targets = 1 - y

        # Calculating Probabilities
        self.xs_pos = torch.sigmoid(x)
        self.xs_neg = 1.0 - self.xs_pos

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            self.xs_neg.add_(self.clip).clamp_(max=1)

        # Basic CE calculation
        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))
        self.loss.add_(self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps)))

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            self.xs_pos = self.xs_pos * self.targets
            self.xs_neg = self.xs_neg * self.anti_targets
            self.asymmetric_w = torch.pow(1 - self.xs_pos - self.xs_neg,
                                          self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            self.loss *= self.asymmetric_w

        return -self.loss.sum()


class ASLSingleLabel(nn.Module):
    def __init__(self, gamma_pos=0, gamma_neg=4, eps: float = 0.1, reduction='mean'):
        super(ASLSingleLabel, self).__init__()

        self.eps = eps
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.targets_classes = []  # prevent gpu repeated memory allocation
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.reduction = reduction

    def forward(self, inputs, target, reduction=None):
        num_classes = inputs.size()[-1]
        log_preds = self.logsoftmax(inputs)
        self.targets_classes = torch.zeros_like(inputs).scatter_(1, target.long().unsqueeze(1), 1)

        # ASL weights
        targets = self.targets_classes
        anti_targets = 1 - targets
        xs_pos = torch.exp(log_preds)
        xs_neg = 1 - xs_pos
        xs_pos = xs_pos * targets
        xs_neg = xs_neg * anti_targets
        asymmetric_w = torch.pow(1 - xs_pos - xs_neg,
                                 self.gamma_pos * targets + self.gamma_neg * anti_targets)
        log_preds = log_preds * asymmetric_w

        if self.eps > 0:  # label smoothing
            self.targets_classes.mul_(1 - self.eps).add_(self.eps / num_classes)

        # loss calculation
        loss = - self.targets_classes.mul(log_preds)

        loss = loss.sum(dim=-1)
        if self.reduction == 'mean':
            loss = loss.mean()

        return loss


#%%
if __name__ == "__main__":
    gamma_neg=0
    gamma_pos=0
    clip=0.00
    disable_torch_grad_focal_loss=False

    batch_size = 32
    num_label = 26
    logits = torch.randn(batch_size, num_label)
    labels = torch.randint(0,2, size=(batch_size, num_label))


    criteria = AsymmetricLossOptimized(gamma_neg=gamma_neg, gamma_pos=gamma_neg, clip=clip, disable_torch_grad_focal_loss=clip)
    loss = criteria(logits, labels)
    print(loss)

    criteria = AsymmetricLoss(gamma_neg=gamma_neg, gamma_pos=gamma_neg, clip=clip, disable_torch_grad_focal_loss=clip)
    loss = criteria(logits, labels)
    print(loss)


#%%
    # https://github.com/Alibaba-MIIL/ASL/issues/22
    pred = torch.tensor([[-0.4089, -1.2471, 0.5907], [-0.4897, -0.8267, -0.7349], [0.5241, -0.1246, -0.4751]])
    label = torch.tensor([[0, 1, 1], [0, 0, 1], [1, 0, 1]]).float()

    crition1 = torch.nn.BCEWithLogitsLoss()
    loss1 = crition1(logits, labels)
    print(loss1)

    crition2 = AsymmetricLoss(gamma_neg = 0,gamma_pos = 0,clip = 0,disable_torch_grad_focal_loss=True)

    loss2 = crition2(pred, label)
    print(loss2)

    crition3 = AsymmetricLossOptimized(gamma_neg = 0,gamma_pos = 0,clip = 0)
    loss3 = crition3(pred, label)
    print(loss3)