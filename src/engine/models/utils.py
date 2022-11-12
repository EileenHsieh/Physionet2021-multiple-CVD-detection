import torch
from torch import optim
import torch.nn as nn
import numpy as np

class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        # lr_factor = self.warmup_steps ** 0.5 * min(epoch ** (-0.5), epoch * self.warmup_steps ** (-1.5))
        
        return lr_factor

#%% run scheduler warmup
# if __name__=='__main__':
#     import seaborn as sns
#     import matplotlib.pyplot as plt
#     sns.reset_orig()

#     p = nn.Parameter(torch.empty(4,4))
#     optimizer = optim.Adam([p], lr=1e-3)
#     lr_scheduler = CosineWarmupScheduler(optimizer=optimizer, warmup=100, max_iters=2000)

#     # Plotting
#     epochs = list(range(2000))
#     sns.set()
#     plt.figure(figsize=(8,3))
#     plt.plot(epochs, [lr_scheduler.get_lr_factor(e) for e in epochs])
#     plt.ylabel("Learning rate factor")
#     plt.xlabel("Iterations (in batches)")
#     plt.title("Cosine Warm-up Learning Rate Scheduler")
#     plt.show()
#     sns.reset_orig()