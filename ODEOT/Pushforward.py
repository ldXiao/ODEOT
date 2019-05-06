from .ANODE import InjAugNODE
import torch
import torch.nn as nn
import numpy as np

# class ModifiedSinkhorn(SinkhornLoss):
#     def __init__(self, eps=1e-3, max_iters=300, labd=10):
#         super(ModifiedSinkhorn, self).__init__(eps=eps, max_iters=max_iters)
#
#     def forward(self, predicted, expected):

class Pushforward(nn.Module):
    def __init__(self, phi:InjAugNODE, f:callable, device:str="cuda"):
        super(Pushforward, self).__init__()
        self.phi = phi
        self.f = f
        self.device = device
    def forward(self, y):
        with torch.no_grad():
            y = torch.from_numpy(y).to(self.device)
            y = self.phi.invert(y).cpu().numpy()

            return np.array([self.f(s[0],s[1]) for s in y])