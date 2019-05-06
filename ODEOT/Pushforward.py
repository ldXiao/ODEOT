from .ANODE import InjAugNODE
import torch
import torch.nn as nn

# class ModifiedSinkhorn(SinkhornLoss):
#     def __init__(self, eps=1e-3, max_iters=300, labd=10):
#         super(ModifiedSinkhorn, self).__init__(eps=eps, max_iters=max_iters)
#
#     def forward(self, predicted, expected):

class Pushforward(nn.Module):
    def __init__(self, phi:InjAugNODE, f:callable):
        super(Pushforward, self).__init__()
        self.phi = phi
        self.f = f
    def forward(self, y):
        return self.f(self.phi.invert(y))