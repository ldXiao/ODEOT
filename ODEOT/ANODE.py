import numpy as np
import torch
import torch.nn as nn
from torchdiffeq import odeint

class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.fc1 = nn.Linear(in_dim, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 512)
        self.fc4 = nn.Linear(512, 512)
        self.fc5 = nn.Linear(128, out_dim)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        # x = self.relu(self.fc2(x))
        # x = self.relu(self.fc3(x))
        # x = self.relu(self.fc4(x))
        x = self.fc5(x)
        return x

class ODEfunc(nn.Module):

    def __init__(self, var_dim: int, hid_dims: list, device: str = "cuda"):
        super(ODEfunc, self).__init__()
        # self.norm1 = norm(dim)
        self.relu = nn.ReLU()
        self.fcs = nn.ModuleList()
        self.nfe = 0
        self.device = device
        self.num_layers = len(hid_dims) + 1
        dims = [var_dim] + [hd for hd in hid_dims] + [var_dim]
        for dim1, dim2 in zip(dims[:-1], dims[1:]):
            self.fcs.append(nn.Linear(dim1 + 1, dim2))
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        out = x
        # print(self.fcs)
        n = self.num_layers
        for count, fc in enumerate(self.fcs):
            dummy = torch.zeros(out.shape[0], out.shape[1] + 1).to(self.device)
            dummy[:, 0:out.shape[1]] = out.to(self.device)
            dummy[:, out.shape[1]:] = torch.ones(dummy.shape[0], 1).to(self.device) * t
            if count < n - 1:
                out = self.relu(fc(dummy))
            else:
                out = fc(dummy)
        return out

class ODEBlock(nn.Module):

    def __init__(self, odefunc, tol:float):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.integration_time = torch.tensor([0, 1]).float()
        self.tol = tol

    def forward(self, x):
        self.integration_time = self.integration_time.type_as(x)
        out = odeint(self.odefunc, x, self.integration_time, rtol=self.tol, atol=self.tol)
        return out[1]

    def event_t(self, t, x):
        timescale = torch.tensor([0, t]).type_as(x)
        # print(x)
        # print(t)
        out = odeint(self.odefunc, x, timescale, rtol=self.tol, atol=self.tol)
        return out[1]


    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value

class ParametrizationNet(nn.Module):

    def __init__(self, in_dim=2, out_dim=3, var_dim=50, ker_dims:list=[1024,1024,1024,1024], device:str="cude"):
        super(ParametrizationNet, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.var_dim = var_dim
        self.odeblock = ODEBlock(ODEfunc(var_dim, ker_dims))
        self.device = device
        self.augment_part = None

    def forward(self, x):
        out = torch.zeros(x.shape[0], self.var_dim).to(self.device)
        out[:, 0: x.shape[1]] = x
        out = self.odeblock(out)
        out = out[:, 0:self.out_dim]

        return out

    def event_t(self, t, x):
        out = torch.zeros(x.shape[0], self.var_dim).to(self.device)
        out[:, 0: x.shape[1]] = x
        out = self.odeblock.event_t(t, out)
        out = out[:, 0:self.out_dim]
        return out

    def augment_part(self, x):
        out = torch.zeros(x.shape[0], self.var_dim).to(self.device)
        out[:, 0: x.shape[1]] = x
        out = out[:, 0:self.out_dim]
        return out


