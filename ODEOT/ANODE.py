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

    def __init__(self, odefunc, tol:float=0.0001):
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
        out = odeint(self.odefunc, x, timescale, rtol=self.tol, atol=self.tol)
        return out[1]

    def flow(self,t0,tf,x):
        timescale = torch.tensor([t0, tf]).type_as(x)
        out = odeint(self.odefunc, x, timescale, rtol=self.tol, atol=self.tol)
        return out[1]

    def invert(self, y):
        timescale = torch.tensor([1,0]).type_as(y)
        input = odeint(self.odefunc, y, timescale, rtol=self.tol, atol=self.tol)
        return input[1]

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value

class ParametrizationNet(nn.Module):

    def __init__(self, in_dim=2, out_dim=3, var_dim=50, device="cuda"):
        super(ParametrizationNet, self).__init__()
        self.out_dim = out_dim
        self.device = device
        self.var_dim = var_dim
        self.odeblock = ODEBlock(ODEfunc(var_dim, [1024,1024,1024,1024], device=self.device))
        self.MLP = MLP(in_dim=var_dim - in_dim, out_dim=var_dim-in_dim)

    def forward(self, x):
        out = torch.zeros(x.shape[0], self.var_dim).to(x.device)
        out[:, 0: x.shape[1]] = x
        # out[:, x.shape[1]:] = self.MLP(torch.ones(x.shape[0], self.var_dim-x.shape[1]).to(args.device))
        out = self.odeblock(out)
        out = out[:, 0:self.out_dim]
        # out = self.MLP(out)

        return out

    def event_t(self, t, x):
        # timescale = torch.tensor([0,t]).type_as(x)
        # print(x)
        # print(t)
        out = torch.zeros(x.shape[0], self.var_dim).to(x.device)
        out[:, 0: x.shape[1]] = x
        # out[:, x.shape[1]:] = self.MLP(torch.ones(x.shape[0], self.var_dim-x.shape[1]).to(args.device))
        out = self.odeblock.event_t(t, out)
        out = out[:, 0:self.out_dim]
        # out = self.MLP(out)
        return out



class AugNODE(nn.Module):
    def __init__(self, in_dim=2, out_dim=3, var_dim=50, sample_size=100, ker_dims:list=[1024,1024,1024,1024], device:str="cuda"):
        super(AugNODE, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.var_dim = var_dim
        self.odeblock = ODEBlock(ODEfunc(var_dim, ker_dims))
        self.device = device
        self.augment_part = nn.Parameter(torch.ones((sample_size, var_dim-self.in_dim),requires_grad=False, device=self.device))
        # self.augMLP = MLP()
        self.register_parameter("params",self.augment_part)
        self.augout_part = None

    def forward(self, x):
        out = torch.zeros(x.shape[0], self.var_dim).to(self.device)
        out[:, 0: x.shape[1]] = x
        out[:, x.shape[1]:] = self.augment_part
        out = self.odeblock(out)
        self.augout_part=out[:,x.shape[1]:]
        out = out[:, 0:self.out_dim]
        return out

class InjAugNODE(nn.Module):
    def __init__(self, in_dim=2, out_dim=3, var_dim=50, ker_dims:list=[1024,1024,1024,1024], device:str="cuda"):
        super(InjAugNODE, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.var_dim = var_dim
        self.device = device
        self.odeblock = ODEBlock(ODEfunc(var_dim, ker_dims, device=self.device))
        self.augout_part = None

    def forward(self, x):
        out = torch.ones(x.shape[0], self.var_dim).to(self.device)
        out[:, 0: x.shape[1]] = x
        # out[:, x.shape[1]:] = torch.ones((x.shape[0], self.var_dim-x.shape[1]), device=self.device)
        out = self.odeblock(out)
        self.augout_part=out[:,x.shape[1]:]
        return out

    def proj_forward(self, x):
        out = self.forward(x)
        out = out[:, 0:self.out_dim]
        return out

    def invert(self, y):
        input = torch.ones(y.shape[0],self.var_dim).to(self.device)
        input[:,0:y.shape[1]] = y
        input = self.odeblock.invert(input)
        input = input[:,0:self.in_dim]
        return input

    def event_t(self, t, x):
        # timescale = torch.tensor([0,t]).type_as(x)
        # print(x)
        # print(t)
        out = torch.ones(x.shape[0], self.var_dim).to(self.device)
        out[:, 0: x.shape[1]] = x
        # out[:, x.shape[1]:] = self.MLP(torch.ones(x.shape[0], self.var_dim-x.shape[1]).to(args.device))
        out = self.odeblock.event_t(t, out)
        out = out[:, 0:self.out_dim]
        # out = self.MLP(out)
        return out

    def flow(self, t0,tf, x):
        out = torch.zeros(x.shape[0], self.var_dim).to(x.device)
        out[:, 0: x.shape[1]] = x
        # out[:, x.shape[1]:] = self.MLP(torch.ones(x.shape[0], self.var_dim-x.shape[1]).to(args.device))
        out = self.odeblock.flow(t0,tf, out)
        out = out[:, 0:self.out_dim]
        return out


