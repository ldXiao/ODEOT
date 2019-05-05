import argparse

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import point_cloud_utils as pcu
from fml.nn import SinkhornLoss
from ODEOT.ANODE import InjAugODE
from ODEOT.utils import load_mesh_by_file_extension, plot_flow, embed_3d


argparser = argparse.ArgumentParser()
argparser.add_argument("mesh_filename", type=str, help="Point cloud to reconstruct")
argparser.add_argument("--max-sinkhorn-iters", "-si", type=int, default=16,
                       help="Maximum number of Sinkhorn iterations")
argparser.add_argument("--sinkhorn-eps", "-se", type=float, default=1e-3,
                       help="Maximum number of Sinkhorn iterations")
argparser.add_argument("--learning-rate", "-lr", type=float, default=1e-3, help="Step size for gradient descent")
argparser.add_argument("--num-epochs", "-n", type=int, default=250, help="Number of training epochs")
argparser.add_argument("--device", "-d", type=str, default="cuda")
argparser.add_argument("--adjoint", "-adj", type=eval, default=False, choices=[True,False])
argparser.add_argument("--tol", "-tol", type=float, default=1e-3)
args = argparser.parse_args()
if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint




# def norm(dim):
#     return nn.GroupNorm(min(32, dim), dim)

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

    def __init__(self, var_dim:int, hid_dims:list, device:str="cuda"):
        super(ODEfunc, self).__init__()
        # self.norm1 = norm(dim)
        self.relu = nn.ReLU()
        self.fcs = nn.ModuleList()
        self.nfe = 0
        self.device = device
        self.num_layers = len(hid_dims)+1
        dims = [var_dim]+ [hd for hd in hid_dims] + [var_dim]
        for dim1, dim2 in zip(dims[:-1], dims[1:]):
            self.fcs.append(nn.Linear(dim1+1, dim2))
        self.nfe = 0

    def forward(self, t, x):
        self.nfe+=1
        out = x
        # print(self.fcs)
        n = self.num_layers
        for count, fc in enumerate(self.fcs):
            dummy = torch.zeros((out.shape[0], out.shape[1]+1),device=self.device)
            dummy[:, 0:out.shape[1]] = out.to(self.device)
            # print( dummy[:, out.shape[1]:].device)
            # print((torch.ones((dummy.shape[0],1),device="cuda") * t).device)
            dummy[:, out.shape[1]:] = torch.ones((dummy.shape[0],1),device=self.device) * t
            if count < n-1:
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
        # print(x)
        # print(t)
        out = odeint(self.odefunc, x, timescale, rtol=self.tol, atol=self.tol)
        return out[1]

    def invert(self, y):
        # print(y)
        timescale = torch.tensor([1,0]).type_as(y)
        # print(timescale.device)
        input = odeint(self.odefunc, y, timescale, rtol=self.tol, atol=self.tol)
        return input[1]

class ParametrizationNet(nn.Module):

    def __init__(self, in_dim=2, out_dim=3, var_dim=50):
        super(ParametrizationNet, self).__init__()
        self.out_dim = out_dim
        self.var_dim = var_dim
        self.odeblock = ODEBlock(ODEfunc(var_dim, [1024,1024,1024,1024]))
        self.MLP = MLP(in_dim=var_dim - in_dim, out_dim=var_dim-in_dim)

    def forward(self, x):
        out = torch.zeros(x.shape[0], self.var_dim).to(args.device)
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




def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# class InjAugODE(nn.Module):
#     def __init__(self, in_dim=2, out_dim=3, var_dim=50, sample_size=100, ker_dims:list=[1024,1024,1024,1024], device:str="cuda"):
#         super(InjAugODE, self).__init__()
#         self.in_dim = in_dim
#         self.out_dim = out_dim
#         self.var_dim = var_dim
#         self.odeblock = ODEBlock(ODEfunc(var_dim, ker_dims))
#         self.device = device
#         # self.augment_part = torch.ones((sample_size, var_dim-self.in_dim),requires_grad=False, device="cuda")
#         # self.augMLP = MLP()
#         self.augout_part = None
#
#     def forward(self, x):
#         out = torch.zeros(x.shape[0], self.var_dim).to(self.device)
#         out[:, 0: x.shape[1]] = x
#         # out[:, x.shape[1]:] = torch.ones((x.shape[0], self.var_dim-x.shape[1]), device=self.device)
#         out = self.odeblock(out)
#         self.augout_part=out[:,x.shape[1]:]
#         return out
#
#     def proj_forward(self, x):
#         out = self.forward(x)
#         out = out[:, 0:self.out_dim]
#         return out
#
#     def invert(self, y):
#         input = torch.ones(y.shape[0],self.var_dim).to(self.device)
#         input[:,0:y.shape[1]] = y
#         input = self.odeblock.invert(input)
#         input = input[:,0:self.in_dim]
#         return input
#
#     def event_t(self, t, x):
#         # timescale = torch.tensor([0,t]).type_as(x)
#         # print(x)
#         # print(t)
#         out = torch.zeros(x.shape[0], self.var_dim).to(args.device)
#         out[:, 0: x.shape[1]] = x
#         # out[:, x.shape[1]:] = self.MLP(torch.ones(x.shape[0], self.var_dim-x.shape[1]).to(args.device))
#         out = self.odeblock.event_t(t, out)
#         out = out[:, 0:self.out_dim]
#         # out = self.MLP(out)
#         return out

def main():
    # NOTE: We're doing everything in float32 and numpy defaults to float64, so you need to make sure you cast
    # everything to the right type

    # x is a tensor of shape [n, 3] containing the positions of the points we are trying to fit
    x = torch.from_numpy(load_mesh_by_file_extension(args.mesh_filename)).to(args.device)

    # y = torch.zeros_like(x).to(args.device)
    # y[:, 0]= x[:,0]
    # y[:, 1] = x[:, 2]
    # y[:, 2] = x[:, 1]
    # x = y
    # x_mean = x.sum(0) / x.shape[0]
    # x = x - x_mean
    # x /= x.max()
    # x[:,2] += torch.ones_like(x[:,2]).to(args.device)
    # x[:, 1] += torch.ones_like(x[:, 1]).to(args.device)
    # x*= 2
    # print(x.max())

    # t is a tensor of shape [n, 2] containing a set of nicely distributed samples in the unit square
    t = embed_3d(torch.from_numpy(pcu.lloyd_2d(x.shape[0]).astype(np.float32)).to(args.device),0)
    # t_mean = t.sum(0) / t.shape[0]
    # t = t - t_mean

    # The model is a simple fully connected network mapping a 2D parameter point to 3D
    # phi = ParametrizationNet(in_dim=3, out_dim=3, var_dim=4).to(args.device)
    phi = InjAugODE(in_dim=3, out_dim=3, var_dim=10, sample_size=x.shape[0], ker_dims=[1024,1024,1024,1024], device="cuda").to(args.device)
    # Eps is 1/lambda and max_iters is the maximum number of Sinkhorn iterations to do
    loss_fun = SinkhornLoss(eps=args.sinkhorn_eps, max_iters=args.max_sinkhorn_iters)
    dummy = torch.ones(x.shape[0], 10).to(args.device)
    dummy[:,0:x.shape[1]]=x
    x = dummy

    # Here I'm using the Adam optimizer just as an example, you'll need to replace this with your thing
    optimizer = torch.optim.Adam(phi.parameters(), lr=0.0001)
    # optimizer.add_param_group({"params": phi.augment_part})
    print("Number of Parameters=", count_parameters(phi))

    for epoch in range(1, args.num_epochs+1):
        optimizer.zero_grad()

        # Do the forward pass of the neural net, evaluating the function at the parametric points
        y = phi(t)


        loss = loss_fun(y.unsqueeze(0), x.unsqueeze(0))

        loss.backward()
        optimizer.step()
        print("Epoch %d, loss = %f" % (epoch, loss.item()))
        # b = x[0:1,:]
        # print(phi.invert(x))

    print(phi.invert(x)[:,-1])
    # print(x)
    torch.save(phi.state_dict(), "../models/phi.pt")
    phi.load_state_dict(torch.load("../models/phi.pt"))
    plot_flow(x[:,0:3], t, phi, 128,  t.shape[0]//100)


if __name__ == "__main__":
    main()



