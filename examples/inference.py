from ODEOT.ANODE import InjAugNODE
from ODEOT.Pushforward import Pushforward
from ODEOT.utils import gen2Dsample_square, gen2Dsample_disk
import matplotlib.pylab as plt
import numpy as np
import torch
import torch.nn as nn
from fml.nn import SinkhornLoss
import argparse


argparser = argparse.ArgumentParser()
# argparser.add_argument("mesh_filename", type=str, help="Point cloud to reconstruct")
argparser.add_argument("--max-sinkhorn-iters", "-si", type=int, default=16,
                       help="Maximum number of Sinkhorn iterations")
argparser.add_argument("--sinkhorn-eps", "-se", type=float, default=1e-3,
                       help="Maximum number of Sinkhorn iterations")
argparser.add_argument("--learning-rate", "-lr", type=float, default=1e-3, help="Step size for gradient descent")
argparser.add_argument("--num-epochs", "-n", type=int, default=250, help="Number of training epochs")
argparser.add_argument("--device", "-d", type=str, default="cuda")
# argparser.add_argument("--adjoint", "-adj", type=eval, default=False, choices=[True,False])
argparser.add_argument("--tol", "-tol", type=float, default=1e-3)
args = argparser.parse_args()


def func_square(x,y):
    xt = (x <1 and x >2/3) or (x>0 and x < 1/3)
    yt = (y <1 and y >2/3) or (y>0 and y < 1/3)
    if (xt and yt):
        return 1
    elif (not xt and not yt):
        return 1
    else:
        return 0

def func_disk(x,y):
    return np.exp(- 3 * (x **2 + y** 2))* np.cos(2.5 * np.pi * np.sqrt((x **2 + y** 2))) ** 2

def PlotInference(phi, t, x):
    with torch.no_grad():
        y = phi(t).cpu().numpy()[:,0:2]
        x = x.cpu().numpy()[:,0:2]
        plt.plot(x[:,0], x[:,1],'o', label="target")
        plt.plot(y[:,0], y[:,1], 'x', label="fit")
        plt.legend()
        plt.show()

class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.fc1 = nn.Linear(in_dim, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 512)
        self.fc4 = nn.Linear(512, 512)
        self.fc5 = nn.Linear(512, out_dim)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.fc5(x)
        return x

def main():
    sample_x = gen2Dsample_disk(1000, func_disk, 1).astype(np.float32)
    sample_x = sample_x - np.ones_like(sample_x) * 3
    sample_t = gen2Dsample_square(1000, func_square, 1).astype(np.float32)

    t = torch.from_numpy(sample_t).to(args.device)
    x = torch.from_numpy(sample_x).to(args.device)
    vardim = 3
    phi = InjAugNODE(in_dim=2, out_dim=2, var_dim=vardim, sample_size=x.shape[0], ker_dims=[1024, 1024, 1024, 1024],
                     device="cuda").to(args.device)
    # phi = MLP(in_dim=2,out_dim=2).to(args.device)
    loss_fun = SinkhornLoss(eps=args.sinkhorn_eps, max_iters=args.max_sinkhorn_iters)
    dummy = torch.ones(x.shape[0], vardim).to(args.device)
    dummy[:, 0:x.shape[1]] = x
    x = dummy

    optimizer = torch.optim.Adam(phi.parameters(), lr=0.0001)

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
    # print(phi.invert(x)[:, -1])
    # print(x)
    torch.save(phi.state_dict(), "../models/phi_inference.pt")
    phi.load_state_dict(torch.load("../models/phi_inference.pt"))
    PlotInference(phi=phi, x=x, t=t)

if __name__ == "__main__":
    main()
