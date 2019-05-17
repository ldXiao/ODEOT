from ODEOT.ANODE import InjAugNODE, AugNODE
from ODEOT.Pushforward import Pushforward
from ODEOT.utils import gen2Dsample_square, gen2Dsample_disk
import matplotlib.pyplot as plt
from matplotlib import cm, animation
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import torch
import torch.nn as nn
from fml.nn import SinkhornLoss
import argparse


argparser = argparse.ArgumentParser()
# argparser.add_argument("mesh_filename", type=str, help="Point cloud to reconstruct")
argparser.add_argument("--max-sinkhorn-iters", "-si", type=int, default=32,
                       help="Maximum number of Sinkhorn iterations")
argparser.add_argument("--sinkhorn-eps", "-se", type=float, default=1e-3,
                       help="Maximum number of Sinkhorn iterations")
argparser.add_argument("--learning-rate", "-lr", type=float, default=1e-3, help="Step size for gradient descent")
argparser.add_argument("--num-epochs", "-n", type=int, default=250, help="Number of training epochs")
argparser.add_argument("--device", "-d", type=str, default="cuda")
# argparser.add_argument("--adjoint", "-adj", type=eval, default=False, choices=[True,False])
argparser.add_argument("--tol", "-tol", type=float, default=1e-3)
args = argparser.parse_args()


def func_square0(x,y):
    if (x<1 and x>0) and (y<1 and y>0):
        # xt = (x <1 and x >2/3) or (x>0 and x < 1/3)
        # yt = (y <1 and y >2/3) or (y>0 and y < 1/3)
        # if (xt and yt):
        #     return 1
        # elif (not xt and not yt):
        #     return 1
        # else:
        #     return 0
        absx = abs(x - 0.5)
        absy = abs(y - 0.5)
        maxabs = max(absx, absy)
        # if maxabs >0.2 and maxabs <0.3:
        #     return 0.5
        if maxabs>0.4:
            return 0.5
        else:
            return 0
    else:
        return 0
    # return 1

def func_square1(x,y):
    absx = abs(x - 0.5)
    absy = abs(y - 0.5)
    maxabs = max(absx, absy)
    if (x < 1 and x > 0) and (y < 1 and y > 0):
        if maxabs < 0.1:
            return 1
        else:
            return 0
    else:
        return 0

def pattern_square(x,y):
    # if (x<1 and x>0) and (y<1 and y>0):
        # xt = (x <1 and x >2/3) or (x>0 and x < 1/3)
        # yt = (y <1 and y >2/3) or (y>0 and y < 1/3)
        # if (xt and yt):
        #     return 1
        # elif (not xt and not yt):
        #     return 1
        # else:
        #     return 0
        # absx = abs(x - 0.5)
        # absy = abs(y - 0.5)
        # maxabs = max(absx, absy)
        # if maxabs < 0.1:
        #     return 1
        # elif maxabs >0.2 and maxabs <0.3:
        #     return 0.5
        # elif maxabs>0.4:
        #     return 0.5
        # else:
        #     return 0
    return func_square0(x,y)+func_square1(x,y)
    # else:
    #     return 0

def func_disk0(x,y):
    # dist0 = np.sqrt((x+0.7)** 2 + (y+0.7)**2)
    dist1 = np.sqrt( (x-0.3)**2/2 + (y-0.3)**2)
    # if dist1 > 0.3 and dist1<0.4:
    #     return 0.5
    if dist1 > 0.5 and dist1 < 0.6:
        return 0.3
    else:
        return 0
    # return np.exp(- 3 * (x **2/2 + y** 2))* np.cos(2.5 * np.pi * np.sqrt((x **2/2 + y** 2))) ** 2
def func_disk1(x,y):
    dist1 = np.sqrt((x+0.5) ** 2 + (y+0.5)**2)
    if dist1 < 0.1:
        return 1
    else:
        return 0



def PlotFit(phi, t0, x0, t1, x1):
    with torch.no_grad():
        y0 = phi(t0).cpu().numpy()[:,0:2]
        x0 = x0.cpu().numpy()[:,0:2]
        y1 = phi(t1).cpu().numpy()[:,0:2]
        x1 = x1.cpu().numpy()[:,0:2]

        plt.plot(y0[:, 0], y0[:, 1], 'x', label="fit0")
        plt.plot(y1[:, 0], y1[:, 1], 'x', label="fit1")
        plt.plot(x0[:, 0], x0[:, 1], 'o', label="target0")

        plt.plot(x1[:, 0], x1[:, 1], 'x', label="target1")
        plt.legend()
        plt.show()

def PlotInterence(phi, z, func:callable):
    with torch.no_grad():
        z = z.cpu().numpy()[:,0:2]
        x = z[:,0]
        y = z[:,1]
        # Define the borders
        deltaX = (max(x) - min(x)) / 10
        deltaY = (max(y) - min(y)) / 10

        xmin = min(x) - deltaX
        xmax = max(x) + deltaX

        ymin = min(y) - deltaY
        ymax = max(y) + deltaY
        print(xmin, xmax, ymin, ymax)

        # Create meshgrid
        xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
        positions = np.vstack([xx.ravel(), yy.ravel()])

        pf= Pushforward(phi=phi, f=func)
        zz=np.reshape(pf(positions.T), xx.shape)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(xx, yy,zz, rstride=1, cstride=1, cmap='coolwarm', edgecolor='none')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('PDF')
        ax.set_title('Pushforwarded distribution function')
        fig.colorbar(surf, shrink=0.5, aspect=5)  # add color bar indicating the PDF
        # ax.view_init(60, 35)
        plt.show()


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.fc1 = nn.Linear(in_dim, 128)
        self.fc2 = nn.Linear(128, 1024)
        self.fc3 = nn.Linear(1024, 1024)
        self.fc4 = nn.Linear(1024, 1024)
        self.fc5 = nn.Linear(1024, out_dim)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.fc5(x)
        return x

def main():
    sample_x0 = gen2Dsample_disk(500, func_disk0, 1).astype(np.float32)
    sample_x0 = sample_x0 - np.ones_like(sample_x0) * 3
    sample_x1 = gen2Dsample_disk(500, func_disk1, 1).astype(np.float32)
    sample_x1 = sample_x1 - np.ones_like(sample_x1) * 3
    sample_t0 = gen2Dsample_square(500, func_square0, 1).astype(np.float32)
    sample_t1 = gen2Dsample_square(500, func_square1, 1).astype(np.float32)
    plt.plot(sample_t0[:,0],sample_t0[:,1], 'x', label="source0")
    plt.plot(sample_t1[:, 0], sample_t1[:, 1], 'x', label="source1")
    plt.plot(sample_x0[:, 0], sample_x0[:, 1], 'o', label="target0")
    plt.plot(sample_x1[:, 0], sample_x1[:, 1], 'o', label="target1")
    plt.legend()
    plt.show()
    t0 = torch.from_numpy(sample_t0).to(args.device)
    x0 = torch.from_numpy(sample_x0).to(args.device)
    t1 = torch.from_numpy(sample_t1).to(args.device)
    x1 = torch.from_numpy(sample_x1).to(args.device)
    vardim = 4
    phi = InjAugNODE(in_dim=2, out_dim=2, var_dim=vardim, ker_dims=[1024, 1024, 1024, 1024,1024],
                     device="cuda").to(args.device)
    # phi = AugNODE(in_dim=2, out_dim=2, sample_size=x0.shape[0]+x1.shape[0], ker_dims=[1024, 1024, 1024, 1024]).to(args.device)
    # phi = MLP(in_dim=2,out_dim=2).to(args.device)
    loss_fun0 = SinkhornLoss(eps=args.sinkhorn_eps, max_iters=args.max_sinkhorn_iters)
    loss_fun1 = SinkhornLoss(eps=args.sinkhorn_eps, max_iters=args.max_sinkhorn_iters)
    dummy0 = torch.ones(x0.shape[0], vardim).to(args.device)
    dummy0[:, 0:x0.shape[1]] = x0
    x0 = dummy0
    #
    dummy1 = torch.ones(x1.shape[0], vardim).to(args.device)
    dummy1[:, 0:x1.shape[1]] = x1
    x1 = dummy1

    optimizer = torch.optim.Adam(phi.parameters(), lr=0.0001)
    for epoch in range(1, args.num_epochs+1):
        optimizer.zero_grad()

        # Do the forward pass of the neural net, evaluating the function at the parametric points
        y0 = phi(t0)
        y1= phi(t1)
        # print(phi.augment_part[0:3,0:3])


        loss = loss_fun0(y0.unsqueeze(0), x0.unsqueeze(0))+ loss_fun1(y1.unsqueeze(0), x1.unsqueeze(0))

        loss.backward()
        optimizer.step()
        print("Epoch %d, loss = %f" % (epoch, loss.item()))
        # b = x[0:1,:]
        # print(phi.invert(x))
    # print(phi.invert(x)[:, -1])
    # print(x)
    # torch.save(phi.state_dict(), "../models/phi_inference2.pt")
    phi.load_state_dict(torch.load("../models/phi_inference2.pt"))
    PlotFit(phi=phi, x0=x0, t0=t0, x1=x1, t1=t1)
    # distr = Pushforward(phi, func_square)
    # xsample = np.linspace(-4,-2, 1000)
    # ysample = np.linspace(-4,-2, 1000)
    # zsample = np.meshgrid(xsample, ysample)
    z = torch.ones(x0.shape[0]+x1.shape[0], x0.shape[1]).to(args.device)
    z[0:x0.shape[0],:]=x0
    z[x0.shape[0]:,:]=x1
    PlotInterence(phi=phi, z=z, func=pattern_square)
if __name__ == "__main__":
    main()
