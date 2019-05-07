import argparse
import open3d
import torch
import torch.nn as nn

import matplotlib.pyplot as plt
import numpy as np

import point_cloud_utils as pcu
from fml.nn import SinkhornLoss
from ODEOT.ANODE import InjAugNODE
from ODEOT.utils import load_mesh_by_file_extension, plot_flow, embed_3d, animate_flow

argparser = argparse.ArgumentParser()
argparser.add_argument("--mesh_filename0", "-f0", type=str, help="Point cloud to reconstruct")
argparser.add_argument("--mesh_filename1", "-f1", type=str, help="Point cloud to reconstruct")
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


def main():
    # NOTE: We're doing everything in float32 and numpy defaults to float64, so you need to make sure you cast
    # everything to the right type

    # x is a tensor of shape [n, 3] containing the positions of the points we are trying to fit
    mesh0 = pcu.read_obj(args.mesh_filename0)
    mesh1 = pcu.read_obj(args.mesh_filename1)

    t = torch.from_numpy(pcu.sample_mesh_lloyd(mesh0[0],mesh0[1][:,0:3],1000).astype(np.float32)).to(args.device)
    x = torch.from_numpy(pcu.sample_mesh_lloyd(mesh1[0],mesh1[1][:,0:3],1000).astype(np.float32)).to(args.device)
    x += torch.ones_like(x)
    # print(t)

    vardim = 3
    phi = InjAugNODE(in_dim=3, out_dim=3, var_dim=vardim, ker_dims=[1024, 1024, 1024, 1024], device="cuda").to(
        args.device)
    # Eps is 1/lambda and max_iters is the maximum number of Sinkhorn iterations to do
    loss_fun = SinkhornLoss(eps=args.sinkhorn_eps, max_iters=args.max_sinkhorn_iters)
    dummy = torch.ones(x.shape[0], vardim).to(args.device)
    dummy[:, 0:x.shape[1]] = x
    x = dummy

    # Here I'm using the Adam optimizer just as an example, you'll need to replace this with your thing
    optimizer = torch.optim.Adam(phi.parameters(), lr=0.0001)
    # optimizer.add_param_group({"params": phi.augment_part})
    # print("Number of Parameters=", count_parameters(phi))

    for epoch in range(1, args.num_epochs + 1):
        optimizer.zero_grad()

        # Do the forward pass of the neural net, evaluating the function at the parametric points
        y = phi(t)

        loss = loss_fun(y.unsqueeze(0), x.unsqueeze(0))

        loss.backward()
        optimizer.step()
        print("Epoch %d, loss = %f" % (epoch, loss.item()))
        # b = x[0:1,:]
        # print(phi.invert(x))

    print(phi.invert(x)[:, -1])
    # print(x)
    torch.save(phi.state_dict(), "../models/phi_itpl.pt")
    phi.load_state_dict(torch.load("../models/phi_itpl.pt"))
    # plot_flow(x[:,0:3], t, phi, 128,  t.shape[0]//100)
    animate_flow(x[:, 0:3], t, phi, 128, t.shape[0] // 100, mesh0=mesh0)


if __name__ == "__main__":
    main()