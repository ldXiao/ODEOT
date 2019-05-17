import argparse
import open3d
import torch
import torch.nn as nn

import matplotlib.pyplot as plt
import numpy as np

import point_cloud_utils as pcu
from fml.nn import SinkhornLoss
from ODEOT.ANODE import InjAugNODE
from ODEOT.utils import load_mesh_by_file_extension, plot_flow, embed_3d, animate_flow, precond, seed_everything

print("everything imported correctly")