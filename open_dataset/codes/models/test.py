from tkinter import N
import torch
import numpy as np

t = torch.tensor([[ 0.4880, -0.7527, -0.4419],
        [-0.3703, -0.9181, -0.1416],
        [ 0.0538, -0.9650, -0.2567],
        [ 0.0679, -0.9021, -0.4262],
        [ 0.3004, -0.8195, -0.4880],
        [ 0.4332, -0.7890, -0.4357],
        [ 0.3793, -0.8545, -0.3550],
        [ float('nan'), float('nan'), float('nan')]])
n = np.isnan(t)
print(n)
s = n.shape
print(s[0]*s[1] - np.count_nonzero(n))
# if np.count_nonzero(torch.isnan(t)):
#     print("yes")