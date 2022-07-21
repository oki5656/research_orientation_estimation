import os
import sys
import math
import torch
import torch.nn as nn
import pandas as pd
from transformer_enc_dec import Transformer
from transformer_enc import TransAm
from matplotlib import pyplot as plt
import numpy as np



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# model = Transformer(d_model=6, nhead = 3, seq_len = 2)
# model = nn.Transformer(d_model=6, nhead=3)
model = TransAm()
model = model.float()
model.to(device)


model.eval()

src = torch.rand((30, 8, 6)).to(device)
tgt = torch.rand((20, 8, 6)).to(device)
# src = torch.rand((30, 32, 6)).to(device)# (src sequence, batch, feature)
# tgt = torch.rand((40, 32, 6)).to(device)# (tgt sequence, batch, feature)
print(src)
# print(tgt)

print("src.shape : ", src.shape)
# print("tgt.shape : ", tgt.shape)
out = model(src)
print("out.shape : ", out.shape)