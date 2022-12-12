
import os
import sys
import re
import math
import random
import json
import torch
import torch.nn as nn
import pandas as pd
from torch.optim import SGD, lr_scheduler
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
from os.path import join
import decimal
import optuna
import datetime
import argparse
from distutils.util import strtobool

from models import choose_model, MODEL_DICT


parser = argparse.ArgumentParser(description='training argument')
##########################################################################################################################
parser.add_argument('--model', type=str, default="lstm", help=f'choose model from {MODEL_DICT.keys()}')
parser.add_argument('-s', '--sequence_length', type=int, default=21, help='select train data sequence length')
parser.add_argument('-p', '--pred_future_time', type=int, default=12, help='How many seconds later would you like to predict?')
parser.add_argument('--input_shift', type=int, default=1, help='specify input (src, tgt) shift size for transformer_encdec.')
weight_path = os.path.join("..", "images", "2212081618_lstm_seq15_pred57", "trial18_MAE10.88619_MDE500.63548_lr0.005615_batch8_nhead3_num_layers5_hiddensize46_seq15_pred57.pth")
sequence_length = 15
hidden_size = 46
num_layers = 5
nhead = 3
batch_size = 8
##########################################################################################################################
output_dim = 3 # 進行方向ベクトルの要素数
selected_train_columns = ['gyroX', 'gyroY', 'gyroZ', 'accX', 'accY', 'accZ']
selected_correct_columns = ['pX', 'pY', 'pZ', 'qW', 'qX', 'qY', 'qZ']
args = parser.parse_args()

# mode; setting
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = choose_model(args.model, len(selected_train_columns), hidden_size, num_layers, nhead, output_dim, sequence_length, args.input_shift)
model = model.float()
model.to(device)
model.load_state_dict(torch.load(weight_path))

# predict
data = torch.rand((sequence_length, batch_size, 6)).to(device)
print("input data", data)
if args.model == "transformer_encdec":
    shift = args.input_shift
    src = data[:sequence_length-shift, :, :]
    tgt = data[shift:, :, :]
    output = model(src=src.float().to(device), tgt=tgt.float().to(device))
    # output = output.contiguous().view(-1, output.size(-1))
    # print("output.shape", output.shape)
    # output = output.contiguous().view(-1, output.size(-1))
elif args.model == "lstm" or args.model == "transformer_enc" or args.model == "imu_transformer":
    output = model(data.float().to(device))
    # print("output.shape", output.shape)
else:
    print(" specify light model name")
print("output", output)