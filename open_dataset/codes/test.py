# このプログラムはtrain, val, testのtest工程を行う

import os
import sys
import re
import math
import random
import json
import glob
import torch
import torch.nn as nn
import pandas as pd
from torch.optim import SGD, lr_scheduler
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
from os.path import join
import optuna
import datetime
import argparse
from statistics import mean
from distutils.util import strtobool

from models import choose_model, MODEL_DICT

###############################################################################################################################################################
result_save_path = join("..", "images2")
test_data_path = join("..","datasets", "large_space", "nan_removed", "interpolation_under_15", "harf_test_20220809_002_nan_under15_nan_removed.csv")
test_data_weight_path = join("..", "images3", "stopped_2212112335_transformer_encdec_seq15_pred21", "trial1_MAE4.7246_MDE113.81507_lr0.000266_batch8_nhead6_num_layers5_hiddensize10_seq15_pred21.pth")
selected_train_columns = ['gyroX', 'gyroY', 'gyroZ', 'accX', 'accY', 'accZ']
selected_correct_columns = ['pX', 'pY', 'pZ', 'qW', 'qX', 'qY', 'qZ', 'imu_position_x', 'imu_position_y', 'imu_position_z']
Non_duplicate_length = 10
output_dim = 3
batch_size = 8
parser = argparse.ArgumentParser(description='training argument')
parser.add_argument('--weight_save', type=strtobool, default=True, help='specify weight file save(True) or not(False).')
parser.add_argument('--model', type=str, default="transformer_encdec", help=f'choose model from {MODEL_DICT.keys()}')
parser.add_argument('--epoch', type=int, default=100, help='specify epochs number')
parser.add_argument('-s', '--sequence_length', type=int, default=27, help='select train data sequence length')
parser.add_argument('-p', '--pred_future_time', type=int, default=33, help='How many seconds later would you like to predict?')
parser.add_argument("--is_output_unit", type=str, default="false", help='select output format from unit vector or normal vector(including distance)')
parser.add_argument("--is_train_smp2foot", type=str, default="true", help='select training Position2Position or smpPosition2footPosition')
parser.add_argument('--input_shift', type=int, default=1, help='specify input (src, tgt) shift size for transformer_encdec.')
args = parser.parse_args()
###############################################################################################################################################################


def dataloader(path, train_columns, correct_columns):
    df = pd.read_csv(path)
    train_x_df = df[train_columns]
    train_t_df = df[correct_columns]

    return train_x_df, train_t_df


def CalcAngle(NowPosi, EstDir, CorrDir):
    L1, L2, D, cos_theta, theta = 0, 0, 0, 0, 0
    L1 = math.sqrt((0 - EstDir[0])**2 + (0 - EstDir[1])**2 + (0 - EstDir[2])**2)
    L2 = math.sqrt((0 - CorrDir[0])**2 + (0 - CorrDir[1])**2 + (0 - CorrDir[2])**2)
    D = math.sqrt((EstDir[0] - CorrDir[0])**2 + (EstDir[1] - CorrDir[1])**2 + (EstDir[2] - CorrDir[2])**2)
    cos_theta = (L1**2 + L2**2 - D**2 + 0.000000000000001-0.000000000000001)/(2*L1*L2 + 0.000000000000001)
    theta_rad = math.acos(np.clip(cos_theta, -1.0, 1.0))# thetaはラジアン
    theta_deg = math.degrees(theta_rad)# ラジアンからdegreeに変換

    return theta_deg


def CalcAngleErr(output, label, batch_size):
    angleErrSum = 0.0
    distanceErrSum = 0.0
    for i in range(batch_size):
        angleErr = CalcAngle(label[i, :], output[i, :], label[i, :])
        angleErrSum += angleErr
        distanceErr = math.sqrt((label[i, 0] - output[i, 0])**2 + (label[i, 1] -  output[i, 1])**2 + (label[i, 2] -  output[i, 2])**2)
        distanceErrSum += distanceErr

    return angleErrSum/batch_size, distanceErrSum/batch_size


def ConvertUnitVec(dir_vec):
    batch_size, _ = dir_vec.shape
    unit_dir_vec = np.empty((batch_size, 3))
    for i in range(batch_size):
        bunbo = math.sqrt(dir_vec[i][0]**2 + dir_vec[i][1]**2 + dir_vec[i][2]**2) + 0.0000000000000000001
        unit_dir_vec[i][0] = dir_vec[i][0]/bunbo
        unit_dir_vec[i][1] = dir_vec[i][1]/bunbo
        unit_dir_vec[i][2] = dir_vec[i][2]/bunbo

    return unit_dir_vec


def MakeBatch(train_x_df, train_t_df, batch_size, sequence_length, selected_train_columns,
              selected_correct_columns, mini_batch_random_list, pred_future_time, is_output_unit,
              Non_duplicate_length, Non_duplicate_length_offset, is_train_smp2foot):
    """train_x, train_tを受け取ってbatch_x_df_np(sequence_length, batch_size, input_size)と
    dir_vec(sequence_length, batch_size, input_size)を返す
    Args : 

    Returns : 
        torch.tensor(batch_x_df_np) : batch size分の入力データ。shapeは(sequence_length, batch_size, input_size)
        torch.tensor(dir_vec) : スマホ座標系の正解進行方向ベクトルshapeは(sequence_length, batch_size, input_size)
    """
    out_x = list()
    out_t = list()
    batch_length = len(mini_batch_random_list)
    if batch_length == 8:
        batch_size = 8
    else:
        batch_size=len(mini_batch_random_list)

    for i in range(batch_size):
        idx = mini_batch_random_list[i]*Non_duplicate_length + Non_duplicate_length_offset
        out_x.append(np.array(train_x_df[idx : idx + sequence_length]))
        out_t.append(np.array(train_t_df[idx + sequence_length - 1: idx + sequence_length + pred_future_time]))
        # out_t.append(np.array(train_t_df.loc[idx + sequence_length + pred_future_time]))
    out_x = np.array(out_x)
    out_t = np.array(out_t)
    batch_x_df_np = out_x.transpose(1, 0, 2)
    batch_t_df_np = out_t.transpose(1, 0, 2)
    # print("out_x.shape", out_x.shape)# out_x.shape (19, 30, 6)=(batch size, sequence length, x-feature num)
    # print("out_t.shape", out_t.shape)# out_t.shape (19, 41, 7)=(batch size, now to future length, t-feature num)

    # スマホ座標系正解ベクトル生成
    if is_train_smp2foot:
        dir_vec = TransWithQuatSMP2P(batch_t_df_np, batch_size, sequence_length, pred_future_time)
    else:
        dir_vec = TransWithQuatP2P(batch_t_df_np, batch_size, sequence_length, pred_future_time)# dir_vec.shape : (batch size, 3)

    if is_output_unit == "true":
        unit_dir_vec = ConvertUnitVec(dir_vec) # dir_vecのある行が0だとnanを生成してしまう。
        return torch.tensor(batch_x_df_np), torch.tensor(unit_dir_vec)
    else:
        return torch.tensor(batch_x_df_np), torch.tensor(dir_vec)


def TransWithQuatSMP2P(batch_t_df_np, batch_size, sequence_length, pred_future_time):
    """
    Returns : 
        dirvec (ndarray) : 現在スマホ位置から未来すらわちpred_future_timeの足の位置への方向ベクトル
    """
    dir_vec = np.ones((batch_size, 3))

    for j in range(batch_size):
        qW, qX, qY, qZ = batch_t_df_np[0][j][3], batch_t_df_np[0][j][4], batch_t_df_np[0][j][5], batch_t_df_np[0][j][6]

        # クォータニオン表現による回転行列
        E = np.array([[qX**2 - qY**2 - qZ**2 + qW**2, 2*(qX*qY - qZ*qW), 2*(qX*qZ + qY*qW)],
                [2*(qX*qY + qZ*qW), -qX**2 + qY**2 - qZ**2 + qW**2, 2*(qY*qZ - qX*qW)],
                [2*(qX*qZ - qY*qW), 2*(qY*qZ + qX*qW), -qX**2 - qY**2 + qZ**2 + qW**2]])

        # ２点(現在スマホ位置から未来すらわちpred_future_timeの足の位置)から方向ベクトルを求める
        corr_dir_vec = np.array([batch_t_df_np[pred_future_time][j][0] - batch_t_df_np[0][j][7],
                                 batch_t_df_np[pred_future_time][j][1] - batch_t_df_np[0][j][8],
                                 batch_t_df_np[pred_future_time][j][2] - batch_t_df_np[0][j][9]])
        smh_dir_vec = np.matmul(E.T, corr_dir_vec.T)#  転地するかも。スマートフォン座標系進行方向ベクトル生成。
        dir_vec[j][0], dir_vec[j][1], dir_vec[j][2] = smh_dir_vec[0], smh_dir_vec[1], smh_dir_vec[2]

    return dir_vec


def TransWithQuatP2P(batch_t_df_np, batch_size, sequence_length, pred_future_time):
    """
    Returns : 
    dirvec (ndarray) : 現在位置(px, py, pz)から未来位置(px, py, pz)への方向ベクトル。px, py, pzを予測する
    """
    dir_vec = np.ones((batch_size, 3))

    for j in range(batch_size):
        qW, qX, qY, qZ = batch_t_df_np[0][j][3], batch_t_df_np[0][j][4], batch_t_df_np[0][j][5], batch_t_df_np[0][j][6]
        E = np.array([[qX**2 - qY**2 - qZ**2 + qW**2, 2*(qX*qY - qZ*qW), 2*(qX*qZ + qY*qW)],
                [2*(qX*qY + qZ*qW), -qX**2 + qY**2 - qZ**2 + qW**2, 2*(qY*qZ - qX*qW)],
                [2*(qX*qZ - qY*qW), 2*(qY*qZ + qX*qW), -qX**2 - qY**2 + qZ**2 + qW**2]])#クォータニオン表現による回転行列
        corr_dir_vec = np.array([batch_t_df_np[pred_future_time][j][0] - batch_t_df_np[0][j][0],
                                 batch_t_df_np[pred_future_time][j][1] - batch_t_df_np[0][j][1],
                                 batch_t_df_np[pred_future_time][j][2] - batch_t_df_np[0][j][2]])#２点(現在とpred_future_time)の位置から進行方向ベクトルを求めた
        smh_dir_vec = np.matmul(E.T, corr_dir_vec.T)##############################  転地するかも。スマートフォン座標系進行方向ベクトル生成。
        dir_vec[j][0], dir_vec[j][1], dir_vec[j][2] = smh_dir_vec[0], smh_dir_vec[1], smh_dir_vec[2]

    return dir_vec


def test():
    """train, val, testの中のtest工程を行う。結果はtest_result.txtに記録される
    """
    print("\ntest starting...")
    test_data_weight_file_name = os.path.basename(test_data_weight_path)
    num_layers = int((re.search(r'num_layers(.+)_hid', test_data_weight_file_name).group(1)))
    hidden_size = int((re.search(r'hiddensize(.+)_seq', test_data_weight_file_name).group(1)))
    nhead = int((re.search(r'nhead(.+)_num', test_data_weight_file_name).group(1)))
    sequence_length = int((re.search(r'seq(.+)_pre', test_data_weight_file_name).group(1)))
    pred_future_time = int((re.search(r'pred(.+).pth', test_data_weight_file_name).group(1)))

    test_x_df, test_t_df = dataloader(test_data_path, selected_train_columns, selected_correct_columns)
    test_frame_num = len(test_x_df)
    test_mini_data_num = int(test_frame_num/Non_duplicate_length)
    test_use_data_num = test_mini_data_num - (sequence_length+pred_future_time)//Non_duplicate_length - 2
    test_iter_num = test_use_data_num//batch_size

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = choose_model(args.model, len(selected_train_columns), hidden_size, num_layers, nhead, output_dim, sequence_length, args.input_shift)
    model = model.float()
    model.to(device)
    model.load_state_dict(torch.load(test_data_weight_path))
    model.eval()
    TestAngleErrSum = 0
    TestDistanceErrSum = 0
    TestLossSum = 0
    mini_batch_test_random_list =[]
    Non_duplicate_length_offset = np.random.randint(0, Non_duplicate_length)
    criterion = nn.L1Loss()

    test_random_num_list = random.sample(range(1, test_use_data_num + 1),
                                        k=test_use_data_num)
    for _ in range(batch_size):
        mini_batch_test_random_list.append(test_random_num_list.pop())

    for i in tqdm(range(test_iter_num)):
        data, label = MakeBatch(test_x_df, test_t_df, batch_size, sequence_length, selected_train_columns, selected_correct_columns,
                                mini_batch_test_random_list, pred_future_time, args.is_output_unit, Non_duplicate_length,
                                Non_duplicate_length_offset, args.is_train_smp2foot)
        data.to(device)
        label.to(device)

        if args.model == "transformer_encdec":
            shift = args.input_shift
            src = data[:sequence_length-shift, :, :]
            tgt = data[shift:, :, :]
            output = model(src=src.float().to(device), tgt=tgt.float().to(device))
        elif args.model == "lstm" or args.model == "transformer_enc" or args.model == "imu_transformer":
            output = model(data.float().to(device))
        else:
            print("specify light model name")

        angleErr, distanceErr = CalcAngleErr(output, label, batch_size)
        loss = criterion(output.float().to(device), label.float().to(device))
        TestAngleErrSum += angleErr
        TestDistanceErrSum += distanceErr
        TestLossSum += float(loss)
        
    MAE_te = TestAngleErrSum/test_iter_num
    MDE_te = TestDistanceErrSum/test_iter_num
    MTL_te = TestLossSum/test_iter_num
    tqdm.write(f"Test mean angle and distance error = {MAE_te}, {MDE_te}")
    now = datetime.datetime.now()
    save_file_name = f"test_result_{now.month}{now.day}_{now.hour}{now.hour}_{args.model}"\
                     f"_seq{sequence_length}_pred{pred_future_time}.txt"
    save_file_path = join(result_save_path, save_file_name) 
    with open(save_file_path, 'w') as f:
        s = f"test data weight path: {test_data_weight_path}\n"\
            f"model name: {args.model}\n\n"\
            f"test mean angle error: {MAE_te}\n"\
            f"test mean distance error:{MDE_te}\n"\
            f"test mean loss:{MTL_te}"
        f.write(s)


if __name__ == "__main__":
    test()
