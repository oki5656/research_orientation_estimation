
from faulthandler import disable
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
parser.add_argument('--model', type=str, default="transformer_encdec", help=f'choose model from {MODEL_DICT.keys()}')
# parser.add_argument('-s', '--sequence_length', type=int, default=21, help='select train data sequence length')
# parser.add_argument('-p', '--pred_future_time', type=int, default=12, help='How many seconds later would you like to predict?')
parser.add_argument('--input_shift', type=int, default=1, help='specify input (src, tgt) shift size for transformer_encdec.')
test_data_path = os.path.join("..","datasets", "large_space", "nan_removed", "Take20220809_083159pm_002nan_removed.csv")
# weight_path = os.path.join("..", "images", "2211021545_transformer_encdec_seq21_pred33_trial25_epoch100_unitfalse_trainsum_Take20220809_083159001and003nan_removed_testTake20220809_083159pm_002nan_removed",
#                            "trial23_MAE6.44.pth")
weight_path = os.path.join("..", "images", "2210271336_transformer_encdec_seq27_pred27_trial25_epoch100_unitfalse_trainsum_Take20220809_083159001and003nan_removed_testTake20220809_083159pm_002nan_removed",
                           "MAE4.26seq27_pred27.pth")
sequence_length = 27
pred_future_frame =27
hidden_size = 13
num_layers = 8
batch_size = 8
test_data_start_col = 30*20
# test_data_end_col = 10
predicted_frequency = 1 # means test data is used 1 in selected "value" lines
number_of_predict_position = 50
##########################################################################################################################
output_dim = 3 # 進行方向ベクトルの要素数
selected_train_columns = ['gyroX', 'gyroY', 'gyroZ', 'accX', 'accY', 'accZ']
selected_correct_columns = ['pX', 'pY', 'pZ', 'qW', 'qX', 'qY', 'qZ', 'imu_position_x', 'imu_position_y', 'imu_position_z',]
args = parser.parse_args()

# mode; setting
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = choose_model(args.model, len(selected_train_columns), hidden_size, num_layers,
                     output_dim, sequence_length, args.input_shift)
model = model.float()
model.to(device)
model.load_state_dict(torch.load(weight_path))
model.eval()


def data_loader(path, train_columns, correct_columns, start_col):
    end_col = start_col+predicted_frequency*number_of_predict_position+sequence_length+pred_future_frame+2
    df = pd.read_csv(path)
    train_x_df = df[train_columns]
    train_t_df = df[correct_columns]
    print("type(train_x_df)", type(train_x_df))
    print("train_x_df[start_col: end_col]", train_x_df[start_col: end_col])

    return train_x_df, train_t_df

def TransWithQuat(batch_t_df_np, pred_future_time):
    """正解の進行方向ベクトルを出力する
    Args : 
        batch_t_df_np (ndarray) : train_t_dfからseq_length+pred_fut_time分を抽出したもの
        pred_future_time (int) : どれくらい未来を予測するか
    Returns : 
        dir_vec (ndarray) : スマートフォン座標系の正解進行方向ベクトル
    """
    dir_vec = np.ones(3)

    # for j in range(batch_size):
    qW, qX, qY, qZ = batch_t_df_np[0][3], batch_t_df_np[0][4], batch_t_df_np[0][5], batch_t_df_np[0][6]
    E = np.array([[qX**2 - qY**2 - qZ**2 + qW**2, 2*(qX*qY - qZ*qW), 2*(qX*qZ + qY*qW)],
            [2*(qX*qY + qZ*qW), -qX**2 + qY**2 - qZ**2 + qW**2, 2*(qY*qZ - qX*qW)],
            [2*(qX*qZ - qY*qW), 2*(qY*qZ + qX*qW), -qX**2 - qY**2 + qZ**2 + qW**2]])#クォータニオン表現による回転行列
    corr_dir_vec = np.array([batch_t_df_np[pred_future_time][0] - batch_t_df_np[0][0],
                                batch_t_df_np[pred_future_time][1] - batch_t_df_np[0][1],
                                batch_t_df_np[pred_future_time][2] - batch_t_df_np[0][2]])#２点(現在とpred_future_time)の位置から進行方向ベクトルを求めた
    smh_dir_vec = np.matmul(E.T, corr_dir_vec.T)##############################  転地するかも。スマートフォン座標系進行方向ベクトル生成。
    dir_vec[0], dir_vec[1], dir_vec[2] = smh_dir_vec[0], smh_dir_vec[1], smh_dir_vec[2]

    return dir_vec

def predict(train_x_df, train_t_df):
    """predict next step position from selected model
    Args: 
        train_x_df: test data. columns = ['gyroX', 'gyroY', 'gyroZ', 'accX', 'accY', 'accZ']
        train_t_df: test correct data. colmuns = ['pX', 'pY', 'pZ', 'qW', 'qX', 'qY', 'qZ']
    Returns: 
        output: predicted next steps position. Shape is (number_of_predict_position, 3)
    """
    outputs = []
    correct_dirctions = []
    smp_positions = []

    # TODO smp_posi, leg_posiの違いに着目して実装する

    if args.model == "transformer_encdec":
        shift = args.input_shift
        # src = data[:sequence_length-shift, :, :]
        # tgt = data[shift:, :, :]
        for i in range(number_of_predict_position):
            start_col = i*predicted_frequency
            src = torch.tensor(np.array(train_x_df[start_col:start_col+sequence_length-shift])).unsqueeze(1)
            tgt = torch.tensor(np.array(train_x_df[start_col+shift:start_col+sequence_length])).unsqueeze(1)
            output = model(src=src.float().to(device), tgt=tgt.float().to(device)).cpu().detach().numpy()
            # correct = np.array(train_t_df.iloc[start_col+sequence_length+pred_future_frame])
            
            correct = TransWithQuat(np.array(train_t_df.iloc[start_col + sequence_length - 1: start_col + sequence_length + pred_future_frame]),
                                    pred_future_frame)

            smp_position = np.array(train_t_df.iloc[start_col+sequence_length])
            outputs.append(output)
            correct_dirctions.append(correct)
            smp_positions.append(smp_position)

    elif args.model == "lstm" or args.model == "transformer_enc" or args.model == "imu_transformer":
        pass
        # output = model(data.float().to(device))
    else:
        print(" specify light model name")

    assert len(outputs) == len(correct_dirctions), "length of outouts and length of corrects is different. you should check th code."

    return outputs, correct_dirctions, smp_positions


def calc_distance(output, correct):
    dis_err = np.sqrt((output[0]-correct[0])**2+(output[1]-correct[1])**2+(output[2]-correct[2])**2)
    
    return dis_err


def convert_err2RGB(dis_error):
    """距離誤差の大きさにお応じて色（RGB値）を出力する
    Args : 
        dis_err(float) : 距離誤差
    Return : 
        rgb(list) : RGB値のリスト
    """
    rgb = []*3
    if dis_error <= 100:
        rgb = [0, 191, 255]
    elif dis_error <= 200:
        rgb = [50, 205, 50]
    elif dis_error <= 300:
        rgb = [255, 230, 0]
    elif dis_error <= 400:
        rgb = [255, 130, 0]
    elif dis_error <= 500:
        rgb = [255, 0, 0]
    else:
        rgb = [139, 0, 0]
    
    return rgb


def calc_err(outputs, corrects):
    """Calcurate error distance from output and correct and correspond to the size of error make RGB color list.
    Args: 
    Returns: 
        RGB_list: 
    """
    color_list = []

    for i in range(number_of_predict_position):
        output, correct = outputs[i], corrects[i]
        dis_err = calc_distance(output, correct)
        rgb = convert_err2RGB(dis_err)
        color_list.append(rgb)

    return color_list


def draw_trajectry(outputs, correct_dirctions, smp_positions):
    color_list = calc_err(outputs, correct_dirctions)
    pred_next_step_positions = []
    for i in range(number_of_predict_position):
        qW, qX, qY, qZ = smp_positions[i][3], smp_positions[i][4], smp_positions[i][5], smp_positions[i][6]
        smt_dir_vec = np.array([outputs[i][0], outputs[i][1], outputs[i][2]])
        R_smt2world = np.array([[qX**2 - qY**2 - qZ**2 + qW**2, 2*(qX*qY - qZ*qW), 2*(qX*qZ + qY*qW)],
                                [2*(qX*qY + qZ*qW), -qX**2 + qY**2 - qZ**2 + qW**2, 2*(qY*qZ - qX*qW)],
                                [2*(qX*qZ - qY*qW), 2*(qY*qZ + qX*qW), -qX**2 - qY**2 + qZ**2 + qW**2]])
        word_dirction_vector = np.matmul(R_smt2world, smt_dir_vec.T)
        pred_next_step_positions.append(word_dirction_vector+smp_positions[i][:3])
        color_list.insert(0, [0, 0, 0])

    # 描画
    correct_dirctions = np.array(correct_dirctions)
    pred_next_step_positions = np.array(pred_next_step_positions)
    color_list = np.array(color_list)/255
    x = np.concatenate([correct_dirctions[:, 0], pred_next_step_positions[:, 0]])
    y = np.concatenate([correct_dirctions[:, 1], pred_next_step_positions[:, 1]])
    z = np.concatenate([correct_dirctions[:, 2], pred_next_step_positions[:, 2]])
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(x, y, z, vmin=0, vmax=1, c=color_list)
    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() * 0.5
    mid_x = (x.max()+x.min()) * 0.5
    mid_y = (y.max()+y.min()) * 0.5
    mid_z = (z.max()+z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    plt.show()
    pass


def main():
    train_x_df, train_t_df = data_loader(test_data_path, selected_train_columns, selected_correct_columns,
                                         test_data_start_col)
    outputs, correct_dirctions, smp_positions = predict(train_x_df, train_t_df)
    draw_trajectry(outputs, correct_dirctions, smp_positions)

main()
