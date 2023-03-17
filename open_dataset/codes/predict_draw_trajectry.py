# 次歩推定軌跡とその真値を描画する
# 入力にはweight_file, sequence_length, pred_future_frame, hidden_size, num_layers, batch_size, nheadなどが必要
# 描画をスタートするフレームNo, 描画するフレーム数，何フレームに１フレームを描画するかが設定できる
# コンソールに距離誤差のmax, minが出力される

import os
import re
import torch
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from os.path import join
import argparse

from models import choose_model, MODEL_DICT


parser = argparse.ArgumentParser(description='training argument')
##########################################################################################################################
parser.add_argument('--model', type=str, default="lstm", help=f'choose model from {MODEL_DICT.keys()}')
# parser.add_argument('-s', '--sequence_length', type=int, default=21, help='select train data sequence length')
# parser.add_argument('-p', '--pred_future_time', type=int, default=12, help='How many seconds later would you like to predict?')
parser.add_argument('--input_shift', type=int, default=1, help='specify input (src, tgt) shift size for transformer_encdec.')
# test_data_path = join("..","datasets", "large_space", "nan_removed", "Take20220809_083159pm_002nan_removed.csv")
test_data_path = join("..","datasets", "large_space", "nan_removed", "interpolation_under_15", "harf_test_20220809_002_nan_under15_nan_removed.csv")
weight_path = join("..", "images5", "2201180522_lstm_seq27_pred21", "trial9_MAE3.98126_MDE90.82089_lr0.020077_batch8_nhead3_num_layers3_hiddensize54_seq27_pred21.pth")
parser.add_argument("--is_train_smp2foot", type=str, default="true", help='select training Position2Position or smpPosition2footPosition')
_2Dor3D = "2D"
sequence_length = 27
pred_future_frame =21
hidden_size = 54
num_layers = 3
batch_size = 8
nhead = 3
# test_data_start_col = 30*0
test_data_start_col = 30*65
predicted_frequency = 1 # means test data is used 1 in selected "value" lines
number_of_predict_position = 30*60
# number_of_predict_position = 9188-21-21
Normalization_or_Not = "Normalization"
##########################################################################################################################
g = 9.80665 # fixed
output_dim = 3 # 進行方向ベクトルの要素数
max_error =0
min_error = 99999999
selected_train_columns = ['gyroX', 'gyroY', 'gyroZ', 'accX', 'accY', 'accZ']
selected_correct_columns = ['pX', 'pY', 'pZ', 'qW', 'qX', 'qY', 'qZ', 'imu_position_x', 'imu_position_y', 'imu_position_z']
args = parser.parse_args()
weight_file_name = os.path.basename(weight_path)

if "hiddensize" in weight_file_name:
    hidden_size = int(re.search(r'hiddensize(.+)_seq', weight_file_name).group(1))
if "num_layer" in weight_file_name: 
    num_layer = int(re.search(r'num_layers(.+)_hid', weight_file_name).group(1))
if "nhead" in weight_file_name:
    nhead = int(re.search(r'nhead(.+)_num', weight_file_name).group(1))
if "seq" in weight_file_name:
    sequence_length = int(re.search(r'seq(.+)_pre', weight_file_name).group(1))

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = choose_model(args.model, len(selected_train_columns), hidden_size, num_layers,
                     nhead, output_dim, sequence_length, args.input_shift)
model = model.float()
model.to(device)
model.load_state_dict(torch.load(weight_path))
model.eval()


def acceleration_normalization(train_x_df):
    """加速度を受け取り正規化（合力方向から距離が9.8分になるようにそれぞれのベクトルの要素から引く）する
    Args: 
        out_x : (batchsize, seq_length, 要素数)で構成される加速度と角速度シーケンス.ndarray
    output: 
        out_x : (batchsize, seq_length, 要素数)で構成される "正規化された" 加速度と角速度シーケンス
    """
    seq_len, element = train_x_df.shape
    for j in range(seq_len):
        l2norm = np.sqrt(train_x_df.iat[j, 3]**2+train_x_df.iat[j, 4]**2+train_x_df.iat[j, 5]**2)
        train_x_df.iat[j, 3] -= g*train_x_df.iat[j, 3]/l2norm
        train_x_df.iat[j, 4] -= g*train_x_df.iat[j, 4]/l2norm
        train_x_df.iat[j, 5] -= g*train_x_df.iat[j, 5]/l2norm

    return train_x_df


def data_loader(path, train_columns, correct_columns, start_col):
    end_col = start_col+predicted_frequency*number_of_predict_position+sequence_length+pred_future_frame+2
    df = pd.read_csv(path)
    train_x_df = df[train_columns]
    train_t_df = df[correct_columns]
    if Normalization_or_Not == "Normalization":
        train_x_df = acceleration_normalization(train_x_df)
    print("type(train_x_df)", type(train_x_df))
    print("train_x_df[start_col: end_col]", train_x_df[start_col: end_col])

    return train_x_df, train_t_df


def TransWithQuatSMP2P(batch_t_df_np, output, pred_future_time):
    """正解の進行方向ベクトルを出力する
    Args : 
        batch_t_df_np (ndarray) : train_t_dfからseq_length+pred_fut_time分を抽出したもの
        pred_future_time (int) : どれくらい未来を予測するか
    Returns : 
        dirvec (ndarray) : 現在スマホ位置から未来すらわちpred_future_timeの足の位置への方向ベクトル（世界座標系）
    """
    qW, qX, qY, qZ = batch_t_df_np[0][3], batch_t_df_np[0][4], batch_t_df_np[0][5], batch_t_df_np[0][6]

    # クォータニオン表現による回転行列
    E = np.array([[qX**2 - qY**2 - qZ**2 + qW**2, 2*(qX*qY - qZ*qW), 2*(qX*qZ + qY*qW)],
            [2*(qX*qY + qZ*qW), -qX**2 + qY**2 - qZ**2 + qW**2, 2*(qY*qZ - qX*qW)],
            [2*(qX*qZ - qY*qW), 2*(qY*qZ + qX*qW), -qX**2 - qY**2 + qZ**2 + qW**2]])

    smp_dir_vec = np.array([output[0], output[1], output[2]])# スマホ座標系次歩推定ベクトル
    world_dir_vec = np.matmul(E, smp_dir_vec.T)#  世界座標系進行方向ベクトル生成。

    return world_dir_vec


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
    world_pred_next_step_positions = []
    world_foot_positions = []

    # if args.model == "transformer_encdec":
    shift = args.input_shift
    for i in range(number_of_predict_position):
        start_col = (i+test_data_start_col)*predicted_frequency
        if args.model == "transformer_encdec":
            src = torch.tensor(np.array(train_x_df[start_col:start_col+sequence_length-shift])).unsqueeze(1)
            tgt = torch.tensor(np.array(train_x_df[start_col+shift:start_col+sequence_length])).unsqueeze(1)
            output = model(src=src.float().to(device), tgt=tgt.float().to(device)).cpu().detach().numpy()
        elif args.model == "lstm" or args.model == "transformer_enc" or args.model == "imu_transformer":
            data = torch.tensor(np.array(train_x_df[start_col:start_col+sequence_length])).unsqueeze(1)
            output = model(data.float().to(device)).cpu().detach().numpy()
        else:
            print(" specify light model name")
        world_dir_vec = TransWithQuatSMP2P(np.array(train_t_df.iloc[start_col + sequence_length - 1: start_col + sequence_length + pred_future_frame]),
                                            output, pred_future_frame)

        world_smp_position = np.array(train_t_df.iloc[start_col+sequence_length-1, 7:10])
        world_foot_position = np.array(train_t_df.iloc[start_col+sequence_length+pred_future_frame, 0:4])
        world_pred_next_step_positions.append(world_smp_position+world_dir_vec)
        world_foot_positions.append(world_foot_position)

    # elif args.model == "lstm" or args.model == "transformer_enc" or args.model == "imu_transformer":
    #     data = torch.tensor(np.array(train_x_df[start_col:start_col+sequence_length])).unsqueeze(1)
    #     output = model(data.float().to(device)).cpu().detach().numpy()


    assert len(outputs) == len(correct_dirctions), "length of outouts and length of corrects is different. you should check th code."

    return world_foot_positions, world_pred_next_step_positions


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
    # if dis_error <= 10000:
    if dis_error <= 100:
        rgb = [0, 150, 255]
    elif dis_error <= 200:
        rgb = [50, 205, 50]
    elif dis_error <= 300:
        rgb = [255, 135, 0]
    elif dis_error <= 400:
        rgb = [255, 10, 0]
    # elif dis_error <= 400:
    #     rgb = [255, 0, 0]
    else:
        rgb = [139, 0, 0]
    
    return rgb


def calc_err(outputs, corrects):
    """Calcurate error distance from output and correct and correspond to the size of error make RGB color list.
    Args: 
        outputs : result of next-step prediction
        corrects : correct data of next-step prediction
    Returns: 
        RGB_list: 
    """
    color_list = []
    global min_error
    global max_error 
    for i in range(number_of_predict_position):
        output, correct = outputs[i], corrects[i]
        dis_err = calc_distance(output, correct)
        min_error = min(dis_err, min_error)
        max_error = max(dis_err, max_error)
        rgb = convert_err2RGB(dis_err)
        color_list.append(rgb)

    return color_list


def draw_trajectry_2D(world_foot_positions, world_pred_next_step_positions):
    """This function draw the next-step prediction result in 2D image. You can choose 2D or 3D using _2Dor3D variable.
    Args :
        world_foot_positions : 世界座標系で表される次歩推定位置
        world_pred_next_step_positions : 世界座標系で表される次歩推定の正解位置
    Returns : 
        None
    """
    color_list = calc_err(world_foot_positions, world_pred_next_step_positions)
    for i in range(number_of_predict_position):
        color_list.insert(0, [0, 0, 0])

    # 描画
    world_foot_positions = np.array(world_foot_positions)/1000
    world_pred_next_step_positions = np.array(world_pred_next_step_positions)/1000
    color_list = np.array(color_list)/255
    x = np.concatenate([world_foot_positions[:, 0], world_pred_next_step_positions[:, 0]])
    y = np.concatenate([world_foot_positions[:, 2], world_pred_next_step_positions[:, 2]])
    # z = np.concatenate([world_foot_positions[:, 2], world_pred_next_step_positions[:, 2]])
    # fig = plt.figure()
    fig, ax = plt.subplots(figsize=(10,8))

    # ax.scatter(x, y, z, vmin=0, vmax=1, c=color_list, marker = ".")
    max_range = np.array([x.max()-x.min(), y.max()-y.min()]).max() * 0.5+1
    mid_x = (x.max()+x.min()) * 0.5
    mid_y = (y.max()+y.min()) * 0.5
    # mid_z = (z.max()+z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range + 4, mid_y + max_range - 4)
    ax.set_xlabel("[m]")
    ax.set_ylabel("[m]")
    plt.xticks([-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10])
    plt.yticks([-4, -2, 0, 2, 4])

    for idx, c in enumerate(color_list):
        ax.plot(x[idx], y[idx], 'bo', color=c, marker = ".")
    ax.legend(loc="upper left")
    plt.grid()
    plt.show()


def draw_trajectry_3D(world_foot_positions, world_pred_next_step_positions):
    """This function draw the next-step prediction result in 3D image. You can choose 2D or 3D using _2Dor3D variable.
    Args :
        world_foot_positions : 世界座標系で表される次歩推定位置
        world_pred_next_step_positions : 世界座標系で表される次歩推定の正解位置
    Returns : 
        None
    """
    color_list = calc_err(world_foot_positions, world_pred_next_step_positions)
    for i in range(number_of_predict_position):
        color_list.insert(0, [0, 0, 0])

    # 描画
    world_foot_positions = np.array(world_foot_positions)/1000
    world_pred_next_step_positions = np.array(world_pred_next_step_positions)/1000
    color_list = np.array(color_list)/255
    x = np.concatenate([world_foot_positions[:, 0], world_pred_next_step_positions[:, 0]])
    y = np.concatenate([world_foot_positions[:, 1], world_pred_next_step_positions[:, 1]])
    z = np.concatenate([world_foot_positions[:, 2], world_pred_next_step_positions[:, 2]])
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(x, y, z, vmin=0, vmax=1, c=color_list, marker = ".")
    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() * 0.5
    mid_x = (x.max()+x.min()) * 0.5
    mid_y = (y.max()+y.min()) * 0.5
    mid_z = (z.max()+z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    # plt.legend(("100mm", "200mm", "300mm", "400mm", "500mm", "others"))
    ax.set_xlabel("[m]")
    ax.set_ylabel("[m]")
    ax.set_zlabel("[m]")
    plt.show()


def main():
    train_x_df, train_t_df = data_loader(test_data_path, selected_train_columns, selected_correct_columns,
                                         test_data_start_col)
    world_foot_positions, world_pred_next_step_positions = predict(train_x_df, train_t_df)
    if _2Dor3D == "2D":
        draw_trajectry_2D(world_foot_positions, world_pred_next_step_positions)
    elif _2Dor3D == "3D":
        draw_trajectry_3D(world_foot_positions, world_pred_next_step_positions)
    else:
        print(f"_2Dor3D you set is something wrong. You set {_2Dor3D}")

    print(f"max error: {max_error}\nmin error: {min_error}")

main()
