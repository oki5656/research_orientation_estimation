# このファイルではデータセットを用いLSTMで学習を行う
########################################################################
from genericpath import isfile
import os
import sys
# a=os.path.dirname(sys.executable)
print(os.path.dirname(sys.executable))
########################################################################

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
import optuna
import datetime
import argparse
from statistics import mean
from distutils.util import strtobool

from models import choose_model, MODEL_DICT

#############################################  config  ##################################################
img_save_path = os.path.join("..", "images")
# train_data_path = os.path.join("..","datasets", "TUM","dataset-room_all", "mav0", "self_made_files", "new_all_in_imu_mocap_13456.csv")
# test_data_path = os.path.join("..","datasets", "TUM","dataset-room2_512_16", "mav0", "self_made_files", "new_all_in_imu_mocap.csv")
# train_data_path = os.path.join("..","datasets", "oxford_IOD","handheld", "data1", "syn", "concate_imu2_vi2.csv")
# train_data_path = os.path.join("..","datasets", "oxford_IOD", "mix", "data1", "syn", "mix_slow_imu3_vi3_handheld_imu235_vi235.csv")#最近まで使ってた8/17
train_data_path = os.path.join("..","datasets", "large_space", "nan_removed", "sum_Take20220809_083159001and003nan_removed.csv")
# train_data_path = os.path.join("..","datasets", "oxford_IOD","handheld", "data1", "syn", "concate_imu3_vi3.csv")
# test_data_path = os.path.join("..","datasets", "oxford_IOD","handheld", "data1", "syn", "concate_imu4_vi4.csv")#最近まで使ってた8/17
test_data_path = os.path.join("..","datasets", "large_space", "nan_removed", "Take20220809_083159pm_002nan_removed.csv")
# test_data_path = os.path.join("..","datasets", "oxford_IOD", "slow walking", "data1", "syn", "concate_imu4_vi4.csv")
selected_train_columns = ['gyroX', 'gyroY', 'gyroZ', 'accX', 'accY', 'accZ']
selected_correct_columns = ['pX', 'pY', 'pZ', 'qW', 'qX', 'qY', 'qZ', 'imu_position_x', 'imu_position_y', 'imu_position_z']
# selected_correct_columns = ['pX', 'pY', 'pZ', 'qW', 'qX', 'qY', 'qZ']
Non_duplicate_length = 10
#########################################################################################################


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
    # print("label.shape", label.shape) # torch.Size([32, 3]) = (batch size, outout feature num)
    # print("output.shape", output.shape) # (sequence, batch size, outout feature num)
    for i in range(batch_size):
        angleErr = CalcAngle(label[i, :], output[i, :], label[i, :])# とりあえずsequenceのidxは1。後でmodelの出力経形式を要検討。# for transformer2 
        angleErrSum += angleErr
        # angleErrSum += CalcAngle(label[i, :], output[0, i, :], label[i, :])# とりあえずsequenceのidxは1。後でmodelの出力経形式を要検討。# for lstm
        # distanceErrSum += math.sqrt((label[i, 0] - output[0, i, 0])**2 + (label[i, 1] -  output[0, i, 1])**2 + (label[i, 2] -  output[0, i, 2])**2) # for lstm
        distanceErr = math.sqrt((label[i, 0] - output[i, 0])**2 + (label[i, 1] -  output[i, 1])**2 + (label[i, 2] -  output[i, 2])**2) # for transformer2
        distanceErrSum += distanceErr

    return angleErrSum/batch_size, distanceErrSum/batch_size

def ConvertUnitVec(dir_vec):
    batch_size, _ = dir_vec.shape
    unit_dir_vec = np.empty((batch_size, 3))
    # unit_dir_vec = [[0]*3 for j in range(batch_size)]
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
    # print("batch_t_df_np.shape", batch_t_df_np.shape) # (now to future length, batch size, t-feature num)
    # dir_vec = np.ones((sequence_length, batch_size, 3))

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


class Log():
    def __init__(self):
        self.LastEpochResults = {"LastEpochTestAngleErr" : [], "LastEpochTestDistanceErr" : [], "LastEpochTestLoss" : []}
        dt_now = datetime.datetime.now()
        self.TrainStartTime = "22" + str(dt_now.month).zfill(2) + str(dt_now.day).zfill(2) + str(dt_now.hour).zfill(2) + str(dt_now.minute).zfill(2)


    def train_info_log(self, train_info_log_path):
        if os.path.isfile(train_info_log_path):
            with open(train_info_log_path, mode='a') as f:
                s = f"\ntrial:{self.trial_num} BestMAE:{self.BestMAE} lr:{self.lr} "\
                    f"batch_size:{self.batch_size} num_layers:{self.num_layers} hidden_size:{self.hidden_size} "\
                    f"nhead:{self.nhead}"
                f.write(s)
        else:
            with open(train_info_log_path, 'w') as f:
                s = f"startTime:{self.TrainStartTime} model:{self.model} seq:{self.sequence_length} "\
                    f"pred:{self.pred_future_time} maxEpoch:{self.max_epoch} \ntrain_filename:{self.train_filename} "\
                    f"\ntest_filename:{self.test_filename} is_output_unit:{self.is_output_unit}\n"
                f.write(s)


    def LastEpochResultLog(self, now_epoch, max_epoch, LastEpochTestAngleErr, LastEpochTestDistanceErr, LastEpochTestLoss):
        """Record last epochs result every optuna trial.

        Args:
            now_epoch(int) : 
            max_epoch(int?) : 
            LastEpochTestAngleErr : 
            LastEpochTestDistanceErr : 
            LastEpochTestLoss : 
        Returns:
            none
        """
        if now_epoch == max_epoch-1:
            self.LastEpochResults["LastEpochTestAngleErr"].append(round(LastEpochTestAngleErr, 5))
            self.LastEpochResults["LastEpochTestDistanceErr"].append(round(LastEpochTestDistanceErr, 5))
            self.LastEpochResults["LastEpochTestLoss"].append(round(LastEpochTestLoss, 5))


    def LastEpochResultsShow(self):
        """ Show all last epoch result to console.
        """
        print("LastEpochTestAngleErr : ", self.LastEpochResults["LastEpochTestAngleErr"])
        print("LastEpochTestDistanceErr : ", self.LastEpochResults["LastEpochTestDistanceErr"])
        print("LastEpochTestLoss : ", self.LastEpochResults["LastEpochTestLoss"])


    def AverageLastEpochResultSave(self):
        nan_remove_last_epoch_test_angle_error = [x for x in self.LastEpochResults["LastEpochTestAngleErr"] if not math.isnan(x)]
        nan_remove_last_epoch_test_distance_error = [x for x in self.LastEpochResults["LastEpochTestDistanceErr"] if not math.isnan(x)]
        nan_remove_last_epoch_test_loss = [x for x in self.LastEpochResults["LastEpochTestLoss"] if not math.isnan(x)]

        RankingLastEpochTestAngleError = sorted(nan_remove_last_epoch_test_angle_error)
        RankingLastEpochTestDistanceErr = sorted(nan_remove_last_epoch_test_distance_error)
        RankingLastEpochTestLoss = sorted(nan_remove_last_epoch_test_loss)

        average_last_epoch_test_angle_error = round(mean(RankingLastEpochTestAngleError[:5]), 5)
        average_last_epoch_test_distance_error = round(mean(RankingLastEpochTestDistanceErr[:5]), 5)
        average_last_epoch_test_loss = round(mean(RankingLastEpochTestLoss[:5]), 5)
        save_file_path = join(self.train_info_dir_path, "last_epoch_average.txt")
        with open(save_file_path, 'w') as f:
            s = f"average_last_epoch_test_angle_error:{average_last_epoch_test_angle_error}"\
                f"\naverage_last_epoch_test_distance_error:{average_last_epoch_test_distance_error}"\
                f"\naverage_last_epoch_test_loss:{average_last_epoch_test_loss}"
            f.write(s)
        # return average_last_epoch_test_angle_error, average_last_epoch_test_distance_error, average_last_epoch_test_loss 


    def LastEpochResultSave(self, save_path):
        """ Save all last epoch result to specified json derectory.
        """
        save_file_name = join(save_path, "last_epoch_results.json")
        with open(save_file_name, 'w') as f:
            json.dump(self.LastEpochResults, f, indent=4)


def main(trial):
    parser = argparse.ArgumentParser(description='training argument')
    parser.add_argument('--weight_save', type=strtobool, default=True, help='specify weight file save(True) or not(False).')
    parser.add_argument('--model', type=str, default="transformer_encdec", help=f'choose model from {MODEL_DICT.keys()}')
    parser.add_argument('--epoch', type=int, default=100, help='specify epochs number')
    parser.add_argument('-s', '--sequence_length', type=int, default=27, help='select train data sequence length')
    parser.add_argument('-p', '--pred_future_time', type=int, default=33, help='How many seconds later would you like to predict?')
    parser.add_argument("--is_output_unit", type=str, default="false", help='select output format from unit vector or normal vector(including distance)')
    parser.add_argument("--is_train_smp2foot", type=str, default="true", help='select training Position2Position or smpPosition2footPosition')
    parser.add_argument('--input_shift', type=int, default=1, help='specify input (src, tgt) shift size for transformer_encdec.')
    # parser.add_argument('-t', '--trial_num', type=int, default=30, help='select optuna trial number')
    args = parser.parse_args()

    if args.model == "transformer_encdec":
        hidden_size = 10
        num_layers = 5
        nhead = trial.suggest_categorical('nhead', [1, 2, 3, 6])
    elif args.model == "lstm":
        hidden_size = trial.suggest_int('hidden_size', 5, 100)
        num_layers = trial.suggest_int('num_layers', 1, 10)
        nhead = 3
    else:
        hidden_size = trial.suggest_int('hidden_size', 5, 100)
        num_layers = trial.suggest_int('num_layers', 1, 10)
        nhead = trial.suggest_categorical('nhead', [1, 2, 3, 6])
    # batch_size = trial.suggest_int('batch_size', 4, 32)
    batch_size = 8
    sequence_length = args.sequence_length # 30 # x means this model use previous time for 0.01*x seconds 
    pred_future_time = args.pred_future_time # 40 # x means this model predict 0.01*x seconds later
    output_dim = 3 # 進行方向ベクトルの要素数
    lr = trial.suggest_float('lr', 0.0001, 0.1, log=True)
    img_save_flag = False#fixed

    train_x_df, train_t_df = dataloader(train_data_path, selected_train_columns, selected_correct_columns)
    test_x_df, test_t_df = dataloader(test_data_path, selected_train_columns, selected_correct_columns)
    train_frame_num = len(train_x_df)
    test_frame_num = len(test_x_df)

    # 基本
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = choose_model(args.model, len(selected_train_columns), hidden_size, num_layers, nhead, output_dim, sequence_length, args.input_shift)
    model = model.float()
    model.to(device)
    criterion = nn.L1Loss()#nn.MSELoss()
    optimizer = SGD(model.parameters(), lr=lr)
    # scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=8, eta_min=lr*0.01, last_epoch=- 1, verbose=True)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 40], gamma=0.5) # 10 < 20 < 40
    old_test_accurancy=0
    TrainAngleErrResult = []
    TestAngleErrResult = []
    TrainDistanceErrResult = []
    TestDistanceErrResult = []
    TrainLossResult = []
    TestLossResult = []
    train_mini_data_num = int(train_frame_num/Non_duplicate_length)
    test_mini_data_num = int(test_frame_num/Non_duplicate_length)
    BestMAE = 360
    BestMDE = 100000
    log.sequence_length = args.sequence_length
    log.pred_future_time = args.pred_future_time
    log.model = args.model
    log.max_epoch = args.epoch
    log.is_output_unit = args.is_output_unit
    log.lr = round(lr, 6)
    log.trial_num = trial.number
    log.batch_size = batch_size
    log.num_layers = num_layers
    log.hidden_size = hidden_size
    log.nhead = nhead

    # イテレーション数計算
    train_use_data_num = train_mini_data_num - (sequence_length+pred_future_time)//Non_duplicate_length - 2
    test_use_data_num = test_mini_data_num - (sequence_length+pred_future_time)//Non_duplicate_length - 2
    train_iter_num = train_use_data_num//batch_size
    test_iter_num = test_use_data_num//batch_size
    print(
    f"max_epoch:{args.epoch}, model_name:{args.model}, batch_size:{batch_size}, hidden_size:{hidden_size}, "\
    f"num_layers:{num_layers}, sequence_length:{sequence_length}, pred_future_time:"\
    f"{pred_future_time}, nhead:{nhead}"
    )
    for epoch in range(args.epoch):
        print("\nstart", epoch, "epoch")
        log.epoch = epoch
        running_loss = 0.0
        angleErrSum = 0
        distanceErrSum = 0

        Non_duplicate_length_offset = np.random.randint(0, Non_duplicate_length)
        train_random_num_list = random.sample(range(1, train_use_data_num + 1),
                                              k=train_use_data_num)
        test_random_num_list = random.sample(range(1, test_use_data_num + 1),
                                             k=test_use_data_num)

        # iteration loop
        model.train()
        for i in tqdm(range(train_iter_num)):
            optimizer.zero_grad()
            # make mini batch random list
            mini_batch_train_random_list =[]
            for _ in range(batch_size):
                mini_batch_train_random_list.append(train_random_num_list.pop())

            data, label = MakeBatch(train_x_df, train_t_df, batch_size, sequence_length, selected_train_columns,
                                    selected_correct_columns, mini_batch_train_random_list, pred_future_time,
                                    args.is_output_unit, Non_duplicate_length, Non_duplicate_length_offset,
                                    args.is_train_smp2foot)

            data = data.squeeze()  
            # print("data.shape", data.shape)#torch.Size([30, 32, 6])sequence, batch size, feature num # (T, N, E)
            # print(label.shape)#torch.Size([30, 32, 3])sequence, batch size, feature num # (T, N, E)
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

            angleErr, distanceErr = CalcAngleErr(output, label, batch_size) # decimal.Decimal(CalcAngleErr(output, label, batch_size))
            angleErrSum += angleErr
            distanceErrSum += distanceErr
            
            loss = criterion(output.float().to(device), label.float().to(device))
            loss.backward()
            optimizer.step()

            running_loss += loss.data # ??????????????? todo : loss.dataとlossの違いは？
        scheduler.step()
        TrainLossResult.append(loss.cpu().detach().numpy())

        ## 絶対平均誤差と平均距離誤差を算出 ##
        MAE_tr = angleErrSum/(train_iter_num-1)
        MDE_tr = distanceErrSum/(train_iter_num-1)
        TrainAngleErrResult.append(MAE_tr)
        TrainDistanceErrResult.append(MDE_tr)
        tqdm.write(f"train mean angle and distance error = {MAE_tr}[度], {MDE_tr}[m]")

        # test
        model.eval()
        TestAngleErrSum = 0
        TestDistanceErrSum = 0
        TestLossSum = 0
        mini_batch_test_random_list =[]
        for _ in range(batch_size):
            mini_batch_test_random_list.append(test_random_num_list.pop())

        for i in tqdm(range(test_iter_num)):
            data, label = MakeBatch(test_x_df, test_t_df, batch_size, sequence_length, selected_train_columns, selected_correct_columns,
                                    mini_batch_test_random_list, pred_future_time, args.is_output_unit, Non_duplicate_length,
                                    Non_duplicate_length_offset, args.is_train_smp2foot)
            data.to(device)
            label.to(device)

            if args.model == "transformer_encdec":
                # src = data[:30, :, :]
                # tgt = data[1:, :, :]
                src = data[:sequence_length-shift, :, :]
                tgt = data[shift:, :, :]
                output = model(src=src.float().to(device), tgt=tgt.float().to(device))
                # output = output.contiguous().view(-1, output.size(-1))
                # print("output.shape", output.shape)
            elif args.model == "lstm" or args.model == "transformer_enc" or args.model == "imu_transformer":
                output = model(data.float().to(device))
                # print("output.shape", output.shape)
            else:
                print("specify light model name")

            angleErr, distanceErr = CalcAngleErr(output, label, batch_size)# decimal.Decimal(CalcAngleErr(output, label, batch_size))
            loss = criterion(output.float().to(device), label.float().to(device))
            TestAngleErrSum += angleErr
            TestDistanceErrSum += distanceErr
            TestLossSum += float(loss)
            
        TestLossResult.append(loss.cpu().detach().numpy())
        MAE_te = TestAngleErrSum/test_iter_num
        MDE_te = TestDistanceErrSum/test_iter_num
        MTL_te = TestLossSum/test_iter_num
        TestAngleErrResult.append(MAE_te)
        TestDistanceErrResult.append(MDE_te)
        log.LastEpochResultLog(epoch, args.epoch, MAE_te, MDE_te, MTL_te) 
        tqdm.write(f"Test mean angle and distance error = {MAE_te}, {MDE_te}")
        BestMAE = min(BestMAE, MAE_te)
        BestMDE = min(BestMDE, MDE_te)
        if BestMAE == MAE_te:
            img_save_flag = True
        tqdm.write(f"Best mean absolute test error, mean distance error = {BestMAE}, {BestMDE}")

    # graph plotting
    fig = plt.figure(figsize = [9.0, 6.0])# [横幅, 高さ]
    ax1 = fig.add_subplot(2, 3, 1)
    ax2 = fig.add_subplot(2, 3, 2)
    ax3 = fig.add_subplot(2, 3, 3)
    ax4 = fig.add_subplot(2, 3, 4)
    ax5 = fig.add_subplot(2, 3, 5)
    ax6 = fig.add_subplot(2, 3, 6)
    ax1.plot(TrainAngleErrResult)
    ax2.plot(TrainDistanceErrResult)
    ax3.plot(TrainLossResult)
    ax4.plot(TestAngleErrResult)
    ax5.plot(TestDistanceErrResult)
    ax6.plot(TestLossResult)
    ax1.set_title("train angle error")
    ax2.set_title("train distance error")
    ax3.set_title("train loss")
    ax4.set_xlabel("test angle error")
    ax5.set_xlabel("test distance error")
    ax6.set_xlabel("test loss")
    # ax1.set_ylim(0, 50)
    # ax2.set_ylim(0, 0.6)
    # ax3.set_ylim(0, 0.5)
    # ax4.set_ylim(0, 50)
    # ax5.set_ylim(0, 0.6)
    # ax6.set_ylim(0, 0.5)

    log.train_filename = os.path.splitext(os.path.basename(train_data_path))[0]
    log.test_filename = os.path.splitext(os.path.basename(test_data_path))[0]
    log.BestMAE = str(f"{BestMAE:.02f}")
    
    StartTime = log.TrainStartTime
    
    dir_name = f"{StartTime}_{args.model}_seq{sequence_length}_pred{pred_future_time}"
    dir_name = dir_name.replace(".", "").replace(" ", "").replace("-", "")
    dir_path = join(img_save_path, dir_name)
    os.makedirs(dir_path, exist_ok=True)

    train_info_log_path = join(dir_path, "train_info_log.txt")
    log.train_info_dir_path = dir_path
    log.train_info_log(train_info_log_path)
    log.LastEpochResultsShow()
    log.LastEpochResultSave(dir_path)

    # trialごとにweight fileをsaveする
    rounded_mae = round(MAE_te, 5)
    rouded_mde = round(MDE_te, 5)
    weight_file_name = f"trial{trial.number}_MAE{rounded_mae}_MDE{rouded_mde}_lr{lr:.06f}_batch_size_{batch_size}_num_layers{num_layers}_hiddensize{hidden_size}_seq{sequence_length}_pred{pred_future_time}.pth"
    weight_save_path = join(img_save_path, dir_name, weight_file_name)
    torch.save(model.state_dict(), weight_save_path)

    # trialごとに推論結果のグラフを保存
    if img_save_flag == True:
        file_name = f"err_{BestMAE:.02f}_trial{trial.number}.png"
        fig.savefig(join(img_save_path, dir_name, file_name))
    print(f"model name : {args.model}, batch size : {batch_size}, hidden_size : {hidden_size}, num_layers : {num_layers}, sequence_length : {sequence_length}, pred_future_time : {pred_future_time}")
    print("finished objective")
    return MAE_te


if __name__ == '__main__':
    cwd = os.getcwd()
    print("now directory is", cwd)
    log = Log()
    # analysis = AnalysisLastEpochAverage()
    TRIAL_NUM = 25
    study = optuna.create_study()
    study.optimize(main, n_trials=TRIAL_NUM)
    log.AverageLastEpochResultSave()
