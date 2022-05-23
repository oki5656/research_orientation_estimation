# このファイルではデータセットを用いLSTMで学習を行う
########################################################################
import os
import sys
a=os.path.dirname(sys.executable)
print(os.path.dirname(sys.executable))
########################################################################

from cmath import isnan
import torch
import torch.nn as nn
import pandas as pd
from torch.optim import SGD, lr_scheduler
from matplotlib import pyplot as plt
import math
import random
import numpy as np
from tqdm import tqdm
import decimal
import optuna

#############################################  config  ##################################################
weight_path = os.path.join("..","weights")
img_save_path = os.path.join("..", "images")
train_data_path = os.path.join("..","datasets", "dataset-room_all", "mav0", "self_made_files", "new_all_in_imu_mocap_13456.csv")
test_data_path = os.path.join("..","datasets", "dataset-room2_512_16", "mav0", "self_made_files", "new_all_in_imu_mocap.csv")
selected_train_columns = ['gyroX', 'gyroY', 'gyroZ', 'accX', 'accY', 'accZ']
selected_correct_columns = ['pX', 'pY', 'pZ', 'qW', 'qX', 'qY', 'qZ']
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
    cos_theta = (L1**2 + L2**2 - D**2 + 0.000000000000001-0.000000000000001)/(2*L1*L2 + 0.000000000000001-0.000000000000001)
    theta_rad = math.acos(np.clip(cos_theta, -1.0, 1.0))# thetaはラジアン
    theta_deg = math.degrees(theta_rad)# ラジアンからdegreeに変換

    return theta_deg


def CalcAngleErr(output, label, batch_size):
    angleErrSum = 0.0
    for i in range(batch_size):
        angleErrSum += CalcAngle(label[0, i, :], output[0, i, :], label[0, i, :])

    return angleErrSum/batch_size


def MakeBatch(train_x_df, train_t_df, batch_size, sequence_length, selected_train_columns, selected_correct_columns, mini_batch_random_list, pred_future_time):
    """
    train_x, train_tを受け取ってbatch_x_df_np(sequence_length, batch_size, input_size)とdir_vec(sequence_length, batch_size, input_size)を返す
    """
    batch_x_df = pd.DataFrame(columns=selected_train_columns)
    batch_t_df = pd.DataFrame(columns=selected_correct_columns)
    #####################################################################################################################
    # idx = np.random.randint(3, len(train_x_df) - batch_size - 10)
    # for j in range(batch_size):
    #     # batch_x_df = batch_x_df.append(train_x_df[idx + j: idx + j + 1], ignore_index=True)
    #     # batch_t_df = batch_t_df.append(train_t_df[idx + j: idx + j + 1], ignore_index=True)
    #     batch_x_df = pd.concat([batch_x_df, train_x_df[idx + j: idx + j + 1]])
    #     batch_t_df = pd.concat([batch_t_df, train_t_df[idx + j: idx + j + 1]])
    # # lossを求める際に未来の情報が必要なので１個プラスでappend
    # # batch_t_df = batch_t_df.append(train_t_df[idx + j + 1: idx + j + 2], ignore_index=True)
    # batch_t_df= pd.concat([batch_t_df, train_t_df[idx + j + 1: idx + j + 2]])
    #####################################################################################################################
    out_x = list()
    out_t = list()
    batch_length = len(mini_batch_random_list)
    if batch_length == 8:
        batch_size = 8
    else:
        batch_size=len(mini_batch_random_list)

    for i in range(batch_size):
        # idx = np.random.randint(3, len(train_x_df) - batch_size - sequence_length -10 )
        idx = mini_batch_random_list[i]*sequence_length
        out_x.append(np.array(train_x_df[idx : idx + sequence_length]))
        out_t.append(np.array(train_t_df[idx : idx + sequence_length + pred_future_time]))
    out_x = np.array(out_x)
    out_t = np.array(out_t)
    # print("out_x.shape", out_x.shape)
    # print("out_t.shape", out_t.shape)
    batch_x_df_np = out_x.transpose(1, 0, 2)
    # print("out_t.shape", out_t.shape)
    batch_t_df_np = out_t.transpose(1, 0, 2)

    #####################################################################################################################

    # numpy形式に変換
    # batch_x_df_np = batch_x_df.to_numpy()[np.newaxis, :, :].astype(np.float32)#'gyroX', 'gyroY', 'gyroZ', 'accX', 'accY', 'accZ'
    # batch_t_df_np = batch_t_df.to_numpy()[np.newaxis, :, :].astype(np.float32)#'pX', 'pY', 'pZ', 'qW', 'qX', 'qY', 'qZ'

    # スマホ座標系に変換
    dir_vec = TransWithQuat(batch_t_df_np, batch_size, sequence_length)
    # print("batch_x_df_np.shape", batch_x_df_np.shape)(5, 8, 6)
    # print("dir_vec.shape", dir_vec.shape)(5, 8, 3)
    return torch.tensor(batch_x_df_np), torch.tensor(dir_vec)


def TransWithQuat(batch_t_df_np, batch_size, sequence_length):
    dir_vec = np.ones((sequence_length, batch_size, 3))
    ####################################################################################################################################################################################
    # for i in range(batch_size):
    #     qW, qX, qY, qZ = batch_t_df_np[0][i][3], batch_t_df_np[0][i][4], batch_t_df_np[0][i][5], batch_t_df_np[0][i][6]
    #     E = np.array([[qX**2 - qY**2 - qZ**2 + qW**2, 2*(qX*qY - qZ*qW), 2*(qX*qZ + qY*qW)],
    #             [2*(qX*qY + qZ*qW), -qX**2 + qY**2 - qZ**2 + qW**2, 2*(qY*qZ - qX*qW)],
    #             [2*(qX*qZ - qY*qW), 2*(qY*qZ + qX*qW), -qX**2 - qY**2 + qZ**2 + qW**2]])#クォータニオン表現による回転行列
    #     old_dir = np.array([batch_t_df_np[0][i+1][0] - batch_t_df_np[0][i][0], batch_t_df_np[0][i+1][1] - batch_t_df_np[0][i][1], batch_t_df_np[0][i+1][2] - batch_t_df_np[0][i][2]])#２点の位置から進行方向ベクトルを求めた
    #     new_dir = np.matmul(E, old_dir.T)
    #     dir_vec[0][i][0], dir_vec[0][i][1], dir_vec[0][i][2] = new_dir[0], new_dir[1], new_dir[2]
    # return dir_vec
    ####################################################################################################################################################################################
    for i in range(sequence_length):
        for j in range(batch_size):
            qW, qX, qY, qZ = batch_t_df_np[i][j][3], batch_t_df_np[i][j][4], batch_t_df_np[i][j][5], batch_t_df_np[i][j][6]
            E = np.array([[qX**2 - qY**2 - qZ**2 + qW**2, 2*(qX*qY - qZ*qW), 2*(qX*qZ + qY*qW)],
                    [2*(qX*qY + qZ*qW), -qX**2 + qY**2 - qZ**2 + qW**2, 2*(qY*qZ - qX*qW)],
                    [2*(qX*qZ - qY*qW), 2*(qY*qZ + qX*qW), -qX**2 - qY**2 + qZ**2 + qW**2]])#クォータニオン表現による回転行列
            old_dir = np.array([batch_t_df_np[i+1][j][0] - batch_t_df_np[i][j][0], batch_t_df_np[i+1][j][1] - batch_t_df_np[i][j][1], batch_t_df_np[i+1][j][2] - batch_t_df_np[i][j][2]])#２点の位置から進行方向ベクトルを求めた
            new_dir = np.matmul(E.T, old_dir.T)##############################################################  転地するかも
            dir_vec[i][j][0], dir_vec[i][j][1], dir_vec[i][j][2] = new_dir[0], new_dir[1], new_dir[2]
    # print("dir_vec.shape", dir_vec.shape)
    return dir_vec
    ####################################################################################################################################################################################

class Predictor(nn.Module):
    def __init__(self, inputDim, hiddenDim, num_layers, outputDim):
        super(Predictor, self).__init__()

        self.rnn = nn.LSTM(input_size = inputDim,
                            hidden_size = hiddenDim,
                            num_layers = num_layers,
                            batch_first = False)
        self.output_layer = nn.Linear(hiddenDim, outputDim)
    
    def forward(self, inputs, hidden0=None):
        output, (hidden, cell) = self.rnn(inputs, hidden0) #LSTM層
        # output = self.output_layer(output[:, -1, :]) #全結合層
        output = self.output_layer(output)

        return output


def objective(trial):
    print("start objective")
    hidden_size = trial.suggest_int('hidden_size', 5, 100)
    # hidden_size = 50
    num_layers = trial.suggest_int('num_layers', 1, 10)
    batch_size = trial.suggest_int('batch_size', 4, 32)
    # batch_size = 8
    epochs_num = 100
    sequence_length = 10 # x means this model use previous time for 0.01*x seconds 
    pred_future_time = 10 # x means this model predict 0.01*x seconds later
    output_dim = 3#進行方向ベクトル
    lr = trial.suggest_float('lr', 0.0001, 0.1, log=True)
    # lr = 0.0005
    img_save_flag = False

    train_x_df, train_t_df = dataloader(train_data_path, selected_train_columns, selected_correct_columns)
    test_x_df, test_t_df = dataloader(test_data_path, selected_train_columns, selected_correct_columns)
    train_data_num = len(train_x_df)
    test_data_num = len(test_x_df)

    # 基本
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = Predictor(len(selected_train_columns), hidden_size, num_layers, output_dim)
    model = model.float()
    model.to(device)
    criterion = nn.L1Loss()#nn.MSELoss()
    optimizer = SGD(model.parameters(), lr=lr)
    # scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=8, eta_min=lr*0.01, last_epoch=- 1, verbose=True)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 40], gamma=0.5) # 10 < 20 < 40
    old_test_accurancy=0
    TrainAngleErrResult = []
    TestAngleErrResult = []
    TrainLossResult = []
    TestLossResult = []
    BestMAE = 360

    # イテレーション数計算
    train_iter_num = int(train_data_num/(sequence_length*batch_size))
    test_iter_num = int(test_data_num/(sequence_length*batch_size))

    for epoch in range(epochs_num):
        print("\nstart", epoch, "epoch")
        running_loss = 0.0
        # training_accuracy = 0.0
        angleErrSum = 0
        train_mini_data_num = int(train_data_num/sequence_length)
        test_mini_data_num = int(test_data_num/sequence_length)
        train_random_num_list = random.sample(range(1, train_mini_data_num), k=train_mini_data_num-1)
        test_random_num_list = random.sample(range(1, test_mini_data_num), k=test_mini_data_num-1)

        # iteration loop
        for i in tqdm(range(train_iter_num-1)):
            optimizer.zero_grad()
            # make mini batch random list
            mini_batch_train_random_list =[]
            for _ in range(batch_size):
                mini_batch_train_random_list.append(train_random_num_list.pop())

            data, label = MakeBatch(train_x_df, train_t_df, batch_size, sequence_length, selected_train_columns, selected_correct_columns, mini_batch_train_random_list, pred_future_time)
            output = model(data.float().to(device))
            if np.isnan(output.cpu().detach().numpy()).any():
                print('Nan was found. So system break.')
                sys.exit()
            # print("############# output.shape = ", output.shape)#torch.Size([1, 8, 7])
            # print("############# label.shape = ", label.shape)#torch.Size([1, 8, 3])
            angleErrSum += decimal.Decimal(CalcAngleErr(output, label, batch_size))
            
            loss = criterion(output.float().to(device), label.float().to(device))
            loss.backward()
            optimizer.step()

            running_loss += loss.data
        scheduler.step()
        TrainLossResult.append(loss.cpu().detach().numpy())

        ## 絶対平均誤差を算出 ##
        MAE_tr = angleErrSum/train_iter_num
        TrainAngleErrResult.append(MAE_tr)
        # print(angleErrSum)
        tqdm.write(("train mean angle error = "+ str(MAE_tr)))

        # test
        TestAngleErrSum = 0
        mini_batch_test_random_list =[]
        for _ in range(batch_size):
            mini_batch_test_random_list.append(test_random_num_list.pop())

        for i in tqdm(range(test_iter_num)):
            data, label = MakeBatch(test_x_df, test_t_df, batch_size, sequence_length, selected_train_columns, selected_correct_columns, mini_batch_test_random_list, pred_future_time)
            data.to(device)
            label.to(device)
            output = model(data.float().to(device), None)
            TestAngleErrSum += decimal.Decimal(CalcAngleErr(output, label, batch_size))
            loss = criterion(output.float().to(device), label.float().to(device))
        TestLossResult.append(loss.cpu().detach().numpy())
        MAE_te = TestAngleErrSum/test_iter_num
        TestAngleErrResult.append(MAE_te)
        tqdm.write(("Test mean angle error = "+ str(MAE_te)))
        BestMAE = min(BestMAE, MAE_te)
        if BestMAE == MAE_te:
            img_save_flag = True
        tqdm.write("Best mean absolute test error = "+ str(BestMAE))

    # graph plotting
    fig = plt.figure(figsize = [5.8, 4])
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)
    ax1.plot(TrainAngleErrResult)
    ax2.plot(TrainLossResult)
    ax3.plot(TestAngleErrResult)
    ax4.plot(TestLossResult)
    ax1.set_title("train angle error")
    ax2.set_title("train loss")
    ax3.set_xlabel("test angle error")
    ax4.set_xlabel("test loss")
    # plt.show()
    # 今までで一番精度がいい時に推論結果を保存
    if img_save_flag == True:
        fig.savefig(os.path.join(img_save_path, f"err_{BestMAE:.02f}_lr_{lr:.06f}_batch_size_{batch_size}_num_layers_{num_layers}_hidden_size{hidden_size}_seq_length{sequence_length}_pred_future_time{pred_future_time}.png"))
    print("finished objective")
    return MAE_te


if __name__ == '__main__':
    if not os.path.isdir(weight_path):
        os.mkdir(weight_path)
    # main()
    TRIAL_SIZE = 50
    study = optuna.create_study()
    study.optimize(objective, n_trials=TRIAL_SIZE)