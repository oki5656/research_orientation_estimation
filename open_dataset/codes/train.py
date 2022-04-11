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
from torch.optim import SGD
from matplotlib import pyplot as plt
import os
import sys
import math
import time
import numpy as np
from tqdm import tqdm
import decimal

#############################################  config  ##################################################
weight_path = os.path.join("..","weights")
train_data_path = os.path.join("..","datasets", "dataset-room1_512_16", "mav0", "self_made_files", "new_all_in_imu_mocap.csv")
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


def MakeBatch(train_x_df, train_t_df, batch_size, selected_train_columns, selected_correct_columns):
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
    sequence_length = 20
    for _ in range(batch_size):
        idx = np.random.randint(3, len(train_x_df) - batch_size - sequence_length -10 )
        out_x.append(np.array(train_x_df[idx : idx + sequence_length]))
        out_t.append(np.array(train_t_df[idx : idx + sequence_length + 1]))
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
            new_dir = np.matmul(E, old_dir.T)##############################################################  転地するかも
            dir_vec[i][j][0], dir_vec[i][j][1], dir_vec[i][j][2] = new_dir[0], new_dir[1], new_dir[2]
    # print("dir_vec.shape", dir_vec.shape)
    return dir_vec
    ####################################################################################################################################################################################

class Predictor(nn.Module):
    def __init__(self, inputDim, hiddenDim, outputDim):
        super(Predictor, self).__init__()

        self.rnn = nn.LSTM(input_size = inputDim,
                            hidden_size = hiddenDim,
                            batch_first = False)
        self.output_layer = nn.Linear(hiddenDim, outputDim)
    
    def forward(self, inputs, hidden0=None):
        output, (hidden, cell) = self.rnn(inputs, hidden0) #LSTM層
        # output = self.output_layer(output[:, -1, :]) #全結合層
        output = self.output_layer(output)

        return output


def main():
    training_size = 10000
    test_size = 200
    epochs_num = 50
    hidden_size = 50
    batch_size = 8
    output_dim = 3#進行方向ベクトル

    train_x_df, train_t_df = dataloader(train_data_path, selected_train_columns, selected_correct_columns)
    test_x_df, test_t_df = dataloader(test_data_path, selected_train_columns, selected_correct_columns)
    print()

    # 基本
    model = Predictor(len(selected_train_columns), hidden_size, output_dim)
    model = model.float()
    criterion = nn.MSELoss()
    optimizer = SGD(model.parameters(), lr=0.01)
    old_test_accurancy=0

    for epoch in range(epochs_num):
        print("start", epoch, "epoch")
        running_loss = 0.0
        # training_accuracy = 0.0
        angleErrSum = 0
        for i in tqdm(range(int(training_size/batch_size))):##############################################ここが微妙
            optimizer.zero_grad()

            data, label = MakeBatch(train_x_df, train_t_df, batch_size, selected_train_columns, selected_correct_columns)
            output = model(data.float())
            if np.isnan(output.detach().numpy()).any():
                print('Nan was found. So system break.')
                sys.exit()
                # break
            # print("############# output.shape = ", output.shape)#torch.Size([1, 8, 7])
            # print("############# label.shape = ", label.shape)#torch.Size([1, 8, 3])
            angleErrSum += decimal.Decimal(CalcAngleErr(output, label, batch_size))
            
            loss = criterion(output.float(), label.float())
            # tqdm.write(str(label))# + str(loss) + str(label))
            loss.backward()
            optimizer.step()

            running_loss += loss.data
            
        ## 絶対平均誤差を算出 ##
        # MAE = 0.0
        MAE_tr = angleErrSum/int(training_size/batch_size)
        # print(angleErrSum)
        tqdm.write(("train mean angle error = "+ str(MAE_tr)))
        # tqdm.write('end of epoch !!%d loss: %.3f, training_accuracy: %.5f' % (epoch + 1, running_loss, training_accuracy))

        #test
        test_accuracy = 0.0
        angleErrSum = 0
        for i in range(int(test_size / batch_size)):
            offset = i * batch_size
            # data, label = torch.tensor(test_x[offset:offset+batch_size]), torch.tensor(test_t[offset:offset+batch_size])
            data, label = MakeBatch(test_x_df, test_t_df, batch_size, selected_train_columns, selected_correct_columns)
            output = model(data.float(), None)
            angleErrSum += decimal.Decimal(CalcAngleErr(output, label, batch_size))
            loss = criterion(output.float(), label.float())

        MAE_te = angleErrSum/int(test_size/batch_size)
        tqdm.write(("test mean angle error = "+ str(MAE_te)))
            # test_accuracy += np.sum(np.abs((output.data - label.data).numpy()) < 0.1)
        
        # training_accuracy /= training_size
        # test_accuracy /= test_size
        # if test_accuracy>old_test_accurancy:
        #     model_weight_path=os.path.join(weight_path,"best_acc_weight.pt")
        #     torch.save(model.state_dict(), model_weight_path)
        # old_test_accurancy=test_accuracy

        # tqdm.write('%d loss: %.3f, training_accuracy: %.5f, test_accuracy: %.5f' % (
        #     epoch + 1, running_loss, training_accuracy, test_accuracy))


if __name__ == '__main__':
    if not os.path.isdir(weight_path):
        os.mkdir(weight_path)
    main()