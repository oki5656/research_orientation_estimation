from cmath import isnan
import torch
import torch.nn as nn
import pandas as pd
from torch.optim import SGD
from matplotlib import pyplot as plt
import os
import math
import time
import numpy as np
from tqdm import tqdm
import decimal

#############################################  config  ##################################################
weight_path = os.path.join("..","weights")
train_data_path = os.path.join("..","datasets", "dataset-room1_512_16", "mav0", "self_made_files", "all_in_imu_mocap.csv")
test_data_path = os.path.join("..","datasets", "dataset-room2_512_16", "mav0", "self_made_files", "all_in_imu_mocap.csv")
selected_train_columns = ['gyroX', 'gyroY', 'gyroZ', 'accX', 'accY', 'accZ']
selected_correct_columns = ['pX', 'pY', 'pZ', 'qW', 'qX', 'qY', 'qZ']
#########################################################################################################


def mkDataSet(data_size, data_length=50, freq=60., noise=0.02):
    """
    params
      data_size : データセットサイズ
      data_length : 各データの時系列長
      freq : 周波数
      noise : ノイズの振幅
    returns
      train_x : トレーニングデータ（t=1,2,...,size-1の値)
      train_t : トレーニングデータのラベル（t=sizeの値）
    """
    train_x = []
    train_t = []

    for offset in range(data_size):
        train_x.append([[math.sin(2 * math.pi * (offset + i) / freq) + np.random.normal(loc=0.0, scale=noise)] for i in range(data_length)])
        train_t.append([math.sin(2 * math.pi * (offset + data_length) / freq)])

    return train_x, train_t


def dataloader(path, train_columns, correct_columns):
    df = pd.read_csv(path)
    train_x_df = df[train_columns]
    train_t_df = df[correct_columns]

    return train_x_df, train_t_df


def CalcAngle(NowPosi, EstPosi, CorrPosi):
    L1, L2, D, cos_theta, theta = 0, 0, 0, 0, 0
    # print('Now position : ',NowPosi)
    # print('Est position : ',EstPosi)
    # print('Cor position : ',CorrPosi)
    L1 = math.sqrt((NowPosi[0] - EstPosi[0])**2 + (NowPosi[1] - EstPosi[1])**2 + (NowPosi[2] - EstPosi[2])**2)
    L2 = math.sqrt((NowPosi[0] - CorrPosi[0])**2 + (NowPosi[1] - CorrPosi[1])**2 + (NowPosi[2] - CorrPosi[2])**2)
    D = math.sqrt((EstPosi[0] - CorrPosi[0])**2 + (EstPosi[1] - CorrPosi[1])**2 + (EstPosi[2] - CorrPosi[2])**2)
    cos_theta = (L1**2 + L2**2 - D**2 + 0.00001)/(2*L1*L2 + 0.00000000001)
    theta = math.acos(np.clip(cos_theta, -1.0, 1.0))# thetaはラジアン
    # print('result',L1, L2, D)
    # print('\n')
    return theta


def CalcAngleErr(output, label, batch_size):
    angleErrSum = 0.0
    for i in range(batch_size):
        angleErrSum += CalcAngle(label[0, i, :], output[0, i, :], label[0, i+1, :])

    return angleErrSum


def mkDataSetFromCsv(csvPath, data_size, data_length=50, freq=60., noise=0.02):
    """
    params
      csvPath : csvファイルのpath
      data_size : データセットサイズ
      data_length : 各データの時系列長
      freq : 周波数
    returns
      train_x : トレーニングデータ（t=1,2,...,size-1の値)
      train_t : トレーニングデータのラベル（t=sizeの値）
    """
    df = pd.read_csv(csvPath, names=('day', 'min', 'start', 'high', 'low', 'close', 'value'))
    start=df["start"]

    train_x = []
    train_t = []

    for offset in range(data_size):
        train_x.append(start[offset:offset+data_length])
        train_t.append(start[offset+data_length])

    return train_x, train_t


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


def MakeBatch(train_x_df, train_t_df, batch_size, selected_train_columns, selected_correct_columns):
    """
    train_x, train_tを受け取ってbatch_x, batch_tを返す。
    """
    # batch_x = []
    # batch_t = []
    batch_x_df = pd.DataFrame(columns=selected_train_columns)
    batch_t_df = pd.DataFrame(columns=selected_correct_columns)
    idx = np.random.randint(3, len(train_x_df) - batch_size - 10)
    for j in range(batch_size):
        batch_x_df = batch_x_df.append(train_x_df[idx + j: idx + j + 1], ignore_index=True)
        batch_t_df = batch_t_df.append(train_t_df[idx + j: idx + j + 1], ignore_index=True)
    
    # lossを求める際に未来の情報が必要なので１個プラスでappend
    batch_t_df = batch_t_df.append(train_t_df[idx + j: idx + j + 1], ignore_index=True)

    # numpy形式に変換
    batch_x_df_np = batch_x_df.to_numpy()[np.newaxis, :, :].astype(np.float32)
    batch_t_df_np = batch_t_df.to_numpy()[np.newaxis, :, :].astype(np.float32)

    return torch.tensor(batch_x_df_np), torch.tensor(batch_t_df_np)


def main():
    training_size = 400
    test_size = 100
    epochs_num = 10
    hidden_size = 9
    batch_size = 8
    # print("len(selected_train_columns) = ", len(selected_train_columns))

    # train_x, train_t = mkDataSet(training_size)
    # test_x, test_t = mkDataSet(test_size)
    # print(np.array(train_x).shape)

    train_x_df, train_t_df = dataloader(train_data_path, selected_train_columns, selected_correct_columns)
    test_x_df, test_t_df = dataloader(test_data_path, selected_train_columns, selected_correct_columns)
    # all_row_num = train_t_df.shape[0]

    model = Predictor(len(selected_train_columns), hidden_size, len(selected_correct_columns))
    model = model.float()
    criterion = nn.MSELoss()
    optimizer = SGD(model.parameters(), lr=0.01)
    old_test_accurancy=0
    decimal.getcontext().prec = 4
    for epoch in range(epochs_num):
        print("start", epoch, "epoch")
        running_loss = 0.0
        # training_accuracy = 0.0
        angleErrSum = 0
        for i in range(int(training_size / batch_size)):
            optimizer.zero_grad()

            data, label = MakeBatch(train_x_df, train_t_df, batch_size, selected_train_columns, selected_correct_columns)
            # print(label)
            #print("data.shape, label.shape = ", data.shape, label.shape)
            data = data.double()
            output = model(data.float())
            # print(output)
            if np.isnan(output.detach().numpy()).any():
                print('break')
                break
            # print("############# output.shape = ", output.shape)#長さ7のベクトル
            # print("############# label.shape = ", label.shape)#12 × 7のベクトル
            # time.sleep(2)
            # print("############# output = ", output)
            #print("output.float(), label.float() : ", label.float())#, label.float())
            angleErrSum += decimal.Decimal(CalcAngleErr(output, label, batch_size))
            
            # print(angleErrSum, end=" ")
            # print("###############angleErrSum = ", angleErrSum)
            loss = criterion(output.float(), label[:, 1:, :].float())
            # tqdm.write(str(label))# + str(loss) + str(label))
            loss.backward()
            optimizer.step()

            running_loss += loss.data
            
        ## 絶対平均誤差を算出 ##
        # MAE = 0.0
        MAE = angleErrSum/int(training_size / batch_size)
        # print(angleErrSum)
        tqdm.write(("mean angle error = "+ str(MAE)))

        # tqdm.write('end of epoch !!%d loss: %.3f, training_accuracy: %.5f' % (epoch + 1, running_loss, training_accuracy))

        #test
        test_accuracy = 0.0
        for i in range(int(test_size / batch_size)):
            offset = i * batch_size
            # data, label = torch.tensor(test_x[offset:offset+batch_size]), torch.tensor(test_t[offset:offset+batch_size])
            data, label = MakeBatch(test_x_df, train_t_df, batch_size, selected_train_columns, selected_correct_columns)
            output = model(data.float(), None)

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
    # print(weight_path)
    if not os.path.isdir(weight_path):
        os.mkdir(weight_path)
    main()