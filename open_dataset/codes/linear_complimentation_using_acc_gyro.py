# このファイルではTUMのデータセットを用い、時系列データ学習用にcsvファイルの整形を行う。
# 具体的にはIMUのファイルを200Hz→100Hzにダウンサンプリングする処理(down_sampling.py)ののち、mocapのファイルを120Hz→100Hzへダウンサンプリングすると同時にIMUのタイムスタンプを参考に線形補完を行う
import os
from os import path
import numpy as np
import pandas as pd
from tqdm import tqdm

####################################### path setting #################################################
imu_path = path.join("..", "datasets", "dataset-room3_512_16", "mav0", "imu0", "data_imu.csv")
new_imu_path = path.join("..", "datasets", "dataset-room3_512_16", "mav0", "imu0", "new_data_imu.csv")
mocap_path = path.join("..", "datasets", "dataset-room3_512_16", "mav0", "mocap0", "data_mocap.csv")
# out_new_imu_path = path.join("..", "datasets", "dataset-room1_512_16", "mav0", "imu0", "new_data_imu.csv")
out_new_mocap_path = path.join("..", "datasets", "dataset-room3_512_16", "mav0", "mocap0", "new_data_mocap.csv")
out_all_in_df_path = path.join("..", "datasets", "dataset-room3_512_16", "mav0", "self_made_files", "all_in_imu_mocap.csv")
######################################################################################################

## 関数 ##
def AveTwoValue(imu_df, idx):
    NewTimestamp = (imu_df.timestamp[idx*2] + imu_df.timestamp[idx*2 + 1])/2
    NewGyroX = (imu_df.gyroX[idx*2] + imu_df.gyroX[idx*2 + 1])/2
    NewGyroY = (imu_df.gyroY[idx*2] + imu_df.gyroY[idx*2 + 1])/2
    NewGyroZ = (imu_df.gyroZ[idx*2] + imu_df.gyroZ[idx*2 + 1])/2
    NewAccX = (imu_df.aX[idx*2] + imu_df.aX[idx*2 + 1])/2
    NewAccY = (imu_df.aY[idx*2] + imu_df.aY[idx*2 + 1])/2
    NewAccZ = (imu_df.aZ[idx*2] + imu_df.aZ[idx*2 + 1])/2

    return NewTimestamp, NewGyroX, NewGyroY, NewGyroZ, NewAccX, NewAccY, NewAccZ


def linear_com(new_imu_df, mocap_df, idx):
    # 内分率計算
    #print(int(idx), end=" ")
    #print("new_imu_df")
    #print(new_imu_df)
    # print(list(np.where(mocap_df.timestamp <= new_imu_df.timestamp[idx])))
    #print(new_imu_df.timestamp[idx])
    #print(np.where(mocap_df.timestamp <= out_imu_df.timestamp[idx]))
    SmallIdx = max(list(np.where(mocap_df.timestamp <= new_imu_df.timestamp[idx])[0]))
    BigIdx = min(list(np.where(new_imu_df.timestamp[idx] <= mocap_df.timestamp)[0]))
    base_line = mocap_df.timestamp[BigIdx] - mocap_df.timestamp[SmallIdx]
    upper_rate = (mocap_df.timestamp[BigIdx] - imu_df.timestamp[idx])/base_line
    lower_rate = (imu_df.timestamp[idx] - mocap_df.timestamp[SmallIdx])/base_line
    #print(mocap_df.timestamp[SmallIdx], mocap_df.timestamp[BigIdx])

    NewTimestamp = mocap_df.timestamp[SmallIdx]*upper_rate + mocap_df.timestamp[BigIdx]*lower_rate
    NewpX = mocap_df.pX[SmallIdx]*upper_rate + mocap_df.pX[BigIdx]*lower_rate
    NewpY = mocap_df.pY[SmallIdx]*upper_rate + mocap_df.pY[BigIdx]*lower_rate
    NewpZ = mocap_df.pZ[SmallIdx]*upper_rate + mocap_df.pZ[BigIdx]*lower_rate
    NewqW = mocap_df.qW[SmallIdx]*upper_rate + mocap_df.qW[BigIdx]*lower_rate
    NewqX = mocap_df.qX[SmallIdx]*upper_rate + mocap_df.qX[BigIdx]*lower_rate
    NewqY = mocap_df.qY[SmallIdx]*upper_rate + mocap_df.qY[BigIdx]*lower_rate
    NewqZ = mocap_df.qZ[SmallIdx]*upper_rate + mocap_df.qZ[BigIdx]*lower_rate
    
    # return NewTimestamp, NewGyroX, NewGyroY, NewGyroZ, NewAccX, NewAccY, NewAccZ
    return NewTimestamp, NewpX, NewpY, NewpZ, NewqW, NewqX, NewqY, NewqZ


def take_imu_value(new_imu_df, idx):
    Timestamp = new_imu_df.timestamp[idx]
    GyroX = new_imu_df.gyroX[idx]
    GyroY = new_imu_df.gyroY[idx]
    GyroZ = new_imu_df.gyroZ[idx]
    AccX = new_imu_df.accX[idx]
    AccY = new_imu_df.accY[idx]
    AccZ = new_imu_df.accZ[idx]

    return Timestamp, GyroX, GyroY, GyroZ, AccX, AccY, AccZ

## 出力csvが格納されるフォルダ作成
if not path.isdir(path.dirname(out_all_in_df_path)):
    os.mkdir(path.dirname(out_all_in_df_path))

imu_df = pd.read_csv(imu_path)
new_imu_df = pd.read_csv(new_imu_path)
mocap_df = pd.read_csv(mocap_path)
print("new_imu_df")
print(new_imu_df)
print("mocap_df")
print(mocap_df)
new_imu_df_row_num = int(imu_df.shape[0]/2)
print("new_imu_df_row_num", new_imu_df_row_num)

## mocap fileの線形補完 and IMUとmocapの統合ファイル生成 ##
out_mocap_df = pd.DataFrame(columns=['timestamp', 'pX', 'pY', 'pZ', 'qW', 'qX', 'qY', 'qZ'])
all_in_df = pd.DataFrame(columns=['timestamp', 'gyroX', 'gyroY', 'gyroZ', 'accX', 'accY', 'accZ', 'pX', 'pY', 'pZ', 'qW', 'qX', 'qY', 'qZ'])
for i in tqdm(range(5, new_imu_df_row_num -5)):
    NewTimestamp, NewpX, NewpY, NewpZ, NewqW, NewqX, NewqY, NewqZ = linear_com(new_imu_df, mocap_df, i)
    _, GyroX, GyroY, GyroZ, AccX, AccY, AccZ = take_imu_value(new_imu_df, i)
    out_mocap_df = out_mocap_df.append({'timestamp': NewTimestamp, 'pX': NewpX, 'pY': NewpY, 'pZ': NewpZ, 'qW': NewqW, 'qX': NewqX, 'qY': NewqY, \
                                        'qZ': NewqZ}, ignore_index=True)
    all_in_df = all_in_df.append({'timestamp': NewTimestamp, 'pX': NewpX, 'pY': NewpY, 'pZ': NewpZ, 'qW': NewqW, 'qX': NewqX, 'qY': NewqY, 'qZ': NewqZ,\
                                    'gyroX': GyroX, 'gyroY': GyroY, 'gyroZ': GyroZ, 'accX': AccX, 'accY': AccY, 'accZ': AccZ}, ignore_index=True)


## 出力 ##
# out_imu_df.to_csv(out_new_imu_path)
# out_mocap_df.to_csv(out_new_mocap_path)
all_in_df.to_csv(out_all_in_df_path)
