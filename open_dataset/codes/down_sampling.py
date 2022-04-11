# このファイルではTUMのデータセットを用い、時系列データ学習用にcsvファイルの整形を行う。
# 具体的にはIMUのファイルを200Hz→100Hzにダウンサンプリングする部分のみを行う。

from os import path
import numpy as np
import pandas as pd
from tqdm import tqdm

####################################### path setting #################################################
imu_path = path.join("..", "datasets", "dataset-room2_512_16", "mav0", "imu0", "data_imu.csv")
mocap_path = path.join("..", "datasets", "dataset-room2_512_16", "mav0", "mocap0", "data_mocap.csv")
out_new_imu_path = path.join("..", "datasets", "dataset-room2_512_16", "mav0", "imu0", "new_data_imu.csv")
out_new_mocap_path = path.join("..", "datasets", "dataset-room2_512_16", "mav0", "mocap0", "new_data_mocap.csv")
######################################################################################################

## 関数 ##
def AveTwoValue(imu_df, idx):
    NewTimestamp = (imu_df.timestamp[idx*2] + imu_df.timestamp[idx*2 + 1])/2
    NewGyroX = (imu_df.gyroX[idx*2] + imu_df.gyroX[idx*2 + 1])/2
    NewGyroY = (imu_df.gyroY[idx*2] + imu_df.gyroY[idx*2 + 1])/2
    NewGyroZ = (imu_df.gyroZ[idx*2] + imu_df.gyroZ[idx*2 + 1])/2
    NewAccX = (imu_df.accX[idx*2] + imu_df.accX[idx*2 + 1])/2
    NewAccY = (imu_df.accY[idx*2] + imu_df.accY[idx*2 + 1])/2
    NewAccZ = (imu_df.accZ[idx*2] + imu_df.accZ[idx*2 + 1])/2

    return NewTimestamp, NewGyroX, NewGyroY, NewGyroZ, NewAccX, NewAccY, NewAccZ


imu_df = pd.read_csv(imu_path)
mocap_df = pd.read_csv(mocap_path)

## IMU fileのダウンサンプリング ##
out_imu_df = pd.DataFrame(columns=['timestamp', 'gyroX', 'gyroY', 'gyroZ', 'accX', 'accY', 'accZ'])
new_imu_df_row_num = int(imu_df.shape[0]/2)
for i in tqdm(range(new_imu_df_row_num)):
    NewTimestamp, NewGyroX, NewGyroY, NewGyroZ, NewAccX, NewAccY, NewAccZ = AveTwoValue(imu_df, i)
    out_imu_df = out_imu_df.append({'timestamp': NewTimestamp, 'gyroX': NewGyroX, 'gyroY': NewGyroY, 'gyroZ': NewGyroZ, 'accX': NewAccX, 'accY': NewAccY, 'accZ': NewAccZ}, ignore_index=True)
print(out_imu_df)

## 出力 ##
out_imu_df.to_csv(out_new_imu_path)