# このファイルではTUMのデータセットを用い、時系列データ学習用にcsvファイルの整形を行うための簡易なコーディングテストに用いる

from os import path
import numpy as np
import pandas as pd


####################################### path setting #################################################
imu_path = path.join("..", "datasets", "dataset-room1_512_16", "mav0", "imu0", "data_imu.csv")
mocap_path = path.join("..", "datasets", "dataset-room1_512_16", "mav0", "mocap0", "data_mocap.csv")
out_new_imu_path = path.join("..", "datasets", "dataset-room1_512_16", "mav0", "imu0", "new_data_imu.csv")
out_new_mocap_path = path.join("..", "datasets", "dataset-room1_512_16", "mav0", "mocap0", "new_data_mocap.csv")
######################################################################################################
# pd.options.display.precision = 15
imu_df = pd.read_csv(imu_path)
mocap_df = pd.read_csv(mocap_path)

print(imu_df)
print(mocap_df)
# print(imu_df.columns)
# print(imu_df.timestamp)
print("imu_df.timestamp[5] = ", imu_df.timestamp[5])
SmallIdx = max(list(np.where(mocap_df.timestamp <= imu_df.timestamp[5])[0]))
BigIdx = min(list(np.where(imu_df.timestamp[5] <= mocap_df.timestamp)[0]))
base_line = mocap_df.timestamp[BigIdx] - mocap_df.timestamp[SmallIdx]
upper_rate = (mocap_df.timestamp[BigIdx] - imu_df.timestamp[5])/base_line
lower_rate = (imu_df.timestamp[5] - mocap_df.timestamp[SmallIdx])/base_line
print(mocap_df.timestamp[SmallIdx], mocap_df.timestamp[BigIdx])
print("lower_rate, upper_rate", lower_rate, upper_rate)
