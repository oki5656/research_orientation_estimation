# linear_complimentation_using_acc_gyro.pyを実行後、欠損を埋めるためにこのファイルを実行する
# all_in_imu_mocap.csvをもとに欠損が保管されたnew_all_in_imu_mocap.csvを生成する

from cmath import isnan
import pandas as pd
from matplotlib import pyplot as plt
import os
import numpy as np
from tqdm import tqdm


#############################################  config  ##################################################
train_data_path = os.path.join("..","datasets", "dataset-room3_512_16", "mav0", "self_made_files", "all_in_imu_mocap.csv")
new_train_data_path = os.path.join("..","datasets", "dataset-room3_512_16", "mav0", "self_made_files", "new_all_in_imu_mocap.csv")
selected_train_columns = ['gyroX', 'gyroY', 'gyroZ', 'accX', 'accY', 'accZ']
selected_correct_columns = ['pX', 'pY', 'pZ', 'qW', 'qX', 'qY', 'qZ']
#########################################################################################################

df = pd.read_csv(train_data_path)
print(df.shape)

for i in tqdm(range(df.shape[0])):
    if np.isnan(df.at[i, "pX"]) or np.isnan(df.at[i, "pY"]) or np.isnan(df.at[i, "pZ"]):
        # print(i, df.at[i, "pX"], df.at[i, "pY"], df.at[i, "pZ"])
        tqdm.write(str(i) + str(df.at[i, "pX"]) + str(df.at[i, "pY"]) + str(df.at[i, "pZ"]))
        df.at[i, "pX"] = (df.at[i - 1, "pX"] + df.at[i + 1, "pX"])/2
        df.at[i, "pY"] = (df.at[i - 1, "pY"] + df.at[i + 1, "pY"])/2
        df.at[i, "pZ"] = (df.at[i - 1, "pZ"] + df.at[i + 1, "pZ"])/2
        df.at[i, "qW"] = (df.at[i - 1, "qW"] + df.at[i + 1, "qW"])/2
        df.at[i, "qX"] = (df.at[i - 1, "qX"] + df.at[i + 1, "qX"])/2
        df.at[i, "qY"] = (df.at[i - 1, "qY"] + df.at[i + 1, "qY"])/2
        df.at[i, "qZ"] = (df.at[i - 1, "qZ"] + df.at[i + 1, "qZ"])/2
df.drop(columns='Unnamed: 0')
print(df.columns)
df.to_csv(new_train_data_path)
print("completed!!!")