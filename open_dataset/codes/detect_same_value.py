from cmath import isnan
import pandas as pd
from matplotlib import pyplot as plt
import os
import numpy as np


#############################################  config  ##################################################
train_data_path = os.path.join("..","datasets", "dataset-room1_512_16", "mav0", "self_made_files", "all_in_imu_mocap.csv")
selected_train_columns = ['gyroX', 'gyroY', 'gyroZ', 'accX', 'accY', 'accZ']
selected_correct_columns = ['pX', 'pY', 'pZ', 'qW', 'qX', 'qY', 'qZ']
#########################################################################################################

def dataloader(path, train_columns, correct_columns):
    df = pd.read_csv(path)
    train_x_df = df[train_columns]
    train_t_df = df[correct_columns]

    return train_x_df, train_t_df

def main():
    # train_x_df, train_t_df = dataloader(train_data_path, selected_train_columns, selected_correct_columns)
    train_t_df = pd.read_csv(os.path.join("..","datasets", "dataset-room1_512_16", "mav0", "mocap0", "data_mocap.csv"))
    # columsの確認
    print(train_t_df.columns)
    print("columns num = ", len(train_t_df.columns))
    columns_num = len(train_t_df.columns)
    print(train_t_df.shape)
    same_count = 0
    for i in range(train_t_df.shape[0] - 1):
        if train_t_df["pZ"][i] == train_t_df["pZ"][i+1]:
            same_count+=1
    print("same_count = ", same_count)
    print("nanをいくつふくむか", train_t_df.isnull().sum())
    print("nanを含む行", train_t_df[train_t_df['pX'].isnull()])
    # print(train_t_df["pX"][1])
    # print(train_t_df["pX"][2])

    #重複確認
    # for i in range(columns_num):
    #     print(train_t_df.columns[i])
    #     print(len(train_t_df) == len(train_t_df[train_t_df.columns[i]].unique()))
    #     print()



if __name__ == '__main__':
    main()