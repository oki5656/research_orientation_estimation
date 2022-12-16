import os
import re
import math
import random
import json
import glob
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
from os.path import join
import argparse
from statistics import mean

###########################################################################################################
train_data_path = os.path.join("..", "datasets", "large_space", "nan_removed", "interpolation_under_15",
                               "harf_test_20220809_002_nan_under15_nan_removed.csv")
position_columns = ['pX', 'pY', 'pZ']
dataset_frequency = 30
near_length_threshold = 40#[mm]
start_frame = 30*(10)
draw_length = 30*(15)
###########################################################################################################

class MeasureLength():
    def __init__(self):
        pass


    def process_all(self):
        """
        """
        position_df = self.position_data_loader(train_data_path)
        foot_position = self.extract_foot_position(position_df)


    def position_data_loader(self, csv_path):
        """
        """
        df = pd.read_csv(csv_path)
        position_df = df[position_columns]
        self.df_row_num = df.shape[0]
        # position_df = position_df.iloc[start_frame: start_frame+draw_length, :]

        return position_df


    def measure_distance(self, posi_A, posi_B):
        dis = math.sqrt((posi_A[0]-posi_B[0])**2+(posi_A[1]-posi_B[1])**2+(posi_A[2]-posi_B[2])**2)
        
        return dis



    def count_near_foot_position(self, position_df, i, near_threshoul):
        """近い距離（閾値以内に存在）にあるfoot positionの個数をカウントする。閾値は初期情報でユーザーが与える。
        """
        count = 0
        for j in range(-dataset_frequency, dataset_frequency+1):
            if j == 0:
                continue
            dis = self.measure_distance(position_df.iloc[i, :], position_df.iloc[i+j, :])
            if dis <= near_length_threshold:
                count+=1

        return count


    def extract_foot_position(self, position_df):
        """足の位置ベクトルのみのリストを出力する（今のところ近くにある位置数のグラフを表示）
        """
        self.near_foot_posi_num_list = []
        print("count number of near positions...")
        # for i in tqdm(range(dataset_frequency, self.df_row_num - dataset_frequency)):
        for i in tqdm(range(start_frame, start_frame+draw_length)):
            near_foot_posi_num = self.count_near_foot_position(position_df, i, near_length_threshold)
            self.near_foot_posi_num_list.append(near_foot_posi_num)
        # x = list(range(dataset_frequency, self.df_row_num - dataset_frequency))
        x = list(range(start_frame, start_frame+draw_length))
        y = self.near_foot_posi_num_list
        plt.plot(x, y, color="k")
        plt.show()



if __name__ == "__main__":
    measure_length = MeasureLength()
    measure_length.process_all()
