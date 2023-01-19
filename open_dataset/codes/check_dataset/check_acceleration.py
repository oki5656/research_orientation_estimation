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
                               "sum_train_20220809_001and003_nan_under15_nan_removed.csv")
acceleration_columns = ['accX', 'accY', 'accZ']
dataset_frequency = 30

###########################################################################################################

class CheckAcc():
    def __init__(self):
        pass


    def process_all(self):
        """
        """
        acceleration_df = self.acceleration_data_loader(train_data_path)
        self.check_each_xyz_min_max(acceleration_df)
        self.check_distance_min_max(acceleration_df)


    def acceleration_data_loader(self, csv_path):
        """
        """
        df = pd.read_csv(csv_path)
        acceleration_df = df[acceleration_columns]
        self.df_row_num = df.shape[0]

        return acceleration_df


    def check_each_xyz_min_max(self, acceleration_df):
        """
        """
        accX_max, accY_max, accZ_max = 0, 0, 0
        accX_min, accY_min, accZ_min = 99, 99, 99
        for i in tqdm(range(self.df_row_num)):
            row_i_df = acceleration_df.iloc[i]
            accX_max = max(row_i_df[0], accX_max)
            accY_max = max(row_i_df[1], accY_max)
            accZ_max = max(row_i_df[2], accZ_max)
            accX_min = min(row_i_df[0], accX_min)
            accY_min = min(row_i_df[1], accY_min)
            accZ_min = min(row_i_df[2], accZ_min)
        print(accX_min, accY_min, accZ_min, accX_max, accY_max, accZ_max)


    def check_distance_min_max(self, acceleration_df):
        """
        """
        distance_max, distance_min, = 0, 999
        for i in tqdm(range(self.df_row_num)):
            row_i_df = acceleration_df.iloc[i]
            distance = math.sqrt(row_i_df[0]**2+row_i_df[1]**2+row_i_df[2]**2)
            distance_max = max(distance, distance_max)
            distance_min = min(distance, distance_min)
        print(distance_min, distance_max)
         

if __name__ == "__main__":
    check_acc = CheckAcc()
    check_acc .process_all()
