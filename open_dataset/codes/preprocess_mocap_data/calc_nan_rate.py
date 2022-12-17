# モーションキャプチャによる記録がされたファイルにおいて，各列にどれくらいNanが含まれているか計算する
# また，Nanがk連続して含まれる数をk（1, 2, 3..）によって集計する

import os
import numpy as np
import pandas as pd
from os.path import join


################################################################################
# csv_file_path = join("csvdata", "all_complete_001_v1", "Take 2022-08-09 08.31.59 PM_003_v1_topbuttom_cut_all_complete.csv")
csv_file_path = join("..", "datasets", "large_space", "mocap", "foot_maker_processed_0row_interpolation",
"Take 2022-08-09 08.31.59 PM_001_v1_topbuttom_cut_foot_maker_processed_nan_remain_test.csv")
foot_marker_check_col_name = "foot_maker_position_X"
################################################################################


class CalcNan():
    def __init__(self):
        self.df = pd.read_csv(csv_file_path)
        self.row_num, self.col_num = self.df.shape
        self.file_name = os.path.basename(csv_file_path)


    def process_all(self):
        self.calc_nan_rate_of_each_col()
        self.foot_marker_nan()


    def calc_nan_rate_of_each_col(self):
        print(f"{self.file_name} nan rate of each col..")
        for i in range(self.col_num):
            print(f"{self.df.columns[i]}: {round((self.df.isnull().sum()[i]/self.row_num)*100, 3)}%")


    def foot_marker_nan(self):
        print("\nfoot maker nan rate")
        nan_count_list = [0]*60
        contiguous_nan_num = 0
        nan_num = 0
        for i in range(self.row_num):
            if np.isnan(self.df[foot_marker_check_col_name][i]):
                contiguous_nan_num+=1
                nan_num+=1
            else:
                nan_count_list[contiguous_nan_num]+=1
                contiguous_nan_num=0

        print("nan_num", nan_num)
        print("nan_count_list ", nan_count_list)

        cumulative_sum = 0
        for i in range(len(nan_count_list)):
            row_i_percent = round(100*nan_count_list[i]*i/self.row_num, 2)
            cumulative_sum+=row_i_percent
            print(f"row{i}: {row_i_percent}%   cumulative sum{round(cumulative_sum, 2)}%")


if __name__ == "__main__":
    calc_nan = CalcNan()
    calc_nan.process_all()
