# モーションキャプチャによる記録がされたファイルにおいて，各列にどれくらいNanが含まれているか計算する
# また，Nanがk連続して含まれる数をk（1, 2, 3..）によって集計する

import os
import numpy as np
import pandas as pd
from os.path import join


################################################################################
# csv_file_path = join("csvdata", "all_complete_001_v1", "Take 2022-08-09 08.31.59 PM_003_v1_topbuttom_cut_all_complete.csv")
csv_file_path = join("..", "datasets", "large_space", "mocap", "foot_maker_processed_15row_interpolation",
"Take 2022-08-09 08.31.59 PM_001_v1_topbuttom_cut_foot_maker_processed_15nan_remain.csv")
################################################################################
df = pd.read_csv(csv_file_path)
row_num, col_num = df.shape
file_name = os.path.basename(csv_file_path)
print(f"{file_name} nan rate of each row..")
for i in range(col_num):
    print(f"{df.columns[i]}: {round((df.isnull().sum()[i]/row_num)*100, 3)}%")

print("\nfoot maker nan rate")
nan_count_list = [0]*60
contiguous_nan_num = 0
nan_num = 0
for i in range(row_num):
    if np.isnan(df.iat[i, 7]):
        contiguous_nan_num+=1
        nan_num+=1
    else:
        nan_count_list[contiguous_nan_num]+=1
        contiguous_nan_num=0

print("nan_num", nan_num)
print("nan_count_list ", nan_count_list)

cumulative_sum = 0
for i in range(len(nan_count_list)):
    row_i_percent = round(100*nan_count_list[i]*i/row_num, 2)
    cumulative_sum+=row_i_percent
    print(f"row{i}: {row_i_percent}%   cumulative sum{round(cumulative_sum, 2)}%")
    
