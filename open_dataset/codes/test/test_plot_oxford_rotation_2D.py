# oxfordのvi*.csvファイルの連続数値列(今回は謎のrotation_x, rotation_y, rotation_z, rotation_w)を可視化（2Dプロット）するためのコード
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# xyzテキストファイルパス
# input_csv_file_path=os.path.join(".." ,".." ,"datasets" ,"oxford_IOD" ,"handheld", "data1", "syn", "vi1.csv")
input_csv_file_path=os.path.join(".." ,".." ,"datasets" ,"oxford_IOD" ,"handheld", "data1", "syn", "imu1.csv")
start = 1000
goal = 1500

assert os.path.exists(os.path.join("..", "..", "datasets")), "None"
all_df = pd.read_csv(input_csv_file_path)
all_df = all_df[start:goal]
print(all_df)
# print(plot_df.dtypes)
# plot_df = pd.DataFrame(columns=['rotation_x', 'rotation_x', 'rotation_x', 'rotation_w'])
# all_df['rotation_x']=all_df['rotation_x'].astype(float)# all_df['accX']=all_df['accX'].astype(float)
# all_df['rotation_y']=all_df['rotation_y'].astype(float)# all_df['accY']=all_df['accY'].astype(float)
# all_df['rotation_z']=all_df['rotation_z'].astype(float)# all_df['accZ']=all_df['accZ'].astype(float)
# all_df['rotation_w']=all_df['rotation_w'].astype(float)# all_df['accZ']=all_df['accZ'].astype(float)
# all_df[['rotation_x', 'rotation_y', 'rotation_z', 'rotation_w']].plot()# all_df[['accX', 'accY', 'accZ']].plot()
plot_df = pd.DataFrame(columns=['user_acc_x(G)', 'user_acc_y(G)', 'user_acc_z(G)'])
all_df['user_acc_x(G)']=all_df['user_acc_x(G)'].astype(float)# all_df['accX']=all_df['accX'].astype(float)
all_df['user_acc_y(G)']=all_df['user_acc_y(G)'].astype(float)# all_df['accY']=all_df['accY'].astype(float)
all_df['user_acc_z(G)']=all_df['user_acc_z(G)'].astype(float)# all_df['accZ']=all_df['accZ'].astype(float)
all_df[['user_acc_x(G)', 'user_acc_y(G)', 'user_acc_z(G)']].plot()# all_df[['accX', 'accY', 'accZ']].plot()
plt.show()