# oxfordのvi*.csvファイルの連続数値列(今回は謎のrotation_x, rotation_y, rotation_z, rotation_w)を可視化（2Dプロット）するためのコード
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# xyzテキストファイルパス
# input_csv_file_path=os.path.join(".." ,".." ,"datasets" ,"oxford_IOD" ,"running", "data1", "syn", "concate_imu3_vi3.csv")
# input_csv_file_path=os.path.join(".." ,".." ,"datasets" ,"oxford_IOD" ,"slow walking", "data1", "syn", "concate_imu3_vi3.csv")
input_csv_file_path=os.path.join("..", "datasets" ,"oxford_IOD", "mix", "data1", "syn", "mix_slow_imu3_vi3_handheld_imu235_vi235.csv")
# input_csv_file_path=os.path.join(".." ,".." ,"datasets" ,"oxford_IOD" ,"handheld", "data1", "syn", "concate_imu3_vi3.csv")
offset = 102000
show_length = 1000
start = 0 + offset
goal = show_length + offset

assert os.path.exists(input_csv_file_path), "The dataset path which you set is not correct."
all_df = pd.read_csv(input_csv_file_path)
print(all_df)
all_df = all_df[start:goal]

# print(plot_df.dtypes)
# plot_df = pd.DataFrame(columns=['rotation_x', 'rotation_x', 'rotation_x', 'rotation_w'])
# all_df['rotation_x']=all_df['rotation_x'].astype(float)# all_df['accX']=all_df['accX'].astype(float)
# all_df['rotation_y']=all_df['rotation_y'].astype(float)# all_df['accY']=all_df['accY'].astype(float)
# all_df['rotation_z']=all_df['rotation_z'].astype(float)# all_df['accZ']=all_df['accZ'].astype(float)
# all_df['rotation_w']=all_df['rotation_w'].astype(float)# all_df['accZ']=all_df['accZ'].astype(float)
# all_df[['rotation_x', 'rotation_y', 'rotation_z', 'rotation_w']].plot()# all_df[['accX', 'accY', 'accZ']].plot()

# plot_df = pd.DataFrame(columns=['user_acc_x(G)', 'user_acc_y(G)', 'user_acc_z(G)'])
# all_df['user_acc_x(G)']=all_df['user_acc_x(G)'].astype(float)
# all_df['user_acc_y(G)']=all_df['user_acc_y(G)'].astype(float)
# all_df['user_acc_z(G)']=all_df['user_acc_z(G)'].astype(float)
# all_df[['user_acc_x(G)', 'user_acc_y(G)', 'user_acc_z(G)']].plot()

# plot_df = pd.DataFrame(columns=['pX', 'pY', 'pZ'])
# all_df['pZ']=all_df['pX'].astype(float)
# all_df['pY']=all_df['pY'].astype(float)
# all_df['pZ']=all_df['pZ'].astype(float)
# all_df[['pX', 'pY', 'pZ']].plot()

plot_df = pd.DataFrame(columns=['accX', 'accY', 'accZ'])
all_df['accX']=all_df['accX'].astype(float)
all_df['accY']=all_df['accY'].astype(float)
all_df['accZ']=all_df['accZ'].astype(float)
all_df[['accX', 'accY', 'accZ']].plot()

# plot_df = pd.DataFrame(columns=['gyroX', 'gyroY', 'gyroZ'])#['accX', 'accY', 'accZ'])
# all_df['gyroX']=all_df['gyroX'].astype(float)# all_df['accX']=all_df['accX'].astype(float)
# all_df['gyroY']=all_df['gyroY'].astype(float)# all_df['accY']=all_df['accY'].astype(float)
# all_df['gyroZ']=all_df['gyroZ'].astype(float)# all_df['accZ']=all_df['accZ'].astype(float)
# all_df[['gyroX', 'gyroY', 'gyroZ']].plot()# all_df[['accX', 'accY', 'accZ']].plot()
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=18)
plt.show()