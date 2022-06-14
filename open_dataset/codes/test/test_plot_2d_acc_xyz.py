# csvファイルの連続数値列を可視化（2Dプロット）するためのコード
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# xyzテキストファイルパス
# input_text_file_path=os.path.join("..","datasets","running-easy-vi_gt_data","mocap_data.txt")
out_all_in_df_path = os.path.join("..", "..", "datasets", "TUM", "dataset-room2_512_16", "mav0", "self_made_files", "new_all_in_imu_mocap.csv")

all_df = pd.read_csv(out_all_in_df_path)
print(all_df)
all_df = all_df[5000:6000]
# plot_df = pd.DataFrame(columns=['gyroX', 'gyroY', 'gyroZ'])#['accX', 'accY', 'accZ'])
# all_df['gyroX']=all_df['gyroX'].astype(float)# all_df['accX']=all_df['accX'].astype(float)
# all_df['gyroY']=all_df['gyroY'].astype(float)# all_df['accY']=all_df['accY'].astype(float)
# all_df['gyroZ']=all_df['gyroZ'].astype(float)# all_df['accZ']=all_df['accZ'].astype(float)
# all_df[['gyroX', 'gyroY', 'gyroZ']].plot()# all_df[['accX', 'accY', 'accZ']].plot()
plot_df = pd.DataFrame(columns=["qW","qX","qY","qZ"])
all_df['qW']=all_df['qW'].astype(float)
all_df['qX']=all_df['qX'].astype(float)
all_df['qY']=all_df['qY'].astype(float)
all_df['qZ']=all_df['qZ'].astype(float)
all_df[['qW', 'qX', 'qY', 'qZ']].plot()
plt.ylim(-100, 100)
plt.show()
