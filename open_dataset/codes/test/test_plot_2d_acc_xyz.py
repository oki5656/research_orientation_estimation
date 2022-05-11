# csvファイルの連続数値列を可視化（2Dプロット）するためのコード
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# xyzテキストファイルパス
# input_text_file_path=os.path.join("..","datasets","running-easy-vi_gt_data","mocap_data.txt")
out_all_in_df_path = os.path.join("..", "..", "datasets", "dataset-room5_512_16", "mav0", "self_made_files", "new_all_in_imu_mocap.csv")

all_df = pd.read_csv(out_all_in_df_path)
all_df = all_df[0:100]
print(all_df)
plot_df = pd.DataFrame(columns=['gyroX', 'gyroY', 'gyroZ'])#['accX', 'accY', 'accZ'])
print(plot_df.dtypes)
all_df['gyroX']=all_df['gyroX'].astype(float)# all_df['accX']=all_df['accX'].astype(float)
all_df['gyroY']=all_df['gyroY'].astype(float)# all_df['accY']=all_df['accY'].astype(float)
all_df['gyroZ']=all_df['gyroZ'].astype(float)# all_df['accZ']=all_df['accZ'].astype(float)
all_df[['gyroX', 'gyroY', 'gyroZ']].plot()# all_df[['accX', 'accY', 'accZ']].plot()
plt.show()
