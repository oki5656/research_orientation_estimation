# csvファイルの連続数値列を可視化（プロット）するためのコード
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# xyzテキストファイルパス
# input_text_file_path=os.path.join("..","datasets","running-easy-vi_gt_data","mocap_data.txt")
out_all_in_df_path = os.path.join("..", "..", "datasets", "dataset-room1_512_16", "mav0", "self_made_files", "new_all_in_imu_mocap.csv")

all_df = pd.read_csv(out_all_in_df_path)
all_df = all_df[1000:2000]
print(all_df)
plot_df = pd.DataFrame(columns=['accX', 'accY', 'accZ'])
print(plot_df.dtypes)
all_df['accX']=all_df['accX'].astype(float)
all_df['accY']=all_df['accY'].astype(float)
all_df['accZ']=all_df['accZ'].astype(float)
# plot_df.plot()
all_df[['accX', 'accY', 'accZ']].plot()
plt.show()
