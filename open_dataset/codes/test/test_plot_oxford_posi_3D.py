# oxfordのvi*.csvファイルの中身を３次元プロットする
# 中身は直交座標系の位置だと思われる。単位は[m]っぽい

import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.style as mplstyle
from os import path
import pandas as pd


# position csvファイルパス
# input_text_file_path=os.path.join(".." ,".." ,"datasets","running-hard-vi_gt_data","mocap_data.txt")
input_text_file_path=os.path.join(".." ,".." ,"datasets" ,"oxford_IOD" ,"handheld", "data1", "syn", "concate_imu2_vi2.csv")

# plotを高速化
mplstyle.use('fast')

# Figureを追加
fig = plt.figure(figsize = (8, 8))

# 3DAxesを追加
ax = fig.add_subplot(111, projection='3d')

# Axesのタイトルを設定
ax.set_title("", size = 40)

# 軸ラベルを設定
ax.set_xlabel("x", size = 14, color = "r")
ax.set_ylabel("y", size = 14, color = "r")
ax.set_zlabel("z", size = 14, color = "r")

# 軸目盛を設定
# ax.set_xticks([-5.0, -2.5, 0.0, 2.5, 5.0, 7.5, 10.0])
# ax.set_yticks([-5.0, -2.5, 0.0, 2.5, 5.0, 7.5, 10.0])

# テキストファイル読み込み
# x, y, z = np.arange(1), np.arange(1), np.arange(1)
count=1
csv_df = pd.read_csv(input_text_file_path)
start=3000
goal=4000
csv_df_row_num = int(csv_df.shape[0])
print("csv_df_row_num=", csv_df_row_num)
print(f"plotting from row {start} to {goal}.")
with open(input_text_file_path) as f:
    for i in tqdm(range(start, goal)):
        # pX = csv_df.translation_x[i + 1]
        # pY = csv_df.translation_y[i + 1]
        # pZ = csv_df.translation_z[i + 1]
        pX = csv_df.pX[i + 1]
        pY = csv_df.pY[i + 1]
        pZ = csv_df.pZ[i + 1]
        # 曲線を描画
        color_set = [(count/(goal - start), 0, 0, 1)]
        ax.scatter(float(pX), float(pY), float(pZ), s = 5, c = color_set)
        count += 1
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_zlim(-1.8, 4.2)
plt.show()

# print(csv_df)
