# PYTHON_MATPLOTLIB_3D_PLOT_02

# 3次元散布図
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# xyzテキストファイルパス
# input_text_file_path=os.path.join("..","..","datasets","running-easy-vi_gt_data","mocap_data.txt")
input_text_file_path=os.path.join("..","..","datasets","running-hard-vi_gt_data","mocap_data.txt")
# input_text_file_path=os.path.join("..","..","running-easy-vi_gt_data","test.txt")

# テキストファイル読み込み
x, y, z = np.arange(1), np.arange(1), np.arange(1)
count=0
with open(input_text_file_path) as f:
    for line in tqdm(f):
        if count==0:
            count+=1
        # elif 1460<count:
        #     break
        else:
            line = line.rstrip()  # 読み込んだ行の末尾には改行文字があるので削除
            listedLine = line.split()
            x = np.append(x, float(listedLine[1]))# set col number
            y = np.append(y, float(listedLine[2]))# set col number
            z = np.append(z, float(listedLine[3]))# set col number
            count+=1
            #print(listedLine)
x = np.delete(x, 0)
y = np.delete(y, 0)
z = np.delete(z, 0)

f.close()
#print(x.shape)


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

# -5～5の乱数配列(100要素)
x[:, np.newaxis, np.newaxis]
y[:, np.newaxis, np.newaxis]
z[:, np.newaxis, np.newaxis]

# 曲線を描画
ax.scatter(x, y, z, s = 5, c = "blue")

plt.show()