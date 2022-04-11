import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# xyzテキストファイルパス
# input_text_file_path=os.path.join("..","datasets","running-easy-vi_gt_data","mocap_data.txt")
input_text_file_path=os.path.join(".." ,".." ,"datasets" ,"running-hard-vi_gt_data" ,"mocap_data.txt")
# input_text_file_path=os.path.join("..","running-easy-vi_gt_data","test.txt")

# テキストファイル読み込み
timestamps = np.arange(1)
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
            timestamps = np.append(timestamps, float(listedLine[0]))
            count+=1
            #print(listedLine)
timestamps = np.delete(timestamps, 0)
f.close()

#グラフのタイトル＆軸
plt.title("Visualizing Timestamps")
plt.xlabel('number of plot')
plt.ylabel('timestamp value')

x=np.linspace(0, count-2, count-1)
plt.plot(x, timestamps)
plt.show()