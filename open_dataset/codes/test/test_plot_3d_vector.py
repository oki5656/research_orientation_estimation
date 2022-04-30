# csvファイルの中身を３次元プロットする

import os
import math
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from os import path
import pandas as pd

# position csvファイルパス#####################################################################################################################
input_text_file_path=os.path.join(".." ,".." ,"datasets", "dataset-room2_512_16", "mav0", "self_made_files", "new_all_in_imu_mocap.csv")
##############################################################################################################################################

def TransWithQuat(pX, pY, pZ, qW, qX, qY, qZ):
    E = np.array([[qX**2 - qY**2 - qZ**2 + qW**2, 2*(qX*qY - qZ*qW), 2*(qX*qZ + qY*qW)],
                [2*(qX*qY + qZ*qW), -qX**2 + qY**2 - qZ**2 + qW**2, 2*(qY*qZ - qX*qW)],
                [2*(qX*qZ - qY*qW), 2*(qY*qZ + qX*qW), -qX**2 - qY**2 + qZ**2 + qW**2]])#クォータニオン表現による回転行列
    # old_vector = np.array([pX, pY, pZ])
    old_vector = np.array([1, 0, 0])
    new_vector = np.matmul(E, old_vector.T)
    return new_vector

# テキストファイル読み込み
count=1
csv_df = pd.read_csv(input_text_file_path)
start=4000
goal=9500
csv_df_row_num = int(csv_df.shape[0])
# with open(input_text_file_path) as f:
u = []
v = []
w = []
for i in tqdm(range(start, csv_df_row_num - goal)):
    pX = csv_df.pX[i]
    pY = csv_df.pY[i]
    pZ = csv_df.pZ[i]
    qW = csv_df.qW[i]
    qX = csv_df.qX[i]
    qY = csv_df.qY[i]
    qZ = csv_df.qZ[i]
    new_vector = TransWithQuat(pX, pY, pZ, qW, qX, qY, qZ)
    unit_value = math.sqrt(new_vector[0]**2 + new_vector[1]**2 + new_vector[2]**2)
    u.append(4*new_vector[0]/unit_value)
    v.append(4*new_vector[1]/unit_value)
    w.append(4*new_vector[2]/unit_value)
######################################################################
plt.style.use('ggplot')
plt.rcParams["axes.facecolor"] = 'white'
fig = plt.figure()
ax = fig.gca(projection='3d')
# Make xyz
x = csv_df.pX[start:csv_df_row_num - goal]
y = csv_df.pY[start:csv_df_row_num - goal]
z = csv_df.pZ[start:csv_df_row_num - goal]

# Make the direction data for the arrows
# u = 0.1*np.ones((3,3,3))
# v = 0.1*np.ones((3,3,3))
# w = -0.1*np.ones((3,3,3))
ax.set(xlabel='x',ylabel='y',zlabel='z')
ax.quiver(x, y, z, u, v, w)
ax.plot(x.ravel(),y.ravel(),z.ravel(),'go')
######################################################################

plt.show()