# このファイルではファイルの1,2列目についている番号を1,2,3...と貼りなおす
########################################################################
import os
import sys
a=os.path.dirname(sys.executable)
print(os.path.dirname(sys.executable))
########################################################################

import pandas as pd
from matplotlib import pyplot as plt
import math
import numpy as np
from tqdm import tqdm
import decimal


#########################################################################################################################################################
# input_path = os.path.join("..","datasets", "dataset-room2_512_16", "mav0", "self_made_files", "new_all_in_imu_mocap.csv")
input_path = os.path.join("..","datasets", "dataset-room_all", "mav0", "self_made_files", "new_all_in_imu_mocap_13456.csv")

#########################################################################################################################################################

df = pd.read_csv(input_path)
df_rows_num = len(df)

for i in tqdm(range(df_rows_num)):
    df.A[i] = i
    df.B[i] = i

    
print("Done")