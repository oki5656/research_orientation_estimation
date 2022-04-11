import numpy as np
from os import path
import os
import pandas as pd

input_text_file_path=os.path.join(".." ,".." ,"datasets", "dataset-room1_512_16", "mav0", "self_made_files", "new_all_in_imu_mocap.csv")
csv_df = pd.read_csv(input_text_file_path)
x = csv_df.pX[5000:5010]
print(x)
u = 0.1*np.ones((3,3,3))
print(u)

# color_set = np.array([0.5, 0.5, 0.5, 1.0])
# print(color_set.shape)
# print(color_set)
# print(type(color_set))