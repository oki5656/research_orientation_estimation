# ラージスペースデータセットの列名を選択することで任意の列の３次元位置をプロットすることができる

import os
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from os.path import join


foot_position_columns = ['pX', 'pY', 'pZ']
imu_position_colmuns = ["imu_position_x", "imu_position_y", "imu_position_z"]
position_columns = {"foot": foot_position_columns, "imu": imu_position_colmuns}
###########################################################################################################
csv_path = join("..", "datasets", "large_space", "nan_removed", "interpolation_under_15",
                               "harf_test_20220809_002_nan_under15_nan_removed.csv")
imu_or_foot = "imu"
dataset_frequency = 30
start_frame = 30*(5)
draw_length = 30*(10)
###########################################################################################################


class DrawTrajectory():
    def __init__(self):
        self.selected_colmuns = position_columns[imu_or_foot]


    def data_loader(self, path):
        assert os.path.isfile(path), "The path you set is something wrong..."
        df = pd.read_csv(path)
        selected_position_df = df[self.selected_colmuns]
        selected_position_df = selected_position_df.iloc[start_frame: start_frame+draw_length, :]

        return selected_position_df


    def draw_trajectry(self, selected_position_df):
        # 描画
        selected_world_positions = np.array(selected_position_df)/1000
        x = selected_world_positions[:, 0]
        y = selected_world_positions[:, 1]
        z = selected_world_positions[:, 2]
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(x, y, z, vmin=0, vmax=1, color="black", marker = ".")
        max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() * 0.5
        mid_x = (x.max()+x.min()) * 0.5
        mid_y = (y.max()+y.min()) * 0.5
        mid_z = (z.max()+z.min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        ax.set_xlabel("[m]")
        ax.set_ylabel("[m]")
        ax.set_zlabel("[m]")
        plt.show()


    def process_all(self):
        selected_position_df = self.data_loader(csv_path)
        self.draw_trajectry(selected_position_df)


if __name__ == "__main__":
    draw_trajectory = DrawTrajectory()
    draw_trajectory.process_all()
