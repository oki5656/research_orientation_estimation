# データの上下カットがなされたものを使用して、足のマーカー座標を整形する。

from cmath import isnan, nan
import os
import math
from turtle import distance
import pandas as pd
import numpy as np
from tqdm import tqdm
from os.path import join
from argparse import ArgumentParser, Namespace


#######################################################################################################
csv_file_path = join("..", "datasets", "large_space", "mocap", "top_buttom_cut", "Take 2022-08-09 08.31.59 PM_001_v1_topbuttom_cut.csv")
new_csv_file_folder_path = join("..", "datasets", "large_space", "mocap", "foot_maker_processed_15row_interpolation")
new_csv_file_name = os.path.splitext(os.path.basename(csv_file_path))[0] + "_foot_maker_processed_nan_remain_test.csv"
XYZorQuaternion = "Quaternion"
NotRowsNumber = 3 # not value exept for columns name. For example, imu 6F512D5A17D511EDA5E3221A4204BB97, Rotation.
Smp_maker_number = 3
OkSameMakerDistance = 50
max_allowed_continuous_nan_num = 15
#######################################################################################################
cwd = os.getcwd()
print("now directory is", cwd)


class PreprocessMocapFile(object):
    """Class for preprocess mocap file."""

    def __init__(self) -> None:
        self.csv_file_path = csv_file_path
        self.df = pd.read_csv(csv_file_path)
        self.RotationColNumber = {"XYZ" : 3, "Quaternion" : 4 }
        self.rotation_columns_number = self.RotationColNumber[XYZorQuaternion]
        self.imu_col_number = 3 + self.rotation_columns_number
        self.all_rows_number, self.all_columns_number = self.df.shape
        self.foot_marker_col_name = ["foot_maker_position_X", "foot_maker_position_Y", "foot_maker_position_Z"]
        self.imu_position_col_name = ["imu_position_x", "imu_position_y", "imu_position_z"]
        self.imu_xyz_rotation_col_name = ["imu_rotation_x", "imu_rotation_y", "imu_rotation_z"]
        self.imu_quaternion_col_name = ["imu_quaternion_x", "imu_quaternion_y", "imu_quaternion_z", "imu_quaternion_w"]
        self.imu_xyz_rotation_col_name.extend(self.imu_position_col_name)
        self.imu_quaternion_col_name.extend(self.imu_position_col_name)
        self.imu_col_name = {"XYZ" : self.imu_xyz_rotation_col_name,
                             "Quaternion" : self.imu_quaternion_col_name}


    def preprocess_all(self) -> None:
        """全てのプロセスがこの関数に基づいて行われる
        """
        new_df = self.make_new_dataframe()
        new_df = self.calc_foot_maker_position(new_df)
        new_df_rows_number, new_df_cols_number = new_df.shape
        new_df = self.interpolation_all_process(new_df, new_df_rows_number, new_df_cols_number)
        os.makedirs(new_csv_file_folder_path, exist_ok=True)
        new_df.to_csv(join(new_csv_file_folder_path, new_csv_file_name))
        print("Preorocess was all completed and csv file was created. ")


    def make_new_dataframe(self):
        """ making new dataframe and setting imu_maker rataion, position
        Args :
            None
        Returns :
            new_df(pd.dataframe) : new pd.dataframe to create
        """
        new_df = pd.DataFrame(
            data={'time': self.df.iloc[3:, 1]}
        )

        for i in range(self.imu_col_number):
            new_col_name = self.imu_col_name[XYZorQuaternion][i]
            new_df[new_col_name] = self.df.iloc[NotRowsNumber:, i + 2]

        return new_df


    def calcurate_distance(self, reference_posi, separeted_maker_position) -> float:
        """calcurate distance between imu and a maker.
        Args :
            reference_posi(list) : first position of maker
            separeted_maker_position(list) : second position of maker
        Returns :
            distance(float) : distance between imu and a maker
        """
        distance = math.sqrt((reference_posi[0]-separeted_maker_position[0])**2 +\
                 (reference_posi[1]-separeted_maker_position[1])**2 + (reference_posi[2]-separeted_maker_position[2])**2)
        return distance


    def is_distance_between_makers_correct(self, distance_between_makers: float, min_distance: int, max_distance: int) -> bool:
        """check distance between two makers is correct or not
        Args :
            distance_between_makers : distance between two makers
        Returns :
            IsDistanceOK(bool) : If distance between two makers is correct become 'True', incorrect become 'False'.

        """
        IsDistanceOK = False
        if min_distance <= distance_between_makers and distance_between_makers <= max_distance:
            IsDistanceOK = True

        return IsDistanceOK


    def calcurate_correct_maker_posi(self, imu_position, feature_col):
        """calcurate correct maker position
        Args :
            imu_position : imu position which is reference point
            maker_position : maker position to check
        Returns :
            correct_maker_posi
        """
        correct_flag = False
        correct_maker_posi = []
        separeted_maker_position = []
        reference_posi = []
        reference_posi.append(float(imu_position[0]))
        reference_posi.append(float(imu_position[1]))
        reference_posi.append(float(imu_position[2]))
        foot_maker_cols_number = (self.all_columns_number - self.imu_col_number - Smp_maker_number*3 - 2)//3

        # Maker loop (direction of columns)，値が入っているマーカーの２次元リストを作成．
        for j in range(foot_maker_cols_number):
            if (not math.isnan(float(feature_col[j*3]))) and (not math.isnan(float(feature_col[j*3+1]))) and (not math.isnan(float(feature_col[j*3+2]))):
                separeted_maker_position.append([float(feature_col[j*3]), float(feature_col[j*3+1]), float(feature_col[j*3+2])])
        # print(separeted_maker_position)

        recognized_maker_number = len(separeted_maker_position)

        # 1行に認識されたマーカー数が１のとき
        if recognized_maker_number == 1:
            # distance_between_makers = math.sqrt((reference_posi[0]-separeted_maker_position[0][0])**2 +\
            #      (reference_posi[1]-separeted_maker_position[0][1])**2 + (reference_posi[2]-separeted_maker_position[0][2])**2)
            distance_between_makers = self.calcurate_distance(reference_posi, separeted_maker_position[0])
            if self.is_distance_between_makers_correct(distance_between_makers, 700, 1700):
                correct_maker_posi = separeted_maker_position[0]
            else:
                correct_maker_posi = [np.nan, np.nan, np.nan]

        # 1行に認識されたマーカー数が2以上のとき
        else:
            # makers loop, マーカー一個に着目
            for k in reversed(range(recognized_maker_number)):
                distance_between_makers = self.calcurate_distance(reference_posi, separeted_maker_position[k])

                # IMUマーカーからの距離で判断
                if self.is_distance_between_makers_correct(distance_between_makers, 700, 1700):
                    pass
                else:
                    # IMUマーカーからの距離がおかしいマーカーを除去
                    separeted_maker_position.pop(k)
                
            # 正しい距離のマーカーに対する処理
            correct_makers_number = len(separeted_maker_position)
            if correct_makers_number == 0:
                correct_maker_posi = [np.nan, np.nan, np.nan]
            elif correct_makers_number == 1:
                correct_maker_posi = separeted_maker_position[0]
            else:
                IsMakerCorrect = False
                for l in range(correct_makers_number):
                    for m in range(l+1, correct_makers_number):
                        distance_between_makers = self.calcurate_distance(separeted_maker_position[l], separeted_maker_position[m])
                        if not self.is_distance_between_makers_correct(distance_between_makers, 0, OkSameMakerDistance):
                            IsMakerCorrect = False
                            break
                    else:
                        continue
                    break
                if IsMakerCorrect:
                    np_makers_posi = np.array(separeted_maker_position)
                    correct_maker_posi = np.mean(np_makers_posi, axis=0)
                else:
                    correct_maker_posi = [np.nan, np.nan, np.nan]

        return correct_maker_posi


    def calc_foot_maker_position(self, new_df):
        """calcurate foot maker position from ambiguous makers position
        Args : 
            new_df(pandas.dataframe) : only have imu information
        Returns :
            new_df(pandas.dataframe) : added foot maker information
        """
        foot_maker_position_X = []
        foot_maker_position_Y = []
        foot_maker_position_Z = []

        print("calcurating foot maker position")
        # Rows loop
        for i in tqdm(range(NotRowsNumber, self.all_rows_number)):
            featureCol= self.df.iloc[i, self.imu_col_number + Smp_maker_number*3 + 2: ]
            imu_position = self.df.iloc[i, 2 + self.rotation_columns_number: 5 + self.rotation_columns_number]

            correct_maker_posi = self.calcurate_correct_maker_posi(imu_position, featureCol)
            foot_maker_position_X.append(correct_maker_posi[0])
            foot_maker_position_Y.append(correct_maker_posi[1])
            foot_maker_position_Z.append(correct_maker_posi[2])
        new_df[self.foot_marker_col_name[0]] = foot_maker_position_X
        new_df[self.foot_marker_col_name[1]] = foot_maker_position_Y
        new_df[self.foot_marker_col_name[2]] = foot_maker_position_Z

        return new_df


    def nan_check_per_row(self, df_iloc):
        """1行にNanが含まれているかチェックする
        Args: 
            df_iloc: pandas.dataframeのある行のimu or foot_marker部分
        Return: 1(Nanが含まれている場合), 0(Nanが含まれていない場合)
        """
        if 1 <= df_iloc.isna().sum():
            return 1
        else:
            return 0


    def interpolation_imu(self, new_df, i, nan_include_row_num, col_num_except_foot_marker):
        """dfのimuデータに対して線形補間の処理を実際に行う。欠損を含む任意の連続行数を補間可能。
        Args: 
            new_df: 欠損を含むpandas.dataframe
            i: 欠損を含まないフレーム。i-1フレーム以下で欠損を含んでいる。
            nan_include_row_num: nanを含む連続行数
            col_num_except_foot_marker: foot markerを除く列数（imuとtimeの列数）
        Return: 
            new_df: imu部分の線形補間がなされたpandas.dataframe
        """
        N = nan_include_row_num+1
        for j in range(1, N):# 行方向
            for k in range(col_num_except_foot_marker):# 列方向
                if math.isnan(float(new_df.iloc[i-j, k])):
                    new_df.iloc[i-j, k] = j*float(new_df.iloc[i-N, k])/(N) + (N-j)*float(new_df.iloc[i, k])/(N)

        return new_df


    def interpolation_foot_marker(self, new_df, i, nan_include_row_num, col_num_except_foot_marker, new_df_cols_number):
        """dfのfoot markerデータに対して線形補間の処理を実際に行う。欠損を含む任意の連続行数を補間可能。
        Args: 
            new_df: 欠損を含むpandas.dataframe
            i: 欠損を含まないフレーム。i-1フレーム以下で欠損を含んでいる。
            nan_include_row_num: nanを含む連続行数
            col_num_except_foot_marker: foot markerを除く列数（imuとtimeの列数）
            new_df_cols_number: 入力のpandas.dataframeの列数
        Return: 
            new_df: foot marker部分の線形補間がなされたpandas.dataframe
        """
        N = nan_include_row_num+1
        for j in range(1, N):# 行方向
            for k in range(col_num_except_foot_marker, new_df_cols_number):# 列方向
                if math.isnan(float(new_df.iloc[i-j, k])):
                    new_df.iloc[i-j, k] = j*float(new_df.iloc[i-N, k])/(N) + (N-j)*float(new_df.iloc[i, k])/(N)

        return new_df


    def interpolation_all_process(self, new_df, new_df_rows_number, new_df_cols_number):
        """
        Args :
            new_df(pandas.dataframe) : include imu maker and calcurated foot maker information
            new_df_cols_number(int) : number of new_df columns
        Returns :
            new_df(pandas.dataframe) : added interpolation to rack of only one row 
        """
        col_num_except_foot_marker = new_df_cols_number - len(self.foot_marker_col_name)

        print("interpolating for removing nan in imu")
        nan_include_row_num = 0
        for i in tqdm(range(new_df_rows_number)):
            if self.nan_check_per_row(new_df.iloc[i, :col_num_except_foot_marker]):
                nan_include_row_num += 1
            elif (self.nan_check_per_row(new_df.iloc[i, :col_num_except_foot_marker]) == 0) and (1 <= nan_include_row_num) and (nan_include_row_num <= max_allowed_continuous_nan_num ):
                new_df = self.interpolation_imu(new_df, i, nan_include_row_num, col_num_except_foot_marker)
                nan_include_row_num = 0
            elif (self.nan_check_per_row(new_df.iloc[i, :col_num_except_foot_marker]) == 0) and (1 <= nan_include_row_num) and (max_allowed_continuous_nan_num < nan_include_row_num):
                nan_include_row_num = 0
            elif (self.nan_check_per_row(new_df.iloc[i, :col_num_except_foot_marker]) == 0) and (0 == nan_include_row_num):
                pass

        print("interpolating for removing nan in foot marker")
        nan_include_row_num = 0
        for i in tqdm(range(new_df_rows_number)):
            if self.nan_check_per_row(new_df.iloc[i, col_num_except_foot_marker:]):
                nan_include_row_num += 1
            elif (self.nan_check_per_row(new_df.iloc[i, col_num_except_foot_marker:]) == 0) and (1 <= nan_include_row_num) and (nan_include_row_num <= max_allowed_continuous_nan_num ):
                new_df = self.interpolation_foot_marker(new_df, i, nan_include_row_num, col_num_except_foot_marker, new_df_cols_number)
                nan_include_row_num = 0
            elif (self.nan_check_per_row(new_df.iloc[i, col_num_except_foot_marker:]) == 0) and (1 <= nan_include_row_num) and (max_allowed_continuous_nan_num < nan_include_row_num):
                nan_include_row_num = 0
            elif (self.nan_check_per_row(new_df.iloc[i, col_num_except_foot_marker:]) == 0) and (0 == nan_include_row_num):
                pass

        return new_df


if __name__ == "__main__":
    preprocess_mocap_file = PreprocessMocapFile()
    preprocess_mocap_file.preprocess_all()

