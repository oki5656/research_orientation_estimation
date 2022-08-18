# mocap, imuファイルから同期されたデータを作成する。足のマーカー座標が整形されたものを用いる
# mocap, imuそれぞれのスタート、エンドポイントを同一時刻として同期させる
# imuが100Hz、mocapが30Hzより全体としては30Hzのデータを作成する
# 入力ファイルのtimestampはないものを使用する（あらかじめ削除しておくこと）

import os
import sys
import pandas as pd
from tqdm import tqdm
from os.path import join

#######################################################################################################################################
# mocap_file_path = join("csvdata", "all_complete_001_v1", "Take 2022-08-09 08.31.59 PM_003_v1_topbuttom_cut_all_complete.csv")
mocap_file_path  = join("..", "datasets", "large_space", "mocap", "foot_maker_processed", "Take 2022-08-09 08.31.59 PM_002_v1_topbuttom_cut_foot_maker_processed.csv")
imu_file_path = join("..", "datasets", "large_space", "smartphone_imu", "sensorlog_20220809_210823_trial002_topbuttom_cut.csv")
new_synchronize_dir = join("..", "datasets", "large_space", "synchronize")
new_synchronize_file_name = os.path.splitext(os.path.basename(mocap_file_path))[0] + "synchronize.csv"
XYZorQuaternion = "Quaternion"
new_timestamp_Hz = 30

#######################################################################################################################################

class Synchronize(object):
    def __init__(self) -> None:
        cwd = os.getcwd()
        print("now directory is", cwd)
        self.mocap_df = pd.read_csv(mocap_file_path)
        self.imu_df = pd.read_csv(imu_file_path)
        self.mocap_rows, self.mocap_cols = self.mocap_df.shape
        self.imu_rows, self.imu_cols = self.imu_df.shape
        self.RotationColNumber = {"XYZ" : 3, "Quaternion" : 4 }
        self.correct_mocap_cols = self.RotationColNumber[XYZorQuaternion] + 6
        self.all_time_length = (self.imu_rows-1)*0.01


    def process_all(self) -> None:
        self.__check_number_of_input_files_columns(self.mocap_cols, self.imu_cols)
        self.__add_imu_timestamp(self.imu_df)
        self.__mocap_timestamp(self.mocap_df)
        new_df = self.__make_new_dataframe()
        new_df = self.__synchronize(new_df, self.mocap_df, self.imu_df)

        # new_df = self.__add_mocap_dataframe(new_df)
        # new_df = self.__add_imu_dataframe(new_df)
        os.makedirs(new_synchronize_dir, exist_ok=True)
        new_df.to_csv(join(new_synchronize_dir, new_synchronize_file_name))
        print("Preorocess was all completed and synchronized csv file was created. ")

    def __check_number_of_input_files_columns(self, mocap_cols, imu_cols) -> None:
        """Check number of mocap, imu file columns. Number of imu file columns is 6, mocap is 10.
        Args:
            mocap_df, imu_df
        Returns: 
            none
        """
        correct_mocap_cols = self.RotationColNumber[XYZorQuaternion] + 6
        correct_imu_cols = 6
        if (mocap_cols != correct_mocap_cols) or (imu_cols != correct_imu_cols):
            print("moccap or imu file columns is not corret. program is ended.")
            sys.exit()

    def __add_imu_timestamp(self, imu_df):
        """add timestamp to imu_df
        Args :
            imu_df : original imu_df which have smp_acc, smp_ang
        Returns :
            imu_df : added timestamp based on imu_df and 100 Hz.
        """
        imu_timestamp = [i*0.01 for i in range(self.imu_rows)]
        imu_df.insert(loc = 0, column= 'imu_timestamp', value= imu_timestamp)

        return imu_df

    def __mocap_timestamp(self, mocap_df):
        """add timestamp to mocap_df
        Args :

        Returns :
        """
        mocap_unit_length = self.all_time_length/(self.mocap_rows-1)
        self.mocap_timestamp = [i*mocap_unit_length for i in range(self.mocap_rows)]
        mocap_df.insert(loc = 0, column= 'mocap_timestamp', value= self.mocap_timestamp)

        return mocap_df

    def __make_new_dataframe(self):
        """ making new dataframe .
        Args :
            None
        Returns :
            new_df(pd.dataframe) : new pd.dataframe to create
        """
        # new_timestamp = [i/new_timestamp_Hz for i in range(self.mocap_rows)]
        new_df = pd.DataFrame(
            data={'timestamp': self.mocap_timestamp}
        )
        return new_df

    def __append_to_list(self, X_acc, Y_acc, Z_acc, X_ang, Y_ang, Z_ang, new_imu_row) -> list:
        x_acc, y_acc, z_acc, x_ang, y_ang, z_ang = new_imu_row[0], new_imu_row[1], new_imu_row[2], new_imu_row[3], new_imu_row[4], new_imu_row[5] 
        X_acc.append(x_acc)
        Y_acc.append(y_acc)
        Z_acc.append(z_acc)
        X_ang.append(x_ang)
        Y_ang.append(y_ang)
        Z_ang.append(z_ang)

        return X_acc, Y_acc, Z_acc, X_ang, Y_ang, Z_ang

    def __linear_interpolation(self, imu_front, imu_back, mocap_between):
        """imuとmocapのtimestampから線形補間をする。imu側にmocap timestampに合わせた新規値を作る。
        Args :
            imu_front :
            imu_back :
            mocap :
        Returns :


        """
        imu_front_time = imu_front[0]
        imu_back_time = imu_back[0]
        mocap_between_time = mocap_between[0]
        base_time_length = imu_back_time - imu_front_time
        front_time_length = mocap_between_time - imu_front_time
        back_time_length = imu_back_time - mocap_between_time
        front_rate = front_time_length/base_time_length
        back_rate = back_time_length/base_time_length

        new_imu_row = []
        for i in range(1, 7):
            new_imu_row.append(back_rate*imu_front[i] + front_rate*imu_back[i])
        
        return new_imu_row



    def __synchronize(self, new_df, mocap_df, imu_df):
        X_acc = []
        Y_acc = []
        Z_acc = []
        X_ang = []
        Y_ang = []
        Z_ang = []
        mocap_flag = 0
        print("start shnchronizing...")
        for searchFlag in tqdm(range(self.imu_rows-1)):
            front_time = imu_df["imu_timestamp"][searchFlag]
            back_time = imu_df["imu_timestamp"][searchFlag+1]
            mocap_feature_time = round(mocap_df["mocap_timestamp"][mocap_flag], 6)
            if front_time <= mocap_feature_time and mocap_feature_time <= back_time:
                new_imu_row = self.__linear_interpolation(imu_df.iloc[searchFlag, :], imu_df.iloc[searchFlag+1, :], mocap_df.iloc[mocap_flag, :])
                X_acc, Y_acc, Z_acc, X_ang, Y_ang, Z_ang = self.__append_to_list(X_acc, Y_acc, Z_acc, X_ang, Y_ang, Z_ang, new_imu_row)
                mocap_flag+=1

        new_df["X_acc"] = X_acc
        new_df["Y_acc"] = Y_acc
        new_df["Z_acc"] = Z_acc
        new_df["X_ang"] = X_ang
        new_df["Y_ang"] = Y_ang
        new_df["Z_ang"] = Z_ang

        new_df = pd.concat([new_df, mocap_df.iloc[:, 1:]], axis=1)

        return new_df
    

if __name__ == "__main__":
    synchronize_mocap_imu = Synchronize()
    synchronize_mocap_imu.process_all()
