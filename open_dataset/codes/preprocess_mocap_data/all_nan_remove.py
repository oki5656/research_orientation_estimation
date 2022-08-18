# This program can remove rows that include nan. This program intend to use after shnchronizing.

import os
import sys
import pandas as pd
from tqdm import tqdm
from os.path import join

#######################################################################################################################################
synchronized_file_path = join("..", "datasets", "large_space", "synchronize", "Take 2022-08-09 08.31.59 PM_003_v1_topbuttom_cut_foot_maker_processedsynchronize.csv")
new_nan_removed_dir = join("..", "datasets", "large_space", "nan_removed")
new_nan_removed_file_name = os.path.splitext(os.path.basename(synchronized_file_path))[0] + "_nan_removed.csv"
####################################################################################################################################### 

class NanRemove(object):
    def __init__(self) -> None:
        cwd = os.getcwd()
        print("now directory is", cwd)
        self.synchro_df = pd.read_csv(synchronized_file_path)
        self.synchro_rows, self.synchro_cols = self.synchro_df.shape
    
    def process_all(self):
        self.__nan_remove()
        os.makedirs(new_nan_removed_dir, exist_ok=True)
        self.synchro_df.to_csv(join(new_nan_removed_dir, new_nan_removed_file_name))
        print("Preorocess was all completed and synchronized csv file was created. ")

    def __nan_remove(self):
        for i in tqdm(reversed(range(self.synchro_rows))):
            if 1 <= self.synchro_df.iloc[i, :].isna().sum():
                self.synchro_df = self.synchro_df.drop(self.synchro_df.index[i])




if __name__ == "__main__":
    nan_remove = NanRemove()
    nan_remove.process_all()