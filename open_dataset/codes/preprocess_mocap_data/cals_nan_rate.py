import os
import pandas as pd
from os.path import join



################################################################################
csv_file_path = join("csvdata", "all_complete_001_v1", "Take 2022-08-09 08.31.59 PM_003_v1_topbuttom_cut_all_complete.csv")
# csv_file_path = join("csvdata", "preprocessed", "Take 2022-08-09 08.31.59 PM_001_v1_topbuttom_cut.csv")
################################################################################
cwd = os.getcwd()
print("now directory is", cwd)
df = pd.read_csv(csv_file_path)

print(df)
