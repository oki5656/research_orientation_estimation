# このファイルは次歩推定結果をスマホで撮影した画像上に描画するプログラムである．
# 入力にはsynchronizeされたIMUとビデオ（画像），必要な予測機の分だけの重みファイルが必要．
# ビデオ→画像への分割はffmpeg, IMU(.mat)→IMU(.csv)への変換はmatlabを想定する

# このファイルは次歩推定結果をスマホ画角を模した領域にプロットし濃淡画像（2Dヒストグラム？）を作成する

import os
import re
import sys
import cv2
import glob
import torch
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from os.path import join
from models import choose_model, MODEL_DICT


selected_train_columns = ['gyroX', 'gyroY', 'gyroZ', 'accX', 'accY', 'accZ']
selected_correct_columns = ['pX', 'pY', 'pZ', 'qW', 'qX', 'qY', 'qZ', 'imu_position_x', 'imu_position_y', 'imu_position_z']

class NextStepVectorHistogram():
    def __init__(self, args):
        self.selected_IMU_columns = ['X_acc', 'Y_acc', 'Z_acc', 'X_ang', 'Y_ang', 'Z_ang']
        self.drawed_img_dir = args.drawed_img_dir
        self.csv_path = args.csv_path
        self.weight_path = args.weight_path


    def process_all(self):
        """全ての処理を行う
        """
        os.makedirs(self.drawed_img_dir, exist_ok=True)
        df = self.dataloader(self.csv_path)
        self.predict_and_draw(self.weight_path)


    def dataloader(self, path):
        """csvのパスを入力に加速度、各速度に相当する列のみのpandas-dataframeを出力
        Args : 
            path : path to csv file which include IMU time series data
        Return : 
            df(pd.dataframe) : time-series data of acceleration and angular velocity
        """
        assert os.path.isfile(path), "The data csv path you set is not correct. Youshoudl check the path."
        df = pd.read_csv(path)
        df = df[self.selected_IMU_columns]
        self.all_frame_num = len(df)
        
        return df


    def predict_and_draw(self, weight_path):
        weight_file_name = os.path.basename(weight_path)
        if "hiddensize" in weight_file_name:
            hidden_size = int(re.search(r'hiddensize(.+)_seq', weight_file_name).group(1))
        if "num_layer" in weight_file_name: 
            num_layer = int(re.search(r'num_layers(.+)_hid', weight_file_name).group(1))
        if "nhead" in weight_file_name:
            nhead = int(re.search(r'nhead(.+)_', weight_file_name).group(1))
        else:
            nhead = self.nhead

        if "seq" in weight_file_name:
            sequence_length = int(re.search(r'seq(.+)_pre', weight_file_name).group(1))
        else:
            sequence_length = self.sequence_length

        model = choose_model(args.model, len(selected_train_columns), hidden_size,
                             num_layer, nhead, 3, sequence_length, args.input_shift)
        model = model.float()
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.load_state_dict(torch.load(weight_path))
        model.eval()
        .................................




if __name__ == '__main__':
    cwd = os.getcwd()
    print("now directory is", cwd)

    parser = argparse.ArgumentParser(description='training argument')
    parser.add_argument('--weight_path', type=str, default= "C:/Users/admin/Desktop/orientation_estimation/open_dataset/images/2211140648_lstm_seq27_pred21/trial13_MAE3.00714_MDE70.11617_lr0.001630_batch_size_8_num_layers3_hiddensize76_seq27_pred21.pth")
    parser.add_argument('--csv_path', type=str, default="C:/Users/admin/Desktop/orientation_estimation/open_dataset/datasets/large_space/nan_removed/interpolation_under_15/harf_test_20220809_002_nan_under15_nan_removed.csv", help='specify csv file path.')
    parser.add_argument('--drawed_img_dir', type=str, default="C:/Users/admin/Desktop/orientation_estimation/open_dataset/experiment_result/test_field_experiment/test7_1209_1916_no_block/drawed_img_multi_model", help='specify drawed images folder path.')
    parser.add_argument('--model', type=str, default="lstm", help=f'choose model from {MODEL_DICT.keys()}')
    parser.add_argument('--hidden_size', type=int, default=76, help='select hidden size of LSTM')
    parser.add_argument('-l', '--num_layer', type=int, default=3, help='select number of layer for LSTM')
    parser.add_argument('-n', '--nhead', type=int, default=3, help='select nhead for Transformer')
    parser.add_argument('-i', '--input_shift', type=int, default=1, help='select number of input shift Transformer')
    parser.add_argument('-s', '--sequence_length', type=int, default=27, help='select train data sequence length')
    parser.add_argument('-p', '--pred_future_time', type=int, default=33, help='How many seconds later would you like to predict?')
    parser.add_argument('--horizontal_img_range', type=float, default=36.1, help='horizontal image range')
    parser.add_argument('--vertical_img_range', type=float, default=58.7, help='horizontal image range')
    parser.add_argument("--is_train_smp2foot", type=str, default="true", help='select training Position2Position or smpPosition2footPosition')

    args = parser.parse_args()
    histogram = NextStepVectorHistogram(args)
    histogram.process_all()
