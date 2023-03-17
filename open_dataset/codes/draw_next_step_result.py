# このファイルは次歩推定結果をスマホで撮影した画像上に描画するプログラムである．
# 入力にはsynchronizeされたIMUとビデオ（画像），必要な予測機の分だけの重みファイルが必要．
# ビデオ→画像への分割はffmpeg, IMU(.mat)→IMU(.csv)への変換はmatlabを想定する

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

class DrawNextStepResult():
    def __init__(self, args):
        self.model = args.model
        self.hidden_size = args.hidden_size
        self.num_layer = args.num_layer
        self.nhead = args.nhead
        self.input_shift = args.input_shift
        self.sequence_length = args.sequence_length
        self.pred_future_time = args.pred_future_time
        self.horizontal_img_range = args.horizontal_img_range
        self.vertical_img_range = args.vertical_img_range

        self.weight_path_list = []
        self.weight_path_list.append(args.weight_path1)
        # self.weight_path_list.append(args.weight_path2)
        # self.weight_path_list.append(args.weight_path3)
        
        self.csv_path = args.csv_path
        self.drawed_img_dir = args.drawed_img_dir
        self.images_dir = args.images_dir
        self.all_frame_num = 0
        self.g = 9.80665 # fixed
        self.Normalization_or_Not = "Normalization"
        self.images_pathes = glob.glob(join(self.images_dir, "*"))
        # self.selected_IMU_columns = ['X_acc', 'Y_acc', 'Z_acc', 'X_ang', 'Y_ang', 'Z_ang']
        self.selected_IMU_columns = ['X_ang', 'Y_ang', 'Z_ang', 'X_acc', 'Y_acc', 'Z_acc']
        self.df = self.dataloader(self.csv_path)


    def process_all(self):
        """全ての処理を行う
        """
        assert len(self.images_pathes) == self.all_frame_num, "number of images and IMU frames is not equal"
        os.makedirs(self.drawed_img_dir, exist_ok=True)
        self.predict_draw()


    def dataloader(self, path):
        """csvのパスを入力に加速度、各速度に相当する列のみのpandas-dataframeを出力
        Args : 
            path : path to csv file which include IMU time series data
        Return : 
            df(pd.dataframe) : time-series data of acceleration and angular velocity
        """
        print("#######", os.path.isfile(path))
        df = pd.read_csv(path)
        df = df[self.selected_IMU_columns]
        self.all_frame_num = len(df)
        
        return df


    def acceleration_normalization(self, imu):
        """加速度を受け取り正規化（合力方向から距離が9.8分になるようにそれぞれのベクトルの要素から引く）する
        Args: 
            out_x : (batchsize, seq_length, 要素数)で構成される加速度と角速度シーケンス.ndarray
        output: 
            out_x : (batchsize, seq_length, 要素数)で構成される "正規化された" 加速度と角速度シーケンス
        """
        seq_len, element = imu.shape
        for j in range(seq_len):
            l2norm = np.sqrt(imu[j, 3]**2+imu[j, 4]**2+imu[j, 5]**2)
            imu[j, 3] -= self.g*imu[j, 3]/l2norm
            imu[j, 4] -= self.g*imu[j, 4]/l2norm
            imu[j, 5] -= self.g*imu[j, 5]/l2norm
    
        return imu


    def load_imu(self, frame, weight_file_name):
        """frameを受け取ってbatch_x_df_np(sequence_length, batch_size, input_size)を返す
        Args : 
            frame : 連番画像中の画像の番号．
        Returns : 
            imu : imu出力。shapeは(sequence_length, 6)
        """
        if "seq" in weight_file_name:
            sequence_length = int(re.search(r'seq(.+)_pre', weight_file_name).group(1))
        else:
            sequence_length = self.sequence_length
        imu = np.array(self.df[frame - sequence_length: frame])
        if self.Normalization_or_Not == "Normalization":
            imu = self.acceleration_normalization(imu)

        return imu


    def pridict(self, frame, weight_path):
        """frameを受け取ってbatch_x_df_np(sequence_length, batch_size, input_size)を返す
        Args : 
            frame : 連番画像中の画像の番号．
        Returns : 
            next_step_vector : スマホ座標系次歩推定結果。shapeは(3)
        """
        weight_file_name = os.path.basename(weight_path)
        if "hiddensize" in weight_file_name:
            hidden_size = int(re.search(r'hiddensize(.+)_seq', weight_file_name).group(1))
        if "num_layer" in weight_file_name: 
            num_layer = int(re.search(r'num_layers(.+)_hid', weight_file_name).group(1))
        if "nhead" in weight_file_name:
            nhead = int(re.search(r'nhead(.+)_num', weight_file_name).group(1))
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
        imu  = self.load_imu(frame, weight_file_name)

        if self.model == "transformer_encdec":
            src = torch.from_numpy(imu[:sequence_length-self.input_shift, :].astype(np.float32)).unsqueeze(1)
            tgt = torch.from_numpy(imu[self.input_shift:, :].astype(np.float32)).unsqueeze(1)
            next_step_vector = model(src=src.float().to(device), tgt=tgt.float().to(device)) 
        elif self.model == "lstm" or args.model == "transformer_enc" or args.model == "imu_transformer":
            input = torch.from_numpy(imu).unsqueeze(1)
            next_step_vector = model(input.float().to(device))

        return next_step_vector, weight_file_name


    def convert_next_step_position(self, next_step_vector, img_height, img_width):
        """スマートフォン座標系次歩推定値を画像に描画可能な座標に変換する
        Args : 
            next_step_vector :(x, y, z) next step vector
            img_height : image height
            img_width : image width
        Return :
            x : x next-step position of image
            y : y next-step position of image
        """
        X, Y, Z = next_step_vector[0]*1000, next_step_vector[1]*1000, next_step_vector[2]*1000
        Z = -1*Z

        
        normlized_coodinate_plane_h_max = np.tan(np.deg2rad(self.horizontal_img_range/2))*2
        normlized_coodinate_plane_v_max = np.tan(np.deg2rad(self.vertical_img_range/2))*2

        img_magnification_rate_h = img_width/normlized_coodinate_plane_h_max
        img_magnification_rate_v = img_height/normlized_coodinate_plane_v_max

        x = X/Z
        y = Y/Z

        u = x*img_magnification_rate_h+img_width//2
        v = -y*img_magnification_rate_v+img_height//2

        return int(u), int(v)


    def draw_next_step(self, predict_number, img, draw_canvas, next_step_vector, weight_file_name):
        """何も描画されていないキャンバスに次歩推定ベクトル（実際には点というか円）を描画する
        Args:
            predict_number: 何番目の予測器を使用するか
            img: 次歩推定ベクトルを描画する画像
            draw_canvas: 次歩推定ベクトルが描画されるor描画された画像
            next_step_vector: スマホ座標系次歩推定結果。shapeは(3)
            weight_file_name: 重みファイル名
        Return: 
            draw_canvas: 次歩推定ベクトルが描画された画像
        """
        height, width, _ = img.shape

        color_list = [
                        (0, 120, 238),
                        (50, 205, 50), 
                        (255, 130, 0), 
                        (255, 0, 0)
                      ]

        u, v = self.convert_next_step_position(next_step_vector, height, width)
        if 0 <= u and u < width and 0 <= v and v < height:
            cv2.circle(draw_canvas,
                center=(u, v),
                radius=(height+width)//50,
                color=color_list[predict_number],
                thickness=-1,
                lineType=cv2.LINE_AA,
            shift=0)
        else:
            if u < 0:
                u = 0
            if width <= u:
                u = width
            if v < 0:
                v = 0
            if height <= v:
                v = height
            cv2.circle(draw_canvas,
                center=(u, v),
                radius=(height+width)//50,
                color=color_list[predict_number],
                thickness=-1,
                lineType=cv2.LINE_AA,
            shift=0)

        if "pred" in weight_file_name:
            pred_future_time = int(re.search(r'pred(.+).pth', weight_file_name).group(1))
        else:
            pred_future_time = self.pred_future_time
        cv2.putText(draw_canvas,
            text=f'prediction time: {pred_future_time/30} ',
            org=(20+320*predict_number, height-20),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1.0,
            color=color_list[predict_number],
            thickness=2,
            lineType=cv2.LINE_8)

        return draw_canvas

    
    def read_img(self, frame):
        """連番画像の中から指定されたフレームの画像を読み込み，
           画像(np.array)，ファイル名，白紙の画像(np.array)を出力
        Args: 
            frame: 連番画像中の処理に用いられる画像の番号
        returns: 
            img(np.array): 読み込んだ画像
            img_name(str): 読み込んだ画像のファイル名
            draw_canvas(np.array): 値がすべて0で読み込んだ画像と同じshapeの画像
        """
        img_path = self.images_pathes[frame]
        img = cv2.imread(img_path)
        img_name = os.path.basename(img_path)
        height, width, ch = img.shape
        draw_canvas = np.zeros((height, width, ch))

        return img, img_name, draw_canvas


    def predict_draw(self):
        """それぞれの画像に次歩推定ベクトルを描画する
        """
        predict_model_num = len(self.weight_path_list)
        for frame in tqdm(range(self.sequence_length, self.all_frame_num-self.pred_future_time)):
            img, img_name, draw_canvas = self.read_img(frame)
            for predict_number in range(predict_model_num):
                next_step_vector, weight_file_name = self.pridict(frame, self.weight_path_list[predict_number])
                draw_canvas = self.draw_next_step(predict_number, img, draw_canvas, next_step_vector, weight_file_name)

            blend_img = cv2.addWeighted(img, 0.5, draw_canvas.astype(np.uint8), 0.5, 0)
            cv2.imwrite(join(self.drawed_img_dir, img_name), blend_img)


if __name__ == '__main__':
    cwd = os.getcwd()
    print("now directory is", cwd)

    parser = argparse.ArgumentParser(description='training argument')
    parser.add_argument('--weight_path1', type=str, default= "C:/Users/admin/Desktop/orientation_estimation/open_dataset/images5/2201180522_lstm_seq27_pred21/trial9_MAE3.98126_MDE90.82089_lr0.020077_batch8_nhead3_num_layers3_hiddensize54_seq27_pred21.pth", help='specify weight file path 1.')
    parser.add_argument('--weight_path2', type=str, default= "C:/Users/admin/Desktop/orientation_estimation/open_dataset/images/2211221722_lstm_seq21_pred33/trial22_MAE4.88546_MDE137.91135_lr0.004426_batch_size_8_num_layers4_hiddensize45_seq21_pred33.pth", help='specify weight file path 2.')
    parser.add_argument('--weight_path3', type=str, default= "C:/Users/admin/Desktop/orientation_estimation/open_dataset/images/2211281428_lstm_seq15_pred45/trial21_MAE5.08919_MDE222.3561_lr0.006873_batch_size_8_num_layers4_hiddensize44_seq15_pred45.pth", help='specify weight file path 3.')
    parser.add_argument('--csv_path', type=str, default="C:/Users/admin/Desktop/orientation_estimation/open_dataset/experiment_result/test_field_experiment/test16_corner_fast/cut_corner_fast_ooki_sensorlog_20221225_173307.csv", help='specify csv file path.')
    parser.add_argument('--images_dir', type=str, default="C:/Users/admin/Desktop/orientation_estimation/open_dataset/experiment_result/test_field_experiment/test16_corner_fast/cut_images", help='specify images folder path.')
    parser.add_argument('--drawed_img_dir', type=str, default="C:/Users/admin/Desktop/orientation_estimation/open_dataset/experiment_result/test_field_experiment/test16_corner_fast/drawed_img_single_model", help='specify drawed images folder path.')
    parser.add_argument('--model', type=str, default="lstm", help=f'choose model from {MODEL_DICT.keys()}')
    parser.add_argument('--hidden_size', type=int, default=76, help='select hidden size of LSTM')
    parser.add_argument('-l', '--num_layer', type=int, default=3, help='select number of layer for LSTM')
    parser.add_argument('-n', '--nhead', type=int, default=3, help='select nhead for Transformer')
    parser.add_argument('-i', '--input_shift', type=int, default=1, help='select number of input shift Transformer')
    parser.add_argument('-s', '--sequence_length', type=int, default=27, help='select train data sequence length')
    parser.add_argument('-p', '--pred_future_time', type=int, default=33, help='How many seconds later would you like to predict?')
    parser.add_argument('--horizontal_img_range', type=float, default=36.2, help='horizontal image range')
    parser.add_argument('--vertical_img_range', type=float, default=63.1, help='horizontal image range')
    parser.add_argument("--is_train_smp2foot", type=str, default="true", help='select training Position2Position or smpPosition2footPosition')

    args = parser.parse_args()
    draw_result = DrawNextStepResult(args)
    draw_result.process_all()
