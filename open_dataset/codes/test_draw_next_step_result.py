import os
import sys
import cv2
import glob
import torch
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from os.path import join
# sys.path.append('../')
from models import choose_model, MODEL_DICT
# from draw_next_step_result import 

# selected_train_columns = ['gyroX', 'gyroY', 'gyroZ', 'accX', 'accY', 'accZ']
# selected_correct_columns = ['pX', 'pY', 'pZ', 'qW', 'qX', 'qY', 'qZ', 'imu_position_x', 'imu_position_y', 'imu_position_z']


class DrawNextStepResult():
    def __init__(self, args):
        self.horizontal_img_range = args.horizontal_img_range
        self.vertical_img_range = args.vertical_img_range

        self.drawed_img_dir = args.drawed_img_dir
        self.images_dir = args.images_dir
        self.all_frame_num = 0
        self.images_pathes = glob.glob(join(self.images_dir, "*"))
        self.selected_IMU_columns = ['X_acc', 'Y_acc', 'Z_acc', 'X_ang', 'Y_ang', 'Z_ang']

        self.landmarks_coodinates = [
            (0.5, -0.7, -3.5), (0.5, -0.7, -3.0), (0.5, -0.7, -2.5), (0.5, -0.7, -2.0),
            (0, -0.7, -3.5), (0, -0.7, -2.5), (0, -0.7, -1.5),
            (-0.5, -0.7, -3.5), (-0.5, -0.7, -3.0), (-0.5, -0.7, -2.5), (-0.5, -0.7, -2.0)
        ]
        self.coodinates_num = len(self.landmarks_coodinates)


    def process_all(self):
        """全ての処理
        """
        os.makedirs(self.drawed_img_dir, exist_ok=True)
        self.predict_draw()


    def load_imu(self, frame):
        """frameを受け取ってbatch_x_df_np(sequence_length, batch_size, input_size)を返す
        Args : 
            frame : 
        Returns : 
            imu : imu出力。shapeは(sequence_length, 6)
        """
        imu = np.array(self.df[frame - self.sequence_length: frame])

        return imu


    def pridict(self, frame):
        """frameを受け取ってbatch_x_df_np(sequence_length, batch_size, input_size)を返す
        Args : 
            frame : 
        Returns : 
            next_step_vector : スマホ座標系次歩推定結果。shapeは(?)
        """
        next_step_vector  = self.landmarks_coodinates[frame]

        return next_step_vector #torch.Size([3])


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


    def draw_next_step(self, frame, next_step_vector):
        img_path = self.images_pathes[frame]
        img = cv2.imread(img_path)
        img_name = os.path.basename(img_path)
        height, width, ch = img.shape
        draw_canvas = np.zeros((height, width, ch))

        u, v = self.convert_next_step_position(next_step_vector, height, width)
        if 0 <= u and u < width and 0 <= v and v < height:
            cv2.circle(draw_canvas,
                center=(u, v),
                radius=(height+width)//100,
                color=(0, 0, 255),
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
                radius=(height+width)//150,
                color=(0, 0, 255),
                thickness=-1,
                lineType=cv2.LINE_AA,
            shift=0)

        cv2.putText(draw_canvas,
            text=f'X={next_step_vector[0]}, Y={next_step_vector[1]}, Z={next_step_vector[2]}',
            org=(20, height-20),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1.0,
            color=(0, 255, 0),
            thickness=2,
            lineType=cv2.LINE_4)

        blend_img = cv2.addWeighted(img, 0.5, draw_canvas.astype(np.uint8), 0.5, 0)

        return blend_img, img_name


    def predict_draw(self):
        for frame in tqdm(range(self.coodinates_num)):
            next_step_vector = self.pridict(frame)
            blend_img, img_name = self.draw_next_step(frame, next_step_vector)
            cv2.imwrite(join(self.drawed_img_dir, img_name), blend_img)


if __name__ == '__main__':

    cwd = os.getcwd()
    print("now directory is", cwd)

    parser = argparse.ArgumentParser(description='training argument')

    parser.add_argument('--images_dir', type=str, default="C:/Users/admin/Desktop/orientation_estimation/open_dataset/experiment_result/test_draw_next_step_result/images", help='specify images folder path.')
    parser.add_argument('--drawed_img_dir', type=str, default="C:/Users/admin/Desktop/orientation_estimation/open_dataset/experiment_result/test_draw_next_step_result/drawed_landmarks", help='specify drawed images folder path.')
    parser.add_argument('--horizontal_img_range', type=float, default=36.1, help='horizontal image range')
    parser.add_argument('--vertical_img_range', type=float, default=58.7, help='horizontal image range')
    parser.add_argument("--is_train_smp2foot", type=str, default="true", help='select training Position2Position or smpPosition2footPosition')

    args = parser.parse_args()
    draw_result = DrawNextStepResult(args)
    draw_result.process_all()