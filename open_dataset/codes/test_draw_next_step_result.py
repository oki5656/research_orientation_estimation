# スマートフォン座標系における物体の位置が，画像上の想定する位置に描画されるかを確かめるためのプログラム
# self.landmarks_coodinatesにスマートフォン座標系の物体の座標を設定し，描画する画像，画像の画角（水平垂直）
# などを設定することで使用可能

import os
import cv2
import glob
import argparse
import numpy as np
from tqdm import tqdm
from os.path import join


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


    def draw_next_step(self, frame, next_step_vector, draw_canvas):
        """次歩推定ベクトルと座標（x, y, z）を描画する
        """
        # img_path = self.images_pathes[frame]
        # img = cv2.imread(img_path)
        # img_name = os.path.basename(img_path)
        height, width, ch = draw_canvas.shape
        # draw_canvas = np.zeros((height, width, ch))

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

        # cv2.putText(draw_canvas,
        #     text=f'X={next_step_vector[0]}, Y={next_step_vector[1]}, Z={next_step_vector[2]}',
        #     org=(20, height-20),
        #     fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        #     fontScale=1.0,
        #     color=(0, 255, 0),
        #     thickness=2,
        #     lineType=cv2.LINE_4)

        # blend_img = cv2.addWeighted(img, 0.5, draw_canvas.astype(np.uint8), 0.5, 0)

        return draw_canvas


    def predict_draw(self):
        """次歩推定を行い次歩推定ベクトルを描画する．そしてその結果画像を保存する．
        """
        img_path = self.images_pathes[5]
        img = cv2.imread(img_path)
        height, width, ch = img.shape
        draw_canvas = np.zeros((height, width, ch))

        for frame in tqdm(range(self.coodinates_num)):
            next_step_vector = self.pridict(frame)
            draw_canvas = self.draw_next_step(frame, next_step_vector, draw_canvas)

        blend_img = cv2.addWeighted(img, 0.5, draw_canvas.astype(np.uint8), 0.5, 0)
        cv2.imwrite(join(self.drawed_img_dir, "all_position_drawed.png"), blend_img)


if __name__ == '__main__':

    cwd = os.getcwd()
    print("now directory is", cwd)

    parser = argparse.ArgumentParser(description='training argument')

    parser.add_argument('--images_dir', type=str, default="C:/Users/admin/Desktop/orientation_estimation/open_dataset/experiment_result/test_draw_next_step_result/images2", help='specify images folder path.')
    parser.add_argument('--drawed_img_dir', type=str, default="C:/Users/admin/Desktop/orientation_estimation/open_dataset/experiment_result/test_draw_next_step_result/drawed_landmarks_all", help='specify drawed images folder path.')
    parser.add_argument('--horizontal_img_range', type=float, default=36.2, help='horizontal image range')
    parser.add_argument('--vertical_img_range', type=float, default=63.2, help='horizontal image range')
    parser.add_argument("--is_train_smp2foot", type=str, default="true", help='select training Position2Position or smpPosition2footPosition')

    args = parser.parse_args()
    draw_result = DrawNextStepResult(args)
    draw_result.process_all()