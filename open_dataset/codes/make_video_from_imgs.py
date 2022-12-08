# 連番画像（1スタートでなくてもいいはず）から動画を作成できる
# プログラム中の文字列をいい感じに調節して使用する

import glob
import cv2 as cv

def create_movie(dir_path):
    output = dir_path + '/single_model.mp4'
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    outfh = cv.VideoWriter(output, fourcc, 24, (1080, 1920))
    for photo_name in sorted(glob.glob(dir_path + 'drawed_img_single_model/*.png')):
        im = cv.imread(photo_name)
        outfh.write(im)
    outfh.release()

if __name__ == '__main__':
    create_movie('test1_1129_1619/')