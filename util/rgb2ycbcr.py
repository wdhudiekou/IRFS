import random
import cv2
import os
import numpy as np

def rgb2ycbcr(img, only_y=True):
    '''same as matlab rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [65.481, 128.553, 24.966]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786],
                              [24.966, 112.0, -18.214]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)


if __name__ == '__main__':

    vi_path = "/home/zongzong/WD/Datasets/RGBT/VT5000/Train/T/"

    vi_file_list = sorted(os.listdir(vi_path))

    save_vi_path = "/home/zongzong/WD/Datasets/RGBT/VT5000/Train/T_Y/"

    if not os.path.exists(save_vi_path):
        os.makedirs(save_vi_path)

    for idx, vi_filename in enumerate(vi_file_list):

        vi_filepath = os.path.join(vi_path, vi_filename)
        img_vi = cv2.imread(vi_filepath)
        # img_vi = cv2.resize(img_vi,(352, 352), interpolation=cv2.INTER_LINEAR)

        vi_gray = rgb2ycbcr(img_vi)
        vi_save_name = os.path.join(save_vi_path, vi_filename)
        cv2.imwrite(vi_save_name, vi_gray)





