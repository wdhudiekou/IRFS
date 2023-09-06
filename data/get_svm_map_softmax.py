import os
import cv2
import kornia
import torch
import numpy as np

def softmax(map1, map2, c):
    exp_x1 = np.exp(map1*c)
    exp_x2 = np.exp(map2*c)
    exp_sum = exp_x1 + exp_x2
    map1 = exp_x1/exp_sum
    map2 = exp_x2/exp_sum
    return map1, map2

def vsm(img):
    his = np.zeros(256, np.float64)
    for i in range(img.shape[0]): # 256
        for j in range(img.shape[1]): # 256
            his[img[i][j]] += 1
    sal = np.zeros(256, np.float64)
    for i in range(256):
        for j in range(256):
            sal[i] += np.abs(j - i) * his[j]
    map = np.zeros_like(img, np.float64)
    for i in range(256):
        map[np.where(img == i)] = sal[i]
    if map.max() == 0:
        return np.zeros_like(img, np.float64)
    return map / (map.max())

def torch_vsm(img):
    his = torch.zeros((256, 320),  dtype=torch.float32).cuda()
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            his[img[i, j].item()] += 1
    sal = torch.zeros((256, 320), dtype=torch.float32).cuda()
    for i in range(256):
        for j in range(320):
            sal[i] += abs(j - i) * his[j].item()
    map = torch.zeros_like(img, dtype=torch.float32)
    for i in range(256):
        map[torch.where(img == i)] = sal[i]
    if map.max() == 0:
        return torch.zeros_like(img, dtype=torch.float32)
    return map / (map.max())

if __name__ == '__main__':


    T_path = "/home/zongzong/WD/Datasets/RGBT/VT5000/Train/T/"
    RGB_path = "/home/zongzong/WD/Datasets/RGBT/VT5000/Train/RGB/"

    T_file_list = sorted(os.listdir(T_path))
    RGB_file_list = sorted(os.listdir(RGB_path))

    T_map_path = "/home/zongzong/WD/Datasets/RGBT/VT5000/Train/T_map_soft/"
    RGB_map_path = "/home/zongzong/WD/Datasets/RGBT/VT5000/Train/RGB_map_soft/"

    if not os.path.exists(T_map_path):
        os.makedirs(T_map_path)
    if not os.path.exists(RGB_map_path):
        os.makedirs(RGB_map_path)

    for idx, (T_filename, RGB_filename) in enumerate(zip(T_file_list, RGB_file_list)):

        T_filepath = os.path.join(T_path, T_filename)
        RGB_filepath = os.path.join(RGB_path, RGB_filename)

        img_T = cv2.imread(T_filepath, cv2.IMREAD_GRAYSCALE)  # uint8 (256, 256)
        img_RGB = cv2.imread(RGB_filepath, cv2.IMREAD_GRAYSCALE)  # uint8 (256, 256)

        map_T = vsm(img_T)
        map_RGB = vsm(img_RGB)

        w_T, w_RGB = softmax(map_T, map_RGB, c=5)

        img_w_T = (w_T * 255).astype(np.uint8)
        img_w_RGB = (w_RGB * 255).astype(np.uint8)

        T_save_name = os.path.join(T_map_path, T_filename)
        RGB_save_name = os.path.join(RGB_map_path, RGB_filename)

        cv2.imwrite(T_save_name, img_w_T)
        cv2.imwrite(RGB_save_name, img_w_RGB)

