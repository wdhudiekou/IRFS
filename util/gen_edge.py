# -*- coding: utf-8 -*-
"""
@software: PyCharm
@file: gen_edge.py
@time: 2021/5/10 23:36
"""

import cv2
import os
import numpy as np

def sobel(img):
    '''
    edge detection based on sobel

    Parameters
    ----------
    img : TYPE
        the image input.
    threshold : TYPE
         varies for application [0 255].

    Returns
    -------
    mag : TYPE
        output after edge detection.

    '''
    G_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    G_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    rows = np.size(img, 0)
    columns = np.size(img, 1)
    mag = np.zeros(img.shape)
    for i in range(0, rows - 2):
        for j in range(0, columns - 2):
            v = sum(sum(G_x * img[i:i + 3, j:j + 3]))  # vertical
            h = sum(sum(G_y * img[i:i + 3, j:j + 3]))  # horizon
            mag[i + 1, j + 1] = np.sqrt((v ** 2) + (h ** 2))

    # for p in range(0, rows):
    #     for q in range(0, columns):
    #         if mag[p, q] < threshold:
    #             mag[p, q] = 0
    return mag

def Edge_Extract(root):
    img_root = os.path.join(root,'GT')
    edge_root = os.path.join(root,'Edge_sobel')

    if not os.path.exists(edge_root):
        os.mkdir(edge_root)

    file_names = os.listdir(img_root)
    img_name = []

    for name in file_names:
        print(f'Generate Edge Image {name} successful!')
        if not name.endswith('.png'):
            assert "This file %s is not PNG"%(name)
        img_name.append(os.path.join(img_root,name[:-4]+'.png'))

    index = 0
    for image in img_name:
        img = cv2.imread(image, 0)
        img = cv2.resize(img, [352, 352], interpolation=cv2.INTER_LINEAR)
        edge = sobel(img)
        # cv2.imwrite(edge_root+'/'+file_names[index],cv2.Canny(img,30,100))
        cv2.imwrite(edge_root + '/' + file_names[index], edge)
        index += 1
    return 0


# def sobel(img, threshold):
#     '''
#     edge detection based on sobel
#
#     Parameters
#     ----------
#     img : TYPE
#         the image input.
#     threshold : TYPE
#          varies for application [0 255].
#
#     Returns
#     -------
#     mag : TYPE
#         output after edge detection.
#
#     '''
#     G_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
#     G_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
#     rows = np.size(img, 0)
#     columns = np.size(img, 1)
#     mag = np.zeros(img.shape)
#     for i in range(0, rows - 2):
#         for j in range(0, columns - 2):
#             v = sum(sum(G_x * img[i:i + 3, j:j + 3]))  # vertical
#             h = sum(sum(G_y * img[i:i + 3, j:j + 3]))  # horizon
#             mag[i + 1, j + 1] = np.sqrt((v ** 2) + (h ** 2))
#
#     for p in range(0, rows):
#         for q in range(0, columns):
#             if mag[p, q] < threshold:
#                 mag[p, q] = 0
#     return mag


if __name__ == '__main__':
    root = '/home/zongzong/WD/Datasets/RGBT/VT5000/Test/'
    Edge_Extract(root)
