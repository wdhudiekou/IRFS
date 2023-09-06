import os
import cv2
# import base_utils


path = "/media/zongzong/63CBD6C4ED88DCA1/WD/图像融合-博一/JointFusionSOD结果汇总/TNO/U2Fusion/"
save = "/media/zongzong/63CBD6C4ED88DCA1/WD/图像融合-博一/JointFusionSOD结果汇总/TNO/U2Fusion/TNO/"

if not os.path.exists(save):
    os.makedirs(save)

"""os.listdir(path) 操作效果为 返回指定路径(path)文件夹中所有文件名"""
filename_list = sorted(os.listdir(path))  # 扫描目标路径的文件,将文件名存入列表
a = 0
# match = '_fake'
# replace = ''
for i in filename_list:
    used_name = i
    img = cv2.imread(os.path.join(path, used_name))
    num = used_name[:-4]
    new_name = num + '.jpg'
    # new_name = num + '.' + filename_list[a].split('.')[1]
    # new_name = used_name.replace(match, replace)
    # os.rename(os.path.join(path, used_name),os.path.join(path, new_name))
    # print("文件%s重命名成功,新的文件名为%s" %(used_name,new_name))
    savePath = os.path.join(save, new_name)
    cv2.imwrite(savePath, img)
    a += 1