# IRFS
 
[![LICENSE](https://img.shields.io/badge/license-MIT-green)](https://github.com/wdhudiekou/IRFS/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-1.6.0-%237732a8)](https://pytorch.org/)


### An interactively reinforced paradigm for joint infrared-visible image fusion and saliency object detection [Information Fusion]

By Di Wang, Jinyuan Liu, Risheng Liu, and Xin Fan*


<div align=center>
<img src="https://github.com/wdhudiekou/IRFS/blob/main/Fig/network.png" width="90%">
</div>

## Updates
[2023-05-17] Our paper is available online! [[arXiv version](https://arxiv.org/abs/2305.09999)]  

## Requirements
- CUDA 10.1
- Python 3.6 (or later)
- Pytorch 1.6.0
- Torchvision 0.7.0
- OpenCV 3.4
- Kornia 0.5.11

## Dataset
Please download the following datasets:

Infrared and Visible Image Fusion Datasets
*   [RoadScene](https://github.com/hanna-xu/RoadScene)
*   [TNO](http://figshare.com/articles/TNO\_Image\_Fusion\_Dataset/1008029)
*   [M3FD](https://github.com/JinyuanLiu-CV/TarDAL)

RGBT SOD Saliency Datasets
*   [VT821](https://github.com/lz118/RGBT-Salient-Object-Detection)
*   [VT1000](https://github.com/lz118/RGBT-Salient-Object-Detection)
*   [VT5000](https://github.com/lz118/RGBT-Salient-Object-Detection)

## Data preparation
1. You can obtain self-visual saliency maps for training image fusion by
    ```python
       cd ./data
       python get_svm_map_softmax.py

## Get start
Firstly, you need to download the pretrained model of [ResNet-34](https://drive.google.com/drive/folders/1vOaToFPI74Uv8Ok7C88zjaOat9wR8dwd) and put it into folder './pretrained/'.
1. You can implement the interactive training of image fusion and SOD. Please check the dataset paths in train_Inter_IR_FSOD.py, and then run:
    ```python
       cd ./Trainer
       python train_Inter_IR_FSOD.py
2. You can also train image fusion or SOD separately. Please check the dataset paths in train_fsfnet.py and train_fgccnet.py, and then run:
   ```python
    ## for image fusion
       cd ./Trainer
       python train_fsfnet.py
    ## for SOD
       cd ./Trainer
       python train_fgccnet.py  
After training, the pretrained models will be saved in folder './checkpoint/'. 
1. You can load pretrained models to evaluate the performance of the IRFS in two tasks (i.e., image fusion, SOD) by running:
    ```python
       cd ./Test
       python test_IR_FSOD.py      
2. You can also test image fusion and SOD separately by running:
    ```python
    ## for image fusion
       cd ./Test
       python test_fsfnet.py
    ## for SOD
        cd ./Test
       python test_fgccnet.py   
       
## Experimental Results

## Any Question
If you have any other questions about the code, please email: diwang1211@mail.dlut.edu.cn

## Citation
If this work has been helpful to you, please feel free to cite our paper!
```
@InProceedings{Wang_2023_IF,
	author = {Di, Wang and Jinyuan, Liu and Risheng Liu and Xin, Fan},
	title = {An interactively reinforced paradigm for joint infrared-visible image fusion and saliency object detection},
	booktitle = {Information Fusion},
	year = {2023}
}
```