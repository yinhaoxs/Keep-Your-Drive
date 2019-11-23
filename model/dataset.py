# coding=utf-8
# /usr/bin/env pythpn

'''
Author: yinhao
Email: yinhao_x@163.com
Wechat: xss_yinhao
Github: http://github.com/yinhaoxs

data: 2019-09-17 20:21
desc:
'''

import torch
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
import torch.utils.data as data
import os

# list -> tuple(for train&&valid)
def default_reader(fileList):
    imgList = []
    for i in range(len(fileList)):
        imgPath1, imgPath2, target = fileList[i][0], fileList[i][1], fileList[i][2]
        imgList.append((imgPath1, imgPath2, float(target)))
    return imgList

# list -> tuple(for gini&&std)
def default_reader_x(fileList):
    imgList = []
    for i in range(len(fileList)):
        imgPath, target = fileList[i][0], fileList[i][1]
        imgList.append((imgPath, target))
    return imgList

# image -> rgb(pil)
def PIL_loader(path):
    try:
        with open(path, 'rb') as f:
            return Image.open(f).convert('RGB')
    except:
        print('Cannot load image' + path)

# read image init
class ImageList(data.Dataset):
    def __init__(self, root, fileList, transform=None, list_reader=default_reader, loader=PIL_loader):
        self.root = root
        self.transform = transform
        self.imgList = list_reader(fileList)
        self.loader = loader

    def __getitem__(self, index):
        imgPath1, imgPath2, target = self.imgList[index]
        img1_path = self.loader(os.path.join(self.root, imgPath1))
        img2_path = self.loader(os.path.join(self.root, imgPath2))

        if self.transform is not None:
            img1 = self.transform(img1_path)
            img2 = self.transform(img2_path)
        return img1, img2, target

    def __len__(self):
        return len(self.imgList)


class ImageList_x(data.Dataset):
    def __init__(self, root, fileList, transform=None, list_reader=default_reader_x, loader=PIL_loader):
        self.root = root
        self.transform = transform
        self.imgList = list_reader(fileList)
        self.loader = loader

    def __getitem__(self, index):
        imgPath, target = self.imgList[index]
        img_path = self.loader(os.path.join(self.root, imgPath))

        if self.transform is not None:
            img = self.transform(img_path)
        return img, target

    def __len__(self):
        return len(self.imgList)


## TODO
## five tuple,  six tuple






