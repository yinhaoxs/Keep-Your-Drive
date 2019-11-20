import torch
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
import torch.utils.data as data
import os


# list -> tuple(for train&&valid)
def default_reader(file_txt):
    f = open(file_txt, 'r')
    imgList = []

    for line in f:
        line = line.strip('\n').strip()
        imgPath, label = line.split()[0], line.split()[1]
        imgList.append((str(imgPath), int(label)))

    return imgList


# image -> rgb(pil)
def PIL_loader(path):
    try:
        # with open(path, 'rb') as f:
        return Image.open(path).convert('RGB')
    except:
        print('Cannot load image' + path)


# read image init
class ImageList(data.Dataset):
    def __init__(self, root, file_txt, transform=None, list_reader=default_reader, loader=PIL_loader):
        self.root = root
        self.transform = transform
        self.imgList = list_reader(file_txt)
        self.loader = loader

    def __getitem__(self, index):
        imgPath, target = self.imgList[index]
        img = self.loader(os.path.join(self.root, imgPath))

        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.imgList)

## TODO
## five tuple,  six tuple



