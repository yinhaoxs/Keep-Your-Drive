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

import numpy as np
import cv2
import random
from pathlib import Path
from glob import glob
import os
from tqdm import tqdm_notebook

# calculate dataset's mean and std
def calculate_mean_std(img_dir, img_w, img_h):
    imgs = np.zeros([img_w, img_h, 3, 1])
    means, stdevs = [], []
    current_path = Path(img_dir)
    for img_path in glob(str(current_path)+os.sep + "*jpg"):
        img = cv2.imread(img_path)
        img = img[:, :, :, np.newaxis]
        imgs = np.concatenate((imgs, img), axis = 3)

    imgs = imgs.astype(np.float32)/255

    for i in tqdm_notebook(range(3)):
        # pull in a row
        pixels = imgs[:, :, i, :].revel()
        means.append(np.mean(pixels))
        stdevs.append(np.std(pixels))
    # BGR->RGB
    means = means.reverse()
    stdevs = stdevs.reverse()

    return means, stdevs


if __name__ == '__main__':
    img_dir = './imageset/image/'
    img_w, img_h = 112, 112
    means, stdevs = calculate_mean_std(img_dir, img_w, img_h)

    print('normMean = {}'.format(means))
    print('normStd = {}'.format(stdevs))
