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

import os
from glob import glob
import pandas as pd
import numpy as np


# full row
def lfw_generate_pairs(image_dir, img_list, file_same, file_diffirent):
    # get face_pairs
    for i in range(len(img_list)):
        former = image_list[i]
        for j in range(i+1, len(img_list)):
            later = image_list[j]
            # former_id and later_id
            former_base = os.path.basename(former)
            later_base = os.path.basename(later)
            former_ID = former_base.split('_')[0] + '_' + former_base.split('_')[1]
            later_ID = later_base.split('_')[0] + '_' + later_base.split('_')[1]
            if former_ID == later_ID:
                file_same.write(os.path.join(image_dir, former_base) + ' ' + os.path.join(image_dir, later_base) + ' ' + '0.5' + '\n')
            else:
                file_diffirent.write(os.path.join(image_dir, former_base) + ' ' + os.path.join(image_dir, later_base) + ' ' + '0.5' + '\n')

# txt->csv
def txt2csv(file_txt):
    result_list = []
    file = open(file_txt)
    for line in file.readlines():
        image_name = line.split()[0]
        loss_ratio = float(line.split()[1])
        new_dict = get_face_dict(image_name, loss_ratio)
        result_list.append(new_dict)
    pd_csv = pd.DataFrame.from_dict(result_list)
    pd_csv.to_csv("kyd_set.csv", encoding="UTF-8", index=False)


def get_face_dict(image_name, loss_ratio):
    new_dict = {'face_picture': image_name,
                'loss_ratio': loss_ratio}

    return new_dict


if __name__ == '__main__':
    image_dir = './imageset/image/'
    same_file = './record/log/same_face.txt'
    diffirent_file = './record/log/diffirent_face.txt'

    image_list = glob(image_dir+'/*'+'.jpg')
    same = open(same_file, 'a+')
    diffirent = open(diffirent_file, 'a+')

    lfw_generate_pairs(image_dir, image_list, same, diffirent)

