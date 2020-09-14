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

import face_recognition
import pandas as pd
#from glob import glob
#from pathlib import Path
import os
import time
from multiprocessing import Pool

def get_file_path(path):
    img_paths = []
    dirs = os.listdir(path)
    for file_dir in dirs:
        file_path = os.path.join(path, file_dir)
        img_names = os.listdir(file_path)
        for img_name in img_names:
            img_path = os.path.join(file_path, img_name)
            img_paths.append(img_path)

    return img_paths


def face_encoding(img_paths):
    face_img = face_recognition.load_image_file(img_paths)
    face_code = face_recognition.api.face_encodings(face_img)
    new_dic = {'IMG_NAME' : face_img.split('/')[-1].split('.').strip(),
               'FACE_CODE' : face_code}

    df = pd.DataFrame.from_dict(new_dic)
    df.to_csv("face_coding" + ".csv", encoding="UTF-8", index=False)


if __name__ == '__main__':
    start = time.time()
    img_path = '/data/home/cv/yinhao/face_picture'
    img_paths = get_file_path(img_path)

    pool = Pool(6)
    pool.map(face_encoding, img_paths)
    pool.close()
    pool.join()
    end = time.time()



