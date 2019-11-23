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
import pandas as pd
import time
from PIL import Image
import random
import os
import copy
from glob import glob
import json
from collections import OrderedDict
from model import model
import dlib
import cv2


# detection_align
def detection_align(predictor, img_path, save_dir):
    # import face detection model
    detector = dlib.get_frontal_face_detector()
    # import face landmark points model
    sp = dlib.shape_predictor(predictor)

    bgr_img = cv2.imread(img_path)
    if bgr_img is not None:
        print("Sorry, we could not load '{}' as an image".format(img_path))
        exit()
    # bgr->rgb
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    # detect the faces of picture
    dets = detector(rgb_img, 1)
    num_faces = len(dets)
    if num_faces == 0:
        print("Sorry, there were no faces found in '{}' as an image".format(img_path))

    else:
        area_list = []
        faces = dlib.full_object_detections()
        for i, d in enumerate(dets):
            # print(type(d))
            y1 = d.top() if d.top() > 0 else 0
            y2 = d.bottom() if d.bottom() > 0 else 0
            x1 = d.left() if d.left() > 0 else 0
            x2 = d.right() if d.right() > 0 else 0

            x = x2 - x1 + 1
            y = y2 - y1 + 1
            d_area = x * y
            area_list.append(d_area)
        # choose max_area det
        max_value = max(area_list)
        max_index = area_list.index(max_value)
        dets = dets[max_index]
        faces.append(sp(rgb_img, dets))

        # face_alignment
        image = dlib.get_face_chips(rgb_img, faces, size=112)

        # save face
        cv_rgb_image = np.array(image).astype(np.uint8)  #->numpy
        cv_bgr_image = cv2.cvtColor(cv_rgb_image, cv2.COLOR_BGR2RGB)
        cv2.imwrite(save_dir, cv_bgr_image)


if __name__ == '__main__':
    img_dir = './imgageset/image_test/'
    save_dir = './imageset/'
    predictor = './imageset/label/shape_predictor_68_face_landmarks.dat'

    for img_path in os.listdir(img_dir):
        detection_align(predictor, img_path, save_dir)

    print('align all face_picture!')

