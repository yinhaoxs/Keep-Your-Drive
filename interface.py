import torch
from torchvision import transforms
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


# set gpu
os.environ['CUDA_VISIBLE_DEVICES'] = "7"


# Encapsulating Interface
class KYD_MODEL(object):
    def __init__(self):
        # 1.1 loading trained_model
        self.pretrained_dict = torch.load('./record/checkpoint/*.pth')

        # 1.2 load parallel_model
        # self.new_state_dict = OrderedDict()
        # for k, v in self.pretrained_dict.items():
        #     name = k[7:]
        #     self.new_state_dict[name] = v

        # 1.3 load model
        self.model = model.Backbone(num_layers=50, drop_ratio=0, mode='ir_se')
        self.model_dict = self.model.state_dict()
        self.pretrained_dict = {k: v for k, v in self.pretrained_dict.items() if k in self.model_dict}
        self.model_dict.update(self.pretrained_dict)
        self.model.load_state_dict(self.model_dict)
        self.model = self.model.to('cuda')
        self.valid_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                 std=[0.2023, 0.1994, 0.2010])])

        # 1.4 load scoreing for object
        f = open('./record/log/*.json', "r")
        for line in f:
            self.scoring_dict = json.loads(line)
        f.close()
        self.aiCentile = self.scoring_dict['aiCentile']
        self.values = self.scoring_dict['values']

        # 1.5 landmark points
        self.predictor = './imageset/label/shape_predictor_68_face_landmarks.dat'


    # ficial_score get
    def Ficial_score(self, image):
        # data enhancement
        self.model.eval()
        with torch.no_grad():
            # img = Image.open(img_path)
            img = image.convert("RGB")
            img = self.valid_transform(img)
            # add one dimension
            img = img.unsqueeze(0)
            img = img.to("cuda")
            pred = self.model(img).data.cpu().numpy()[0]

            Ficial_score = pd.cut(pred, self.values, labels=self.aiCentile)

        return Ficial_score


    def detection_align(self, img_path):
        # import face detection model
        detector = dlib.get_frontal_face_detector()
        # import face landmark points model
        sp = dlib.shape_predictor(self.predictor)

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
            image = dlib.get_face_chips(rgb_img, faces, size=112)[0]

        return image


if __name__ == '__main__':
    img_dir = './imgageset/image_v/'
    model = KYD_MODEL()

    for img_path in os.listdir(img_dir):
        # make dict
        Ficial_dict = {}
        img_path_dir = os.path.join(img_dir, img_path)
        img_name = img_path.split('_')[-1].split('.')

        image = model.detection_align(img_path_dir)
        Ficial_score = model.Ficial_score(image)
        Ficial_dict["Ficial_score"] = Ficial_score
        Ficial_dict["img_name"] = img_name
        print('name: %s, ficial_score: %.10f' % (img_name, Ficial_score))

