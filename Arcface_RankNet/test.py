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

    # ficial_score get
    def Score(self, img_path):
        # data enhancement
        self.model.eval()
        with torch.no_grad():
            img = Image.open(img_path)
            img = img.convert("RGB")
            img = self.valid_transform(img)
            # add one dimension
            img = img.unsqueeze(0)
            img = img.to("cuda")
            pred = self.model(img).data.cpu().numpy()[0]

            score = pd.cut(pred, self.values, labels=self.aiCentile)

        return score


    # feature get
    def Feature(self, img_path):
        # data enhancement
        self.model.eval()
        with torch.no_grad():
            img = Image.open(img_path)
            img = img.convert("RGB")
            img = self.valid_transform(img)
            # add one dimension
            img = img.unsqueeze(0)
            img = img.to("cuda")
            feature = self.model.predict(img).data.cpu().numpy()[0]

        return feature


if __name__ == '__main__':
    img_dir = './imgageset/image_test/'
    model = KYD_MODEL()

    for img_path in os.listdir(img_dir):
        # make dict
        Score_dict = {}
        img_path_dir = os.path.join(img_dir, img_path)
        img_name = img_path.split('_')[-1].split('.')
        score = model.Score(img_path_dir)
        Score_dict["score"] = score
        Score_dict["img_name"] = img_name
        print('name: %s, score: %.10f' % (img_name, score))
