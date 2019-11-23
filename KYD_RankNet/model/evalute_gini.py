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
import torch

##################################  evalute model gini  ################################
def evalute(data_loader, model, device):
    output_lists = []
    with torch.no_grad():
        for batch_index, (img_name, data) in enumerate(data_loader, 1):
            data = data.to(device)
            score = model(data)
            output_lists.append((img_name, score))

    return output_lists

