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
from PIL import Image
import torch
from torchvision import transforms
import numpy as np
import pandas as pd

##################################  Valid Model #############################################################
def valid(valid_loader, model, criterion_A, criterion_B, device):
    model.eval()

    # set batch_loss
    batch_loss = 0
    batch_A_loss = 0
    batch_B_loss = 0

    for batch_idx, (data1, data2, target) in enumerate(valid_loader, 1):
        # iteration = (epoch - 1) * len(valid_loader) + batch_idx
        # select A, B index
        target_A_index = (target != 0.5).nonzero()
        target_B_index = (target == 0.5).nonzero()

        # get 1-dim index
        target_A_indices = target_A_index.view(len(target_A_index))
        target_B_indices = target_B_index.view(len(target_B_index))

        # accoding A, B index select data
        data1_A = torch.index_select(data1, dim=0, index=target_A_indices)
        data2_A = torch.index_select(data2, dim=0, index=target_A_indices)
        target_A = torch.index_select(target, dim=0, index=target_A_indices)
        data1_B = torch.index_select(data1, dim=0, index=target_B_indices)
        data2_B = torch.index_select(data2, dim=0, index=target_B_indices)
        target_B = torch.index_select(target, dim=0, index=target_B_indices)

        # data from cpu->gpu
        data1_A, data2_A, target_A = data1_A.to(device), data2_A.to(device), target_A.to(device)
        data1_B, data2_B, target_B = data1_B.to(device), data2_B.to(device), target_B.to(device)

        # compute A output
        output1_A = model(data1_A)
        output2_A = model(data2_A)
        s = output1_A - output2_A
        output_A = torch.sigmoid(s)
        loss_A = criterion_A(output_A, target_A)

        #compute B output
        output1_B = model(data1_B)
        output2_B = model(data2_B)
        s = output1_B - output2_B
        target_B.detach()
        # output_B = torch.sigmoid(s)
        loss_B = criterion_B(s, target_B)

        # combine two loss
        loss = loss_A + loss_B
        batch_loss += loss.data[0]
        batch_A_loss += loss_A.data[0]
        batch_B_loss += loss_B.data[0]

    # compute epoch loss
    epoch_loss = batch_loss / len(valid_loader.dataset)
    epoch_A_loss = batch_A_loss / len(valid_loader.dataset)
    epoch_B_loss = batch_B_loss / len(valid_loader.dataset)

    return epoch_loss, epoch_A_loss, epoch_B_loss