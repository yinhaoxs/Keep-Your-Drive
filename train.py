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
from torchvision import transforms
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import time
import random
from collections import defaultdict
from PIL import Image
import argparse
import json
import random

from model import dataset, model, train_loss, valid_loss, evalute_gini
from utils import gini, score, splitpairs, dataread


################################## main function #############################################################
#1.1 set parameters
parse = argparse.ArgumentParser(description='kyd_net')
parse.add_argument('--root_path', type=str, default='./images/', help='origin image to train the kyd_net')
parse.add_argument('--epoch', type=int, default=100, help='number of epoches for train(default: 100)')
parse.add_argument('--batch_size', type=int, default=128, help='input batch_szie for training and validing(defalt: 64)')
parse.add_argument('--lr', type=float, default=0.001, help='learning rate(default: 0.1)')
parse.add_argument('--momentum', type=float, default=0.9, help='SGD momentum(default: 0.9)')
parse.add_argument('--weight_decay', type=float, default=0.0005, metavar='W', help='default: 0.0005')
parse.add_argument('--log_save_path', type=str, default='./logs/', help='log save for train and valid')
parse.add_argument('--model_save_path', type=str, default='./ptretrained_model/', help='model save for strain and valid')
parse.add_argument('--inter_log', type=int, default=64, help='print times for train_logs')
parse.add_argument('--no_cuda', type=bool, default=False, help='disable CUDA train')
parse.add_argument('--workers', type=int, default=4, help='how many worker for load data')
## 计算gini的准备文件
parse.add_argument('--data_path', type=str, default='./ptretrained_model/csv/', help='the path for csv files')

#1.2 set gpu
os.environ["CUDA_VASIBLE_DEVICES"] = "0, 1"
args = parse.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
args.device = torch.device("cuda" if args.cuda else "cpu")


#2.1 train the kyd_model
def train_valid_model(train_dataloader, valid_dataloader, model, loss_function_A, loss_function_B, device, batch_size, inter_log):
    train_epoch_loss, train_epoch_A_loss, train_epoch_B_loss = train_loss.train(train_dataloader, model,
                                                                                loss_function_A, loss_function_B, device, batch_size, inter_log)
    valid_epoch_loss, valid_epoch_A_loss, valid_epoch_B_loss = valid_loss.valid(valid_dataloader, model,
                                                                                loss_function_A, loss_function_B, device)

    return train_epoch_loss, train_epoch_A_loss, train_epoch_B_loss, \
           valid_epoch_loss, valid_epoch_A_loss, valid_epoch_B_loss


# 3.1 print log
def print_with_time(string):
    print(time.strftime("%Y-%m-%d %H:%M:%S ", time.localtime()) + string)


if __name__ == '__main__':
    # 4.1 data enhancement
    train_transform = transforms.Compose([
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                            std=[0.2023, 0.1994, 0.2010])])
    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                             std=[0.2023, 0.1994, 0.2010])])

    # 4.2 read data
    # 4.2.1 read kyd_csv_files for train_bce
    KYD_data = dataread.read_csv(args.data_path, 'risk_set.csv')
    train_KYD_data, valid_KYD_data = train_test_split(KYD_data, test_size=0.2, random_state=512)
    # 4.2.2 read_lfw_csv_pair_files for train_mse
    train_LFW_data = dataread.read_csv(args.data_path, 'lfw_train_face_pair.csv')
    valid_LFW_data = dataread.read_csv(args.data_path, 'lfw_test_face_pair.csv')
    # 4.2.3 read lfw_csv_files for calculate std
    LFW_data_train_std = dataread.read_csv(args.data_path, 'lfw_train.csv')
    LFW_data_valid_std = dataread.read_csv(args.data_path, 'lfw_test.csv')
    # 4.2.4 read res_csv_files for gini
    res = dataread.read_csv(args.data_path, 'Res_facepicturename.csv')

    # 4.3 dataloader
    # 4.3.1 KYD valid_dataloader
    KYD_valid = splitpairs.balance(valid_KYD_data, y='loss_ratio').sample(frac=1, random=512)
    KYD_valx1, KYD_valx2, KYD_valy = splitpairs.pair_split(KYD_valid, NE=True)
    KYD_valid_list = dataread.kyd_merge_list(KYD_valx1, KYD_valx2, KYD_valy)
    # 4.3.2 LFW valid_dataloader
    LFW_valx1, LFW_valx2, LFW_valy = valid_LFW_data.iloc[:, 0], valid_LFW_data.iloc[:, 1], valid_LFW_data.iloc[:, 2]
    LFW_valid_list = dataread.lfw_merge_list(LFW_valx1, LFW_valx2, LFW_valy)
    # 4.3.3 merge valid_set
    valid_list = KYD_valid_list + LFW_valid_list
    random.seed(512)
    random.shuffle(valid_list)
    # 4.3.4 KYD gini train_and_valid dataloader
    KYD_train_gini, KYD_valid_gini = [], []
    for value in train_KYD_data.values.tolist():
        KYD_train_gini.append(value)
    for value in valid_KYD_data.values.tolist():
        KYD_valid_gini.append(value)
    # 4.3.5 LFW std train_and_valid dataloader
    LFW_train_std, LFW_valid_std = [], []
    for value in LFW_data_train_std.values.tolist():
        LFW_train_std.append(value)
    for value in LFW_data_valid_std.values.tolist():
        LFW_valid_std.append(value)
    # 4.3.6 valid dataloader
    valid_loader = torch.utils.data.Dataloader(
        dataset.ImageList(root=args.root_path, fileList=valid_list, transform=valid_transform),
        batch_size = args.batch_size, shuffle = False, num_workers = args.workers, pin_memory= False, drop_last = True)
    # 4.3.7 gini dataloader
    KYD_train_loader = torch.utils.data.Dataloader(
        dataset.ImageList_x(root=args.root_path, fileList=KYD_train_gini, transform = valid_transform),
        batch_size = args.batch_size, shuffle = False, num_workers = args.workers, pin_memory=False, drop_last=False)
    KYD_valid_loader = torch.utils.data.Dataloader(
        dataset.ImageList_x(root=args.root_path, fileList=KYD_valid_gini, transform = valid_transform),
        batch_size = args.batch_size, shuffle = False, num_workers = args.workers, pin_memory=False, drop_last=False)
    # 4.3.8 std dataloader
    LFW_train_loader = torch.utils.data.Dataloader(
        dataset.ImageList_x(root=args.root_path, fileList=LFW_train_std, transform = valid_transform),
        batch_size = args.batch_size, shuffle = False, num_workers = args.workers, pin_memory=False, drop_last=False)
    LFW_valid_loader = torch.utils.data.Dataloader(
        dataset.ImageList_x(root=args.root_path, fileList=LFW_valid_std, transform = valid_transform),
        batch_size = args.batch_size, shuffle = False, num_workers = args.workers, pin_memory=False, drop_last=False)


    # 4.4 load model
    model_dir = './ptretrained_model/model_ir_se50.pth'
    pretrained_dict = torch.load(model_dir)
    model = model.Backbone(num_layers=50, drop_ratio=0.6, mode='ir')
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)  # update parameter
    model.load_state_dict(pretrained_dict)
    model =torch.nn.DataParallel(model).to(args.device)

    # 4.5 set loss_function
    loss_function_A = torch.nn.MarginRankingLoss().to(args.device)
    loss_function_B = torch.nn.MSELoss().to(args.device)

    # 4.6 choose optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 100], gamma=0.1, last_epoch=-1)

    # # 4.6.1 fixed the convolution layers
    # weight_p, bais_p = [], []
    # count = 0
    # for k in model.children():
    #     count += 1
    #     if count == 3:
    #         for name, p in k.named_parameters():
    #             if 'bais' in name:
    #                 bais_p += [p]
    #             else:
    #                 weight_p += [p]
    #     else:
    #         for param in k.parameters():
    #             param.required_grad = False

    # 4.7 train the kyd_model, evaluate the kyd_model
    for epoch in range(1, args.epoches+1):
        # 4.7.1 KYD train_dataloader
        KYD_train = splitpairs.balance(train_KYD_data, y='loss_ratio').sample(frac=1)
        KYD_trainx1, KYD_trainx2, KYD_trainy = splitpairs.pair_split(KYD_train, NE=True)
        KYD_train_list = dataread.kyd_merge_list(KYD_trainx1, KYD_trainx2, KYD_trainy)
        # 4.7.2 LFW train_dataloader
        train_LFW_data = train_LFW_data.sample(frac=0.4)
        LFW_trainx1, LFW_trainx2, LFW_trainy = train_LFW_data.iloc[:, 0], train_LFW_data.iloc[:, 1], train_LFW_data.iloc[:, 2]
        LFW_valid_list = dataread.lfw_merge_list(LFW_valx1, LFW_valx2, LFW_valy)
        # 4.7.3 merge train_set
        train_list = KYD_train_list + LFW_valid_list
        # 4.7.4 load train_valid
        train_loader = torch.utils.data.Dataloader(
            dataset.ImageList(root=args.root_path, fileList=train_list, transform=train_transform),
            batch_size = args.batch_size, shuffle = True, num_workers = args.workers, pin_memory=False, drop_last=True)

        # 4.7.5 train and valid loss
        scheduler.step()
        train_epoch_loss, train_epoch_A_loss, train_epoch_B_loss, valid_epoch_loss, valid_epoch_A_loss, \
        valid_epoch_B_loss = train_valid_model(train_loader, valid_loader, model, loss_function_A, loss_function_B, args.device, args.batch_size, args.inter_log)
        torch.save(model.state_dict(), './' + 'mse_bce_' + str(epoch) + '.pth')
        
        # 4.7.6 evalute kyd_lfw dataloader, get(name, score)
        KYD_train_output_lists = evalute_gini.evalute(KYD_train_loader, model, args.device)
        KYD_valid_output_lists = evalute_gini.evalute(KYD_valid_loader, model, args.device)
        LFW_train_output_lists = evalute_gini.evalute(LFW_train_loader, model, args.device)
        LFW_valid_output_lists = evalute_gini.evalute(LFW_valid_loader, model, args.device)

        # 4.7.7 calculate kyd_score and lfw_score
        KYD_train_score_dict = dataread.kyd_train_and_valid_score(KYD_train_output_lists, res)
        KYD_valid_score_dict = dataread.kyd_train_and_valid_score(KYD_valid_output_lists, res)
        LFW_train_output_dict = dataread.mapping(LFW_train_output_lists, num=100)
        LFW_valid_output_dict = dataread.mapping(LFW_valid_output_lists, num=100)
        LFW_train_score_dict = dataread.lfw_train_and_valid_score(LFW_train_output_lists,
                                                                LFW_train_output_dict['values'], LFW_train_output_dict['aiCentile'])
        LFW_valid_score_dict = dataread.lfw_train_and_valid_score(LFW_valid_output_dict,
                                                                 LFW_valid_output_dict['values'], LFW_valid_output_dict['aiCentile'])


        # 4.7.8 calculate gini
        train_gini = gini.gini(dataset=KYD_train_score_dict, pred='FACE_SCORE', income='ex_ncd_prem_base_com', claims='inc_com')
        valid_gini = gini.gini(dataset=KYD_valid_score_dict, pred='FACE_SCORE', income='ex_ncd_prem_base_com', claims='inc_com')

        # 4.7.9 calculate std
        _, _, train_std = dataread.compute_std(LFW_train_score_dict)
        _, _, valid_std = dataread.compute_std(LFW_valid_score_dict)

        print_with_time('train_epoch:{}, train_epoch_loss:{}, valid_epoch_loss:{}, train_epoch_margin_loss:{}, '
                        'valid_epoch_margin_loss:{}, train_epoch_mse_loss:{}, valid_epoch_mse_loss:{}, train_gini:{}, '
                        'valid_gini:{}, train_std:{}, valid_std:{},'.format(epoch, train_epoch_loss, valid_epoch_loss,
                        train_epoch_A_loss, valid_epoch_A_loss, train_epoch_B_loss, valid_epoch_B_loss, train_gini,
                        valid_gini, train_std, valid_std))
        
    print("hello world!")
