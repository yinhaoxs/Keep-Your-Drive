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
import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
from itertools import combinations
from test_kydnet import KYD_MODEL


# calculate cosin_distance
def CosDist(array1, array2):
    num = float(np.sum(array1*array2))
    denom = np.linalg.norm(array1)*np.linalg.norm(array2)
    cosdist = denom / num

    return cosdist


# calculate euclidean_distance
def EuclDist(array1, array2):
    return np.sqrt(np.sum(np.square(array1 - array2)))


# intra-class mean_std
def intra_class_mean_std(lfw_dir, feature_dict):

    Eucl_intra_list, Cos_intra_list = [], []
    Eucl_distance_dict, Cos_distance_list = {}, {}

    for label in os.listdir(lfw_dir):
        Eucl_distance_list, Cos_distance_list = [], []
        face_list = os.listdir(os.path.join(lfw_dir, label))
        length = len(face_list)

        for i in range(0, length):
            for j in range(i+1, length):
                Eucl_distance = EuclDist(np.array(feature_dict[label][face_list[i]]), np.array(feature_dict[label][face_list[j]]))
                Cos_distance = CosDist(np.array(feature_dict[label][face_list[i]]), np.array(feature_dict[label][face_list[j]]))
                Eucl_distance_list.append(Eucl_distance)
                Cos_distance_list.append(Cos_distance)

        num = len([c for c in combinations(range(length), 2)])
        Eucl_intra_list.append(sum(Eucl_distance_list) / num)
        Cos_intra_list.append(sum(Eucl_distance_list) / num)

        Eucl_distance_dict[label] = Eucl_intra_list
        Cos_distance_list[label] = Cos_intra_list

    Eucl_mean = np.mean(np.array(Eucl_intra_list))
    Eucl_std = np.std(np.array(Eucl_intra_list))
    Cos_mean = np.mean(np.array(Cos_intra_list))
    Cos_std = np.std(np.array(Cos_intra_list))

    return Eucl_mean, Eucl_std, Cos_mean, Cos_std


# inter-class mean_std
def inter_class_mean_std(lfw_dir, feature_dict):

    Eucl_intra_list, Cos_intra_list = [], []
    Eucl_distance_dict, Cos_distance_list = {}, {}

    key = os.listdir(lfw_dir)
    length = len(key)

    for i in range(0, length):
        for j in range(i+1, length):

            Eucl_distance_list, Cos_distance_list = [], []

            length_1 = len(os.path.join(lfw_dir, key[i]))
            length_2 = len(os.path.join(lfw_dir, key[j]))
            list_1 = os.path.join(lfw_dir, key[i])
            list_2 = os.path.join(lfw_dir, key[j])

            for m in range(0, length_1):
                for n in range(0, length_2):
                    Eucl_distance = EuclDist(np.array(feature_dict[key[i]][list_1[m]]),
                                             np.array(feature_dict[key[j]][list_2[n]]))
                    Cos_distance = CosDist(np.array(feature_dict[key[i]][list_1[m]]),
                                             np.array(feature_dict[key[j]][list_2[n]]))
                    Eucl_distance_list.append(Eucl_distance)
                    Cos_distance_list.append(Cos_distance)

            # num = len([c for c in combinations(range(int(length_1*length_2)), 2)])
            Eucl_intra_list.append(sum(Eucl_distance_list) / int(length_1*length_2))
            Cos_intra_list.append(sum(Eucl_distance_list) / int(length_1*length_2))

            Eucl_distance_dict[i+j] = sum(Eucl_intra_list) / int(length_1*length_2)
            Cos_distance_list[i+j] = sum(Cos_intra_list) / int(length_1*length_2)

    Eucl_mean = np.mean(np.array(Eucl_intra_list))
    Eucl_std = np.std(np.array(Eucl_intra_list))
    Cos_mean = np.mean(np.array(Cos_intra_list))
    Cos_std = np.std(np.array(Cos_intra_list))

    return Eucl_mean, Eucl_std, Cos_mean, Cos_std


# calculate LDA intra_matrix_det inter_matrix_det
def calculate_LDA_matrix_det(feature, label):
    # Sb: between-class scatter matrix   Sw: Class like dispersion matrix
    W, Sw, Sb = LDA_reduce_dimension(feature, label, 2)
    # Class like dispersion matrix
    Sw_w = np.dot(np.dot(np.transpose(W), Sw), W)
    value_w = np.linalg.det(Sw_w)
    # between-class scatter matrix det
    Sb_b = np.dot(np.dot(np.transpose(W), Sb), W)
    value_b = np.linalg.det(Sb_b)

    # intra_matrix_sum, inter_matrix_sum, intra_matrix_det, inter_matrix_det
    intra_matrix_sum, inter_matrix_sum, intra_matrix_det, inter_matrix_det = Sw.sum(), Sb.sum(), value_w, value_b

    return intra_matrix_sum, inter_matrix_sum, intra_matrix_det, inter_matrix_det


def LDA_reduce_dimension(X, y, nComponents):
    # Converting to a set
    labels = list(set(y))

    xClasses = {}
    for label in labels:
        xClasses[label] = np.array([X[i] for i in range(len(X)) if y[i] == label])

    # calculate all_mean
    meanAll = np.mean(X, axis=0)  # compute mean for column
    meanClasses = {}

    # calculate various_average
    for label in labels:
        meanClasses[label] = np.mean(xClasses[label], axis=0)

    # global scatter_matrix
    St = np.zeros((len(meanAll), len(meanAll)))
    St = np.dot((X - meanAll).T, X - meanAll)

    # Class like dispersion matrix
    Sw = np.zeros((len(meanAll), len(meanAll)))
    for i in labels:
        Sw += np.dot((xClasses[i], meanClasses[i]).T, (xClasses[i], meanClasses[i]))

    # between-class scatter matrix
    Sb = np.zeros((len(meanAll), len(meanAll)))
    Sb = St - Sw

    # calculate
    eigenvalues, eigenvectors = np.linalg.eig(np.dot(np.linalg.inv(Sw), Sb))
    # First few eigenvector
    sortedIndices = np.argsort(eigenvalues)
    W = eigenvectors[:, sortedIndices[:-nComponents-1:-1]]

    return W, Sw, Sb


if __name__ == '__main__':
    lfw_dir = './imageset/image/lfw-align-128/'
    model = KYD_MODEL()
    total_feature = {}
    LDA_total_feature, LDA_total_label = [], []
    for label in os.listdir(lfw_dir):
        image_feature = {}
        for image_path in os.listdir(os.path.join(lfw_dir, label)):
            image_name = image_path.split('.')[0]
            feature = model.predict(image_path)

            LDA_total_feature.append(feature)
            LDA_total_label.append(int(label))
            image_feature[image_name] = feature

        feature = np.array(LDA_total_feature)
        label = np.array(LDA_total_label)
        total_feature[label] = image_feature

    intra_Eucl_mean, intra_Eucl_std, intra_Cos_mean, intra_Cos_std = intra_class_mean_std(lfw_dir, total_feature)
    inter_Eucl_mean, inter_Eucl_std, inter_Cos_mean, inter_Cos_std = inter_class_mean_std(lfw_dir, total_feature)
    intra_matrix_sum, inter_matrix_sum, intra_matrix_det, inter_matrix_det = calculate_LDA_matrix_det(feature, label)









