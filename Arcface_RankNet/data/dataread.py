import numpy as np
import pandas as pd
import os
import random
from collections import defaultdict
from utils import gini, score


##################################  read kyd && face_recognition  ################################
# 1.1 read csv files
def read_csv(data_path, file_name):
    file_path = os.path.join(data_path, file_name)
    pd_file = pd.read_csv(file_path)

    return pd_file

# 1.2 get dict to compute gini
def get_face_dict(img_name, face_score):
    new_dict = {'face_picture': img_name,
                'FACE_SCORE': face_score
                }

    return new_dict


# 2.1 merge kyd list
def kyd_merge_list(pd_x1, pd_x2, pd_y):
    x1, x2 = [], []
    for value in pd_x1.values.tolist():
        x1.append(value[0])
    for value in pd_x2.values.tolist():
        x2.append(value[0])
    merge = [[a, b, y] for a, b, y in zip(x1, x2, pd_y.values)]

    return merge

# 2.2 merge lfw list
def lfw_merge_list(pd_x1, pd_x2, pd_y):
    x1, x2 =[], []
    for value in pd_x1.values.tolist():
        x1.append(value)
    for value in pd_x2.values.tolist():
        x2.append(value)
    merge = [[a, b, y] for a, b, y in zip(x1, x2, pd_y.values)]

    return merge


# 3.1 compute kyd_score(no mapping)
def kyd_train_and_valid_score(output_lists, res):
    score_list = []
    for output_list in output_lists:
        img_names = output_list[0]
        img_scores = output_list[1]
        for i in range(0, len(output_list)):
            img_name = img_names[i]
            img_score = img_scores.cpu().numpy().data[i][0]
            new_dict = get_face_dict(img_name, img_score)
            score_list.append(new_dict)
    pd_score = pd.DataFrame.from_dict(score_list)
    pd_score_merge = pd.merge(pd_score, res, on='face_picture', how='left')
    pd_score_sort = pd_score_merge.sort_values('FACE_SCORE').reset_index(drop=True)

    return pd_score_sort

# 3.2 compute lfw_score(no mapping)
def lfw_train_and_valid_score(output_lists, values, labels):
    score_dict = defaultdict(list)
    for output_list in output_lists:
        img_names = output_list[0]
        img_scores = output_list[1]
        for i in range(0, len(output_list)):
            img_key = img_names[i].split('_')[0] + '_' + img_names[i].split('_')[1]
            img_score = img_scores.cpu().numpy().data[i][0]
            img_mapping_score = int(pd.cut(img_score, values, labels=labels)[0])
            score_dict[img_key].append(img_mapping_score)

    return score_dict


# 3.3 no_mapping_dict -> mapping json files
def mapping(kyd_data, num=100):
    score_dict = {}
    _, values = score.score(data=kyd_data, num=num, sort='FACE_SCORE', premium='ex_ncd_prem_base_com', validate=False)
    labels = np.linspace(1, 100, 100)
    kyd_data.sort_values('FACE_SCORE', inplace=True)
    score_dict['aiCentile'] = labels.tolist()
    score_dict['values'] = values

    return score_dict


# 4.1 calculate std
def compute_std(LFW_output_dict):
    count, means, vars, stds = 0, 0, 0, 0
    cal_dict = defaultdict(dict)
    for key, value in LFW_output_dict.items():
        if len(LFW_output_dict[key]) > 1:
            mean = np.mean(np.array(LFW_output_dict[key]))
            var = np.var(np.array(LFW_output_dict[key]))
            std = np.sqrt(var)
            # push mean, var, std to dict
            cal_dict[key]["mean"] = mean
            cal_dict[key]["var"] = var
            cal_dict[key]["std"] = std
            # count peoples
            means += mean
            vars += var
            stds += std
            count += 1
    # average mean, var, std
    av_mean = means / float(count)
    av_var = vars / float(count)
    av_std = stds / float(count)

    return av_mean, av_var, av_std















