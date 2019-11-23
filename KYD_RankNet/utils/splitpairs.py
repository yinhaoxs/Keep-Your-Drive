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

# module set
import numpy as np
import pandas as pd
import os

################################## data balance  #############################################################
def balance(df, y):
    df0 = df[df[y]==0]
    df1 = df[df[y]!=0]
    del df
    len_0 = len(df0)
    len_1 = len(df1)
    fraction_01 = len_0/len_1

    df_reshape = pd.DataFrame()
    for i in range(0, len(np.ceil(fraction_01))):
        df_reshape = pd.concat([df_reshape, df1], axis=0)
    # len_2 = len(df_reshape)
    df = pd.concat([df0, df_reshape], axis=0)

    return df


def pair_split(df, NE=False):
    # middle cut
    nums = np.int(len(df)/2)
    df1 = df.head(nums).reset_index(drop=True)
    df2 = df.tail(nums).reset_index(drop=True)
    length = df1.shape[1]
    y = df1["loss_ratio"] - df2["loss_ratio"]

    if NE:
        y = pd.DataFrame(y).rename(columns={"loss_ratio": "LABEL"})
        del df
        df = pd.concat([df1, df2, y], axis=1)

        df["LABEL"] = df["LABEL"].map(lambda x: 1 if x > 0 else -1)

        df1 = df.iloc[:, 0:length-1]
        df2 = df.iloc[:, length:-2]
        y = df.iloc[:, -1]
        return df1, df2, y
    else:
        df2 = np.zeros((df.shape[0], df.shape[1]))

        return df, df2


# set sigmoid function
def sigmoid(data):
    data = 1 / (1+np.exp(-1*data))
    return data






