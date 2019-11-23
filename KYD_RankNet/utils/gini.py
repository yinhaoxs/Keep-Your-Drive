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


# calculate gini
def gini(dataset, pred='pred', income='ex_ncd_prem_base_com', claims='inc_com'):
    dataset[income] = dataset[income].astype(float)
    dataset[claims] = dataset[claims].astype(float)

    # sorting by predict loss_ratio
    dataset = dataset.sort_values(pred, ascending=True)

    # accumulating claims
    cum_claims = np.cumsum((np.append(dataset[claims], 0)))
    sum_claims = cum_claims[-1]

    # accumulating premium
    cum_income = np.cumsum((np.append(dataset[income], 0)))
    sum_income = cum_income[-1]

    # calculate percentage
    xarray = cum_claims / sum_claims
    yarray = cum_income / sum_income

    # calculate areas
    B = np.trapz(yarray, x=xarray)
    # gini
    A = 0.5 - B
    G = 2 * A

    return G



