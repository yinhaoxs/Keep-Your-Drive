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
import torch.nn as nn

# define tweedieloss
class tweedielossFunc(nn.Module):
    def __init__(self, rho):
        super(tweedielossFunc, self).__init__()
        self.rho = rho

    def forward(self, pred, true):
        t1 = torch.exp(torch.mul(pred, 2-self.rho, out=None), out=None)
        t = torch.mul(t1, 1/(2-self.rho), out=None)
        t2 = torch.exp(torch.mul(pred, 1-self.rho, out=None), out=None)
        loss = torch.mean(torch.addcmul(t, -1/(1 -self.rho), true, t2, out=None))

        return loss



