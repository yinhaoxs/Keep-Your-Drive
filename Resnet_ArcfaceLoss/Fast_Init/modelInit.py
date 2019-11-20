import numpy as np
from numpy import linalg as LA


def get_basemodel_stat(basew):
    return np.average(np.add.reduce(basew * basew, axis=1))


def weight_normalize(W, B, avgnorm2):
    pass


def calc_epsilon():
    pass


def modelinit():
    pass


def modelinit_muti_class():
    pass



if __name__ == '__main__':
    pass
