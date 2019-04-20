import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


# according log_name to draw_curve
def log_read_draw(log_files):
    with open(log_files, 'r') as f:
        train_epoch_loss, valid_epoch_loss, train_epoch_margin_loss, valid_epoch_margin_loss, train_epoch_mse_loss, \
        valid_epoch_mse_loss, train_gini, valid_gini, train_std, valid_std = [], [], [], [], [], [], [], [], [], []
        for line in f.readlines():
            try:
                log_flag = line.split('=')[0].strip()
                if log_flag == 'train_epoch_loss':
                    train_epoch_loss.append(float(line.split('=')[1].strip()))
                elif log_flag == 'valid_epoch_loss':
                    valid_epoch_loss.append(float(line.split('=')[1].strip()))
                elif log_flag == 'train_epoch_margin_loss':
                    train_epoch_margin_loss.append(float(line.split('=')[1].strip()))
                elif log_flag == 'valid_epoch_margin_loss':
                    valid_epoch_margin_loss.append(float(line.split('=')[1].strip()))
                elif log_flag == 'train_epoch_mse_loss':
                    train_epoch_mse_loss.append(float(line.split('=')[1].strip()))
                elif log_flag == 'valid_epoch_mse_loss':
                    valid_epoch_mse_loss.append(float(line.split('=')[1].strip()))
                elif log_flag == 'train_gini':
                    train_gini.append(float(line.split('=')[1].strip()))
                elif log_flag == 'valid_gini':
                    valid_gini.append(float(line.split('=')[1].strip()))
                elif log_flag == 'train_std':
                    train_std.append(float(line.split('=')[1].strip()))
                elif log_flag == 'valid_std':
                    valid_std.append(float(line.split('=')[1].strip()))
                else:
                    continue
            except Exception:
                print('log_dir is not exit!')
    f.close()

    return train_epoch_loss, valid_epoch_loss, train_epoch_margin_loss, valid_epoch_margin_loss, \
           train_epoch_mse_loss, valid_epoch_mse_loss, train_gini, valid_gini, train_std, valid_std


# draw the loss_curve
def draw_loss_curve(train_loss, valid_loss):
    x_length = len(train_loss)
    x = range(0, x_length, 1)
    plt.plot(x, train_loss, marker='.', color='r', label=u'train_loss')
    plt.plot(x, valid_loss, marker='*', color='r', label=u'valid_loss')
    plt.legend() # legend effective
    plt.xticks(x, rotation=45)
    plt.ylim(0.1, 1)
    plt.margins(0)
    plt.subplots_adjust(bottom=0.15)
    plt.xlabel(u'EPOCHES')
    plt.ylabel(u'LOSS')
    plt.title(u'LOSS')
    plt.show()

# draw the gini_curve
def draw_gini_curve(train_gini, valid_gini):
    x_length = len(train_gini)
    x = range(0, x_length, 1)
    plt.plot(x, train_gini, marker='.', color='r', label=u'train_gini')
    plt.plot(x, valid_gini, marker='*', color='r', label=u'valid_gini')
    plt.legend()  # legend effective
    plt.xticks(x, rotation=45)
    plt.ylim(0.1, 1)
    plt.margins(0)
    plt.subplots_adjust(bottom=0.15)
    plt.xlabel(u'EPOCHES')
    plt.ylabel(u'GINI')
    plt.title(u'GINI')
    plt.show()

# draw the std_curve
def draw_std_curve(train_std, valid_std):
    x_length = len(train_std)
    x = range(0, x_length, 1)
    plt.plot(x, train_std, marker='.', color='r', label=u'train_std')
    plt.plot(x, valid_std, marker='*', color='r', label=u'valid_std')
    plt.legend()  # legend effective
    plt.xticks(x, rotation=45)
    plt.ylim(0.1, 1)
    plt.margins(0)
    plt.subplots_adjust(bottom=0.15)
    plt.xlabel(u'EPOCHES')
    plt.ylabel(u'STD')
    plt.title(u'STD')
    plt.show()


if __name__ == '__main__':
    log_files = './record/log/*txt'
    train_epoch_loss, valid_epoch_loss, train_epoch_margin_loss, valid_epoch_margin_loss, \
    train_epoch_mse_loss, valid_epoch_mse_loss, train_gini, valid_gini, train_std, valid_std = log_read_draw(log_files)

    draw_loss_curve(train_epoch_loss, valid_epoch_loss)
    draw_gini_curve(train_gini, valid_gini)
    draw_std_curve(train_std, valid_std)

