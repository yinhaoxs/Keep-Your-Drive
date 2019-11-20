import torch
import torch.nn as tnn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
from model.model import resnet18, Arcface, ArcMarginProduct
from model import dataset
from model import train_loss, valid_loss
import time
import os

# set gpu
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 3.1 print log
def print_with_time(string):
    print(time.strftime("%Y-%m-%d %H:%M:%S ", time.localtime()) + string)


if __name__ == '__main__':
    ## set hyperparameter
    BATCH_SIZE = 32
    LEARNING_RATE = 0.01
    EPOCH = 100
    root_path = '/Users/yinhao_x/PycharmProjects/search_image/images/'
    log_save = '/Users/yinhao_x/PycharmProjects/search_image/log/'
    train_txt = '/Users/yinhao_x/PycharmProjects/search_image/labels/train_shuffle.txt'
    test_txt = '/Users/yinhao_x/PycharmProjects/search_image/labels/test_shuffle.txt'

    # data enhancement
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    ## set dataloader
    trainLoader = torch.utils.data.DataLoader(dataset.ImageList(root=root_path, file_txt=train_txt, transform=train_transform), batch_size=BATCH_SIZE,
                                              shuffle=True, num_workers = 4, pin_memory= False, drop_last = True)
    testLoader = torch.utils.data.DataLoader(dataset.ImageList(root=root_path, file_txt=test_txt, transform=test_transform), batch_size=BATCH_SIZE,
                                             shuffle=False, num_workers = 4, pin_memory= False, drop_last = True)

    # load model
    # vgg16 = VGG16().cuda()
    # head = Arcface().cuda()
    resnet = torch.nn.DataParallel(resnet18()).to(device)
    # head = torch.nn.DataParallel(Arcface()).to(device)
    head = torch.nn.DataParallel(ArcMarginProduct()).to(device)

    # Loss and Optimizer
    cost = tnn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD({'params': vgg16.parameters() + head.parameters(), 'weight_decay': 5e-4},
    #                             lr = LEARNING_RATE,
    #                             momentum = 0.9)
    optimizer = torch.optim.SGD([{'params': resnet.parameters(), 'weight_decay': 5e-4},
                                 {'params': head.parameters(), 'weight_decay': 5e-4}],
                                lr=LEARNING_RATE,
                                momentum=0.9)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 80, 100], gamma=0.1, last_epoch=-1)

    for epoch in range(EPOCH):
        scheduler.step()
        train_epoch_loss, train_correct = train_loss.train(trainLoader,  optimizer, resnet, head, cost, EPOCH, epoch, device)
        test_epoch_loss, test_correct = valid_loss.test(testLoader, resnet, head, cost, device)

        print_with_time('train_epoch:{}, train_epoch_loss:{}, test_epoch_loss:{}, train_correct:{}, '
                        'test_correct:{}'.format(epoch, train_epoch_loss, test_epoch_loss, train_correct, test_correct))

        torch.save(resnet.state_dict(), log_save + 'resnet_' + str(epoch) + '.pth')
        torch.save(head.state_dict(), log_save + 'head_' + str(epoch) + '.pth')

    print('hello world!')



