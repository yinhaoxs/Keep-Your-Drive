import torch
import torch.nn as nn
import math
import copy
from torchvision import models
import math
from torch.nn import Parameter
import sys
import torch.nn.functional as F

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101','resnet152']

# l2_norm
def l2_norm(input,axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, index=0, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.index = index

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, index=0, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.index = index

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_feature=512):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], 0)
        self.layer2 = self._make_layer(block, 128, layers[1], 1, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], 2, stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], 3, stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc_feature = nn.Linear(512 * block.expansion, num_feature)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        for m in self.modules():
            if isinstance(m, Bottleneck):
                m.bn3.weight.data.fill_(0)
            if isinstance(m, BasicBlock):
                m.bn2.weight.data.fill_(0)


    def _make_layer(self, block, planes, blocks, index, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, index, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        self.fc_input_feature = x
        x = self.fc_feature(x)
        x = l2_norm(x)

        return x



def resnet18(**kwargs):
    """Constructs a ResNet-18 model."""
    resnet18 = models.resnet18(pretrained=True)
    model = ResNet(BasicBlock, [2,2,2,2], **kwargs)
    pretrained_dict = resnet18.state_dict()
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    return model


# def resnet34(**kwargs):
#     """Constructs a ResNet-34 model."""
#     resnet34 = models.resnet34(pretrained=True)
#     model = ResNet(BasicBlock, [3,4,6,3], **kwargs)
#     pretrained_dict = resnet34.state_dict()
#     model_dict = model.state_dict()
#     pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
#     model_dict.update(pretrained_dict)
#     model.load_state_dict(model_dict)
#
#     return model

# def resnet50(**kwargs):
#     """Constructs a ResNet-50 model."""
#     resnet50 = models.resnet50(pretrained=True)
#     model = ResNet(Bottleneck, [3,4,6,3], **kwargs)
#     pretrained_dict = resnet50.state_dict()
#     model_dict = model.state_dict()
#     pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
#     model_dict.update(pretrained_dict)
#     model.load_state_dict(model_dict)
#
#     return model
#
# def resnet101(**kwargs):
#     """Constructs a ResNet-101 model."""
#     resnet101 = models.resnet101(pretrained=True)
#     model = ResNet(Bottleneck, [3,4,23,3], **kwargs)
#     pretrained_dict = resnet101.state_dict()
#     model_dict = model.state_dict()
#     pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
#     model_dict.update(pretrained_dict)
#     model.load_state_dict(model_dict)
#
#     return model
#
# def resnet152(**kwargs):
#     """Constructs a ResNet-152 model."""
#     resnet152 = models.resnet152(pretrained=True)
#     model = ResNet(Bottleneck, [3,8,36,3], **kwargs)
#     pretrained_dict = resnet152.state_dict()
#     model_dict = model.state_dict()
#     pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
#     model_dict.update(pretrained_dict)
#     model.load_state_dict(model_dict)
#
#     return model

##################################  Arcface head #############################################################
class Arcface(nn.Module):
    # implementation of additive margin softmax loss in https://arxiv.org/abs/1801.05599
    def __init__(self, embedding_size=512, classnum=100,  s=64., m=0.5):
        super(Arcface, self).__init__()
        self.classnum = classnum
        self.kernel = Parameter(torch.Tensor(embedding_size,classnum))
        # initial kernel
        self.kernel.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)
        self.m = m # the margin value, default is 0.5
        self.s = s # scalar value default is 64, see normface https://arxiv.org/abs/1704.06369
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.mm = self.sin_m * m  # issue 1
        self.threshold = math.cos(math.pi - m)


    def forward(self, embbedings, label):
        # weights norm
        nB = len(embbedings)
        kernel_norm = l2_norm(self.kernel,axis=0)
        # cos(theta+m)
        cos_theta = torch.mm(embbedings,kernel_norm)
        # output = torch.mm(embbedings,kernel_norm)
        cos_theta = cos_theta.clamp(-1,1) # for numerical stability
        cos_theta_2 = torch.pow(cos_theta, 2)
        sin_theta_2 = 1 - cos_theta_2
        sin_theta = torch.sqrt(sin_theta_2)
        cos_theta_m = (cos_theta * self.cos_m - sin_theta * self.sin_m)
        # this condition controls the theta+m should in range [0, pi]
        #      0<=theta+m<=pi
        #     -m<=theta<=pi-m
        cond_v = cos_theta - self.threshold
        cond_mask = cond_v <= 0
        keep_val = (cos_theta - self.mm) # when theta not in [0,pi], use cosface instead
        cos_theta_m[cond_mask] = keep_val[cond_mask]
        output = cos_theta * 1.0 # a little bit hacky way to prevent in_place operation on cos_theta
        idx_ = torch.arange(0, nB, dtype=torch.long)
        output[idx_, label] = cos_theta_m[idx_, label]
        output *= self.s # scale up in order to make softmax work, first introduced in normface

        return output


# ArcFace
class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin

            cos(theta + m)
        """

    def __init__(self, in_features=512, out_features=100, s=30.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        # Parameter 的用途：
        # 将一个不可训练的类型Tensor转换成可以训练的类型parameter
        # 并将这个parameter绑定到这个module里面
        # net.parameter()中就有这个绑定的parameter，所以在参数优化的时候可以进行优化的
        # https://www.jianshu.com/p/d8b77cc02410
        # 初始化权重
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        # torch.nn.functional.linear(input, weight, bias=None)
        # y=x*W^T+b
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        # cos(a+b)=cos(a)*cos(b)-size(a)*sin(b)
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            # torch.where(condition, x, y) → Tensor
            # condition (ByteTensor) – When True (nonzero), yield x, otherwise yield y
            # x (Tensor) – values selected at indices where condition is True
            # y (Tensor) – values selected at indices where condition is False
            # return:
            # A tensor of shape equal to the broadcasted shape of condition, x, y
            # cosine>0 means two class is similar, thus use the phi which make it
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        # 将cos(\theta + m)更新到tensor相应的位置中
        one_hot = torch.zeros(cosine.size(), device='cuda')
        # scatter_(dim, index, src)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s
        # print(output)

        return output
