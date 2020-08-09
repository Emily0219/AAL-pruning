import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
from torch.autograd import Variable
import torch
import time


# 新剪枝的精简模型

__all__ = ['ResNet_cifar10_small', 'resnet20_small', 'resnet32_small',
           'resnet56_small', 'resnet110_small']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

default_cfg_cifar10 = {
    '18': [16, 16, 16, 16, 16, 32, 32, 32, 32, 64, 64, 64, 64],
    '20': [16, 16, 16, 16, 16, 16, 16, 32, 32, 32, 32, 32, 32, 64, 64, 64, 64, 64, 64],
    '32': [16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
           32, 32, 32, 32, 32, 32, 32, 32, 32, 32,
           64, 64, 64, 64, 64, 64, 64, 64, 64, 64],
    '56': [16,
           16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
           32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32,
           64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64],
    '110': [16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
            16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
            32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32,
            32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32,
            64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
            64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64],
}


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes_before_prune, cfg,
                 index, bn_value, stride=1,
                 downsample=None, flag=False):
        super(BasicBlock, self).__init__()
        if flag:
            self.conv1 = conv3x3(planes_before_prune, cfg[0], stride)
        else:
            self.conv1 = conv3x3(inplanes, cfg[0], stride)
        self.bn1 = nn.BatchNorm2d(cfg[0])
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(cfg[0], cfg[1])
        self.bn2 = nn.BatchNorm2d(cfg[1])
        self.downsample = downsample
        self.stride = stride

        # for residual index match
        self.index = index
        # for bn add
        self.bn_value = bn_value
        self.planes_before_prune = planes_before_prune

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        # setting: without index match
        print("residual size{},out size{} ".format(residual.size(), out.size()))

        # setting: with index match
        if residual.device == torch.device('cpu'):
            residual = residual + self.bn_value
            residual.index_add_(1, self.index, out)
        else:
            residual = residual + self.bn_value.cuda()
            residual.index_add_(1, self.index.cuda(), out)

        residual = self.relu(residual)

        return residual


class Bottleneck(nn.Module):
    # expansion is not accurately equals to 4
    expansion = 4

    def __init__(self, inplanes, planes_before_prune, cfg,
                 index, bn_value, stride=1,
                 downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, cfg[0], kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(cfg[0])
        self.conv2 = nn.Conv2d(cfg[0], cfg[1], kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(cfg[1])

        # setting: for accuracy expansion
        self.conv3 = nn.Conv2d(cfg[1], cfg[2], kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(cfg[2])

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        # for residual index match
        self.index = index
        # for bn add
        self.bn_value = bn_value

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

        # setting: without index match
        print("residual size{},out size{} ".format(residual.size(), out.size()))

        # setting: with index match
        if residual.device == torch.device('cpu'):
            residual = residual + self.bn_value
            residual.index_add_(1, self.index, out)
        else:
            residual = residual + self.bn_value.cuda()
            residual.index_add_(1, self.index.cuda(), out)

        residual = self.relu(residual)

        return residual


class ResNet_cifar10_small(nn.Module):
    def __init__(self, block, index, bn_value, cfg, depth=20, num_classes=10):
        super(ResNet_cifar10_small, self).__init__()
        self.inplanes = 16
        # self.inplanes = cfg[0]
        self.cfg = cfg
        n = int((depth - 2) / 6)
        self.layer = [n, n, n]
        # self.conv1 = nn.Conv2d(3, cfg[0], kernel_size=3, stride=1, padding=1,
        #                        bias=False)
        # self.bn1 = nn.BatchNorm2d(cfg[0])
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = lambda x: x

        # setting: expansion may not accuracy equal to 4
        self.index_layer1 = {key: index[key] for key in index.keys() if 'layer1' in key}
        self.index_layer2 = {key: index[key] for key in index.keys() if 'layer2' in key}
        self.index_layer3 = {key: index[key] for key in index.keys() if 'layer3' in key}
        self.index_layer4 = {key: index[key] for key in index.keys() if 'layer4' in key}
        self.bn_layer1 = {key: bn_value[key] for key in bn_value.keys() if 'layer1' in key}
        self.bn_layer2 = {key: bn_value[key] for key in bn_value.keys() if 'layer2' in key}
        self.bn_layer3 = {key: bn_value[key] for key in bn_value.keys() if 'layer3' in key}
        self.bn_layer4 = {key: bn_value[key] for key in bn_value.keys() if 'layer4' in key}
        # print("bn_layer1", bn_layer1.keys(), bn_layer2.keys(), bn_layer3.keys(), bn_layer4.keys())

        self.layer1 = self._make_layer(block, 16, n, cfg[1:2 * n + 1], self.index_layer1, self.bn_layer1, flag=True)
        self.layer2 = self._make_layer(block, 32, n, cfg[2 * n + 1: 4 * n + 1], self.index_layer2, self.bn_layer2, stride=2)
        self.layer3 = self._make_layer(block, 64, n, cfg[4 * n + 1: 6 * n + 1], self.index_layer3, self.bn_layer3, stride=2)
        self.layer4 = lambda x: x
        # if block == BasicBlock:
        #     self.layer1 = self._make_layer(block, 64, layers[0], cfg[1: 2*layers[0]+1],
        #                                    self.index_layer1, self.bn_layer1, flag=True)
        #     self.layer2 = self._make_layer(block, 128, layers[1],
        #                                    cfg[2*layers[0]+1: 2*(layers[0]+layers[1])+1],
        #                                    self.index_layer2, self.bn_layer2, stride=2)
        #     self.layer3 = self._make_layer(block, 256, layers[2],
        #                                    cfg[2*(layers[0]+layers[1])+1: -2*layers[3]],
        #                                    self.index_layer3, self.bn_layer3, stride=2)
        #     self.layer4 = self._make_layer(block, 512, layers[3], cfg[-2*layers[3]:],
        #                                    self.index_layer4, self.bn_layer4, stride=2)
        # else:
        #     self.layer1 = self._make_layer(block, 64, layers[0], cfg[1: 3*layers[0]+1],
        #                                    self.index_layer1, self.bn_layer1)
        #     self.layer2 = self._make_layer(block, 128, layers[1], cfg[3*layers[0]+1: 3*(layers[0]+layers[1])+1],
        #                                    self.index_layer2, self.bn_layer2, stride=2)
        #     self.layer3 = self._make_layer(block, 256, layers[2], cfg[3*(layers[0]+layers[1])+1: -3*layers[3]],
        #                                    self.index_layer3, self.bn_layer3, stride=2)
        #     self.layer4 = self._make_layer(block, 512, layers[3], cfg[-3*layers[3]:],
        #                                    self.index_layer4, self.bn_layer4, stride=2)

        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def _make_layer(self, block, planes_before_prune, blocks, cfg,
                               index, bn_layer, stride=1, flag=False):
        downsample = None
        if stride != 1 or self.inplanes != planes_before_prune * block.expansion:
            if block == Bottleneck or (block == BasicBlock and self.inplanes != self.cfg[0]):
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes_before_prune * block.expansion,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes_before_prune * block.expansion),
                )

        # setting: accu number for_construct expansion
        if block == BasicBlock:
            index_block_0_dict = {key: index[key] for key in index.keys() if '0.conv2' in key}
            index_block_0_value = list(index_block_0_dict.values())[0]
        else:
            index_block_0_dict = {key: index[key] for key in index.keys() if '0.conv3' in key}
            index_block_0_value = list(index_block_0_dict.values())[0]
        bn_layer_0_value = list(bn_layer.values())[0]
        layers = []
        if block == BasicBlock:
            layers.append(block(self.inplanes, planes_before_prune, cfg[0:2], index_block_0_value,
                                bn_layer_0_value, stride, downsample, flag))
        else:
            layers.append(block(self.inplanes, planes_before_prune, cfg[0:3], index_block_0_value,
                                bn_layer_0_value, stride, downsample))
        self.inplanes = planes_before_prune * block.expansion

        for i in range(1, blocks):
            if block == BasicBlock:
                index_block_i_dict = {key: index[key] for key in index.keys() if (str(i) + '.conv2') in key}
                index_block_i_value = list(index_block_i_dict.values())[0]

                bn_layer_i = {key: bn_layer[key] for key in bn_layer.keys() if (str(i) + '.bn2') in key}
                bn_layer_i_value = list(bn_layer_i.values())[0]
                layers.append(block(self.inplanes, planes_before_prune, cfg[2*i: 2*i+2],
                              index_block_i_value,
                              bn_layer_i_value,
                              ))
            else:
                index_block_i_dict = {key: index[key] for key in index.keys() if (str(i) + '.conv3') in key}
                index_block_i_value = list(index_block_i_dict.values())[0]

                bn_layer_i = {key: bn_layer[key] for key in bn_layer.keys() if (str(i) + '.bn3') in key}
                bn_layer_i_value = list(bn_layer_i.values())[0]
                layers.append(block(self.inplanes, planes_before_prune, cfg[3*i: 3*i+3],
                              index_block_i_value,
                              bn_layer_i_value,
                              ))

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
        x = self.fc(x)

        return x


def resnet20_small(pretrained=False, **kwargs):
    model = ResNet_cifar10_small(BasicBlock, depth=20, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet32_small(pretrained=False, **kwargs):
    model = ResNet_cifar10_small(BasicBlock, depth=32, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet56_small(pretrained=False, **kwargs):
    model = ResNet_cifar10_small(Bottleneck, depth=56, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet110_small(pretrained=False, **kwargs):
    model = ResNet_cifar10_small(Bottleneck, depth=110, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model
