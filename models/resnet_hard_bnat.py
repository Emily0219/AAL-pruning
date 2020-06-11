import torch.nn as nn
import torch
import torchvision.transforms as transforms
import math, logging
from models.binarized_modules import BinarizeAttention, Binarize
import numpy as np


# 是适应cifar10的结构，且剪完枝要去掉bnan层
__all__ = ['resnet_bnat_pruned', 'resnet_hard_prune']


default_cfg = {
    '18': [64,
           64, 64, 64, 64,
           128, 128, 128, 128,
           256, 256, 256, 256,
           512, 512, 512, 512],

    '34': [64,
           64, 64, 64, 64, 64, 64,
           128, 128, 128, 128, 128, 128, 128, 128,
           256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256,
           512, 512, 512, 512, 512, 512],

    '50': [64,   # 0
           64, 64, 256, 64, 64, 256, 64, 64, 256,  # 9
           128, 128, 512, 128, 128, 512, 128, 128, 512, 128, 128, 512,  # 21
           256, 256, 1024, 256, 256, 1024, 256, 256, 1024, 256, 256, 1024, 256, 256, 1024, 256, 256, 1024,  # 39
           512, 512, 2048, 512, 512, 2048, 512, 512, 2048],   # 48
    '101': [64,
            64, 64, 256, 64, 64, 256, 64, 64, 256,
            128, 128, 512, 128, 128, 512, 128, 128, 512, 128, 128, 512,
            256, 256, 1024, 256, 256, 1024, 256, 256, 1024, 256, 256, 1024, 256, 256, 1024, 256, 256, 1024, 256, 256, 1024,
            256, 256, 1024, 256, 256, 1024, 256, 256, 1024, 256, 256, 1024, 256, 256, 1024, 256, 256, 1024, 256, 256, 1024,
            256, 256, 1024, 256, 256, 1024, 256, 256, 1024, 256, 256, 1024, 256, 256, 1024, 256, 256, 1024, 256, 256, 1024,
            256, 256, 1024, 256, 256, 1024,
            512, 512, 2048, 512, 512, 2048, 512, 512, 2048],
    '152': [64,
            64, 64, 256, 64, 64, 256, 64, 64, 256,
            128, 128, 512, 128, 128, 512, 128, 128, 512, 128, 128, 512, 128, 128, 512, 128, 128, 512, 128, 128, 512, 128, 128, 512,
            256, 256, 1024, 256, 256, 1024, 256, 256, 1024, 256, 256, 1024, 256, 256, 1024, 256, 256, 1024, 256, 256, 1024,
            256, 256, 1024, 256, 256, 1024, 256, 256, 1024, 256, 256, 1024, 256, 256, 1024, 256, 256, 1024, 256, 256, 1024,
            256, 256, 1024, 256, 256, 1024, 256, 256, 1024, 256, 256, 1024, 256, 256, 1024, 256, 256, 1024, 256, 256, 1024,
            256, 256, 1024, 256, 256, 1024, 256, 256, 1024, 256, 256, 1024, 256, 256, 1024, 256, 256, 1024, 256, 256, 1024,
            256, 256, 1024, 256, 256, 1024, 256, 256, 1024, 256, 256, 1024, 256, 256, 1024, 256, 256, 1024, 256, 256, 1024,
            256, 256, 1024,
            512, 512, 2048, 512, 512, 2048, 512, 512, 2048]

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


def init_model(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, cfg, stride=1, downsample=None, mode=True):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, cfg[0], stride)
        if mode:
            self.bnan1 = BinarizeAttention(cfg[0])
        self.bn1 = nn.BatchNorm2d(cfg[0])
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(cfg[0], planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.mode = mode

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        if self.mode:
            out = self.bnan1(out)
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

    def __init__(self, inplanes, planes, cfg, stride=1, downsample=None, mode=True):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, cfg[0], kernel_size=1, bias=False)
        if mode:
            self.bnan1 = BinarizeAttention(cfg[0])
        self.bn1 = nn.BatchNorm2d(cfg[0])

        self.conv2 = nn.Conv2d(cfg[0], cfg[1], kernel_size=3, stride=stride,
                               padding=1, bias=False)
        if mode:
            self.bnan2 = BinarizeAttention(cfg[1])
        self.bn2 = nn.BatchNorm2d(cfg[1])

        self.conv3 = nn.Conv2d(cfg[1], planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.mode = mode

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        if self.mode:
            out = self.bnan1(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if self.mode:
            out = self.bnan2(out)
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

    def __init__(self):
        super(ResNet, self).__init__()

    def _make_layer(self, block, planes, blocks, cfg=None, stride=1, mode=True):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        if block == BasicBlock:
            layers.append(block(self.inplanes, planes, cfg[0:2], stride, downsample, mode))
        else:
            layers.append(block(self.inplanes, planes, cfg[0:3], stride, downsample, mode))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            if block == BasicBlock:
                layers.append(block(self.inplanes, planes, cfg[2*i: 2*i+2], mode=mode))
            else:
                layers.append(block(self.inplanes, planes, cfg[3*i: 3*i+3], mode=mode))

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
        # x = self.bn(x)

        return x


class ResNet_imagenet(ResNet):
    def __init__(self,  mode=True, cfg=None, num_classes=1000,
                 block=Bottleneck, layers=[3, 4, 23, 3]):
        super(ResNet_imagenet, self).__init__()
        self.mode = mode  # 可以关掉量化感知层
        self.block = block
        self.layer = layers
        self.cfg = cfg

        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, cfg[0], kernel_size=7, stride=2, padding=3,     # imagenet
                               bias=False)
        self.bn1 = nn.BatchNorm2d(cfg[0])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        if block == BasicBlock:
            self.layer1 = self._make_layer(block, 64, layers[0], cfg=cfg[1: 2*layers[0]+1], mode=self.mode)  # layer是堆叠的块个数
            self.layer2 = self._make_layer(block, 128, layers[1],
                                           cfg=cfg[2*layers[0]+1: 2*(layers[0]+layers[1])+1], stride=2, mode=self.mode)
            self.layer3 = self._make_layer(block, 256, layers[2],
                                           cfg=cfg[2*(layers[0]+layers[1])+1: -2*layers[3]], stride=2, mode=self.mode)
            self.layer4 = self._make_layer(block, 512, layers[3], cfg=cfg[-2*layers[3]:], stride=2, mode=self.mode)  # 去掉了avgpool fc
        else:
            self.layer1 = self._make_layer(block, 64, layers[0], cfg=cfg[1: 3*layers[0]+1], mode=self.mode)
            self.layer2 = self._make_layer(block, 128, layers[1], cfg=cfg[3*layers[0]+1: 3*(layers[0]+layers[1])+1], stride=2, mode=self.mode)
            self.layer3 = self._make_layer(block, 256, layers[2], cfg=cfg[3*(layers[0]+layers[1])+1: -3*layers[3]], stride=2, mode=self.mode)
            self.layer4 = self._make_layer(block, 512, layers[3], cfg=cfg[-3*layers[3]:], stride=2, mode=self.mode)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        init_model(self)
        self.regime = {
            0: {'optimizer': 'SGD', 'lr': 1e-1,
                'weight_decay': 1e-4, 'momentum': 0.9},
            30: {'lr': 1e-2},
            60: {'lr': 1e-3, 'weight_decay': 0},
            90: {'lr': 1e-4}
        }


class ResNet_cifar10(ResNet):
    def __init__(self,  mode=True, cfg=None, num_classes=10,
                 block=BasicBlock, depth=56):
        super(ResNet_cifar10, self).__init__()
        self.mode = mode  # 可以关掉量化感知层
        self.block = block
        self.cfg = cfg

        self.inplanes = 16
        n = int((depth - 2) / 6)
        self.layer = [n, n, n]

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = lambda x: x
        self.layer1 = self._make_layer(block, 16, n, cfg=cfg[1:2*n+1], mode=self.mode)
        self.layer2 = self._make_layer(block, 32, n, cfg=cfg[2*n+1: 4*n+1], stride=2, mode=self.mode)
        self.layer3 = self._make_layer(block, 64, n, cfg=cfg[4*n+1: 6*n+1], stride=2, mode=self.mode)
        self.layer4 = lambda x: x
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                m.bias.data.zero_()

        self.regime = {  # he best
            0: {'optimizer': 'SGD', 'lr': 1e-1,
                'weight_decay': 5e-4, 'momentum': 0.9},
            60: {'lr': 2e-2},
            120: {'lr': 4e-3},
            160: {'lr': 8e-4}
        }


class ResNet_cifar100(ResNet):
    def __init__(self, mode=True, cfg=None, num_classes=100,
                 block=Bottleneck, layers=[3, 4, 23, 3]):
        super(ResNet_cifar100, self).__init__()
        self.mode = mode  # 可以关掉量化感知层
        self.block = block
        self.layer = layers
        self.cfg = cfg

        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1,   # !!!!!!!
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = lambda x: x
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        if block == BasicBlock:
            self.layer1 = self._make_layer(block, 64, layers[0], cfg=cfg[1: 2*layers[0]+1], mode=self.mode)  # layer是堆叠的块个数
            self.layer2 = self._make_layer(block, 128, layers[1],
                                           cfg=cfg[2*layers[0]+1: 2*(layers[0]+layers[1])+1], stride=2, mode=self.mode)
            self.layer3 = self._make_layer(block, 256, layers[2],
                                           cfg=cfg[2*(layers[0]+layers[1])+1: -2*layers[3]], stride=2, mode=self.mode)
            self.layer4 = self._make_layer(block, 512, layers[3], cfg=cfg[-2*layers[3]:], stride=2, mode=self.mode)  # 去掉了avgpool fc
        else:
            self.layer1 = self._make_layer(block, 64, layers[0], cfg=cfg[1: 3*layers[0]+1], mode=self.mode)
            self.layer2 = self._make_layer(block, 128, layers[1], cfg=cfg[3*layers[0]+1: 3*(layers[0]+layers[1])+1],
                                           stride=2, mode=self.mode)
            self.layer3 = self._make_layer(block, 256, layers[2], cfg=cfg[3*(layers[0]+layers[1])+1: -3*layers[3]],
                                           stride=2, mode=self.mode)
            self.layer4 = self._make_layer(block, 512, layers[3], cfg=cfg[-3*layers[3]:], stride=2, mode=self.mode)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.regime = {  # he  ---best   cifar100  resnet50/32/18/101
            0: {'optimizer': 'SGD', 'lr': 0.1,  # 0.01
                'weight_decay': 5e-4, 'momentum': 0.9},
            60: {'lr': 0.02},
            120: {'lr': 0.004},
            160: {'lr': 0.0008}
        }


def resnet_bnat_pruned(**kwargs):
    mode, cfg, num_classes, depth, dataset = map(
        kwargs.get, ['mode', 'cfg', 'num_classes', 'depth', 'dataset'])
    cfg = cfg or None
    mode = mode or False
    if dataset == 'imagenet':
        depth = depth or 50
        if cfg is None:
            cfg = default_cfg[str(depth)]
        num_classes = num_classes or 1000
        if depth == 18:
            return ResNet_imagenet(mode=mode, cfg=cfg, num_classes=num_classes,
                                   block=BasicBlock, layers=[2, 2, 2, 2])
        if depth == 34:
            return ResNet_imagenet(mode=mode, cfg=cfg, num_classes=num_classes,
                                   block=BasicBlock, layers=[3, 4, 6, 3])
        if depth == 50:
            return ResNet_imagenet(mode=mode, cfg=cfg, num_classes=num_classes,
                                   block=Bottleneck, layers=[3, 4, 6, 3])
        if depth == 101:
            return ResNet_imagenet(mode=mode, cfg=cfg, num_classes=num_classes,
                                   block=Bottleneck, layers=[3, 4, 23, 3])
        if depth == 152:
            return ResNet_imagenet(mode=mode, cfg=cfg, num_classes=num_classes,
                                   block=Bottleneck, layers=[3, 8, 36, 3])

    elif dataset == 'cifar10':
        num_classes = num_classes or 10
        depth = depth or 56  # 56     # resnet18
        if cfg is None:
            cfg = default_cfg_cifar10[str(depth)]
        return ResNet_cifar10(mode=mode, cfg=cfg, num_classes=num_classes,
                              block=BasicBlock, depth=depth)

    elif dataset == 'cifar100':
        depth = depth or 50
        if cfg is None:
            cfg = default_cfg[str(depth)]
        num_classes = num_classes or 100
        if depth == 18:
            return ResNet_cifar100(mode=mode, cfg=cfg, num_classes=num_classes,
                                   block=BasicBlock, layers=[2, 2, 2, 2])
        if depth == 34:
            return ResNet_cifar100(mode=mode, cfg=cfg, num_classes=num_classes,
                                   block=BasicBlock, layers=[3, 4, 6, 3])
        if depth == 50:
            return ResNet_cifar100(mode=mode, cfg=cfg, num_classes=num_classes,
                                   block=Bottleneck, layers=[3, 4, 6, 3])
        if depth == 101:
            return ResNet_cifar100(mode=mode, cfg=cfg, num_classes=num_classes,
                                   block=Bottleneck, layers=[3, 4, 23, 3])
        if depth == 152:
            return ResNet_cifar100(mode=mode, cfg=cfg, num_classes=num_classes,
                                   block=Bottleneck, layers=[3, 8, 36, 3])
    else:
        print('Error! Now only support ImageNet and CIFAR10/100!')


def resnet_hard_prune(model, mode, depth=50, dataset='cifar10'):
    if dataset=='cifar10':
        cfg = [16]
        cfg_mask = [torch.ones(16)]
    else:
        cfg = [64]
        cfg_mask = [torch.ones(64)]
    block = model.block
    layer = model.layer
    for n, p in list(model.named_parameters()):
        if hasattr(p, 'org'):
            cfg.append(int(torch.sum(torch.abs(p.data))))
            cfg_mask.append(p.data.clone().squeeze())  # p.data就是mask
            # print(sum((Binarize(p.org.data))), p.size(0))
        if block == BasicBlock and 'bn2.weight' in n:
            cfg.append(int(p.size(0)))
            cfg_mask.append(torch.ones(p.data.shape))
            # print(n, p.size(0))
        if block == Bottleneck and 'bn3.weight' in n:
            cfg.append(int(p.size(0)))
            cfg_mask.append(torch.ones(p.data.shape))
    print(cfg)

    if block == BasicBlock:
        # 降采样层不太一样？layer1没有降采样，234都有
        down_index = 2 * layer[0] + 3
        downsample = [down_index]
        bn3 = [2]
        bn3_per_block = 2
        for i in range(1, layer[0]):
            bn3_per_block += 2
            bn3.append(bn3_per_block)
        for i in range(1, len(layer)-1):
            down_index += 2 * layer[i] + 1
            downsample.append(down_index)
        for i in range(len(downsample)):
            bn3_per_block = downsample[i]
            bn3.append(bn3_per_block - 1)  # 降采样的之前的bn3
            for j in range(1, layer[i]):
                bn3_per_block += 2
                bn3.append(bn3_per_block)  # 剩余的bn3
    else:
        down_index = 4
        downsample = [4]
        bn3 = []
        for i in range(3):
            down_index += 3 * layer[i] + 1
            downsample.append(down_index)
        for i in range(len(downsample)):
            bn3_per_block = downsample[i]
            bn3.append(bn3_per_block - 1)  # 降采样的之前的bn3
            for j in range(1, layer[i]):
                bn3_per_block += 3
                bn3.append(bn3_per_block)  # 剩余的bn3
    # print(downsample)
    # print(bn3)
    if dataset=='cifar10':
        func = default_cfg_cifar10[str(depth)]
    else:
        func = default_cfg[str(depth)]
    oldmodel = resnet_bnat_pruned(mode=False, cfg=func, depth=depth, dataset=dataset)
    oldstate = oldmodel.state_dict()
    tmp = model.state_dict()
    for k, v in oldstate.items():
        if k in tmp:
            oldstate[k] = tmp[k]
    oldmodel.load_state_dict(oldstate)
    newmodel = resnet_bnat_pruned(mode=mode, cfg=cfg, depth=depth, dataset=dataset)
    # print(newmodel)

    layer_id_in_cfg = 0
    start_mask = torch.ones(3)
    end_mask = cfg_mask[layer_id_in_cfg]
    bn_counter = 0
    conv_counter = 0
    for [m0, m1] in zip(oldmodel.modules(), newmodel.modules()):
    # for [m0, m1] in zip(model.modules(), newmodel.modules()):
        # print(type(m0))
        if isinstance(m0, nn.BatchNorm2d):
            # if bn_counter > bn3[-1]:
            #     break
            if bn_counter in bn3 or bn_counter in downsample \
                    or bn_counter > bn3[-1] or bn_counter == 0:  # 不剪的层即downsample和bn3和第一层
                # print(bn_counter)
                m1.weight.data = m0.weight.data.clone()
                m1.bias.data = m0.bias.data.clone()
                m1.running_mean = m0.running_mean.clone()
                m1.running_var = m0.running_var.clone()
                m1.num_batches_tracked = m0.num_batches_tracked.clone()
            else:  # 剩余的bn层只跟上一个卷积层的输出通道都有关系
                idx_out = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
                if idx_out.size == 1:
                    idx_out = np.resize(idx_out, (1,))
                m1.weight.data = m0.weight.data[idx_out.tolist()].clone()
                m1.bias.data = m0.bias.data[idx_out.tolist()].clone()
                m1.running_mean = m0.running_mean[idx_out.tolist()].clone()
                m1.running_var = m0.running_var[idx_out.tolist()].clone()
                m1.num_batches_tracked = m0.num_batches_tracked.clone()
            # 不是支路上的BN层才会换下一个mask
            if not bn_counter in downsample:
                layer_id_in_cfg += 1
                start_mask = end_mask.clone()
                if layer_id_in_cfg < len(cfg_mask):
                    end_mask = cfg_mask[layer_id_in_cfg]
            # print(bn_counter)
            bn_counter += 1
        elif isinstance(m0, nn.Conv2d):
            if conv_counter in downsample or conv_counter > bn3[-1] \
                    or conv_counter == 0:  # 只有shortcut支路上的卷积层和第一层不变
                m1.weight.data = m0.weight.data.clone()
                if m0.bias is not None:
                    m1.bias.data = m0.bias.data.clone()
                print('CONV: {:d}, In shape: {:d}, Out shape {:d}.'.format(conv_counter, m1.weight.data.shape[1],
                                                                           m1.weight.data.shape[0]))
            elif conv_counter in bn3:  # conv3只变输入通道，输出通道数不变
                idx_in = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
                m1.weight.data = m0.weight.data[:, idx_in.tolist(), :, :].clone()
                if m0.bias is not None:
                    m1.bias.data = m0.bias.data.clone()
                print('CONV: {:d}, In shape: {:d}, Out shape {:d}.'.format(conv_counter, idx_in.size,
                                                                           m1.weight.data.shape[0]))
            else:  # 其余都变
                idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
                idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
                print('CONV: {:d}, In shape: {:d}, Out shape {:d}.'.format(conv_counter, idx0.size, idx1.size))
                if idx0.size == 1:
                    idx0 = np.resize(idx0, (1,))
                if idx1.size == 1:
                    idx1 = np.resize(idx1, (1,))
                w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()
                w1 = w1[idx1.tolist(), :, :, :].clone()
                m1.weight.data = w1.clone()
                if m0.bias is not None:
                    b1 = m0.bias.data[idx1.tolist()].clone()
                    m1.bias.data = b1.clone()
            conv_counter += 1
        elif isinstance(m0, nn.Linear):
            m1.weight.data = m0.weight.data.clone()
            if m0.bias is not None:
                m1.bias.data = m0.bias.data.clone()
        elif isinstance(m0, BinarizeAttention):
            idx_out = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            if idx_out.size == 1:
                idx_out = np.resize(idx_out, (1,))
            m1.weight.data = m0.weight.data[idx_out.tolist()].clone()
            m1.weight.org = m0.weight.org[idx_out.tolist()].clone()

    num_params_ori = sum([param.nelement() for param in model.parameters()])
    num_params_prune = sum([param.nelement() for param in newmodel.parameters()])
    compress_ratio = num_params_prune/num_params_ori
    print('total pruning ratio {} '.format(1 - compress_ratio))
    # logging.info('total pruning ratio {} '.format(1 - compress_ratio))

    float_conv = sum([p.nelement() if 'conv' in n else 0 for n, p in model.named_parameters()])
    prune_conv = sum([p.nelement() if 'conv' in n else 0 for n, p in newmodel.named_parameters()])
    compress_ratio = prune_conv / float_conv
    print(compress_ratio)
    return newmodel, compress_ratio, cfg, cfg_mask


if __name__ == '__main__':
    model = resnet_bnat_pruned(dataset='cifar10', depth=56, mode=True)
    print(model)
    print(model.cfg)
    x = torch.rand(1, 3, 32, 32)
    # x = torch.rand(1, 3, 224, 224)
    output = model.forward(x)
    print(output)
    # mydict = model.state_dict()

    # print(mydict.keys)
    newmodel, _, _, _ = resnet_hard_prune(model, False, depth=56, dataset='cifar10')

    for p in list(model.parameters()):
        if hasattr(p, 'org'):
            p.data.copy_(p.org)

    output = newmodel.forward(x)
    print(output)