# https://github.com/pytorch/vision/blob/master/torchvision/models/__init__.py
import argparse
import os,sys
import shutil

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import models
import random
import numpy as np
import copy
from ast import literal_eval
from datetime import datetime
from models.binarized_modules import BinarizeAttention
from thop import profile, clever_format
from pytorch_tools import print_model_param_flops


model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--save_dir', type=str, default='eval/', help='Folder to save checkpoints and log.')
parser.add_argument('--model', '-a', metavar='MODEL', default='resnet_bnat',
                    choices=model_names,
                    help='model architecture: ' +
                    ' | '.join(model_names) +
                    ' (default: alexnet)')
parser.add_argument('--input_size', type=int, default=None,
                    help='image input size')
parser.add_argument('--model_config', default='',
                    help='additional architecture configuration')
parser.add_argument('--depth', type=int, default=18,   #34
                    help='resnet depth')

parser.add_argument('--resume',
                    default='resnet_bnat18_2020-05-01_01-40-24/checkpoint.pth.tar',
                    type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--get_small', dest='get_small', action='store_true', help='whether a big or small model')
parser.add_argument('--use_cuda', default=False, help='whether to use cuda')

args = parser.parse_args()
args.use_cuda = torch.cuda.is_available()

def main():
    best_prec1 = 0
    save = args.model + str(args.depth) + '_' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    save_path = os.path.join(args.save_dir, save)
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    # create model
    model = models.__dict__[args.model]
    model_config = {'dataset': 'imagenet', 'depth': args.depth}

    if args.model_config is not '':
        model_config = dict(model_config, **literal_eval(args.model_config))

    model = model(**model_config)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume, map_location='cpu')
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            bnan_state = checkpoint['bnan']
            state_dict = checkpoint['state_dict']
            model.load_state_dict(state_dict)

    cudnn.benchmark = True

    if args.get_small:
        # big_path = os.path.join(args.save_dir, "big_model.pth.tar")
        # torch.save(model, big_path)

        x = torch.rand(1, 3, 224, 224)

        small_model = get_small_model(model.cpu())
        small_path = os.path.join(save_path, "small_model.pth.tar")
        torch.save(small_model, small_path)

        output = small_model.forward(x)
        print(output.shape)
        if args.use_cuda:
            x = x.cuda()
            model = model.cuda()
            small_model = small_model.cuda()

        output = small_model(x)
        # MAC
        MAC, params = profile(model, inputs=(x,))
        print('MAC: {}, params: {}'.format(MAC, params))
        MAC, params = profile(small_model, inputs=(x,))
        print('MAC: {}, params: {}'.format(MAC, params))

        # FLOPs
        print_model_param_flops(model, [224, 224])
        print_model_param_flops(small_model, [224, 224])

        # compress ratio
        num_params_ori = sum([param.nelement() for param in model.parameters()])
        num_params_prune = sum([param.nelement() for param in small_model.parameters()])
        compress_ratio = num_params_prune / num_params_ori
        print(num_params_ori, num_params_prune)
        print('total pruning ratio {} '.format(1 - compress_ratio))

        float_conv = sum([p.nelement() if 'conv' in n else 0 for n, p in model.named_parameters()])
        prune_conv = sum([p.nelement() if 'conv' in n else 0 for n, p in small_model.named_parameters()])
        compress_ratio = prune_conv / float_conv
        print(float_conv, prune_conv, 1 - compress_ratio)


def save_checkpoint(state, is_best, filename, bestname):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, bestname)


def check_channel(tensor, cfg_mask, bnan_index):
    # AAL的mask与对应卷积核相乘，得到稀疏卷积核
    w1 = tensor.clone()
    tmp_mask = np.asarray(cfg_mask[bnan_index].cpu().numpy())
    for i in range(len(tmp_mask)):
        if tmp_mask[i]==0:
            w1[i, :, :, :] *= tmp_mask[i]
    size_0 = tensor.size()[0]
    tensor_resize = w1.view(size_0, -1)
    # indicator: if the channel contain all zeros
    channel_if_zero = np.zeros(size_0)
    for x in range(0, size_0, 1):
        channel_if_zero[x] = np.count_nonzero(tensor_resize[x].cpu().numpy()) != 0

    indices_nonzero = torch.LongTensor((channel_if_zero != 0).nonzero()[0])

    zeros = (channel_if_zero == 0).nonzero()[0]
    indices_zero = torch.LongTensor(zeros) if zeros != [] else []

    return indices_zero, indices_nonzero


def extract_para(big_model):
    '''
    :param model:
    :param batch_size:
    :return: num_for_construc: number of remaining filter,
             [conv1,stage1,stage1_expend,stage2,stage2_expend,stage3,stage3_expend,stage4,stage4_expend]

             kept_filter_per_layer: number of remaining filters for every layer
             kept_index_per_layer: filter index of remaining channel
             model: small model
    '''
    item = list(big_model.state_dict().items())
    print("length of state dict is", len(item))
    try:
        # assert len(item) in [102, 182, 267, 522]   # torch0.3版本
        # assert len(item) in [122, 218, 320, 626]  # torch1.3版本, without bnan
        assert len(item) in [139, 251]   # torch1.3版本, with bnan
        print("state dict length is one of 139, 251")
    except AssertionError as e:
        print("False state dict")

    cfg_mask = []
    for m in big_model.modules():
        if isinstance(m, BinarizeAttention):  # 剪枝前保存二值的mask
            cfg_mask.append(m.weight.data.clone().squeeze())  # p.data就是mask

    kept_index_per_layer = {}
    kept_filter_per_layer = {}
    pruned_index_per_layer = {}

    bnan_index = 0
    for x in range(0, len(item) - 2):  # 卷积层的weight，跳过所有bn\bnan\fc
        if 'conv' in item[x][0]:
            indices_zero, indices_nonzero = check_channel(item[x][1], cfg_mask, bnan_index)
            pruned_index_per_layer[item[x][0]] = indices_zero
            kept_index_per_layer[item[x][0]] = indices_nonzero
            kept_filter_per_layer[item[x][0]] = indices_nonzero.shape[0]
            bnan_index += 1
        elif 'downsample.0.weight' in item[x][0]:
            size_0 = item[x][1].size()[0]
            channel_if_zero = np.ones(size_0)
            indices_nonzero = torch.LongTensor((channel_if_zero != 0).nonzero()[0])

            pruned_index_per_layer[item[x][0]] = torch.LongTensor([])
            kept_index_per_layer[item[x][0]] = indices_nonzero
            # kept_filter_per_layer[item[x][0]] = indices_nonzero.shape[0]

    if len(item) == 139 or len(item) == 251:
        block_flag = "conv2"

    # number of nonzero channel in conv1, and four stages
    num_for_construct = []
    # for key in constrct_flag:
    #     num_for_construct.append(kept_filter_per_layer[key])
    for k, v in kept_filter_per_layer.items():  # 只统计卷积核保留的个数
        num_for_construct.append(v)

    bn_value = get_bn_value(big_model, block_flag, pruned_index_per_layer, kept_index_per_layer, num_for_construct[0])
    if len(item) == 139:
        small_model = models.resnet18_small(index=kept_index_per_layer, bn_value=bn_value,
                                            cfg=num_for_construct)
    if len(item) == 251:
        small_model = models.resnet34_small(index=kept_index_per_layer, bn_value=bn_value,
                                            cfg=num_for_construct)
    return kept_index_per_layer, pruned_index_per_layer, block_flag, small_model


def get_bn_value(big_model, block_flag, pruned_index_per_layer, kept_index_per_layer, start_conv_nums):
    big_model.eval()
    bn_flag = "bn3" if block_flag == "conv3" else "bn2"
    key_bn = [x for x in big_model.state_dict().keys() if bn_flag in x]
    # layer_flag_list = [[x[0:6], x[7], x[9:12], x] for x in key_bn if "weight" in x]
    layer_flag_list = [[x.split('.')[0], x.split('.')[1], x.split('.')[2], x] for x in key_bn if "weight" in x]
    # layer_flag_list = [['layer1', "0", "bn3",'layer1.0.bn3.weight']]
    bn_value = {}

    for layer_flag in layer_flag_list:
        module_bn = big_model._modules.get(layer_flag[0])._modules.get(layer_flag[1])._modules.get(layer_flag[2])
        num_feature = module_bn.num_features

        act_bn = module_bn(torch.zeros(1, num_feature, 1, 1))
        index_name = layer_flag[3].replace("bn", "conv")  # conv2.weight
        index = torch.LongTensor(pruned_index_per_layer[index_name])
        act_bn = torch.index_select(act_bn, 1, index)

        select = torch.zeros(1, num_feature, 1, 1)
        select.index_add_(1, index, act_bn)

        bn_value[layer_flag[3]] = select
    return bn_value


def get_small_model(big_model):
    indice_dict, pruned_index_per_layer, block_flag, small_model = extract_para(big_model)
    big_state_dict = big_model.state_dict()
    small_state_dict = {}
    keys_list = list(big_state_dict.keys())
    # print("keys_list", keys_list)
    for index, [key, value] in enumerate(big_state_dict.items()):
        # all the conv layer excluding downsample layer
        flag_conv_ex_down = not 'bn' in key and not 'downsample' in key and not 'fc' in key
        # downsample conv layer
        flag_down = 'downsample.0' in key
        # value for 'output' dimension: all the conv layer including downsample layer
        if flag_conv_ex_down or flag_down:
            if key == 'conv1.weight':
                small_state_dict[key] = value
            else:
                small_state_dict[key] = torch.index_select(value, 0, indice_dict[key])
            conv_index = keys_list.index(key)
            # 4 following bn layer, bn_weight, bn_bias, bn_runningmean, bn_runningvar
            if flag_conv_ex_down:
                for offset in range(2, 6, 1):  # conv要跳过bnan， down不用，要分开考虑
                    bn_key = keys_list[conv_index + offset]
                    if key == 'conv1.weight':
                        small_state_dict[bn_key] = big_state_dict[bn_key]
                    else:
                        small_state_dict[bn_key] = torch.index_select(big_state_dict[bn_key], 0, indice_dict[key])
                bn_key = keys_list[conv_index + 6]  # num_batchs_tracked
                small_state_dict[bn_key] = big_state_dict[bn_key]
            else:
                for offset in range(1, 5, 1):  # downsample
                    bn_key = keys_list[conv_index + offset]
                    small_state_dict[bn_key] = torch.index_select(big_state_dict[bn_key], 0, indice_dict[key])
                bn_key = keys_list[conv_index + 5]  # num_batchs_tracked
                small_state_dict[bn_key] = big_state_dict[bn_key]
            # value for 'input' dimension
            if flag_conv_ex_down:
                # first layer of first block
                if 'layer1.0.conv1.weight' in key:
                    continue
                # just conv1 of block, the input dimension should not change for shortcut
                elif not "conv1" in key:
                    conv_index = keys_list.index(key)
                    # get the last con layer
                    key_for_input = keys_list[conv_index - 7]
                    # print("key_for_input", key, key_for_input)
                    small_state_dict[key] = torch.index_select(small_state_dict[key], 1, indice_dict[key_for_input])
            # only the first downsample layer should change as conv1 reduced
            elif 'layer1.0.downsample.0.weight' in key:
                small_state_dict[key] = torch.index_select(small_state_dict[key], 1, indice_dict['conv1.weight'])
        elif 'fc' in key:
            small_state_dict[key] = value

    # if len(set(big_state_dict.keys()) - set(small_state_dict.keys())) != 0:
    #     print("different keys of big and small model",
    #           sorted(set(big_state_dict.keys()) - set(small_state_dict.keys())))
    #     for x, y in zip(small_state_dict.keys(), small_model.state_dict().keys()):
    #         if small_state_dict[x].size() != small_model.state_dict()[y].size():
    #             print("difference with model and dict", x, small_state_dict[x].size(),
    #                   small_model.state_dict()[y].size())

    small_model.load_state_dict(small_state_dict)

    return small_model


if __name__ == '__main__':
    main()