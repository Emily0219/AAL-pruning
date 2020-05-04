# BNN.pytorch
Binarized Neural Network (BNN) for pytorch
This is the pytorch version for the BNN code, fro VGG and resnet models
Link to the paper: https://papers.nips.cc/paper/6573-binarized-neural-networks

The code is based on https://github.com/eladhoffer/convNet.pytorch
Please install torch and torchvision by following the instructions at: http://pytorch.org/
To run resnet18 for cifar10 dataset use: python main_binary.py --model resnet_binary --save resnet18_binary --dataset cifar10

##主要修改部分：
* binarized_modules(重定义Binarize函数，增加Channel Attention层)，
* vgg16（full precison vgg16定义）
* ~~vgg_cifar10_binary(加入attention系数层之后的vgg16)~~ 这个应该是可量化的vgg
* vgg16_cifar10_bnat.py(加入attention系数层之后的vgg16)
###新增：
* resnet_bnat.py(加入attention系数层之后的vgg16)

##训练全精度：
```
python3 main_binary.py --model vgg16 --save vgg16_cifar10 --dataset cifar10 --gpus 0 --epoch 300
python3 main_resnet_cifar10.py --model resnet  --depth 56 \
                               --dataset cifar10 --gpus 1 --epoch 200

```

##训练带attention量化系数：
###从零训练
```
python3 main_binary.py --model vgg16_cifar10_bnat --save vgg16_cifar10_bnat --dataset cifar10 --gpus 0 --epoch 300
python3 main_resnet_cifar10.py --model resnet_bnat  --depth 56 \
                               --dataset cifar10 --gpus 1 --epoch 200
                               
python main_resnet_cifar10.py --epochs 300 --model resnet_bnat --norm True      # norm就是加了一个L1正则项                         

```
###接着全精度模型带着量化层训练
```
python3 main_resnet_cifar10_pretrained.py --model resnet_bnat --depth 56 \
                                          --dataset cifar10 --gpus 1 --epoch 200 \
                                          --pretrained [path to float model]
                                          
python3 main_resnet_cifar10_pretrained.py --model resnet_bnat --depth 56 \
                                          --dataset cifar10 --gpus 1 --epoch 200 \
                                          --pretrained results/resnet_bnat56_2020-01-09_13-02-22/checkpoint.pth.tar

```

###根据训练好的量化层剪枝后微调
```
python3 finetune_binary.py --model resnet_bnat_pruned --depth 56 \
                           --dataset cifar10 --gpus 1 --epoch 200 \
                           --resume [path to model with bnan layers]


```


##初步结果： 
 
* vgg16 (full precision)  91.35 
* vgg16_bnat (-0.5< x <0.5 量化为0，其它为1)       90.52
* vgg16_bnat (-0.25< x <0.25 量化为0，其它为1)     90.95
* vgg16_bnat (-0.1< x <0.1 量化为0，其它为1)       91.39
* vgg16_bnat (x <0 量化为0，其它为1)               89.32
* vgg16_bnat (x <0 量化为-1，其它为1)               91.67

##下一步：
* Resnet实验
   * 先只剪第一层,　resnet18；如果是resnet50，则可以剪前两层
* 真正剪枝finetuning后实验