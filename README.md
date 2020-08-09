# Filter Pruning for Deep Convolutional Neural Networks via Auxiliary Attention

Implementation with PyTorch. 

## Requirements
- Python 3.6
- PyTorch 1.3.1
- TorchVision 0.4.2

## Dataset prepare
### ImageNet
Download the ImageNet dataset from [here](http://image-net.org/download-images).
No need to split the val set into corresponding folders because we provide api codes to load data:
```
from mydataset.imagenet_dataset import ImageNetDataset
```
### CIFAR10
Use ```torchvision.datasets.CIFAR10()```


## Train original model from scratch
```
python main_resnet_imagenet_org_multigpu.py --model resnet --depth 50 -b 256 -j 16 --gpus 0,1 --epoch 100 
python main_resnet_cifar10_org.py --model resnet --depth 56 -b 256 -j 8 --gpus 0 --epoch 200
```

## Train a pruned model with AAL
- Train each model from scratch by default.
```
python prune_resnet_imagenet_multigpu.py --model resnet_bnat --depth 50 -b 256 -j 16 --gpus 0,1 --epoch 100 
python prune_resnet_ciafr10.py --model resnet_bnat --depth 56 -b 256 -j 8 --gpus 0 --epoch 200            
```
- Generate a pruned model.
    * resnet18/resnet34     ```get_imagenet_small_resnet18_34.py```
    * resnet50/resnet101    ```get_imagenet_small_resnet50_101.py```
    * resnet_cifar10_20/resnet_cifar10_32/resnet_cifar10_56/resnet_cifar10_110    ```get_cifar10_small_resnet.py```
```
python get_imagenet_small_resnet18_34.py --model resnet_bnat --depth 18 --get_small --resume [path to the model with AAL]
python get_imagenet_small_resnet50_101.py --model resnet_bnat --depth 50 --get_small --resume [path to the model with AAL]
python get_cifar10_small_resnet.py --model resnet_bnat --depth 56 --get_small --resume [path to the model with AAL]
```
