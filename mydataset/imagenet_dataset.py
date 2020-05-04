import os
import pandas as pd
import numpy as np
import random
from torchvision import transforms
import cv2
from torch.utils.data import Dataset, DataLoader
import torch
from PIL import Image


# train_dir = '/public/ImageNet/ILSVRC/Data/CLS-LOC/train/'
# val_dir = '/public/ImageNet/ILSVRC/Data/CLS-LOC/val/'
train_dir = '/public/datasets/ILSVRC2012/train/'
val_dir = '/public/datasets/ILSVRC2012/img_val/'
info_dir = '/home/syr/'


class ImageNetDataset(Dataset):
    r"""
        Args:
            root_dir (str): path to n_frames_jpg folders.
            info_list (str): path to annotation file.
            clip_len (int): Determines how many frames are there in each clip. Defaults to 16.
            transform : Data augmentation.  Defaults is None.
    """

    def __init__(self, split='train'):
        self.split = split
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        if self.split == 'train':
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
            self.root_dir = train_dir
            self.info_list = info_dir + 'AAL-pruning/mydataset/train.txt'
        else:
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])
            self.root_dir = val_dir
            self.info_list = info_dir + 'AAL-pruning/mydataset/val.txt'

        self.filenames_labels = pd.read_csv(self.info_list, delimiter=' ', header=None)

    def __len__(self):
        return len(self.filenames_labels)

    def __getitem__(self, index):
        # Loading and preprocessing.
        image_name = self.filenames_labels.iloc[index, 0]
        image_path = os.path.join(self.root_dir, image_name)
        with Image.open(image_path) as img:
            image = img.convert('RGB')

        labels = self.filenames_labels.iloc[index, 1]
        if self.transform:
            image = self.transform(image)

        return image, torch.from_numpy(np.array(labels))


if __name__ == '__main__':
    # usage

    valset = ImageNetDataset('val')
    print(len(valset))
    dataloader = DataLoader(valset, batch_size=8, shuffle=False, num_workers=0)

    valset = ImageNetDataset('train')
    print(len(valset))
    dataloader = DataLoader(valset, batch_size=8, shuffle=False, num_workers=0)
    for i_batch, (images, targets) in enumerate(dataloader):
        print(i_batch, images.size(), targets.size())
