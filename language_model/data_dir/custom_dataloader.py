import sys


sys.path.append('/home/mlspeech/gshalev/anaconda3/envs/python3_env/lib')
sys.path.append('/home/mlspeech/gshalev/gal/image_captioning')

# from __future__ import print_function, division
import os

import torch

from PIL import Image
from torch.utils.data import Dataset


class Custom_Image_Dataset(torch.utils.data.Dataset):

    def __init__(self, image_path, transform, transform_normalization, caption_path=None):
        self.image_path = image_path
        self.caption_path = caption_path
        self.transform = transform
        self.transform_normalization = transform_normalization
        self.imgs = []

        for f in os.listdir(image_path):
            img = Image.open(os.path.join(image_path, f))
            if self.transform is not None:
                img = self.transform(img)

            if img.shape[0] != 3:
                continue

            if self.transform_normalization is not None:
                img = self.transform_normalization(img)

            image_title = f.split('.')[0]
            self.imgs.append((img, image_title))


    def __len__(self):
        return len(self.imgs)


    def __getitem__(self, index):
        img = self.imgs[index]
        return img

# custom_dataloader