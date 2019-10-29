from __future__ import print_function, division
import os

import torch

from PIL import Image
from torch.utils.data import Dataset


class Custom_Image_Dataset(torch.utils.data.Dataset):

    def __init__(self, image_path, transform, caption_path=None):
        self.image_path = image_path
        self.caption_path = caption_path
        self.transform = transform
        self.imgs = []

        for f in os.listdir(image_path):
            img = Image.open(os.path.join(image_path, f))
            if self.transform is not None:
                img = self.transform(img)

            image_title = f.split('.')[0]
            self.imgs.append((img, image_title))


    def __len__(self):
        return len(self.imgs)


    def __getitem__(self, index):
        img = self.imgs[index]
        return img
