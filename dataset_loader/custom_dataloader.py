from __future__ import print_function, division
import os
import torch
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision
from PIL import Image

class Custom_Image_Dataset(torch.utils.data.Dataset):

    def __init__(self, image_path, transform, caption_path=None
                 ):
        'Initialization'
        self.image_path = image_path
        self.caption_path = caption_path
        self.transform = transform

        self.imgs = []
        for f in os.listdir(image_path):
            img = Image.open(os.path.join(image_path, f))
            if self.transform is not None:
                img = self.transform(img)

            self.imgs.append(img)


    def __len__(self):
        'Denotes the total number of samples'
        return len(self.imgs)


    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        img = self.imgs[index]

        return img

