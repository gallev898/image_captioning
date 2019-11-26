import sys


sys.path.append('/home/mlspeech/gshalev/anaconda3/envs/python3_env/lib')
sys.path.append('/home/mlspeech/gshalev/gal/image_cap')
# sys.path.append('/home/mlspeech/gshalev/gal/image_captioning')



import numpy as np
import imgaug.augmenters as iaa


class ImgAugTransform:

    def __init__(self):
        self.aug = iaa.Sequential([
            #iaa.Resize(32),
            iaa.Sometimes(0.25, iaa.GaussianBlur(sigma=(0, 1.0))),
            iaa.Fliplr(0.5),
            iaa.Affine(rotate=(-10, 10), mode='symmetric'),
            iaa.Sometimes(0.25,
                          iaa.OneOf([iaa.Dropout(p=(0, 0.1)),
                                     iaa.CoarseDropout(0.1, size_percent=0.5)])),
            iaa.AddToHueAndSaturation(value=(-10, 10), per_channel=True)
            ])


    def __call__(self, img):
        img = np.array(img)
        return self.aug.augment_image(img)


class ImgAugTransformSnow:

    def __init__(self):
        self.aug = iaa.Sequential([
            #iaa.Resize(32),
            iaa.Snowflakes(density=(0.5, 0.01))
            ])


    def __call__(self, img):
        img = np.array(img)
        return self.aug.augment_image(img)


class ImgAugTransformFog:

    def __init__(self):
        self.aug = iaa.Sequential([
            #iaa.Resize(22),
            iaa.Fog()
            ])


    def __call__(self, img):
        img = np.array(img)
        return self.aug.augment_image(img)


class ImgAugTransformClouds:

    def __init__(self):
        self.aug = iaa.Sequential([
            #iaa.Resize(32),
            iaa.Clouds()
            ])


    def __call__(self, img):
        img = np.array(img)
        return self.aug.augment_image(img)


class ImgAugTransformMedianBlur:

    def __init__(self):
        self.aug = iaa.Sequential([
            # iaa.Resize(32),
            iaa.MedianBlur(k=3)
            ])


    def __call__(self, img):
        img = np.array(img)
        return self.aug.augment_image(img)


class ImgAugTransformSaltAndPepper:

    def __init__(self):
        self.aug = iaa.Sequential([
            #iaa.Resize(32),
            iaa.SaltAndPepper(p=0.10)
            ])


    def __call__(self, img):
        img = np.array(img)
        return self.aug.augment_image(img)


# Pertubation.py