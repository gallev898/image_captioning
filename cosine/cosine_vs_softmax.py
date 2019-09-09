import sys


sys.path.append('/home/mlspeech/gshalev/gal/image_captioning')
sys.path.append('/home/mlspeech/gshalev/anaconda3/envs/python3_env/lib')

import os
import tqdm
import json
import torch
import random
import argparse
import skimage.transform

import numpy as np
import matplotlib.cm as cm
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.transforms as transforms

from PIL import Image
from scipy.misc import imread, imresize


# args
parser = argparse.ArgumentParser(description='Show, Attend, and Tell - Tutorial - Generate Caption')
parser.add_argument('--model', type=str)
parser.add_argument('--run_local', default=False, action='store_true')
parser.add_argument('--ood', default=False, action='store_true')
parser.add_argument('--all_data', default=False, action='store_true')
parser.add_argument('--limit_ex', type=int, default=1)
parser.add_argument('--ood_limit_ex', type=int, default=1)
parser.add_argument('--beam_size', default=5, type=int)
parser.add_argument('--dont_smooth', dest='smooth', action='store_false', help='do not smooth alpha overlay')
args = parser.parse_args()

# global
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_name = 'coco_5_cap_per_img_5_min_word_freq'
softmax_filename = 'BEST_checkpoint_' + data_name + '.pth.tar'
cosine_filename = 'BEST_checkpoint_' + data_name + '.pth.tar'

if __name__ == '__main__':

    if args.run_local:
        softmax_model_path = softmax_filename
        cosine_model_path = cosine_filename
        if args.ood:
            save_dir = "cosine_vs_softmax_OOD_GIFs"
        else:
            save_dir = "cosine_vs_softmax_GIFs"
    else:
        model_path = "/yoav_stg/gshalev/image_captioning/{}/{}".format(args.model, filename)
        if args.ood:
            save_dir = "/yoav_stg/gshalev/image_captioning/{}/OOD_GIFs".format(args.model)
        else:
            save_dir = "/yoav_stg/gshalev/image_captioning/{}/GIFs".format(args.model)
        # create dir
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # Load model
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    decoder = checkpoint['decoder']
    decoder = decoder.to(device)
    decoder.eval()
    encoder = checkpoint['encoder']
    encoder = encoder.to(device)
    encoder.eval()