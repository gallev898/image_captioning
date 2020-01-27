import sys


sys.path.append('/home/mlspeech/gshalev/gal/image_cap2')
sys.path.append('/home/mlspeech/gshalev/anaconda3/envs/python3_env/lib')

import torch
import random
import sys
import argparse

from tqdm import tqdm
from dataset_loader.datasets import CaptionDataset
import torchvision.transforms as transforms

from utils import data_normalization, get_word_map


parser = argparse.ArgumentParser()
parser.add_argument('--run_local', action='store_true', default=False)
parser.add_argument('--batch_size', default=1, type=int)
args = parser.parse_args()

data_folder = '../output_folder'  # folder with data files saved by create_input_files.py
data_name = 'coco_5_cap_per_img_5_min_word_freq'


if __name__ == '__main__':

    if not args.run_local:
        data_f = '/yoav_stg/gshalev/image_captioning/output_folder'
    else:
        data_f = data_folder

    train_loader = torch.utils.data.DataLoader(
        CaptionDataset(data_f, data_name, 'TRAIN', transform=transforms.Compose([data_normalization])),
        batch_size=args.batch_size, shuffle=True, num_workers=1, pin_memory=True)

    wrong_caps_lst = []
    for i, (image, caps, caplens) in enumerate(tqdm(train_loader)):
        wrong_caps_lst.append((caps, caplens))
        if args.run_local and i == 5:
            break

    save_dir = '../output_folder/train_caps_lst' if args.run_local else '/yoav_stg/gshalev/image_captioning/output_folder/train_caps_lst'
    torch.save(wrong_caps_lst, save_dir)



# create_caps_list.py