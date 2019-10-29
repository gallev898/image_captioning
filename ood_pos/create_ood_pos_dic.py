import sys

from utils import data_normalization


sys.path.append('/home/mlspeech/gshalev/gal/image_captioning')
sys.path.append('/home/mlspeech/gshalev/anaconda3/envs/python3_env/lib')

import os
import json
import torch
import argparse
import en_core_web_sm

from tqdm import tqdm
from dataset_loader.datasets import CaptionDataset
from dataset_loader.dataloader import load, flicker_loader

import torchvision.transforms as transforms


# args
parser = argparse.ArgumentParser(description='Show, Attend, and Tell - Tutorial - Generate Caption')
parser.add_argument('--in_data_type', type=str, default='TRAIN')
parser.add_argument('--run_local', default=False, action='store_true')
parser.add_argument('--pos_type', type=str, default='NOUN')
args = parser.parse_args()

if str.islower(args.pos_type):
    raise Exception('args.pos_type must by in capital (NOUN, VERB, DET, ...)')

coco_data_path = os.path.curdir if args.run_local else '../output_folder'
data_name = 'coco_5_cap_per_img_5_min_word_freq'
IN_DATA_TYPE = args.in_data_type
POS_TYPE = args.pos_type


def get_cap():
    word_map_path = coco_data_path
    word_map_file = os.path.join(word_map_path, 'WORDMAP_' + data_name + '.json')

    # Create rev word map
    with open(word_map_file, 'r') as j:
        map = json.load(j)
        print('loaded word map from: {}'.format(word_map_file))

    rev_map = {v: k for k, v in map.items()}

    return map, rev_map


def get_in_dis_loader():
    return torch.utils.data.DataLoader(
        CaptionDataset(coco_data_path, data_name, IN_DATA_TYPE, transform=transforms.Compose([data_normalization])),
        batch_size=1, shuffle=True, num_workers=1, pin_memory=True)


if __name__ == '__main__':
    print('Creating {} dic'.format(args.pos_type))
    nlp = en_core_web_sm.load()
    flickr_noun_dic = dict()

    dataloader = flicker_loader(args.run_local)
    flickr_noun_counter = 0
    for i, data in enumerate(tqdm(dataloader)):
        for c in data[1]:
            doc = nlp(c)
            for o in doc:
                if o.pos_ == POS_TYPE:
                    flickr_noun_counter += 1
                    word = o.text.lower()

                    if word in flickr_noun_dic.keys():
                        flickr_noun_dic[word] += 1
                    else:
                        flickr_noun_dic[word] = 1

    map, rev_map = get_cap()
    test_loader = get_in_dis_loader()
    in_dis_noun_dic = dict()
    in_dis_noun_counter = 0
    for i, data in enumerate(tqdm(test_loader)):
        if args.in_data_type == 'TEST':
            image, caps, caplens, allcaps = data
        elif args.in_data_type == 'TRAIN':
            imgs, caps, caplens = data

        for c in caps:
            c = ' '.join([rev_map[x] for x in caps.squeeze(0).detach().numpy()[1:caplens.item() - 1]])

            doc = nlp(c)
            for o in doc:
                word = o.text.lower()
                if o.pos_ == POS_TYPE:
                    in_dis_noun_counter += 1
                    if word in in_dis_noun_dic.keys():
                        in_dis_noun_dic[word] += 1
                    else:
                        in_dis_noun_dic[word] = 1

    torch.save({'flickr_dic': flickr_noun_dic, 'in_dis_dic': in_dis_noun_dic},
               '{}/{}_{}_dics'.format(os.path.curdir, IN_DATA_TYPE, POS_TYPE))
