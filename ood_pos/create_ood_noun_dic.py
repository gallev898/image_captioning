from dataset_loader.dataloader import load, flicker_loader
import en_core_web_sm
import os
import torchvision.transforms as transforms
import json

import torch

from dataset_loader.datasets import CaptionDataset
from utils import data_normalization


coco_data = os.path.curdir
data_name = 'coco_5_cap_per_img_5_min_word_freq'
IN_DATA_TYPE = 'TRAIN'


def get_cap():
    word_map_path = coco_data
    word_map_file = os.path.join(word_map_path, 'WORDMAP_' + data_name + '.json')

    # Create rev word map
    with open(word_map_file, 'r') as j:
        map = json.load(j)
        print('loaded word map from: {}'.format(word_map_file))

    rev_map = {v: k for k, v in map.items()}

    return map, rev_map


def get_test_loader():
    return torch.utils.data.DataLoader(
        CaptionDataset(coco_data, data_name, IN_DATA_TYPE, transform=transforms.Compose([data_normalization])),
        batch_size=1, shuffle=True, num_workers=1, pin_memory=True)


if __name__ == '__main__':
    nlp = en_core_web_sm.load()
    flickr_noun_dic = dict()

    dataloader = flicker_loader()
    flickr_noun_counter = 0
    for i, data in enumerate(dataloader):
        for c in data[1]:
            doc = nlp(c)
            for o in doc:
                if o.pos_ == 'NOUN':
                    flickr_noun_counter += 1
                    word = o.text.lower()

                    if word in flickr_noun_dic.keys():
                        flickr_noun_dic[word] += 1
                    else:
                        flickr_noun_dic[word] = 1

    map, rev_map = get_cap()
    test_loader = get_test_loader()
    in_dis_noun_dic = dict()
    in_dis_noun_counter = 0
    for i, (image, caps, caplens, allcaps) in enumerate(test_loader):
        for c in caps:
            c = ' '.join([rev_map[x] for x in caps.squeeze(0).detach().numpy()[1:caplens.item() - 1]])

            doc = nlp(c)
            for o in doc:
                word = o.text.lower()
                if o.pos_ == 'NOUN':
                    in_dis_noun_counter += 1
                    if word in in_dis_noun_dic.keys():
                        in_dis_noun_dic[word] += 1
                    else:
                        in_dis_noun_dic[word] = 1

    torch.save({'flickr_dic': flickr_noun_dic, 'in_dis_dic': in_dis_noun_dic},
               '{}/{}_noun_dics'.format(os.path.curdir, IN_DATA_TYPE))
