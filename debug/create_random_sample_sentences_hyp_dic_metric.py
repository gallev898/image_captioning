import torch
import random
from tqdm import tqdm
w2c = torch.load('word_to_count')


def create_random_sentence(len):
    generated = []
    sum_of_occ = sum(w2c.values())
    for idx in range(len):
        lottery = random.randint(0, sum_of_occ)
        current_counter = 0
        for k,v in w2c.items():
            current_counter += v
            if current_counter >= lottery:
                generated.append(k)
                break

    return generated

del w2c['<start>']
del w2c['<end>']
del w2c['<unk>']

print(create_random_sentence(10))

import os
import json

data_folder = '../output_folder'
data_name = 'coco_5_cap_per_img_5_min_word_freq'
word_map_file = os.path.join(data_folder, 'WORDMAP_' + data_name + '.json')

with open(word_map_file, 'r') as j:
    word_map = json.load(j)

rev_word_map = {v: k for k, v in word_map.items()}


train_caps_lst =  torch.load('train_caps_lst')
hp_metric_dic = {'annotations': list()}
for i in tqdm(range(0, 25000, 5)):
    sample = random.sample(train_caps_lst, 1)[0]
    caplen = sample[1].item()
    cap = sample[0][0][:caplen]
    sen = [rev_word_map[x.item()] for x in cap]

    hp_metric_dic['annotations'].append({u'image_id': i, u'caption': sen})

torch.save(hp_metric_dic, 'random_sample_hp_metric_dic')
