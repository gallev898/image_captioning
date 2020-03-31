import sys
sys.path.append('/home/mlspeech/gshalev/gal/image_cap2')

from tqdm import tqdm
import os
import json
import torchvision.transforms as transforms
import torch
from dataset_loader.datasets4 import CaptionDataset

# data_folder = '../output_folder'
data_folder = '/yoav_stg/gshalev/image_captioning/output_folder'
data_name = 'coco_5_cap_per_img_5_min_word_freq'
data_normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

train_loader = torch.utils.data.DataLoader(
    CaptionDataset(data_folder, data_name, 'TRAIN', transform=transforms.Compose([data_normalization])),
    batch_size=1, shuffle=False, num_workers=1, pin_memory=True)


word_map_file = os.path.join(data_folder, 'WORDMAP_' + data_name + '.json')

with open(word_map_file, 'r') as j:
    word_map = json.load(j)

rev_word_map = {v: k for k, v in word_map.items()}

word_to_count = {}
enumerator = enumerate(train_loader)

for i, (imgs, caps, caplens) in enumerator:

    cap = caps[0][:caplens[0]]

    if i%1000 == 0:
        print('{}/{}'.format(i, len(train_loader)))
        print(' '.join([rev_word_map[x.item()] for x in cap]))

    for w in cap:
        word = rev_word_map[w.item()]
        if word in word_to_count:
            word_to_count[word] += 1
        else:
            word_to_count[word] = 1


# torch.save(word_to_count, 'word_to_count')
torch.save(word_to_count, '/yoav_stg/gshalev/image_captioning/output_folder/train_set_word_to_count')

# train_vocab_count.py