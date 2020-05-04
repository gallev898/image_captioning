import sys
sys.path.append('/home/mlspeech/gshalev/gal/image_cap2')

from tqdm import tqdm
import os
import numpy as np
import json
import torchvision.transforms as transforms
import torch
from dataset_loader.datasets4 import CaptionDataset
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image
data_folder = '../output_folder'
# data_folder = '/yoav_stg/gshalev/image_captioning/output_folder'
data_name = 'coco_5_cap_per_img_5_min_word_freq'
data_normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

train_loader = torch.utils.data.DataLoader(
    CaptionDataset(data_folder, data_name, 'TEST', transform=transforms.Compose([data_normalization])),
    batch_size=1, shuffle=False, num_workers=1, pin_memory=True)


word_map_file = os.path.join(data_folder, 'WORDMAP_' + data_name + '.json')

with open(word_map_file, 'r') as j:
    word_map = json.load(j)

rev_word_map = {v: k for k, v in word_map.items()}

word_to_count = {}
enumerator = enumerate(train_loader)

for i, (imgs, caps, caplens, L) in enumerator:
    [next(enumerator, None) for _ in range(4)]

    # if i in [115, 760, 830, 835, 1150, 2565, 2605, 2930, 3030, 3190, 3390, 3525, 3940, 4025, 4330, 4490, 4500, 4995, 5595, 5790, 5820, 6085, 6450, 6750, 8070, 8125, 8205, 8845, 8875, 9475, 9590, 10030, 10080, 10725, 10820, 11240, 12080, 12185, 12510, 12695, 13725, 13915, 14080, 14330, 14365, 14470, 14515, 14735, 15295, 15710, 15965, 15970, 16220, 16300, 16380, 16425, 16545, 16780, 17260, 17745, 17835, 17920, 17950, 18060, 18065, 18130, 18440, 18760, 19025, 19230, 19545, 19550, 19680, 19760, 19850, 19960, 20040, 20525, 20615, 20710, 20990, 21230, 21545, 21735, 21880, 22335, 22705, 23460, 23835, 24270, 24360, 24390, 24410, 24725]:
    print(i)
    if i in [11970]:
    # if i in [100, 155, 165, 215]:
        im = imgs.squeeze(0).numpy()
        im = im.transpose((1, 2, 0))
        # Undo preprocessing
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        im = std * im + mean
        # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
        im = np.clip(im, 0, 1)
        plt.imshow(im)
        plt.show()
        h =0
        exit()
    # cap = caps[0][:caplens[0]]
    #
    # if i%1000 == 0:
    #     print('{}/{}'.format(i, len(train_loader)))
    #     print(' '.join([rev_word_map[x.item()] for x in cap]))
    #
    # for w in cap:
    #     word = rev_word_map[w.item()]
    #     if word in word_to_count:
    #         word_to_count[word] += 1
    #     else:
    #         word_to_count[word] = 1


# torch.save(word_to_count, 'word_to_count')
# torch.save(word_to_count, '/yoav_stg/gshalev/image_captioning/output_folder/train_set_word_to_count')

# train_vocab_count.py