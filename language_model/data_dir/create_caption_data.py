import torch
import os
import json
# Chane batch saize to 32
# from word_language_model.data_dir.datasets import CaptionDataset
import torchvision.transforms as transforms

from tqdm import tqdm

from dataset_loader.datasets import CaptionDataset


run_local = True
desktop_path = os.path.join(os.path.join(os.path.expanduser('~')), 'Desktop')

data_folder = os.path.join(desktop_path, 'Pycharm_projects/image_captioning/output_folder')
# Read word map
if not run_local:
    data_f = '/yoav_stg/gshalev/image_captioning/output_folder'
else:
    data_f = data_folder

data_name = 'coco_5_cap_per_img_5_min_word_freq'
data_normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

word_map_file = os.path.join(data_f, 'WORDMAP_' + data_name + '.json')
print('word_map_file: {}'.format(word_map_file))

print('loading word map from path: {}'.format(word_map_file))
with open(word_map_file, 'r') as j:
    word_map = json.load(j)
print('load word map COMPLETED')

# rev word map
rev_word_map = {v: k for k, v in word_map.items()}

train_loader = torch.utils.data.DataLoader(CaptionDataset(data_f, data_name, 'TRAIN'), batch_size=1, )

val_loader = torch.utils.data.DataLoader(CaptionDataset(data_f, data_name, 'VAL'), batch_size=1)

test_loader = torch.utils.data.DataLoader(CaptionDataset(data_f, data_name, 'TEST'), batch_size=1)

for i, (imgs, caps, caplens) in enumerate(tqdm(train_loader)):
    f_train = open("train.txt", "a")
    f_train.write(' '.join([rev_word_map[x.item()] for x in caps[0][0:caplens]]) + '\n')
f_train.close()

for i, (imgs, caps, caplens, allcaps) in enumerate(tqdm((val_loader))):
    f_valid = open("valid.txt", "a")
    f_valid.write(' '.join([rev_word_map[x.item()] for x in caps[0][0:caplens]]) + '\n')
f_valid.close()

for i, (image, caps, caplens, allcaps) in enumerate(tqdm((test_loader))):
    f_test = open("test.txt", "a")
    f_test.write(' '.join([rev_word_map[x.item()] for x in caps[0][0:caplens]]) + '\n')
f_test.close()
