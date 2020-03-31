import os
import json
import torch

data_f = '../../output_folder'
data_name = 'coco_5_cap_per_img_5_min_word_freq'  # base name shared by data files

metrics = torch.load(
    '/Users/gallevshalev/Desktop/trained_models/train_show_and_tell_dotproduct/NEW_BEST_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar')

word_map_file = os.path.join(data_f, 'WORDMAP_' + data_name + '.json')
with open(word_map_file, 'r') as j:
    word_map = json.load(j)
rev_word_map = {v: k for k, v in word_map.items()}

# min(torch.norm(metrics_roc_and_more['representations'].t(), p=2,dim=1).detach().numpy())

p = 0
