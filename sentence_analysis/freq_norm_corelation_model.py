import torch
import os
import json
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats


modelA = 'train_fix_show_and_tell_unnormalized'
decoding = 'beam_10'

data_name = 'coco_5_cap_per_img_5_min_word_freq'
path = '/Users/gallevshalev/Desktop/trained_models/{}/inference_data/{}_test_{}'
data_f = '../output_folder'
model_pathA_metric = path.format(modelA, 'metrics_results', decoding)

# sec: load model decoding metric
modelA_metric = torch.load(model_pathA_metric)
generated_sentencesA = modelA_metric['hyp']['annotations']

vocabulary_usage_modelA_set = set()
[vocabulary_usage_modelA_set.update(x['caption']) for x in generated_sentencesA]

A = [x['caption'] for x in generated_sentencesA]
cA = Counter([item for sublist in A for item in sublist])

# sec: load train caps str
# A = torch.load('train_caps_lst_str')
# cA = Counter([item for sublist in A for item in sublist])

# sec: word_map
word_map_file = os.path.join(data_f, 'WORDMAP_' + data_name + '.json')
with open(word_map_file, 'r') as j:
    word_map = json.load(j)
rev_word_map = {v: k for k, v in word_map.items()}

# sec: load model
model= 'NEW_BEST_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar'
local_path = '/Users/gallevshalev/Desktop/trained_models/{}/{}'.format(modelA, model)
dotproduct = torch.load(local_path , map_location='cpu')
representation = dotproduct['representations'].t()

# sec: build freq --> norm --> word
freq_to_norm_to_word = []
for word, freq in cA.items():
    if freq <= 100:
        word_rep_idx = word_map[word]
        rep = representation[word_rep_idx]
        rep_norm = torch.norm(rep).item()
        freq_to_norm_to_word.append((freq, rep_norm, word))

sorted = sorted(freq_to_norm_to_word, key=lambda x: x[0])
freq_lst = [x[0] for x in sorted]
norm_lst = [x[1] for x in sorted]

# sec: plot corelation graph
plt.scatter(freq_lst, norm_lst, c='black')
plt.show()
print('spearman correlation: {}'.format(scipy.stats.spearmanr(freq_lst, norm_lst)))
d=0

#---------------SEC: convert traon caps lst from indexes to strings
# # sec: load train caps lst
# train_caps_lst = torch.load('../output_folder/train_caps_lst')
#
# # sec: word_map
# data_f = '../output_folder'
# word_map_file = os.path.join(data_f, 'WORDMAP_' + data_name + '.json')
#
# with open(word_map_file, 'r') as j:
#     word_map = json.load(j)
# rev_word_map = {v: k for k, v in word_map.items()}
#
# train_caps_lst_str = []
# for cap in train_caps_lst:
#     cap = cap[0][0][1:cap[1]-1]
#     sen = []
#     sen = [rev_word_map[c.item()] for c in cap]
#     train_caps_lst_str.append(sen)
#
# torch.save(train_caps_lst_str, 'train_caps_lst_str')
# j=0