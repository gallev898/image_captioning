import torch
import os
import json
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

# #sec: paths
model = 'train_fix_show_and_tell_unnormalized'
decoding = 'beam_10'
# --------------------------------
model_tar = 'NEW_BEST_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar'
data_name = 'coco_5_cap_per_img_5_min_word_freq'
path = '/Users/gallevshalev/Desktop/trained_models/{}/inference_data/{}_test_{}'
model_pathA_metric = path.format(model, 'metrics_results', decoding)
# --------------------------------

modelA_metric = torch.load(model_pathA_metric)
generated_sentencesA = modelA_metric['hyp']['annotations']

modelA = [x['caption'] for x in generated_sentencesA]
modelA_counter = Counter([item for sublist in modelA for item in sublist])

train_caps_lst_str = torch.load('train_caps_lst_str')
train_caps_lst_str_counter = Counter([item for sublist in train_caps_lst_str for item in sublist])
rest = list(filter(lambda x: x[0] not in modelA_counter.keys(), train_caps_lst_str_counter.items()))

# sec: word_map
data_f = '../output_folder'
word_map_file = os.path.join(data_f, 'WORDMAP_' + data_name + '.json')

with open(word_map_file, 'r') as j:
    word_map = json.load(j)
rev_word_map = {v: k for k, v in word_map.items()}

model_local_path = '/Users/gallevshalev/Desktop/trained_models/{}/{}'.format(model, model_tar)
dotproduct = torch.load(model_local_path, map_location='cpu')
representation = dotproduct['representations'].t()

freq_to_norm_to_word = []
for word, freq in modelA_counter.items():
    if freq > 1000:
        continue
    word_rep_idx = word_map[word]
    rep = representation[word_rep_idx]
    rep_norm = torch.norm(rep).item()
    freq_to_norm_to_word.append((freq, rep_norm, word))

sorted = sorted(freq_to_norm_to_word, key=lambda x: x[0])

freq_lst_by_model = [x[0] for x in sorted]
norm_lst_by_model = [x[1] for x in sorted]
word_lst_by_model = [x[2] for x in sorted]

freq_to_norm_to_word = []
for (word, freq) in rest:
    if freq > 1000:
        continue
    word_rep_idx = word_map[word]
    rep = representation[word_rep_idx]
    rep_norm = torch.norm(rep).item()
    freq_to_norm_to_word.append((freq, rep_norm, word))

freq_to_norm_to_word.sort(key=lambda x: x[0])

freq_lst_rest = [x[0] for x in freq_to_norm_to_word]
norm_lst_rest = [x[1] for x in freq_to_norm_to_word]
word_lst_rest = [x[2] for x in freq_to_norm_to_word]

plt.scatter(freq_lst_by_model, norm_lst_by_model, c='black', alpha=0.8)
plt.scatter(freq_lst_rest, norm_lst_rest, c='yellow', alpha=0.1)
plt.title(model)
plt.show()
d = 0
#-----------------------------------------------------------------------------------


model_tar = 'NEW_BEST_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar'
model_local_path = '/Users/gallevshalev/Desktop/trained_models/{}/{}'.format(model, model_tar)
dotproduct = torch.load(model_local_path, map_location='cpu')
# representation = dotproduct['representations'].t()
representation = dotproduct['decoder']['embedding.weight'].detach()

train_caps_lst_str = torch.load('train_caps_lst_str')
train_caps_lst_str_counter = Counter([item for sublist in train_caps_lst_str for item in sublist])


data_f = '../output_folder'
word_map_file = os.path.join(data_f, 'WORDMAP_' + data_name + '.json')

with open(word_map_file, 'r') as j:
    word_map = json.load(j)
rev_word_map = {v: k for k, v in word_map.items()}

freq_to_norm_to_word = []
for (word, freq) in train_caps_lst_str_counter.items():
    if freq > 100000:
        continue
    word_rep_idx = word_map[word]
    rep = representation[word_rep_idx]
    rep_norm = torch.norm(rep).item()
    freq_to_norm_to_word.append((freq, rep_norm, word))

freq_to_norm_to_word.sort(key=lambda x: x[0])

freq_lst_rest = [x[0] for x in freq_to_norm_to_word]
norm_lst_rest = [x[1] for x in freq_to_norm_to_word]
word_lst_rest = [x[2] for x in freq_to_norm_to_word]

plt.scatter(freq_lst_rest, norm_lst_rest, c='pink')
plt.show()

print('spearman correlation: {}'.format(scipy.stats.spearmanr(freq_lst_rest, norm_lst_rest)))
g = 0

#-----------------------------------------------------------------------------------
#
# two_d_vec = open('2dvec.txt', 'w')
# two_d_metadata = open('2dmetadata.txt', 'w')
# two_d_metadata.write('word\tsource\n')
# for fm, nm, wm in zip(freq_lst_by_model, norm_lst_by_model, word_lst_by_model):
#     two_d_vec.write('{}\t{}'.format(fm, nm))
#     two_d_vec.write('\n')
#
#     two_d_metadata.write('{}\tmodel'.format(wm))
#     two_d_metadata.write('\n')
#
# for fr, nr, wr in zip(freq_lst_rest, norm_lst_rest, word_lst_rest):
#     two_d_vec.write('{}\t{}'.format(fr, nr))
#     two_d_vec.write('\n')
#
#     two_d_metadata.write('{}\trest'.format(wr))
#     two_d_metadata.write('\n')
#
# two_d_metadata.close()
# two_d_vec.close()
# d = 0

#-----------------------------------------------------------------------------------
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
