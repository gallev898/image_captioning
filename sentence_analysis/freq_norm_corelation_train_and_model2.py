import torch
import os
import json
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str)
parser.add_argument('--decoding', type=str, default='beam_10')
args = parser.parse_args()

if not os.path.exists('results/{}'.format(args.model)):
    os.mkdir('results/{}'.format(args.model))

# #sec: paths
model_tar = 'NEW_BEST_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar'
data_name = 'coco_5_cap_per_img_5_min_word_freq'
path = '/Users/gallevshalev/Desktop/trained_models/{}/inference_data/{}_test_{}'
model_pathA_metric = path.format(args.model, 'metrics_results', args.decoding)
# --------------------------------
# S load model metric
modelA_metric = torch.load(model_pathA_metric)
generated_sentencesA = modelA_metric['hyp']['annotations']

#S model words counter
modelA_captions = [x['caption'] for x in generated_sentencesA]
modelA_words_counter = Counter([item for sublist in modelA_captions for item in sublist])
v = open('results/{}/vocab.txt'.format(args.model), 'w')
v.write('vocab usage: {}'.format(len(modelA_words_counter)))
v.close()

#S train data words counter
train_caps_lst_str = torch.load('train_caps_lst_str')
train_caps_lst_str_words_counter = Counter([item for sublist in train_caps_lst_str for item in sublist])

#S all words the model didnt said
rest = list(filter(lambda x: x[0] not in modelA_words_counter.keys(), train_caps_lst_str_words_counter.items()))
print('rest: {}'.format(len(rest)))
# sec: word_map
data_f = '../output_folder'
word_map_file = os.path.join(data_f, 'WORDMAP_' + data_name + '.json')

with open(word_map_file, 'r') as j:
    word_map = json.load(j)
rev_word_map = {v: k for k, v in word_map.items()}

#S load model
model_local_path = '/Users/gallevshalev/Desktop/trained_models/{}/{}'.format(args.model, model_tar)
dotproduct = torch.load(model_local_path, map_location='cpu')
representation = dotproduct['representations'].t()
embedding_weight = dotproduct['decoder']['embedding.weight']


#for spearman
fre2norm = []
for word1, freq1 in train_caps_lst_str_words_counter.items():
    word_rep_idx = word_map[word1]
    rep = representation[word_rep_idx]
    rep_norm = torch.norm(rep).item()
    fre2norm.append((freq1, rep_norm, word1))
    # freq_to_norm_to_word.append((freq, rep_norm, word))

sorted1 = sorted(fre2norm, key=lambda x: x[0])

freq_ = [x[0] for x in sorted1]
norm_ = [x[1] for x in sorted1]
word_ = [x[2] for x in sorted1]






#S collect analizez of the model words
freq_to_norm_to_word = []
for word, freq in modelA_words_counter.items():
    if train_caps_lst_str_words_counter[word] > 5000:
        continue
    word_rep_idx = word_map[word]
    rep = representation[word_rep_idx]
    rep_norm = torch.norm(rep).item()
    freq_to_norm_to_word.append((train_caps_lst_str_words_counter[word], rep_norm, word))
    # freq_to_norm_to_word.append((freq, rep_norm, word))

sorted = sorted(freq_to_norm_to_word, key=lambda x: x[0])

freq_lst_by_model = [x[0] for x in sorted]
norm_lst_by_model = [x[1] for x in sorted]
word_lst_by_model = [x[2] for x in sorted]

#S collect analizez of the words the model didnt use
freq_to_norm_to_word = []
for (word, freq) in rest:
    if freq > 5000:
        continue
    word_rep_idx = word_map[word]
    rep = representation[word_rep_idx]
    rep_norm = torch.norm(rep).item()
    freq_to_norm_to_word.append((freq, rep_norm, word))

freq_to_norm_to_word.sort(key=lambda x: x[0])

freq_lst_rest = [x[0] for x in freq_to_norm_to_word]
norm_lst_rest = [x[1] for x in freq_to_norm_to_word]
word_lst_rest = [x[2] for x in freq_to_norm_to_word]

# S black and yello
# plt.scatter(freq_lst_by_model, norm_lst_by_model, c='black', alpha=0.1)
plt.scatter(freq_lst_rest, norm_lst_rest, c='yellow', alpha=0.1)
plt.xlabel('word frequency')
plt.ylabel('word representation norm')
plt.title(args.model)
plt.savefig('results/{}/{}2'.format(args.model, args.model))
# plt.show()
plt.clf()
d = 0
#S-------------------------------------------
freq_to_norm_to_word = []
for (word, freq) in train_caps_lst_str_words_counter.items():
    if freq > 5000:
        continue
    word_rep_idx = word_map[word]
    rep = representation[word_rep_idx]
    rep_norm = torch.norm(rep).item()
    freq_to_norm_to_word.append((freq, rep_norm, word))

freq_to_norm_to_word.sort(key=lambda x: x[0])

freq_lst = [x[0] for x in freq_to_norm_to_word]
norm_lst = [x[1] for x in freq_to_norm_to_word]
word_lst = [x[2] for x in freq_to_norm_to_word]

spearman = scipy.stats.spearmanr(freq_, norm_)
train_spearmanr = scipy.stats.spearmanr(freq_lst, norm_lst)
rest_spearmanr = scipy.stats.spearmanr(freq_lst_rest, norm_lst_rest)
model_spearmanr = scipy.stats.spearmanr(freq_lst_by_model, norm_lst_by_model)
f = open('results/{}/spearman.txt'.format(args.model), 'w')
f.write('spearman__ correlation by train data: {}\n'.format(spearman))
f.write('spearman correlation by train data: {}\n'.format(train_spearmanr))
f.write('spearman correlation by model data: {}\n'.format(model_spearmanr))
f.write('spearman correlation by rest data: {}'.format(rest_spearmanr))
f.close()
#-----------------------------------------------------------------------------------


#S analyz the embeddings
# representation = dotproduct['decoder']['embedding.weight'].detach()
#
# freq_to_norm_to_word = []
# for (word, freq) in train_caps_lst_str_words_counter.items():
#     if freq > 100000:
#         continue
#     word_rep_idx = word_map[word]
#     rep = representation[word_rep_idx]
#     rep_norm = torch.norm(rep).item()
#     freq_to_norm_to_word.append((freq, rep_norm, word))
#
# freq_to_norm_to_word.sort(key=lambda x: x[0])
#
# freq_lst_rest = [x[0] for x in freq_to_norm_to_word]
# norm_lst_rest = [x[1] for x in freq_to_norm_to_word]
# word_lst_rest = [x[2] for x in freq_to_norm_to_word]
#
# plt.scatter(freq_lst_rest, norm_lst_rest, c='pink')
# plt.xlabel('freq')
# plt.ylabel('norm')
# plt.title('embeddings')
# plt.savefig('results/{}/embeddings'.format(args.model))
#S-----------
# plt.show()

# spearmanr = scipy.stats.spearmanr(freq_lst_rest, norm_lst_rest)
# print('------------------s:{}'.format(scipy.stats.spearmanr(freq_lst_by_model, norm_lst_by_model)))
# f = open('results/{}/spearman.txt'.format(args.model), 'w')
# f.write('spearman correlation: {}'.format(spearmanr))
# f.close()
# print('spearman correlation: {}'.format(spearmanr))
# g = 0

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
