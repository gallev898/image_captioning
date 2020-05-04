import os
import json
import torch.nn.functional as F
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

model = 'train_show_ATTEND_and_tell'
# model = 'V_train_fix_show_and_tell_extra_embedding'
model_tar = 'NEW_BEST_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar'
model = torch.load('/Users/gallevshalev/Desktop/trained_models/{}/{}'.format(model, model_tar), map_location='cpu')
representations = model['representations'].t()
embedding_weight = model['decoder']['embedding.weight']

# sec: word_map
data_f = '../output_folder'
data_name = 'coco_5_cap_per_img_5_min_word_freq'
word_map_file = os.path.join(data_f, 'WORDMAP_' + data_name + '.json')
with open(word_map_file, 'r') as j:
    word_map = json.load(j)
rev_word_map = {v: k for k, v in word_map.items()}

word_norm = []
norms_distribution=[]

examed_words = ['cat']
# examed_words = ['bartender','doctor','learn', 'fix', 'bed', 'couch']

cosine_similarity_out = []
cosine_similarity_in = []
euc_out = []
euc_in = []
###########################
for i in range(10000):
    first_word_idx = random.randint(0,9489)
    second_word_idx = random.randint(0,9489)
    # second_word_idx = first_word_idx

    first_word = rev_word_map[first_word_idx]
    second_word = rev_word_map[second_word_idx]

    first_word_out_rep = representations[first_word_idx]
    second_word_out_rep = representations[second_word_idx]

    first_word_in_rep = embedding_weight[first_word_idx]
    second_word_in_rep = embedding_weight[second_word_idx]

    euc_out.append(np.linalg.norm(first_word_out_rep.detach().numpy()-second_word_out_rep.detach().numpy()))
    cos_out = np.dot((first_word_out_rep.detach().numpy() /torch.norm(first_word_out_rep).item()),
                 (second_word_out_rep.detach().numpy() /torch.norm(second_word_out_rep).item()))

    euc_in.append(np.linalg.norm(first_word_in_rep.detach().numpy()-second_word_in_rep.detach().numpy()))
    cos_in = np.dot((first_word_in_rep.detach().numpy() /torch.norm(first_word_in_rep).item()),
                     (second_word_in_rep.detach().numpy() /torch.norm(second_word_in_rep).item()))
    print('{} {} {} {}'.format(first_word, second_word, cos_in, cos_out))
    cosine_similarity_in.append((first_word, second_word, cos_in))
    cosine_similarity_out.append((first_word, second_word, cos_out))

plt.scatter(euc_out, euc_in, c='yellow', alpha=0.1)
# plt.scatter([x[2] for x in cosine_similarity_out], [x[2] for x in cosine_similarity_in], c='yellow', alpha=0.1)
plt.show()
#########################
# for ew in word_map.keys():
for ew in examed_words:
    ew_idx = word_map[ew]
    ew_norm = torch.norm(representations[ew_idx]).item()

    for idx, rep in enumerate(representations):
        word = rev_word_map[idx]
        word_norm = torch.norm(rep).item()

        cos = np.dot((representations[ew_idx].detach().numpy() /torch.norm(representations[ew_idx]).item()),
                     (rep.detach().numpy() /word_norm))
        cosine_similarity_out.append((ew, word, cos))

    for idx, rep in enumerate(embedding_weight):
        word = rev_word_map[idx]
        word_norm = torch.norm(rep).item()


        cos = np.dot((embedding_weight[ew_idx].detach().numpy() /torch.norm(embedding_weight[ew_idx]).item()),
                     (rep.detach().numpy() /word_norm))
        cosine_similarity_in.append((ew, word, cos))

###############
plt.scatter(cosine_similarity_out, cosine_similarity_in, c='yellow', alpha=0.1)
plt.show()

cosine_similarity_out.sort(key=lambda x: x[1])
list.reverse(cosine_similarity_out)
# print([x[0] for x in cosine_similarity[0:10]])
print(cosine_similarity_out[0:10])
