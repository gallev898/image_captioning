import os
import json
import torch.nn.functional as F
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

model = 'V_train_fix_show_and_tell_extra_embedding'
model_tar = 'BEST_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar'
model = torch.load('/Users/gallevshalev/Desktop/trained_models/{}/{}'.format(model, model_tar), map_location='cpu')
representations = model['representations'].t()

# sec: word_map
data_f = '../output_folder'
data_name = 'coco_5_cap_per_img_5_min_word_freq'
word_map_file = os.path.join(data_f, 'WORDMAP_' + data_name + '.json')
with open(word_map_file, 'r') as j:
    word_map = json.load(j)
rev_word_map = {v: k for k, v in word_map.items()}

word_norm = []
cosine_similarity = []
norms_distribution=[]
for idx, rep in enumerate(representations):
    if idx == 9490:
        word = 'extra_embedding'
        word_norm.append((word, torch.norm(rep)))
    else:
        word = rev_word_map[idx]
        word_norm.append((word, torch.norm(rep)))
    cosine_similarity.append((np.dot((representations[9490].detach().numpy() /torch.norm(representations[9490]).item()),
                                     (rep.detach().numpy() /torch.norm(rep).item())), word, torch.norm(rep).item()))
    norms_distribution.append(torch.norm(rep).item())
f = 0

if idx == 9490:
    norm_9 = [x[0] for x in list(filter(lambda x: x[1].item() >= 9 and x[1].item() <= 10, word_norm))]

    A = torch.load('train_caps_lst_str')
    cA = Counter([item for sublist in A for item in sublist])

    norm_9_freq  = list(filter(lambda x: x[0] in norm_9, cA.items()))
    norm_9_freq.sort(key=lambda x: x[1])

    cosine_similarity.sort(key=lambda x: x[0])
    cosine_similarity.reverse()
    top_cosine=[]
    for i in range(100):
        print(cosine_similarity[i])
        top_cosine.append(cosine_similarity[i])

    plt.hist(norms_distribution, bins=int(max(norms_distribution)))
    plt.show()
h = 0


