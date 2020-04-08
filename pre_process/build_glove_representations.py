import os
import json
import torch
import numpy as np

h = torch.load('glove_representations')


data_f = '/Users/gallevshalev/PycharmProjects/image_captioning/output_folder/'
data_name = 'coco_5_cap_per_img_5_min_word_freq'
word_map_file = os.path.join(data_f, 'WORDMAP_' + data_name + '.json')
with open(word_map_file, 'r') as j:
    word_map = json.load(j)
rev_word_map = {v: k for k, v in word_map.items()}
glove_embeddings = torch.load('glove_embeddings')

glove_representations = torch.zeros((9490, 300))
random_embeddings = 0
for word, index in word_map.items():
    if word in glove_embeddings:
        glove_representations[index] = torch.tensor(glove_embeddings[word])
    else:
        glove_representations[index] = torch.tensor(np.random.randint(low=-100, high=100, size=300))
        random_embeddings += 1

print('random_embeddings: {}'.format(random_embeddings))
torch.save(glove_representations, 'glove_representations')
