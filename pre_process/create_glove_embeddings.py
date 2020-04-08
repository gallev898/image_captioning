import numpy as np
import os
import json
import torch


data_f = '/Users/gallevshalev/PycharmProjects/image_captioning/output_folder/'
data_name = 'coco_5_cap_per_img_5_min_word_freq'
word_map_file = os.path.join(data_f, 'WORDMAP_' + data_name + '.json')
with open(word_map_file, 'r') as j:
    word_map = json.load(j)
rev_word_map = {v: k for k, v in word_map.items()}

gloveFile = '/Users/gallevshalev/Desktop/word_embeddings/glove.42B.300d.txt'
f = open(gloveFile,'r')

model = {}
i=0
for line in f:
    # if i ==3:
    #     break
    # i +=1
    splitLine = line.split()
    word = splitLine[0]
    filtered = list(filter(lambda x: x[0] == word, word_map.items()))
    if len(filtered) > 0 :
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
torch.save(model, 'glove_embeddings')
print('finished')
h=0