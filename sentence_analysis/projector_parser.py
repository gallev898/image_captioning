import torch
import os
import json
from collections import Counter


#sec: settings
ByModel = True
ByTrain = False
model_name = 'train_show_and_tell_dotproduct'
model_tar_file = 'NEW_BEST_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar'
path = '/Users/gallevshalev/Desktop/trained_models/{}/{}'.format(model_name, model_tar_file)
data_name = 'coco_5_cap_per_img_5_min_word_freq'
data_f = '../output_folder'
decoding = 'beam_10'
model_path_metric = '/Users/gallevshalev/Desktop/trained_models/{}/inference_data/metrics_results_test_{}'.format(model_name,decoding )

#sec: load model decoding metric
modelA_metric = torch.load(model_path_metric, map_location='cpu')
generated_sentencesA = modelA_metric['hyp']['annotations']

if ByModel:
    A = [x['caption'] for x in generated_sentencesA]
if ByTrain:
    A = torch.load('train_caps_lst_str')

cA = Counter([item for sublist in A for item in sublist])

#sec: load representation
model = torch.load(path, map_location='cpu')
representations = model['representations'].t()

# sec load word map

word_map_file = os.path.join(data_f, 'WORDMAP_' + data_name + '.json')
with open(word_map_file, 'r') as j:
    word_map = json.load(j)
rev_word_map = {v: k for k, v in word_map.items()}

#sec: parse zip(word, representations) to file
vectors = open('vectors.txt', 'w')
metadata = open('metadata.txt', 'w')

metadata.write('word\tnorm\tfreq')
metadata.write('\n')

for word, freq in cA.items():
    word_rep_idx = word_map[word]
    rep = representations[word_rep_idx]
    rep_norm = torch.norm(rep).item()

    vectors.write('\t'.join([str(x) for x in rep.detach().numpy()]))
    vectors.write('\n')
    metadata.write('{}\t{}\t{}'.format(word,rep_norm, freq))
    metadata.write('\n')

# for i, rep in enumerate(representations):
#     assert all(rep == representations[i])
#     word = rev_word_map[i]
#     vectors.write('\t'.join([str(x) for x in rep.detach().numpy()]))
#     vectors.write('\n')
#     metadata.write('{}\t{}'.format(word,word))
#     metadata.write('\n')

vectors.close()
metadata.close()

f=0
