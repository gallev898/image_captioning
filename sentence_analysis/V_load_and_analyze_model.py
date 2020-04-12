import torch
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import argparse

from torchvision.transforms import transforms
from collections import Counter
from dataset_loader.datasets import CaptionDataset
import itertools

# sec: flags
debug = False
write = True
load_word_map = True

# sec: arg
parser = argparse.ArgumentParser()
parser.add_argument('--run_local', default=False, action='store_true')
parser.add_argument('--model', type=str)
parser.add_argument('--decoding', type=str, default='beam_10')
parser.add_argument('--model_file_name', type=str,
                    default='NEW_BEST_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar')
args = parser.parse_args()

print('args.model: {}'.format(args.model))

coco_data_path = '/Users/gallevshalev/PycharmProjects/image_captioning/output_folder' if args.run_local else '/yoav_stg/gshalev/image_captioning/output_folder'
data_name = 'coco_5_cap_per_img_5_min_word_freq'
data_normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
path = '/Users/gallevshalev/Desktop/trained_models/{}/inference_data/{}_test_{}'

if write:
    f = open('results/{}.txt'.format(args.model), 'w')

# sec: model paths for dic and pos
model_path_metric = path.format(args.model, 'metrics_results', args.decoding)
model_path_pos_dic = path.format(args.model, 'pos_dic', args.decoding)

# sec: load models pos and dic
model_metric = torch.load(model_path_metric)
model_pos_dic = torch.load(model_path_pos_dic)

# sec: models generated sentences
generated_sentences = model_metric['hyp']['annotations']

# sec: avg sentence len
avg_sentence_len = sum([len(x['caption']) for x in generated_sentences]) / len(generated_sentences)
if write:
    f.write('avg_sentence_len: {} \n '.format(avg_sentence_len))

# sec: models sentences likelihood
model_sentence_likelihood = [x[1] for x in model_pos_dic['generated_sentences_likelihood']]

# section: avg sentence likelihood
model_sentences_likelihood_avg = np.average(model_sentence_likelihood)
if write:
    f.write('model_sentences_likelihood_avg: {}\n'.format(model_sentences_likelihood_avg))

# sec: vocabulary usege set
vocabulary_usage_model = set()

[vocabulary_usage_model.update(x['caption']) for x in generated_sentences]
if write:
    f.write('vocabulary_usage_model: {}\n'.format(len(vocabulary_usage_model)))

# sec: vocabulary usege counter
A = [x['caption'] for x in generated_sentences]
cA = Counter([item for sublist in A for item in sublist])

# sec: get sentences with rare words
rare_words = [x[0] for x in list(filter(lambda x: x[1] <= 1, cA.items()))]
sen_with_rare_words = list(filter(lambda x: any(elem in rare_words for elem in x['caption']), generated_sentences))
sen_idx = [x['image_id'] for x in sen_with_rare_words]

rare_file = open('results/rare_words_{}.txt'.format(args.model), 'w')
rare_file.write(', '.join(rare_words))
rare_file.write('\n')
for sa in zip(sen_with_rare_words):
    rare_file.write('model: {}\n-----------\n'.format(' '.format(sa[0]['caption'])))
rare_file.close()

# sec:: vocabulary usage count
vocabulary_usage_modelA_count = len(vocabulary_usage_model)

# sec: word_map
if load_word_map:
    data_f = '../output_folder'
    word_map_file = os.path.join(data_f, 'WORDMAP_' + data_name + '.json')

    with open(word_map_file, 'r') as j:
        word_map = json.load(j)
    rev_word_map = {v: k for k, v in word_map.items()}

# sec: scatter histogram
propA_vec = np.zeros(9490)
hisA = []

# for i in cA.items():
#     word_index = word_map[i[0]]
#     v = np.zeros(i[1])
#     v.fill(word_index)
#     hisA.append(v)
# hisA = np.concatenate(hisA)
# plt.hist(hisA, density=True, bins=9490)
# plt.show()
# plt.clf()


if write:
    f.close()

# sec: debug
if debug == True:
    d = 0
    print(' '.join(list(filter(lambda x: x['image_id'] == 315, generated_sentences))[0]['caption']))

    f = CaptionDataset(coco_data_path, data_name, 'TEST', transform=transforms.Compose([data_normalization]))
    # test_loader = torch.utils.data.DataLoader(
    #     CaptionDataset(coco_data_path, data_name, 'TEST', transform=transforms.Compose([data_normalization])),
    #     batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
    d = 0
