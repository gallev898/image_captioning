import torch
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import argparse
from nltk.translate.bleu_score import corpus_bleu

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
parser.add_argument('--modelA', type=str)
parser.add_argument('--modelB', type=str)
parser.add_argument('--decoding', type=str, default='test_beam_10')
parser.add_argument('--model_file_name', type=str,
                    default='NEW_BEST_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar')
args = parser.parse_args()

print('args.modelA: {}'.format(args.modelA))
print('args.modelB: {}'.format(args.modelB))

# sec: initialization
coco_data_path = '/Users/gallevshalev/PycharmProjects/image_captioning/output_folder' if args.run_local else '/yoav_stg/gshalev/image_captioning/output_folder'
data_name = 'coco_5_cap_per_img_5_min_word_freq'
data_normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
path = '/Users/gallevshalev/Desktop/trained_models/{}/inference_data/{}_{}'

# sec: write to file
if write:
    f = open('results/{}_VS_{}.txt'.format(args.modelA, args.modelB), 'w')

# sec: model paths for dic and pos
model_pathA_metric = path.format(args.modelA, 'metrics_results', args.decoding)
model_pathB_metric = path.format(args.modelB, 'metrics_results', args.decoding)

model_pathA_pos_dic = path.format(args.modelA, 'pos_dic', args.decoding)
model_pathB_pos_dic = path.format(args.modelB, 'pos_dic', args.decoding)

# sec: load models pos and dic
modelA_metric = torch.load(model_pathA_metric)
modelB_metric = torch.load(model_pathB_metric)
#----------------------------
A = torch.load('train_caps_lst_str')
cA = Counter([item for sublist in A for item in sublist])
rare_words = [x[0] for x in sorted([(k, v) for k, v in cA.items()], key=lambda x: x[1]) if 200 <= x[1] <= 500]

modelA_pos_dic = torch.load(model_pathA_pos_dic)
modelB_pos_dic = torch.load(model_pathB_pos_dic)

# sec: models generated sentences
generated_sentencesA = modelA_metric['hyp']['annotations']
generated_sentencesB = modelB_metric['hyp']['annotations']

A = [x['caption'] for x in generated_sentencesA]
B = [x['caption'] for x in generated_sentencesB]
cA = Counter([item for sublist in A for item in sublist])
cB = Counter([item for sublist in B for item in sublist])

#sec: get sentences with rare words
# rare_words = [x[0] for x in list(filter(lambda x: x[1] <=1, cA.items()))]
sen_with_rare_words_A = list(filter(lambda x: any(elem in rare_words for elem in x['caption']), generated_sentencesA))
sen_with_rare_words_B = list(filter(lambda x: any(elem in rare_words for elem in x['caption']), generated_sentencesB))
# len([elem  for x in generated_sentencesA for elem in x['caption'] if elem in rare_words])


g=0






# rare_words = [x[0] for x in sorted([(k, v) for k, v in cA.items()], key=lambda x: x[1])][7000:8000]
print([x for x in sorted([(k, v) for k, v in cA.items()], key=lambda x: x[1])][7000])
print([x for x in sorted([(k, v) for k, v in cA.items()], key=lambda x: x[1])][8000])
ff = open('compare_show_and_tell_dotproduct_and_fixed.txt', 'w')
cap2len = [(x, len(x['caption'])) for x in modelB_metric['hyp']['annotations']]
dotproduct_caps = [x for x in modelA_metric['hyp']['annotations']]
fixed_caps = [x for x in modelB_metric['hyp']['annotations']]
in_ =[]
dc_caps = []
f_caps = []

# data_loader = torch.utils.data.DataLoader(
#     CaptionDataset('../output_folder', data_name, 'TRAIN', transform=transforms.Compose([data_normalization])),
#     batch_size=args.batch_size, shuffle=True, num_workers=1, pin_memory=True)


for dc in dotproduct_caps:
    l = list(filter(lambda x: dc['image_id'] == x['image_id'], fixed_caps))
    if len(l) > 0:
        fixed_model_caption = l[0]['caption']
        words_r = [i for i in fixed_model_caption if i in rare_words]
        if len(words_r) > 0:
        # if len(fixed_model_caption)>len(a[0]['caption'])+1:
            in_.append(dc['image_id'])
            ff.write('-------------------------\n')
            print(words_r)
            for w in words_r:
                ff.write('{}:{}'.format(w, cA[w]))
            ff.write('\n')
            ff.write(str(dc['image_id']))
            ff.write('\n')
            ff.write(' '.join(dc['caption']))
            ff.write('\n')
            ff.write(' '.join(fixed_model_caption))
            ff.write('\n')
            dc_caps.append(' '.join(dc['caption']))
            f_caps.append(' '.join(fixed_model_caption))
print(in_)
print('len: {}'.format(len(in_)))
# corpus_bleu(dc_caps, f_caps)
exit()
#----------------------------
modelA_pos_dic = torch.load(model_pathA_pos_dic)
modelB_pos_dic = torch.load(model_pathB_pos_dic)

# sec: models generated sentences
generated_sentencesA = modelA_metric['hyp']['annotations']
generated_sentencesB = modelB_metric['hyp']['annotations']

# sec: avg sentence len
avg_sentence_lenA = sum([len(x['caption']) for x in generated_sentencesA]) / len(generated_sentencesA)
avg_sentence_lenB = sum([len(x['caption']) for x in generated_sentencesB]) / len(generated_sentencesB)
if write:
    f.write('avg_sentence_lenA: {} \n avg_sentence_lenB: {}\n'.format(avg_sentence_lenA, avg_sentence_lenB))

# sec: models sentences likelihood
modelA_sentence_likelihood = [x[1] for x in modelA_pos_dic['generated_sentences_likelihood']]
modelB_sentence_likelihood = [x[1] for x in modelB_pos_dic['generated_sentences_likelihood']]

# section: avg sentence likelihood
modelA_sentences_likelihood_avg = np.average(modelA_sentence_likelihood)
modelB_sentences_likelihood_avg = np.average(modelB_sentence_likelihood)
if write:
    f.write('modelA_sentences_likelihood_avg: {}\n modelB_sentences_likelihood_avg: {}\n'.format(
    modelA_sentences_likelihood_avg, modelB_sentences_likelihood_avg))

# sec: vocabulary usege set
vocabulary_usage_modelA = set()
vocabulary_usage_modelB = set()

[vocabulary_usage_modelA.update(x['caption']) for x in generated_sentencesA]
[vocabulary_usage_modelB.update(x['caption']) for x in generated_sentencesB]
if write:
    f.write('vocabulary_usage_modelA: {}\n vocabulary_usage_modelB: {}'.format(len(vocabulary_usage_modelA),
                                                                              len(vocabulary_usage_modelB)))

# sec: vocabulary usege counter
A = [x['caption'] for x in generated_sentencesA]
B = [x['caption'] for x in generated_sentencesB]
cA = Counter([item for sublist in A for item in sublist])
cB = Counter([item for sublist in B for item in sublist])

#sec: get sentences with rare words
rare_words = [x[0] for x in list(filter(lambda x: x[1] <=1, cA.items()))]
sen_with_rare_words_A = list(filter(lambda x: any(elem in rare_words  for elem in x['caption']), generated_sentencesA))
sen_idx = [x['image_id'] for x in sen_with_rare_words_A]
sen_with_rare_words_B = list(filter(lambda x: x['image_id'] in sen_idx, generated_sentencesB))

rare_file = open('results/rare_words_{}_VS_{}'.format(args.modelA, args.modelB), 'w')
rare_file.write(', '.join(rare_words))
rare_file.write('\n')
for sa, sb in zip(sen_with_rare_words_A, sen_with_rare_words_B):
    rare_file.write('modelA:\n{}\nmodelB:\n{}\n-----------\n'.format(sa['caption'], sb['caption']))
rare_file.close()

# sec:: vocabulary usage count
vocabulary_usage_modelA_count = len(vocabulary_usage_modelA)
vocabulary_usage_modelB_count = len(vocabulary_usage_modelB)

# sec: word_map
if load_word_map:
    data_f = '../output_folder'
    word_map_file = os.path.join(data_f, 'WORDMAP_' + data_name + '.json')

    with open(word_map_file, 'r') as j:
        word_map = json.load(j)
    rev_word_map = {v: k for k, v in word_map.items()}

# sec: scatter histogram
propA_vec = np.zeros(9490)
propB_vec = np.zeros(9490)
hisA = []
hisB = []

for i in cA.items():
    word_index = word_map[i[0]]
    v = np.zeros(i[1])
    v.fill(word_index)
    hisA.append(v)
hisA = np.concatenate(hisA)
plt.hist(hisA, density=True, bins=9490)
plt.show()
plt.clf()

for i in cB.items():
    word_index = word_map[i[0]]
    v = np.zeros(i[1])
    v.fill(word_index)
    hisB.append(v)
hisB = np.concatenate(hisB)
plt.hist(hisB, density=True, bins=9490)
plt.show()



if write:
    f.close()

# sec: debug
if debug == True:
    d = 0
    print(' '.join(list(filter(lambda x: x['image_id'] == 315, generated_sentencesA))[0]['caption']))
    print(' '.join(list(filter(lambda x: x['image_id'] == 315, generated_sentencesB))[0]['caption']))

    f = CaptionDataset(coco_data_path, data_name, 'TEST', transform=transforms.Compose([data_normalization]))
    # test_loader = torch.utils.data.DataLoader(
    #     CaptionDataset(coco_data_path, data_name, 'TEST', transform=transforms.Compose([data_normalization])),
    #     batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
    d = 0
