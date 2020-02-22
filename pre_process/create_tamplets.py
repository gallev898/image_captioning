import sys


sys.path.append('/home/mlspeech/gshalev/anaconda3/envs/python3_env/lib')
sys.path.append('/home/mlspeech/gshalev/gal/image_cap2')

import os
import torch
import argparse
import torchvision.transforms as transforms

import en_core_web_sm

from dataset_loader.datasets2 import CaptionDataset
from utils import data_normalization, data_name, get_word_map


parser = argparse.ArgumentParser()
parser.add_argument('--run_local', default=False, action='store_true')
args = parser.parse_args()

# section: load data
if not args.run_local:
    data_f = '/yoav_stg/gshalev/image_captioning/output_folder'
else:
    data_f = '../output_folder'

train_loader = torch.utils.data.DataLoader(
    CaptionDataset(data_f, data_name, 'TRAIN', transform=transforms.Compose([data_normalization])),
    batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

# section: get word map
word_map, rev_word_map = get_word_map(args.run_local, '../output_folder/WORDMAP_' + data_name + '.json')

#section: add NOUN VERB
word_map['NOUN'] = 9491
word_map['VERB'] = 9492
rev_word_map[9491] = 'NOUN'
rev_word_map[9492] = 'VERB'

#section : parse
nlp = en_core_web_sm.load()

noun_masking = dict()
verb_masking = dict()
full_masking = dict()

noun_idx_set = set()
verb_idx_set = set()
bad_ex = 0
for i, (imgs, caps, caplens) in enumerate(train_loader):
    caps_key = str(caps[0]).replace('\n', '')
    noun_mask_cap = torch.tensor(caps)
    verb_mask_cap = torch.tensor(caps)
    full_mask_cap = torch.tensor(caps)

    noun_masking_zero_vec = torch.zeros(caplens - 1)
    verb_masking_zero_vec = torch.zeros(caplens - 1)
    full_masking_zero_vec = torch.zeros(caplens - 1)

    noun_indexes = []
    verb_indexes = []

    padded_sentence = [rev_word_map[ind.item()] for ind in caps[0]]
    sentence = padded_sentence[1:padded_sentence.index('<end>')]

    doc = nlp(' '.join(sentence).replace('<unk>', 'unk'))
    pos = []
    for token in doc:
        pos.append(token.pos_)

    if len(pos) != len(sentence):
        noun_masking[caps_key] = (noun_mask_cap, caplens, noun_masking_zero_vec, noun_indexes)
        verb_masking[caps_key] = (verb_mask_cap, caplens, verb_masking_zero_vec, verb_indexes)
        full_masking[caps_key] = (full_mask_cap, caplens, full_masking_zero_vec, (noun_indexes, verb_indexes))
        bad_ex += 1
        continue

    for token in doc:
        if token.pos_ == 'NOUN':
            noun_idx_set.add(9487 if str(token) == 'unk' else word_map[str(token)])

            noun_mask_cap[0][token.i + 1] = word_map[token.pos_]
            full_mask_cap[0][token.i + 1] = word_map[token.pos_]
            noun_masking_zero_vec[token.i] = 1
            full_masking_zero_vec[token.i] = 1
            noun_indexes.append(token.i + 1)
        elif token.pos_ == 'VERB':
            verb_idx_set.add(9487 if str(token) == 'unk' else word_map[str(token)])

            verb_mask_cap[0][token.i + 1] = word_map[token.pos_]
            full_mask_cap[0][token.i + 1] = word_map[token.pos_]
            verb_masking_zero_vec[token.i] = 1
            full_masking_zero_vec[token.i] = 1
            verb_indexes.append(token.i + 1)

    assert noun_mask_cap.shape == caps.shape and full_mask_cap.shape == caps.shape and verb_mask_cap.shape == caps.shape
    noun_masking[caps_key] = (noun_mask_cap, caplens, noun_masking_zero_vec, noun_indexes)
    verb_masking[caps_key] = (verb_mask_cap, caplens, verb_masking_zero_vec, verb_indexes)
    full_masking[caps_key] = (full_mask_cap, caplens, full_masking_zero_vec, (noun_indexes, verb_indexes))

    # section: debug
    noun_new_cap = [rev_word_map[ind.item()] if ind.item() in rev_word_map else '?' for ind in noun_mask_cap[0]]
    verb_new_cap = [rev_word_map[ind.item()] if ind.item() in rev_word_map else '?' for ind in verb_mask_cap[0]]
    full_new_cap = [rev_word_map[ind.item()] if ind.item() in rev_word_map else '?' for ind in full_mask_cap[0]]

    print(' '.join(sentence))
    print(' '.join(noun_new_cap[1:noun_new_cap.index('<end>')]))
    print(' '.join(verb_new_cap[1:noun_new_cap.index('<end>')]))
    print(' '.join(full_new_cap[1:noun_new_cap.index('<end>')]))
    print('{}----------------'.format(i))
    # get the index of the NOUN
    # (noun_mask_cap[0] == word_map['NOUN']).nonzero().flatten()

#section: save to disc
save_path = '../standart_training' if args.run_local else '/yoav_stg/gshalev/image_captioning/output_folder'
torch.save(noun_masking, os.path.join(save_path, 'noun_masking_train_caps'))
torch.save(verb_masking, os.path.join(save_path, 'verb_masking_train_caps'))
torch.save(full_masking, os.path.join(save_path, 'full_masking_train_caps'))
torch.save(noun_idx_set, os.path.join(save_path, 'noun_idx_set'))
torch.save(verb_idx_set, os.path.join(save_path, 'verb_idx_set'))
print('num of bad_ex: {}'.format(bad_ex))
# create_tamplets.py