import sys

from decoding_strategist.beam_search.beam_search_pack_utils import beam_search_decode
from utils import *


sys.path.append('/home/mlspeech/gshalev/gal/image_captioning')
sys.path.append('/home/mlspeech/gshalev/anaconda3/envs/python3_env/lib')

import os
import torch
import argparse
import itertools
import en_core_web_sm

import numpy as np
import torchvision.transforms as transforms

from tqdm import *
# from standart_training.utils import data_normalization
from dataset_loader.datasets import CaptionDataset
from dataset_loader.dataloader import load


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_name = 'coco_5_cap_per_img_5_min_word_freq'
filename = 'BEST_checkpoint_' + data_name + '.pth.tar'
pre_pros_data_dir = '../../output_folder'
noun_phrase_sum_of_log_prop = []
pos_dic_obj = {}

parser = argparse.ArgumentParser(description='Generate Caption')
parser.add_argument('--model', type=str)
parser.add_argument('--run_local', default=False, action='store_true')
parser.add_argument('--debug', default=False, action='store_true')
parser.add_argument('--data', default='random', type=str)
parser.add_argument('--beam_size', default=5, type=int)
args = parser.parse_args()


def caption_image_beam_search(encoder, decoder, image_path, word_map, rev_word_map, beam_size):

    # k = beam_size
    vocab_size = len(word_map)

    image = image_path.to(device)

    seq, alphas, top_seq_total_scors, seq_sum, logits_list = beam_search_decode(encoder, image, beam_size, word_map,
                                                                                decoder, vocab_size)

    words = [rev_word_map[ind] for ind in seq]

    words = words[1:-1]
    top_seq_total = top_seq_total_scors[1:-1]
    ex_top_seq_total = np.exp(top_seq_total_scors[1:-1])

    words_as_str = ' '.join(words)
    nlp = en_core_web_sm.load()
    doc = nlp(words_as_str)

    lats_i = 0
    np_idx = []
    words_copy = words
    original_str = words

    # Collect inexed of noun phrase
    for i in doc.noun_chunks:
        i_str = '{} '.format(i.string)
        if '<unk ' in i_str:
            i_str = i_str.replace('<unk ', '<unk> ')
        i_str = i_str.strip()

        print('*****')
        print(words_copy)
        print(i_str)

        inde = [words_copy.index(str(x)) + lats_i for x in i_str.split()]
        np_idx.append(inde)
        print(inde)
        lats_i = inde[-1] + 1
        words_copy = list(original_str[lats_i:])

    idx_count = 0
    for w, token, score, exp, alph, logit in zip(words, doc, top_seq_total, ex_top_seq_total, alphas, logits_list):
        if args.debug:
            print(w, token.pos_, score, exp)
        alph = list(itertools.chain.from_iterable(alph))
        if token.pos_ in pos_dic_obj:
            pos_dic_obj[token.pos_]['exp'].append(exp)
            pos_dic_obj[token.pos_]['alphas_var'].append(np.var(alph))
            pos_dic_obj[token.pos_]['alphas_max'].append(max(alph))
            pos_dic_obj[token.pos_]['logits'].append(logit)
        else:
            pos_dic_obj[token.pos_] = {'exp': [exp],
                                       'alphas_var': [np.var(alph)],
                                       'alphas_max': [max(alph)],
                                       'logits': [logit]}

        for l in np_idx:
            if idx_count in l:
                l[l.index(idx_count)] = score
                break

        idx_count += 1

    for l in np_idx:
        noun_phrase_sum_of_log_prop.append(sum(l) * -1)

    return seq, alphas, top_seq_total_scors, seq_sum


def get_model_path_and_save_dir(save_dir_name='GIF'):
    if args.run_local:
        desktop_path = os.path.join(os.path.join(os.path.expanduser('~')), 'Desktop')
        model_path = os.path.join(desktop_path, os.path.join(args.model, filename))
        # model_path = os.path.join(os.path.join(desktop_path, 'trained_models'), os.path.join(args.model, filename))
        save_dir = save_dir_name
    else:
        # dir = '/yoav_stg/gshalev/semantic_labeling/mscoco/val2014'
        model_path = "/yoav_stg/gshalev/image_captioning/{}/{}".format(args.model, filename)
        save_dir = "/yoav_stg/gshalev/image_captioning/{}/{}".format(args.model, save_dir_name)

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
        print('created dir: {}'.format(save_dir))

    return model_path, save_dir


if __name__ == '__main__':
    model_path, save_dir = get_model_path_and_save_dir('pos_dic')

    encoder, decoder = get_models(model_path)

    word_map, rev_word_map = get_word_map('../../output_folder/WORDMAP_' + data_name + '.json')

    print('create pos dic for {} data'.format(args.data))

    if args.data == 'test':
        coco_data = os.path.curdir if args.run_local else pre_pros_data_dir

        test_loader = torch.utils.data.DataLoader(
            CaptionDataset(coco_data, data_name, 'TEST', transform=transforms.Compose([data_normalization])),
            batch_size=1, shuffle=True, num_workers=1, pin_memory=True)

        for i, (image, caps, caplens, allcaps) in tqdm(enumerate(test_loader)):
            caption_image_beam_search(encoder, decoder, image, word_map, rev_word_map, args.beam_size)

    if args.data == 'sbu':
        dataloader = load('sbu', args.run_local, 1, 1)

        for i, data in enumerate(dataloader):
            print('index:{} data shape: {}'.format(i, data[0].shape))
            caption_image_beam_search(encoder, decoder, data[0], word_map, rev_word_map, args.beam_size)

    if args.data == 'svhn':
        _, dataloader, _ = load('svhn', args.run_local, 1, 1)

        for i, data in tqdm(enumerate(dataloader)):
            caption_image_beam_search(encoder, decoder, data[0], word_map, rev_word_map, args.beam_size)

    if args.data == 'flicker':
        dataloader = load('flicker', args.run_local, 1, 1)

        for i, data in tqdm(enumerate(dataloader)):
            caption_image_beam_search(encoder, decoder, data[0], word_map, rev_word_map, args.beam_size)

    if args.data == 'random':
        for i in tqdm(range(5)):
            img1 = np.random.uniform(low=0., high=255., size=(256, 256, 3))
            img1 = np.uint8(img1)
            img = img1

            img = img.transpose(2, 0, 1)
            img = img / 255.
            img = torch.FloatTensor(img).to(device)

            transform = transforms.Compose([data_normalization])
            image = transform(img)  # (3, 256, 256)

            # Encode
            image = image.unsqueeze(0)  # (1, 3, 256, 256)

            caption_image_beam_search(encoder, decoder, image, word_map, rev_word_map, args.beam_size)

    if args.data == 'custom':
        dataloader = load('custom', args.run_local, 1, 1)

        for i, data in tqdm(enumerate(dataloader)):
            caption_image_beam_search(encoder, decoder, data[0], word_map, rev_word_map, args.beam_size)

    torch.save({'pos': pos_dic_obj, 'noun_phrase_sum_of_log_prop': noun_phrase_sum_of_log_prop},
               save_dir + '/pos_dic_{}'.format(args.data))
    print('saved dic in: {}/pos_dic_{}'.format(save_dir, args.data))

# run.py
