import PIL
import sys



sys.path.append('/home/mlspeech/gshalev/gal/image_cap2')
sys.path.append('/home/mlspeech/gshalev/anaconda3/envs/python3_env/lib')
from dataset_loader.Pertubation import ImgAugTransformJpegCompression, ImgAugTransformSaltAndPepper

from decoding_strategist_visualizations.top_k_top_p_captions.top_k_p_pack_utils import caption_image
from utils import *
from pos_histograms.create_dic_for_histograms.create_dic_utils import *

import os
import torch
import itertools
import en_core_web_sm

import numpy as np
import torchvision.transforms as transforms

from tqdm import *
from dataset_loader.datasets import CaptionDataset
from dataset_loader.dataloader import load


args = get_args()
device = torch.device("cuda:{}".format(args.cuda) if torch.cuda.is_available() else "cpu")

top_k = args.top_k  # NOTICE: int
top_p = args.top_p  # NOTICE: double


def caption_image_beam_search(encoder, decoder, image, word_map, rev_word_map):
    seq, seqs_alpha, seqs_prop, seqs_logits = caption_image(encoder, decoder, image, word_map, top_k, top_p)

    words = [rev_word_map[ind] for ind in seq]

    words = words[1:-1]
    seqs_prop = seqs_prop[1:-1]
    words_as_str = ' '.join(words)
    nlp = en_core_web_sm.load()
    doc = nlp(words_as_str)

    lats_i = 0
    np_idx = []
    words_copy = words
    original_str = words

    likelihood = np.log(seqs_prop)
    sen_likelihood = sum(likelihood)

    # Collect inexed of noun phrase
    for i in doc.noun_chunks:
        i_str = '{} '.format(i.string)
        if '<unk ' in i_str:
            print('replacing')
            i_str = i_str.replace('<unk ', '<unk> ')
        i_str = i_str.strip()

        print('*****')
        print(words_copy)
        print(i_str)

        try:
            print('split: {}'.format(i_str.split()))
            inde = [words_copy.index(str(x)) + lats_i for x in i_str.split()]
        except ValueError as e:
            print('ValueError {}:'.format(e))
            return None, None, None, None

        np_idx.append(inde)

        print(inde)
        lats_i = inde[-1] + 1
        words_copy = list(original_str[lats_i:])

    idx_count = 0
    for w, token, prop, alph, logit in zip(words, doc, seqs_prop, seqs_alpha, seqs_logits):
        alph = list([x.item() for x in itertools.chain.from_iterable(alph)])
        # alph = list(itertools.chain.from_iterable(alph))
        if token.pos_ in pos_dic_obj:
            pos_dic_obj[token.pos_]['prop'].append(prop)
            pos_dic_obj[token.pos_]['alphas_var'].append(np.var(alph))
            pos_dic_obj[token.pos_]['alphas_max'].append(max(alph))
            pos_dic_obj[token.pos_]['logits'].append(logit)
        else:
            pos_dic_obj[token.pos_] = {
                'prop': [prop],
                'alphas_var': [np.var(alph)],
                'alphas_max': [max(alph)],
                'logits': [logit]}

        for l in np_idx:
            if idx_count in l:
                l[l.index(idx_count)] = prop
                break

        idx_count += 1

    for l in np_idx:  # NOTICE for top k top p
        noun_phrase_sum_of_log_prop.append(sum(l))

    return seq, seqs_alpha, seqs_prop, sen_likelihood


if __name__ == '__main__':

    print('Starting top K: {}'.format(args.top_k) if args.top_k > 0 else 'Starting top_p: {}'.format(
        args.top_p) if args.top_p > 0 else 'GOING TO BREAK')
    if args.top_p <= 0 and args.top_k <= 0:
        exit(" top_k or top _p must have a value")

    model_path, save_dir = get_model_path_and_save_dir(args, 'pos_dic')
    encoder, decoder = get_models(model_path)
    word_map, rev_word_map = get_word_map(args.run_local, '../../output_folder/WORDMAP_' + data_name + '.json')

    print('create pos dic for {} data'.format(args.data))

    if args.data == 'test':
        print('using cuda: {}', format(device))

        print('args.data = {}'.format(args.data))
        p = '/yoav_stg/gshalev/image_captioning/output_folder'
        coco_data = '../../output_folder' if args.run_local else p

        test_loader = torch.utils.data.DataLoader(
            CaptionDataset(coco_data, data_name, 'TEST', transform=transforms.Compose([data_normalization])),
            batch_size=1, shuffle=True, num_workers=1, pin_memory=True)

        print('lev test_loader: {}'.format(len(test_loader)))
        for i, (image, caps, caplens, allcaps) in tqdm(enumerate(test_loader)):
            print(i)
            image = image.to(device)
            _, _, _, sen_likelihood = caption_image_beam_search(encoder, decoder, image, word_map, rev_word_map)
            if not None == sen_likelihood:
                sentences_likelihood.append(sen_likelihood)

    if args.data == 'perturbed_jpeg':
        print('using cuda: {}', format(device))

        print('args.data = {}'.format(args.data))
        p = '/yoav_stg/gshalev/image_captioning/output_folder'
        coco_data = '../../output_folder' if args.run_local else p

        transforms = transforms.Compose([
            transforms.ToPILImage(),
            ImgAugTransformJpegCompression(),
            lambda x: PIL.Image.fromarray(x),
            transforms.ToTensor(),
            data_normalization
            ])
        test_loader = torch.utils.data.DataLoader(
            CaptionDataset(coco_data, data_name, 'TEST', transform=transforms),
            batch_size=1, shuffle=True, num_workers=1, pin_memory=True)

        print('len test_loader: {}'.format(len(test_loader)))
        for i, (image, caps, caplens, allcaps) in tqdm(enumerate(test_loader)):
            print(i)
            image = image.to(device)
            _, _, _, sen_likelihood = caption_image_beam_search(encoder, decoder, image, word_map, rev_word_map)
            if not None == sen_likelihood:
                sentences_likelihood.append(sen_likelihood)

    if args.data == 'perturbed_salt':
        print('using cuda: {}', format(device))

        print('args.data = {}'.format(args.data))
        p = '/yoav_stg/gshalev/image_captioning/output_folder'
        coco_data = '../../output_folder' if args.run_local else p

        transforms = transforms.Compose([
            transforms.ToPILImage(),
            ImgAugTransformSaltAndPepper(),
            lambda x: PIL.Image.fromarray(x),
            transforms.ToTensor(),
            data_normalization
            ])
        test_loader = torch.utils.data.DataLoader(
            CaptionDataset(coco_data, data_name, 'TEST', transform=transforms),
            batch_size=1, shuffle=True, num_workers=1, pin_memory=True)

        print('len test_loader: {}'.format(len(test_loader)))
        for i, (image, caps, caplens, allcaps) in tqdm(enumerate(test_loader)):
            print(i)
            image = image.to(device)
            _, _, _, sen_likelihood = caption_image_beam_search(encoder, decoder, image, word_map, rev_word_map)
            if not None == sen_likelihood:
                sentences_likelihood.append(sen_likelihood)

    if args.data == 'random':
        print('using cuda: {}', format(device))

        for i in tqdm(range(args.random_range)):
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

            _, _, _, sen_likelihood = caption_image_beam_search(encoder, decoder, image, word_map, rev_word_map)
            sentences_likelihood.append(sen_likelihood)

    if args.data == 'custom':
        print('using cuda: {}', format(device))

        dataloader = load('custom', args.run_local, 1, 1)

        for i, data in enumerate(dataloader):
            image = data[0].to(device)
            if image.shape[1] != 3:
                continue
            _, _, _, sen_likelihood = caption_image_beam_search(encoder, decoder, image, word_map, rev_word_map)
            if not None == sen_likelihood:
                sentences_likelihood.append(sen_likelihood)

    if args.data == 'cartoon':
        print('using cuda: {}', format(device))

        dataloader = load('cartoon', args.run_local, 1, 1)

        for i, data in enumerate(dataloader):
            image = data[0].to(device)
            if image.shape[1] != 3:
                continue
            _, _, _, sen_likelihood = caption_image_beam_search(encoder, decoder, image, word_map, rev_word_map)
            if not None == sen_likelihood:
                sentences_likelihood.append(sen_likelihood)

    if args.data == 'cropped_images':

        print('using cuda: {}', format(device))

        dataloader = load('cropped_images', args.run_local, 1, 1)

        for i, data in enumerate(dataloader):

            image = data[0].to(device)
            if image.shape[1] != 3:
                continue
            _, _, _, sen_likelihood = caption_image_beam_search(encoder, decoder, image, word_map, rev_word_map)
            if not None == sen_likelihood:
                sentences_likelihood.append(sen_likelihood)

    dic_name = 'pos_dic_{}_{}_{}'.format(args.data, 'top_k' if args.top_k > 0 else 'top_p',
                                         args.top_k if args.top_k > 0 else args.top_p)
    print('dic name: {}'.format(dic_name))

    save_data_path = os.path.join(save_dir, dic_name)
    print('saving dic in: {}'.format(save_data_path))

    torch.save({'pos': pos_dic_obj,
                'noun_phrase_sum_of_log_prop': noun_phrase_sum_of_log_prop,
                'sentence_likelihood': sentences_likelihood},
               save_data_path)
# run_top_k_top_p.py
