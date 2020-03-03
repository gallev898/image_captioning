import sys


sys.path.append('/home/mlspeech/gshalev/gal/image_cap2')
sys.path.append('/home/mlspeech/gshalev/anaconda3/envs/python3_env/lib')

import PIL

from dataset_loader.Pertubation import *
from decoding_strategist_visualizations.beam_search.beam_search_pack_utils import beam_search_decode
from pos_histograms.create_dic_for_histograms.create_dic_utils import *
from utils import *

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


def caption_image_beam_search(encoder, decoder, image, word_map, rev_word_map, beam_size):
    seq, alphas, top_seq_total_scors, seq_sum, logits_list = beam_search_decode(encoder, image, beam_size, word_map,
                                                                                decoder)

    if seq == None:
        return None, None, None, None, None

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

        try:
            inde = [words_copy.index(str(x)) + lats_i for x in i_str.split()]
        except ValueError:
            print(ValueError.args)
            return None, None, None, None, None

        # inde = [words_copy.index(str(x)) + lats_i for x in i_str.split()]
        np_idx.append(inde)
        print(inde)
        lats_i = inde[-1] + 1
        words_copy = list(original_str[lats_i:])

    idx_count = 0
    for w, token, score, exp, alph, logit in zip(words, doc, top_seq_total, ex_top_seq_total, alphas, logits_list):
        if args.debug:
            print(w, token.pos_, score, exp)
        alph = list(itertools.chain.from_iterable(alph))  # flatten to one list
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

    return seq, alphas, top_seq_total_scors, seq_sum, words


if __name__ == '__main__':
    print('Strating beam search : {}'.format(args.beam_size))
    model_path, save_dir = get_model_path_and_save_dir(args, 'pos_dic')

    encoder, decoder = get_models(model_path, device)
    word_map, rev_word_map = get_word_map(args.run_local, '../../output_folder/WORDMAP_' + data_name + '.json')

    if args.replace_mode:
        word_map['NOUN'] = 9491
        word_map['VERB'] = 9492

    print(len(word_map))
    print('create pos dic for {} data'.format(args.data))
    metrics_data_type_to_save = ['test', 'perturbed_jpeg', 'perturbed_salt']

    # section: build dic
    if args.data == 'test':
        print('using cuda: {}', format(device))
        print('args.data = {}'.format(args.data))

        data_path = '/yoav_stg/gshalev/image_captioning/output_folder'
        coco_data = '../../output_folder' if args.run_local else data_path

        test_loader = torch.utils.data.DataLoader(
            CaptionDataset(coco_data, data_name, 'TEST', transform=transforms.Compose([data_normalization])),
            batch_size=1, shuffle=True, num_workers=1, pin_memory=True)
        print('lev test_loader: {}'.format(len(test_loader)))

        gt_metric_dic = {'annotations': list()}
        hp_metric_dic = {'annotations': list()}
        for i, (image, caps, caplens, allcaps) in tqdm(enumerate(test_loader)):
            for ci in range(allcaps.shape[1]):
                gt = [rev_word_map[ind.item()] for ind in allcaps[0][ci]][1:caplens[0][ci].item() - 1]
                gt_metric_dic['annotations'].append({u'image_id': i, u'caption': gt})

            image = image.to(device)
            _, _, _, seq_sum, words = caption_image_beam_search(encoder, decoder, image, word_map, rev_word_map,
                                                                args.beam_size)
            if not None == words:
                hp_metric_dic['annotations'].append({u'image_id': i, u'caption': words})
            if not None == seq_sum:
                sentences_likelihood.append(seq_sum)

            if args.debug and i == 10:
                break

    if args.data == 'perturbed_jpeg':
        print('Create Dic for \'perturbed_jpeg\'')
        data_path = '/yoav_stg/gshalev/image_captioning/output_folder'
        coco_data = '../../output_folder' if args.run_local else data_path

        test_loader = torch.utils.data.DataLoader(
            CaptionDataset(coco_data, data_name, 'TEST', transform=transforms.Compose([
                transforms.ToPILImage(),
                ImgAugTransformJpegCompression(),
                lambda x: PIL.Image.fromarray(x),
                transforms.ToTensor(),
                data_normalization
                ])),
            batch_size=1, shuffle=True, num_workers=1, pin_memory=True)
        print('size of test_loader: {}'.format(len(test_loader)))

        gt_metric_dic = {'annotations': list()}
        hp_metric_dic = {'annotations': list()}
        for i, (image, caps, caplens, allcaps) in tqdm(enumerate(test_loader)):
            for ci in range(allcaps.shape[1]):
                gt = [rev_word_map[ind.item()] for ind in allcaps[0][ci]][1:caplens[0][ci].item() - 1]
                gt_metric_dic['annotations'].append({u'image_id': i, u'caption': gt})

            if i % 100 == 0:
                print('process : {}/{}'.format(i, len(test_loader)))
            image = image.to(device)
            _, _, _, seq_sum, words = caption_image_beam_search(encoder, decoder, image, word_map, rev_word_map,
                                                                args.beam_size)
            if not None == words:
                hp_metric_dic['annotations'].append({u'image_id': i, u'caption': words})

            if not None == seq_sum:
                sentences_likelihood.append(seq_sum)

    if args.data == 'perturbed_salt':
        data_path = '/yoav_stg/gshalev/image_captioning/output_folder'
        coco_data = '../../output_folder' if args.run_local else data_path

        transform = transforms.Compose([
            transforms.ToPILImage(),
            ImgAugTransformSaltAndPepper(),
            lambda x: PIL.Image.fromarray(x),
            transforms.ToTensor(),
            data_normalization
            ])

        test_loader = torch.utils.data.DataLoader(
            CaptionDataset(coco_data, data_name, 'TEST', transform=transform),
            batch_size=1, shuffle=True, num_workers=1, pin_memory=True)

        gt_metric_dic = {'annotations': list()}
        hp_metric_dic = {'annotations': list()}
        for i, (image, caps, caplens, allcaps) in tqdm(enumerate(test_loader)):
            for ci in range(allcaps.shape[1]):
                gt = [rev_word_map[ind.item()] for ind in allcaps[0][ci]][1:caplens[0][ci].item() - 1]
                gt_metric_dic['annotations'].append({u'image_id': i, u'caption': gt})

            if i % 100 == 0:
                print('process : {}/{}'.format(i, len(test_loader)))
            image = image.to(device)
            _, _, _, seq_sum, words = caption_image_beam_search(encoder, decoder, image, word_map, rev_word_map,
                                                                args.beam_size)
            if not None == words:
                hp_metric_dic['annotations'].append({u'image_id': i, u'caption': words})

            if not None == seq_sum:
                sentences_likelihood.append(seq_sum)

    if args.data == 'custom':
        print('using cuda: {}', format(device))

        dataloader = load('custom', args.run_local, 1, 1)

        for i, data in tqdm(enumerate(dataloader)):
            image = data[0].to(device)
            if image.shape[1] != 3:
                continue
            print('########## image shape:{}'.format(image.shape))
            _, _, _, seq_sum, _ = caption_image_beam_search(encoder, decoder, image, word_map, rev_word_map,
                                                            args.beam_size)
            if not None == seq_sum:
                sentences_likelihood.append(seq_sum)

    if args.data == 'cartoon':
        print('using cuda: {}', format(device))

        dataloader = load('cartoon', args.run_local, 1, 1)
        for i, data in tqdm(enumerate(dataloader)):
            image = data[0].to(device)
            if image.shape[1] != 3:
                continue
            print('########## image shape:{}'.format(image.shape))
            _, _, _, seq_sum, _ = caption_image_beam_search(encoder, decoder, image, word_map, rev_word_map,
                                                            args.beam_size)
            if not None == seq_sum:
                sentences_likelihood.append(seq_sum)

    if args.data == 'cropped_images':
        print('using cuda: {}', format(device))

        dataloader = load('cropped_images', args.run_local, 1, 1)

        for i, data in tqdm(enumerate(dataloader)):
            image = data[0].to(device)
            if image.shape[1] != 3:
                continue
            print('########## image shape:{}'.format(image.shape))
            _, _, _, seq_sum, words = caption_image_beam_search(encoder, decoder, image, word_map, rev_word_map,
                                                                args.beam_size)
            if not None == seq_sum:
                sentences_likelihood.append(seq_sum)

    if args.data == 'random':
        print('using cuda: {}', format(device))
        print('random_range: {}'.format(args.random_range))

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

            _, _, _, seq_sum = caption_image_beam_search(encoder, decoder, image, word_map, rev_word_map,
                                                         args.beam_size)
            if not None == seq_sum:
                sentences_likelihood.append(seq_sum)

    # section: save dic
    dic_name = 'pos_dic_{}_beam_{}'.format(args.data, args.beam_size)
    print('dic name: {}'.format(dic_name))

    save_data_path = os.path.join(save_dir, dic_name)
    print('saving dic in: {}'.format(save_data_path))

    torch.save({'pos': pos_dic_obj,
                'noun_phrase_sum_of_log_prop': noun_phrase_sum_of_log_prop,
                'sentence_likelihood': sentences_likelihood},
               save_data_path)

    # section: seva metrics
    if args.data in metrics_data_type_to_save:
        metrics_save_dir = "/yoav_stg/gshalev/image_captioning/{}/{}".format(args.model, 'metrics')
        if not os.path.exists(metrics_save_dir):
            os.mkdir(metrics_save_dir)
            print('created dir: {}'.format(metrics_save_dir))

        metrics_result_file_name = 'metrics_results_{}_beam_{}'.format(args.data, args.beam_size)
        torch.save({'gt': gt_metric_dic, 'hyp': hp_metric_dic},
                   os.path.join(metrics_save_dir, metrics_result_file_name))
        print('Saved metrics results in {}'.format(os.path.join(metrics_save_dir, metrics_result_file_name)))

# run_beam_search.py
