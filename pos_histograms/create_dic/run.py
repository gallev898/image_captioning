import sys


sys.path.append('/home/mlspeech/gshalev/gal/image_captioning')
sys.path.append('/home/mlspeech/gshalev/anaconda3/envs/python3_env/lib')

import os
import json
import spacy
import torch
import argparse
import itertools
import en_core_web_sm

import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms

from tqdm import *
from standart_training.utils import data_normalization
from dataset_loader.datasets import CaptionDataset
from dataset_loader.dataloader import load


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_name = 'coco_5_cap_per_img_5_min_word_freq'
filename = 'BEST_checkpoint_' + data_name + '.pth.tar'
pre_pros_data_dir = '../../output_folder'
noun_phrase_sum_of_log_prop = []
pos_dic_obj = {}
uncompleted_seq = 0

parser = argparse.ArgumentParser(description='Generate Caption')
parser.add_argument('--model', type=str)
parser.add_argument('--run_local', default=False, action='store_true')
parser.add_argument('--debug', default=False, action='store_true')
parser.add_argument('--data', default='test', type=str)
parser.add_argument('--beam_size', default=5, type=int)
args = parser.parse_args()


def caption_image_beam_search(encoder, decoder, image_path, word_map, rev_word_map, beam_size):
    global uncompleted_seq

    k = beam_size
    vocab_size = len(word_map)

    image = image_path.to(device)

    # The encoder return the image by peaces (14*14*2048)
    encoder_out = encoder(image)
    enc_image_size = encoder_out.size(1)
    encoder_dim = encoder_out.size(3)

    # Flatten encoding (196*2048)
    encoder_out = encoder_out.view(1, -1, encoder_dim)
    num_pixels = encoder_out.size(1)

    # We'll treat the problem as having a batch size of k
    encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)  # (k, num_pixels, encoder_dim)

    # Tensor to store top k previous words at each step; now they're just <start>
    k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device)  # (k, 1)

    # Tensor to store top k sequences; now they're just <start>
    seqs = k_prev_words  # (k, 1)
    logits_list = torch.zeros([5, 1]).to(device)
    seqs_scores = torch.FloatTensor([[0.]] * k).to(device)

    # Tensor to store top k sequences' scores; now they're just 0
    top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

    # Tensor to store top k sequences' alphas; now they're just 1s
    seqs_alpha = torch.ones(k, 1, enc_image_size, enc_image_size).to(
        device)  # (k, 1, enc_image_size, enc_image_size) TODO: for visualization

    # Lists to store completed sequences, their alphas and scores
    complete_seqs = list()
    complete_logits_list = list()
    complete_seqs_alpha = list()
    complete_seqs_scores = list()
    complete_seqs_scores_for_all_steps = list()

    # Start decoding
    step = 1
    h, c = decoder.init_hidden_state(encoder_out)

    # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
    while True:
        embeddings = decoder.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)

        awe, alpha = decoder.attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels)

        alpha = alpha.view(-1, enc_image_size, enc_image_size)  # (s, enc_image_size, enc_image_size)

        gate = decoder.sigmoid(decoder.f_beta(h))  # gating scalar, (s, encoder_dim)# TODO: tihs is bla bla
        awe = gate * awe  # TODO: this is bla bla - motivated by dropout "like" - type of regularization, gate of [0,1] because of sigmoid

        concatination_of_input_and_att = torch.cat([embeddings, awe], dim=1)
        h, c = decoder.decode_step(concatination_of_input_and_att, (
            h, c))  # (s, decoder_dim)

        scores = decoder.fc(h)  # (s, vocab_size)
        logits = scores
        scores = F.log_softmax(scores, dim=1)  # TODO: in our next work this is where we intervert
        scores_copy = scores.clone()

        # Add
        scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size) TODO: can look like aggregated "log"

        # For the first step, all k points will have the same scores (since same k previous words, h, c)
        if step == 1:
            top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
        else:
            # Unroll and find top scores, and their unrolled indices
            top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

        # Convert unrolled indices to actual indices of scores
        prev_word_inds = top_k_words / vocab_size  # (s)
        next_word_inds = top_k_words % vocab_size  # (s)

        logits = [logits[i][next_word_inds[i].item()] for i in range(k)]
        logits = torch.stack(logits, dim=0).unsqueeze(1).to(device)

        # Add new words to sequences, alphas
        logits_list = torch.cat([logits_list[prev_word_inds], logits], dim=1)
        seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)
        seqs_scores = torch.cat([seqs_scores[prev_word_inds], scores_copy.view(-1)[top_k_words].unsqueeze(1)], dim=1)
        seqs_alpha = torch.cat([seqs_alpha[prev_word_inds], alpha[prev_word_inds].unsqueeze(1)],
                               dim=1)  # (s, step+1, enc_image_size, enc_image_size)

        # Which sequences are incomplete (didn't reach <end>)?
        incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                           next_word != word_map['<end>']]
        complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

        # Set aside complete sequences
        if len(complete_inds) > 0:
            complete_logits_list.extend(logits_list[complete_inds].tolist())
            complete_seqs.extend(seqs[complete_inds].tolist())
            complete_seqs_alpha.extend(seqs_alpha[complete_inds].tolist())
            complete_seqs_scores.extend(top_k_scores[complete_inds])
            complete_seqs_scores_for_all_steps.extend(seqs_scores[complete_inds].tolist())
        k -= len(complete_inds)  # reduce beam length accordingly

        # Proceed with incomplete sequences
        if k == 0:
            break
        seqs = seqs[incomplete_inds]
        seqs_alpha = seqs_alpha[incomplete_inds]
        h = h[prev_word_inds[incomplete_inds]]
        c = c[prev_word_inds[incomplete_inds]]
        encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
        top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
        k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

        # Break if things have been going on too long
        if step > 50:
            break
        step += 1

    if len(complete_seqs_scores) == 0:
        print('complete_seqs_scores is empty')
        uncompleted_seq += 1
        return

    i = complete_seqs_scores.index(max(complete_seqs_scores))
    seq_sum = round(max(complete_seqs_scores).item(), 4)

    assertion_error = False
    try:
        assert round(np.array(complete_seqs_scores_for_all_steps[i]).sum(), 4) == seq_sum

    except AssertionError:
        assertion_error = True

        print('------------EXCEPTION ACCRUED---------------')
        print('{} != {}'.format(round(np.array(complete_seqs_scores_for_all_steps[i]).sum(), 4), seq_sum))

    if not assertion_error:
        top_seq_total_scors = complete_seqs_scores_for_all_steps[i]
        seq = complete_seqs[i]
        alphas = complete_seqs_alpha[i][1:-1]
        logits_list = complete_logits_list[i][1:-1]

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
                pos_dic_obj[token.pos_]['exp'].append(exp)  # TODO add logits max of alphas and var of alphas
                pos_dic_obj[token.pos_]['alphas_var'].append(
                    np.var(alph))  # TODO add logits max of alphas and var of alphas
                pos_dic_obj[token.pos_]['alphas_max'].append(
                    max(alph))  # TODO add logits max of alphas and var of alphas
                pos_dic_obj[token.pos_]['logits'].append(logit)  # TODO add logits max of alphas and var of alphas
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


def get_model_path_and_save_dir():
    if args.run_local:
        desktop_path = os.path.join(os.path.join(os.path.expanduser('~')), 'Desktop')
        model_path = os.path.join(os.path.join(desktop_path, 'trained_models'), os.path.join(args.model, filename))
        save_dir = "pos_dic"
    else:
        # dir = '/yoav_stg/gshalev/semantic_labeling/mscoco/val2014'
        model_path = "/yoav_stg/gshalev/image_captioning/{}/{}".format(args.model, filename)
        save_dir = "/yoav_stg/gshalev/image_captioning/{}/pos_dic".format(args.model)

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
        print('created dir: {}'.format(save_dir))

    return model_path, save_dir


def get_models(model_path):
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    decoder = checkpoint['decoder']
    decoder = decoder.to(device)
    decoder.eval()
    encoder = checkpoint['encoder']
    encoder = encoder.to(device)
    encoder.eval()

    return encoder, decoder


def get_word_map():
    word_map_path = '..' if args.run_local else pre_pros_data_dir
    word_map_file = os.path.join(word_map_path, 'WORDMAP_' + data_name + '.json')

    # Create rev word map
    with open(word_map_file, 'r') as j:
        map = json.load(j)
        print('loaded word map from: {}'.format(word_map_file))

    rev_map = {v: k for k, v in map.items()}

    return map, rev_map


if __name__ == '__main__':
    # global uncompleted_seq, pos_dic_obj, device, data_name, filename, pre_pros_data_dir, noun_phrase_sum_of_log_prop
    # Creat save dir
    model_path, save_dir = get_model_path_and_save_dir()

    # Get models
    encoder, decoder = get_models(model_path)

    # Load word map (word2ix)
    word_map, rev_word_map = get_word_map()

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
        for i in tqdm(range(100)):
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
    # print('saved dic in: {}'.format(save_dir + '/pos_dic_{}'.format(args.data)))
    print('saved dic in: {}/pos_dic_{}'.format(save_dir, args.data))
    print('uncompleted_seq = {}'.format(uncompleted_seq))
# run.py
