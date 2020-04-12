import sys

sys.path.append('/home/mlspeech/gshalev/gal/image_cap2')
sys.path.append('/home/mlspeech/gshalev/anaconda3/envs/python3_env/lib')
import torch.nn.functional as F

import argparse
# from decoding_strategist_visualizations.beam_search.beam_search_pack_utils import beam_search_decode
# from pos_histograms.create_dic_for_histograms.create_dic_utils import *
from utils import get_word_map
from standart_training.V_models_with_attention import Encoder, DecoderWithAttention

import os
import torch
import itertools
import en_core_web_sm

import numpy as np
import torchvision.transforms as transforms

from tqdm import *
from dataset_loader.datasets import CaptionDataset

noun_phrase_sum_of_log_prop = []
sentences_likelihood = []
pos_dic_obj = {}

data_name = 'coco_5_cap_per_img_5_min_word_freq'
data_normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

filename = 'NEW_BEST_checkpoint_' + data_name + '.pth.tar'
emb_dim = 512  # dimension of word embeddings
attention_dim = 512  # dimension of attention linear layers
decoder_dim = 512  # dimension of decoder RNN
dropout = 0.5

parser = argparse.ArgumentParser(description='Generate Caption')
parser.add_argument('--model', type=str)
parser.add_argument('--data', type=str, default='test')
parser.add_argument('--beam_size', default=10, type=int)
parser.add_argument('--sphere', default=0, type=int)
parser.add_argument('--cuda', default=0, type=int)
parser.add_argument('--run_local', default=False, action='store_true')
parser.add_argument('--debug', default=False, action='store_true')
parser.add_argument('--cosine', default=False, action='store_true')
args = parser.parse_args()

device = torch.device("cuda:{}".format(args.cuda) if torch.cuda.is_available() else "cpu")
coco_data_path = '/Users/gallevshalev/PycharmProjects/image_captioning/output_folder' if args.run_local else '/yoav_stg/gshalev/image_captioning/output_folder'

def encode(encoder, image, beam_size, word_map, device):
    # Encode - return the output of resnet as 2048 channels
    encoder_out = encoder(image)  # (1, enc_image_size, enc_image_size, encoder_dim)

    enc_image_size = encoder_out.size(1)
    encoder_dim = encoder_out.size(3)

    # Flatten encoding
    encoder_out = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
    num_pixels = encoder_out.size(1)  # num of pixels in each

    # We'll treat the problem as having a batch size of k
    encoder_out = encoder_out.expand(beam_size, num_pixels, encoder_dim)  # (k, num_pixels, encoder_dim)

    # Tensor to store top k previous words at each step; now they're just <start>
    k_prev_words = torch.LongTensor([[word_map['<start>']]] * beam_size).to(device)  # (k, 1)

    # Tensor to store top k sequences; now they're just <start>
    seqs = k_prev_words  # (k, 1)
    seqs_scores = torch.FloatTensor([[0.]] * beam_size).to(device)

    # Tensor to store top k sequences' scores; now they're just 0
    top_k_scores = torch.zeros(beam_size, 1).to(device)  # (k, 1)

    # Tensor to store top k sequences' alphas; now they're just 1s
    seqs_alpha = torch.ones(beam_size, 1, enc_image_size, enc_image_size).to(device)  # NOTICE: for visualization

    return encoder_out, enc_image_size, k_prev_words, seqs, seqs_scores, top_k_scores, seqs_alpha

def beam_search_decode(encoder, image, beam_size, word_map, decoder, device, args, representations):
    vocab_size = len(word_map)
    # Encode
    encoder_out, enc_image_size, k_prev_words, seqs, seqs_scores, top_k_scores, seqs_alpha = encode(encoder, image,
                                                                                                    beam_size,
                                                                                                    word_map, device)

    uncompleted_seq = 0

    # Lists to store completed sequences, their alphas and scores
    complete_seqs = list()
    complete_seqs_alpha = list()
    complete_seqs_scores = list()
    complete_seqs_scores_for_all_steps = list()

    # Start decoding
    step = 1
    h, c = decoder.init_hidden_state(encoder_out)

    complete_logits_list = list()  # NOTICE
    logits_list = torch.zeros([beam_size, 1]).to(device)  # NOTICE!!!!!!!!!!!! fixed from -torch.zeros([5, 1]).to(device)- to: -torch.zeros([beam_size, 1]).to(device)-

    # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
    while True:
        embeddings = decoder.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)

        # NOTICE (s, encoder_dim), (s, num_pixels) האלפות אחרי סופטמקס  והוקטור האטנשיין
        awe, alpha = decoder.attention(encoder_out, h)

        alpha = alpha.view(-1, enc_image_size, enc_image_size)  # (s, enc_image_size, enc_image_size)

        # NOTICE: this is bla bla
        gate = decoder.sigmoid(decoder.f_beta(h))  # gating scalar, (s, encoder_dim)
        # NOTICE: this is bla bla - motivated by dropout "like" - type of regularization
        awe = gate * awe

        concatination_of_input_and_att = torch.cat([embeddings, awe], dim=1)
        h, c = decoder.decode_step(concatination_of_input_and_att, (h, c))  # (s, decoder_dim)

        if args.cosine:
            h = F.normalize(h, dim=1, p=2)
            representations = F.normalize(representations, dim=0, p=2)

        scores = torch.matmul(h, representations).to(device)
        if args.sphere > 0:
            scores *= args.sphere

        logits = scores  # NOTICE
        scores = F.log_softmax(scores, dim=1)
        scores_copy = scores.clone()

        # Add
        # NOTICE: can look like aggregated "log"
        scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

        # For the first step, all k points will have the same scores (since same k previous words, h, c)
        if step == 1:
            top_k_scores, top_k_words = scores[0].topk(beam_size, 0, True, True)  # (s)
        else:
            # Unroll and find top scores, and their unrolled indices
            top_k_scores, top_k_words = scores.view(-1).topk(beam_size, 0, True, True)  # (s)

        # Convert unrolled indices to actual indices of scores
        prev_word_inds = top_k_words / vocab_size  # (s)
        next_word_inds = top_k_words % vocab_size  # (s)

        logits = [logits[i][next_word_inds[i].item()] for i in range(beam_size)]  # NOTICE
        logits = torch.stack(logits, dim=0).unsqueeze(1).to(device)  # NOTICE
        logits_list = torch.cat([logits_list[prev_word_inds], logits], dim=1)  # NOTICE

        # Add new words to sequences, alphas
        seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)
        seqs_scores = torch.cat([seqs_scores[prev_word_inds], scores_copy.view(-1)[top_k_words].unsqueeze(1)], dim=1)
        seqs_alpha = torch.cat([seqs_alpha[prev_word_inds], alpha[prev_word_inds].unsqueeze(1)], dim=1)

        # Which sequences are incomplete (didn't reach <end>)?
        incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                           next_word != word_map['<end>']]
        complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

        # Set aside complete sequences
        if len(complete_inds) > 0:
            complete_logits_list.extend(logits_list[complete_inds].tolist())  # NOTICE

            complete_seqs.extend(seqs[complete_inds].tolist())
            complete_seqs_alpha.extend(seqs_alpha[complete_inds].tolist())
            complete_seqs_scores.extend(top_k_scores[complete_inds])
            complete_seqs_scores_for_all_steps.extend(seqs_scores[complete_inds].tolist())
        beam_size -= len(complete_inds)  # reduce beam length accordingly

        # Proceed with incomplete sequences
        if beam_size == 0:
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

    if len(complete_seqs_scores) == 0:  # NOTICE
        print('complete_seqs_scores is empty')  # NOTICE
        uncompleted_seq += 1  # NOTICE
        return None, None, None, None, None

    # if args.show_all_beam:
    #     seq2 = []
    #     alphas2 = []
    #     top_seq_total_scors2 = []
    #     logits_list2 = []
    #     for i in range(len(complete_seqs_scores)):
    #         top_seq_total_scors2.append(complete_seqs_scores_for_all_steps[i])
    #         seq2.append(complete_seqs[i])
    #         alphas2.append(complete_seqs_alpha[i])
    #         logits_list2.append(complete_logits_list[i][1:-1])  # NOTICE
    #
    #
    #     return seq2, alphas2, top_seq_total_scors2, None, logits_list2
    #
    # else:
    i = complete_seqs_scores.index(max(complete_seqs_scores))
    seq_sum = round(max(complete_seqs_scores).item(), 4)
    try:
        assert round(np.array(complete_seqs_scores_for_all_steps[i]).sum(), 4) == seq_sum
    except AssertionError:
        print('------------EXCEPTION ACCRUED---------------')
        print('{} != {}'.format(round(np.array(complete_seqs_scores_for_all_steps[i]).sum(), 4), seq_sum))

    top_seq_total_scors = complete_seqs_scores_for_all_steps[i]
    seq = complete_seqs[i]
    alphas = complete_seqs_alpha[i]
    logits_list = complete_logits_list[i][1:-1]  # NOTICE

    print(seq_sum)
    return seq, alphas, top_seq_total_scors, seq_sum, logits_list

def caption_image_beam_search(encoder, decoder, image, word_map, rev_word_map, beam_size, representations):
    seq, alphas, top_seq_total_scors, seq_sum, logits_list = beam_search_decode(encoder, image, beam_size, word_map,
                                                                                decoder, device, args, representations)

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


def get_model_path_and_save_dir(args, save_dir_name):
    if args.run_local:
        model_path = '/Users/gallevshalev/Desktop/trained_models/{}/{}'.format(args.model, filename)
        save_dir = save_dir_name
    else:
        model_path = "/yoav_stg/gshalev/image_captioning/{}/{}".format(args.model, filename)
        save_dir = "/yoav_stg/gshalev/image_captioning/{}/{}".format(args.model, save_dir_name)

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
        print('created dir: {}'.format(save_dir))

    if args.run_local:
        save_dir = os.path.join(save_dir, args.model)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
            print('created dir: {}'.format(save_dir))

    print('model path: {}'.format(model_path))
    print('save dir: {}'.format(save_dir))
    return model_path, save_dir


def get_models(model_path, device):
    checkpoint = torch.load(model_path, map_location=torch.device(device))
    decoder = DecoderWithAttention(attention_dim=attention_dim,
                                   embed_dim=emb_dim,
                                   decoder_dim=decoder_dim,
                                   vocab_size=len(word_map),
                                   device=device,
                                   dropout=dropout)

    decoder.load_state_dict(checkpoint['decoder'])
    decoder = decoder.to(device)
    decoder.eval()

    encoder = Encoder()
    encoder.load_state_dict(checkpoint['encoder'])
    encoder = encoder.to(device)
    encoder.eval()

    representations = checkpoint['representations']

    return encoder, decoder, representations


if __name__ == '__main__':


    print('Strating beam search : {}'.format(args.beam_size))
    model_path, save_dir = get_model_path_and_save_dir(args, 'pos_dic')

    word_map, rev_word_map = get_word_map(args.run_local, '../../output_folder/WORDMAP_' + data_name + '.json')
    encoder, decoder, representations = get_models(model_path, device)

    print(len(word_map))
    print('create pos dic for test data')

    # section: build dic
    print('using cuda: {}', format(device))

    data_path = '/yoav_stg/gshalev/image_captioning/output_folder'
    coco_data = '../../output_folder' if args.run_local else data_path

    test_loader = torch.utils.data.DataLoader(
        CaptionDataset(coco_data, data_name, 'TEST', transform=transforms.Compose([data_normalization])),
        batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
    print('lev test_loader: {}'.format(len(test_loader)))

    gt_metric_dic = {'annotations': list()}
    hp_metric_dic = {'annotations': list()}

    enumerator = enumerate(test_loader)
    for i, (image, caps, caplens, allcaps) in tqdm(enumerator):
        [next(enumerator, None) for _ in range(4)]
        for ci in range(allcaps.shape[1]):
            gt = [rev_word_map[ind.item()] for ind in allcaps[0][ci]][1:caplens[0][ci].item() - 1]
            gt_metric_dic['annotations'].append({u'image_id': i, u'caption': gt})

        image = image.to(device)
        _, _, _, seq_sum, words = caption_image_beam_search(encoder, decoder, image, word_map, rev_word_map,
                                                            args.beam_size, representations)
        if not None == words:
            hp_metric_dic['annotations'].append({u'image_id': i, u'caption': words})
        if not None == seq_sum:
            sentences_likelihood.append((i, seq_sum))

        if args.debug and i == 10:
            break



    # section: save dic
    dic_name = 'NEW_pos_dic_test_beam_{}'.format( args.beam_size)
    print('dic name: {}'.format(dic_name))

    save_data_path = os.path.join(save_dir, dic_name)
    print('saving dic in: {}'.format(save_data_path))

    torch.save({'pos': pos_dic_obj,
                'noun_phrase_sum_of_log_prop': noun_phrase_sum_of_log_prop,
                'generated_sentences_likelihood': sentences_likelihood},
               save_data_path)

    # section: seva metrics_roc_and_more
    metrics_save_dir = "/yoav_stg/gshalev/image_captioning/{}/{}".format(args.model,
                                                                         'metrics_roc_and_more') if not args.run_local else os.path.join(
        save_dir, 'metrics_roc_and_more')
    if not os.path.exists(metrics_save_dir):
        os.mkdir(metrics_save_dir)
        print('created dir: {}'.format(metrics_save_dir))

    metrics_result_file_name = 'NEW_metrics_results_{}_beam_{}'.format(args.data, args.beam_size)
    torch.save({'gt': gt_metric_dic, 'hyp': hp_metric_dic},
               os.path.join(metrics_save_dir, metrics_result_file_name))
    print(
        'Saved metrics_roc_and_more results in {}'.format(os.path.join(metrics_save_dir, metrics_result_file_name)))

# V_run_beam_search.py
