import sys


sys.path.append('/home/mlspeech/gshalev/anaconda3/envs/python3_env/lib')
sys.path.append('/home/mlspeech/gshalev/gal/image_cap2')

import argparse

import numpy as np
import torch.nn.functional as F

from utils import *


def encode(encoder, image, beam_size, word_map):
    # Encode - return the output of resnet as 2048 channels
    encoder_out = encoder(image)  # (1, enc_image_size, enc_image_size, encoder_dim)

    enc_image_size = encoder_out.size(1)
    encoder_dim = encoder_out.size(1)

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
    # seqs_alpha = torch.ones(beam_size, 1, enc_image_size, enc_image_size).to(device)  # NOTICE: for visualization

    return encoder_out, enc_image_size, k_prev_words, seqs, seqs_scores, top_k_scores, None


def beam_search_decode(encoder, image, beam_size, word_map, decoder):
    vocab_size = len(word_map)

    # Encode
    encoder_out, enc_image_size, k_prev_words, seqs, seqs_scores, top_k_scores, _ = encode(encoder, image,
                                                                                                    beam_size,
                                                                                                    word_map)
    encoder_out = encoder_out.squeeze(1)
    uncompleted_seq = 0

    # Lists to store completed sequences, their alphas and scores
    complete_seqs = list()
    complete_seqs_scores = list()
    complete_seqs_scores_for_all_steps = list()

    # Start decoding
    step = 1
    h, c = decoder.init_hidden_state(encoder_out)
    h = torch.zeros(h.shape).squeeze(1).to(device)
    c = torch.zeros(c.shape).squeeze(1).to(device)

    complete_logits_list = list()  # NOTICE
    logits_list = torch.zeros([beam_size, 1]).to(device)  # NOTICE!!!!!!!!!!!! fixed from -torch.zeros([5, 1]).to(device)- to: -torch.zeros([beam_size, 1]).to(device)-

    # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
    while True:
        embeddings = decoder.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)

        if step == 1:
            h, c = decoder.decode_step(encoder_out, (h, c))  # (s, decoder_dim)
        else:
            h, c = decoder.decode_step(embeddings, (h, c))  # (s, decoder_dim)

        scores = decoder.fc(h)  # (s, vocab_size)
        logits = scores  # NOTICE
        scores = F.log_softmax(scores, dim=1)
        scores_copy = scores.clone()

        # Add
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

        # Which sequences are incomplete (didn't reach <end>)?
        incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                           next_word != word_map['<end>']]
        complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

        # Set aside complete sequences
        if len(complete_inds) > 0:
            complete_logits_list.extend(logits_list[complete_inds].tolist())  # NOTICE

            complete_seqs.extend(seqs[complete_inds].tolist())
            complete_seqs_scores.extend(top_k_scores[complete_inds])
            complete_seqs_scores_for_all_steps.extend(seqs_scores[complete_inds].tolist())
        beam_size -= len(complete_inds)  # reduce beam length accordingly

        # Proceed with incomplete sequences
        if beam_size == 0:
            break
        seqs = seqs[incomplete_inds]
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
        return seqs, None, None, None, None


    i = complete_seqs_scores.index(max(complete_seqs_scores))
    seq_sum = round(max(complete_seqs_scores).item(), 4)
    try:
        assert round(np.array(complete_seqs_scores_for_all_steps[i]).sum(), 4) == seq_sum
    except AssertionError:
        print('------------EXCEPTION ACCRUED---------------')
        print('{} != {}'.format(round(np.array(complete_seqs_scores_for_all_steps[i]).sum(), 4), seq_sum))

    top_seq_total_scors = complete_seqs_scores_for_all_steps[i]
    seq = complete_seqs[i]
    # alphas = complete_seqs_alpha[i]
    logits_list = complete_logits_list[i][1:-1]  # NOTICE

    print(seq_sum)
    return seq, _, top_seq_total_scors, seq_sum, logits_list


def get_args():
    # args
    parser = argparse.ArgumentParser(description='Show, Attend, and Tell - Tutorial - Generate Caption')

    parser.add_argument('--model', type=str)
    parser.add_argument('--model1', type=str)
    parser.add_argument('--model2', type=str)
    parser.add_argument('--model3', type=str)
    parser.add_argument('--multiple_models', default=False, action='store_true')
    parser.add_argument('--replace_mode', default=False, action='store_true')
    parser.add_argument('--save_dir_name', type=str, default='beam_size')
    parser.add_argument('--run_local', default=False, action='store_true')
    parser.add_argument('--limit_ex', type=int, default=10)
    parser.add_argument('--beam_size', default=5, type=int)
    parser.add_argument('--dont_smooth', dest='smooth', action='store_false', help='do not smooth alpha overlay')
    args = parser.parse_args()

    return args
# beam_search_pack_utils.py
