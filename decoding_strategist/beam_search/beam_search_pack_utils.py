import torch

import numpy as np
import torch.nn.functional as F

from utils import *
import argparse


def beam_search_decode(encoder, image, beam_size, word_map, decoder, vocab_size):
    # Encode
    encoder_out, enc_image_size, k_prev_words, seqs, seqs_scores, top_k_scores, seqs_alpha = encode(encoder, image,
                                                                                                    beam_size,
                                                                                                    word_map)

    global uncompleted_seq

    # Lists to store completed sequences, their alphas and scores
    complete_seqs = list()
    complete_seqs_alpha = list()
    complete_seqs_scores = list()
    complete_seqs_scores_for_all_steps = list()

    # Start decoding
    step = 1
    h, c = decoder.init_hidden_state(encoder_out)

    complete_logits_list = list()  # NOTICE
    logits_list = torch.zeros([5, 1]).to(device)  # NOTICE

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

        scores = decoder.fc(h)  # (s, vocab_size)
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
        return  # NOTICE

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

    return seq, alphas, top_seq_total_scors, seq_sum, logits_list


def get_args():
    # args
    parser = argparse.ArgumentParser(description='Show, Attend, and Tell - Tutorial - Generate Caption')
    parser.add_argument('--model', type=str)
    parser.add_argument('--save_dir_name', type=str, default='beam_size')
    parser.add_argument('--run_local', default=False, action='store_true')
    parser.add_argument('--ood', default=False, action='store_true')
    parser.add_argument('--all_data', default=False, action='store_true')
    parser.add_argument('--limit_ex', type=int, default=5)
    parser.add_argument('--beam_size', default=5, type=int)
    parser.add_argument('--dont_smooth', dest='smooth', action='store_false', help='do not smooth alpha overlay')
    args = parser.parse_args()

    return args
