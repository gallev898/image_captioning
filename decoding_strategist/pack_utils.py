import json
import os

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import skimage.transform
import torch
import torch.nn.functional as F


data_name = 'coco_5_cap_per_img_5_min_word_freq'
filename = 'BEST_checkpoint_' + data_name + '.pth.tar'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_folder = 'output_folder'  # folder with data files saved by create_input_files.py
word_map_file = '../../../output_folder/WORDMAP_' + data_name + '.json'
desktop_path = os.path.join(os.path.join(os.path.expanduser('~')), 'Desktop')


def get_model_path_and_save_path(args, save_dir_name):
    if args.run_local:
        desktop_path = os.path.join(os.path.join(os.path.expanduser('~')), 'Desktop')
        model_path = os.path.join(desktop_path, os.path.join(args.model, filename))
        save_dir = "GIFs"
    else:
        model_path = "/yoav_stg/gshalev/image_captioning/{}/{}".format(args.model, filename)
        save_dir = "/yoav_stg/gshalev/image_captioning/{}/GIFs".format(args.model)

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    beam_size_dir = save_dir_name
    save_dir = os.path.join(save_dir, beam_size_dir)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

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
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)
    rev_word_map = {v: k for k, v in word_map.items()}  # ix2word

    return word_map, rev_word_map


def visualization(image, alphas, words, pos, top_seq_total_scors, top_seq_total_scors_exp, smooth, save_dir):
    for t in range(len(words)):
        if t > 50:
            break
        plt.subplot(np.ceil(len(words) / 5.), 5, t + 1)

        plt.text(0, 1, '{}\n{}\n {:.4f} - \n {:.4f}'.format(words[t], pos[t], top_seq_total_scors[t],
                                                            top_seq_total_scors_exp[t]),
                 color='black', backgroundcolor='white',
                 fontsize=12)
        plt.imshow(image)
        current_alpha = alphas[t, :]
        if smooth:
            alpha = skimage.transform.pyramid_expand(current_alpha.numpy(), upscale=24, sigma=8)
        else:
            alpha = skimage.transform.resize(current_alpha.numpy(), [14 * 24, 14 * 24])
        if t == 0:
            plt.imshow(alpha, alpha=0)
        else:
            plt.imshow(alpha, alpha=0.8)
        plt.set_cmap(cm.Greys_r)
        plt.axis('off')

    plt.savefig(os.path.join(save_dir, 'temp_{}'.format(np.random.randint(0, 10000))))
    plt.clf()
    return words, image


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k >0: keep only top k tokens with highest probability (top-k filtering).
            top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
    """
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits


def encode(encoder, image, k, word_map):
    # Encode
    encoder_out = encoder(image)  # (1, enc_image_size, enc_image_size, encoder_dim)
    enc_image_size = encoder_out.size(1)
    encoder_dim = encoder_out.size(3)

    # Flatten encoding
    encoder_out = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
    num_pixels = encoder_out.size(1)

    # We'll treat the problem as having a batch size of k
    encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)  # (k, num_pixels, encoder_dim)

    # Tensor to store top k previous words at each step; now they're just <start>
    k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device)  # (k, 1)

    # Tensor to store top k sequences; now they're just <start>
    seqs = k_prev_words  # (k, 1)
    seqs_scores = torch.FloatTensor([[0.]] * k).to(device)

    # Tensor to store top k sequences' scores; now they're just 0
    top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

    # Tensor to store top k sequences' alphas; now they're just 1s
    seqs_alpha = torch.ones(k, 1, enc_image_size, enc_image_size).to(device)

    return encoder_out, enc_image_size, k_prev_words, seqs, seqs_scores, top_k_scores, seqs_alpha


def beam_search_decode(encoder, image, k, word_map, decoder, vocab_size):
    # Encode
    encoder_out, enc_image_size, k_prev_words, seqs, seqs_scores, top_k_scores, seqs_alpha = encode(encoder, image, k,
                                                                                                    word_map)

    # Lists to store completed sequences, their alphas and scores
    complete_seqs = list()
    complete_seqs_alpha = list()
    complete_seqs_scores = list()
    complete_seqs_scores_for_all_steps = list()

    # Start decoding
    step = 1
    h, c = decoder.init_hidden_state(encoder_out)

    # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
    while True:
        embeddings = decoder.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)

        awe, alpha = decoder.attention(encoder_out,
                                       h)  # (s, encoder_dim), (s, num_pixels) האלפות אחרי סופטמקס  והוקטור האטנשיין

        alpha = alpha.view(-1, enc_image_size, enc_image_size)  # (s, enc_image_size, enc_image_size)

        gate = decoder.sigmoid(decoder.f_beta(h))  # gating scalar, (s, encoder_dim)# TODO: tihs is bla bla
        awe = gate * awe  # TODO: this is bla bla - motivated by dropout "like" - type of regularization, gate of [0,1] because of sigmoid

        concatination_of_input_and_att = torch.cat([embeddings, awe], dim=1)
        h, c = decoder.decode_step(concatination_of_input_and_att, (  h, c))  # (s, decoder_dim)

        scores = decoder.fc(h)  # (s, vocab_size)
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

        # Add new words to sequences, alphas
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

    return seq, alphas, top_seq_total_scors, seq_sum
