import sys


sys.path.append('/home/mlspeech/gshalev/gal/image_cap2')
sys.path.append('/home/mlspeech/gshalev/anaconda3/envs/python3_env/lib')

import os
import torch
import torch.nn.functional as F


data_name = 'coco_5_cap_per_img_5_min_word_freq'
filename = 'BEST_checkpoint_' + data_name + '.pth.tar'
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_folder = 'output_folder'  # folder with data files saved by create_input_files.py
word_map_file = '../../../output_folder/WORDMAP_' + data_name + '.json'
desktop_path = os.path.join(os.path.join(os.path.expanduser('~')), 'Desktop')


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


def caption_image(encoder, decoder, image, word_map, top_k, top_p, device):
    # Here is how to use this function for top-p sampling
    temperature = 1.0

    # Encode
    encoder_out = encoder(image)  # (1, enc_image_size, enc_image_size, encoder_dim)
    encoder_dim = encoder_out.size(1)

    # Flatten encoding
    encoder_out = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
    # encoder_out = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
    num_pixels = encoder_out.size(1)

    encoder_out = encoder_out.expand(1, num_pixels, encoder_dim)

    prev_word = torch.LongTensor([word_map['<start>']]).to(device)
    seqs = prev_word
    seqs_prop = torch.FloatTensor([0.]).to(device)
    seqs_logits = torch.FloatTensor([0.]).to(device)
    # seqs_alpha = torch.ones(1, enc_image_size, enc_image_size).to(device)

    h, c = decoder.init_hidden_state(encoder_out)
    h = torch.zeros(h.shape).squeeze(1).to(device)##
    c = torch.zeros(c.shape).squeeze(1).to(device)##

    while True:
        embeddings = decoder.embedding(prev_word).squeeze(1)

        h, c = decoder.decode_step(embeddings, (h, c))  # (s, decoder_dim)

        logits = decoder.fc(h)  # (s, vocab_size)

        logits = logits / temperature

        filtered_logits = top_k_top_p_filtering(logits.squeeze(0), top_k=top_k, top_p=top_p)

        # Sample from the filtered distribution
        probabilities = F.softmax(filtered_logits, dim=-1)
        next_token = torch.multinomial(probabilities, 1)
        seqs_prop = torch.cat((seqs_prop, probabilities[next_token]))
        seqs_logits = torch.cat((seqs_logits, filtered_logits[next_token]))
        seqs = torch.cat((seqs, next_token), dim=0)

        prev_word = next_token
        if next_token == word_map['<end>']:
            break

    seqs = [x.item() for x in seqs]
    seqs_prop = [x.item() for x in seqs_prop]
    seqs_logits = [x.item() for x in seqs_logits]
    return seqs, None, seqs_prop, seqs_logits
# top_k_p_pack_utils_no_attention.py



def caption_image1(encoder, decoder, image, word_map, top_k, top_p, device, representations_):
    # Here is how to use this function for top-p sampling
    temperature = 1.0

    # Encode
    encoder_out = encoder(image)  # (1, enc_image_size, enc_image_size, encoder_dim)
    encoder_dim = encoder_out.size(1)

    # Flatten encoding
    encoder_out = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
    # encoder_out = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
    num_pixels = encoder_out.size(1)

    # encoder_out = encoder_out.expand(1, num_pixels, encoder_dim)
    encoder_out = encoder_out.squeeze(1)

    prev_word = torch.LongTensor([word_map['<start>']]).to(device)
    seqs = prev_word
    seqs_prop = torch.FloatTensor([0.]).to(device)
    seqs_logits = torch.FloatTensor([0.]).to(device)
    # seqs_alpha = torch.ones(1, enc_image_size, enc_image_size).to(device)

    h, c = decoder.init_hidden_state(encoder_out)
    h = torch.zeros(h.shape).squeeze(1).to(device)##
    c = torch.zeros(c.shape).squeeze(1).to(device)##
    step = 1
    while True:
        embeddings = decoder.embedding(prev_word).squeeze(1)


        if step == 1:
            h, c = decoder.decode_step(encoder_out, (h, c))  # (s, decoder_dim)
            step +=1
        else:
            h, c = decoder.decode_step(embeddings, (h, c))  # (s, decoder_dim)


        logits = torch.matmul(h, representations_).to(device)


        logits = logits / temperature

        filtered_logits = top_k_top_p_filtering(logits.squeeze(0), top_k=top_k, top_p=top_p)

        # Sample from the filtered distribution
        probabilities = F.softmax(filtered_logits, dim=-1)
        next_token = torch.multinomial(probabilities, 1)
        seqs_prop = torch.cat((seqs_prop, probabilities[next_token]))
        seqs_logits = torch.cat((seqs_logits, filtered_logits[next_token]))
        seqs = torch.cat((seqs, next_token), dim=0)

        prev_word = next_token
        if next_token == word_map['<end>']:
            break

    seqs = [x.item() for x in seqs]
    seqs_prop = [x.item() for x in seqs_prop]
    seqs_logits = [x.item() for x in seqs_logits]
    return seqs, seqs_prop, seqs_logits

# top_k_p_pack_utils_no_attention.py
