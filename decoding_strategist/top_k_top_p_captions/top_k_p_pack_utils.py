import os

import torch
import torch.nn.functional as F


data_name = 'coco_5_cap_per_img_5_min_word_freq'
filename = 'BEST_checkpoint_' + data_name + '.pth.tar'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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


def caption_image(encoder, decoder, image_path, word_map, top_k, top_p):
    image = image_path

    # Here is how to use this function for top-p sampling
    temperature = 1.0

    # Encode
    encoder_out = encoder(image)  # (1, enc_image_size, enc_image_size, encoder_dim)
    enc_image_size = encoder_out.size(1)
    encoder_dim = encoder_out.size(3)

    # Flatten encoding
    encoder_out = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
    num_pixels = encoder_out.size(1)

    k = 1
    encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)

    k_prev_words = torch.LongTensor([word_map['<start>']]).to(device)
    seqs = k_prev_words
    seqs_scores = torch.FloatTensor([0.]).to(device)
    seqs_scores_logits = torch.FloatTensor([0.]).to(device)
    seqs_alpha = torch.ones(1, enc_image_size, enc_image_size).to(device)

    h, c = decoder.init_hidden_state(encoder_out)

    while True:
        embeddings = decoder.embedding(k_prev_words).squeeze(1)

        awe, alpha = decoder.attention(encoder_out, h)

        alpha = alpha.view(-1, enc_image_size, enc_image_size)
        seqs_alpha = torch.cat((seqs_alpha, alpha.detach()), dim=0)

        gate = decoder.sigmoid(decoder.f_beta(h))
        awe = gate * awe

        concatination_of_input_and_att = torch.cat([embeddings, awe], dim=1)
        h, c = decoder.decode_step(concatination_of_input_and_att, (h, c))  # (s, decoder_dim)

        logits = decoder.fc(h)  # (s, vocab_size)

        logits = logits / temperature

        filtered_logits = top_k_top_p_filtering(logits.squeeze(0), top_k=top_k, top_p=top_p)

        # Sample from the filtered distribution
        probabilities = F.softmax(filtered_logits, dim=-1)
        next_token = torch.multinomial(probabilities, 1)
        seqs_scores = torch.cat((seqs_scores, probabilities[torch.multinomial(probabilities, 1)]))
        seqs_scores_logits = torch.cat((seqs_scores_logits, filtered_logits[torch.multinomial(probabilities, 1)]))
        seqs = torch.cat((seqs, next_token), dim=0)

        k_prev_words = next_token
        if k == 0 or next_token == word_map['<end>']:
            break

    seqs = [x.item() for x in seqs]
    seqs_scores = [x.item() for x in seqs_scores]
    seqs_scores_logits = [x.item() for x in seqs_scores_logits]
    return seqs, seqs_alpha, seqs_scores, seqs_scores_logits
