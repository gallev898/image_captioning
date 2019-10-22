import sys

from PIL import Image


sys.path.append('/home/mlspeech/gshalev/anaconda3/envs/python3_env/lib')
sys.path.append('/home/mlspeech/gshalev/gal/image_captioning')

from standart_training.utils import data_normalization
from decoding_strategist.pack_utils import *
from dataset_loader.custom_dataloader import Custom_Image_Dataset

import torch
import argparse
import en_core_web_sm

import numpy as np
import torchvision.transforms as transforms


# args
parser = argparse.ArgumentParser(description='Show, Attend, and Tell - Tutorial - Generate Caption')
parser.add_argument('--model', type=str)
parser.add_argument('--run_local', default=False, action='store_true')
parser.add_argument('--ood', default=False, action='store_true')
parser.add_argument('--all_data', default=False, action='store_true')
parser.add_argument('--limit_ex', type=int, default=1)
parser.add_argument('--beam_size', default=1, type=int)
parser.add_argument('--dont_smooth', dest='smooth', action='store_false', help='do not smooth alpha overlay')
parser.add_argument('--top', type=str, default='k')
args = parser.parse_args()

# global
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_name = 'coco_5_cap_per_img_5_min_word_freq'
filename = 'BEST_checkpoint_' + data_name + '.pth.tar'


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
        seqs = torch.cat((seqs, next_token), dim=0)
        print(rev_word_map[next_token.item()])

        k_prev_words = next_token
        if k == 0 or next_token == word_map['<end>']:
            break

    seqs = [x.item() for x in seqs]
    seqs_scores = [x.item() for x in seqs_scores]
    return seqs, seqs_alpha, seqs_scores


def visualize_att(image_path, seq, alphas, rev_word_map, top_seq_total_scors, save_dir, smooth=True):
    image = image_path.squeeze(0)
    image = transforms.ToPILImage()(image)
    image = image.resize([14 * 24, 14 * 24], Image.LANCZOS)

    words = [rev_word_map[ind] for ind in seq]

    nlp = en_core_web_sm.load()
    doc = nlp(' '.join(words[1:-1]))
    pos = [x.pos_ for x in doc]
    pos.insert(0, '-')
    pos.insert(len(pos), '-')

    top_seq_total_scors_exp = np.exp(top_seq_total_scors)

    return visualization(image, alphas, words, pos, top_seq_total_scors, top_seq_total_scors_exp, smooth, save_dir)


def run(encoder, decoder, word_map, rev_word_map, save_dir, top_k, top_p, image_path=None):
    # seqs, seqs_alpha, seqs_scores
    seq, alphas, top_seq_total_scors = caption_image(encoder,
                                                              decoder,
                                                              image_path,
                                                              word_map,
                                                              top_k, top_p)

    alphas = torch.FloatTensor(alphas)

    visualize_att(image_path, seq, alphas, rev_word_map, top_seq_total_scors, save_dir, args.smooth)



def get_data_loader():
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
        data_normalization
        ])

    data = Custom_Image_Dataset(os.path.join(desktop_path, 'custom_images'), transform)

    dataloader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=False, num_workers=1)

    return dataloader


if __name__ == '__main__':
    top_k = 3
    top_p = 0
    model_path, save_dir = get_model_path_and_save_path(args, 'top_k' if top_k > 0 else 'top_p')

    # Load model
    encoder, decoder = get_models(model_path)

    # Create rev word map
    word_map, rev_word_map = get_word_map()

    dataloader = get_data_loader()


    # top_p = 0.9

    for ind, image in enumerate(dataloader):
        run(encoder, decoder, word_map, rev_word_map, save_dir, top_k, top_p, image_path=image)
