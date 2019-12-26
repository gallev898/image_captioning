import sys

from decoding_strategist_visualizations.decoding_strategist_utils import *


sys.path.append('/home/mlspeech/gshalev/anaconda3/envs/python3_env/lib')
sys.path.append('/home/mlspeech/gshalev/gal/image_captioning')

from utils import *
from dataset_loader.dataloader import load
from decoding_strategist_visualizations.top_k_top_p_captions.top_k_p_pack_utils import *

import torch
import argparse
import en_core_web_sm

import numpy as np

args = get_args()


def visualize_att(image, seq, alphas, rev_word_map, top_seq_total_scors, save_dir, image_name, smooth=True):
    image = image.squeeze(0)  # remove batch dim
    image = image.numpy()

    image = image.transpose((1, 2, 0))

    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)

    words = [rev_word_map[ind] for ind in seq]

    nlp = en_core_web_sm.load()
    doc = nlp(' '.join(words[1:-1]))
    pos = [x.pos_ for x in doc]
    pos.insert(0, '-')
    pos.insert(len(pos), '-')

    top_seq_total_scors_exp = np.exp(top_seq_total_scors)

    return visualization(image, alphas, words, pos, top_seq_total_scors, top_seq_total_scors_exp, smooth, save_dir,
                         image_name)


def run(encoder, decoder, word_map, rev_word_map, save_dir, top_k, top_p, image, image_title):

    seq, alphas, top_seq_total_scors, seqs_scores_logits = caption_image(encoder,
                                                     decoder,
                                                     image,
                                                     word_map,
                                                     top_k, top_p)

    alphas = torch.FloatTensor(alphas)

    visualize_att(image, seq, alphas, rev_word_map, top_seq_total_scors, save_dir, image_title, args.smooth)


if __name__ == '__main__':
    top_k = 5  # NOTICE: inr
    top_p = 0  # NOTICE: double

    model_path, save_dir = get_model_path_and_save_path(args, 'top_k_{}'.format(top_k)
    if top_k > 0 else 'top_p_{}'.format(top_p))

    # Load model
    encoder, decoder = get_models(model_path)

    # Create rev word map
    word_map, rev_word_map = get_word_map()

    dataloader = load('custom', True, 1, 1)

    for ind, image_data in enumerate(dataloader):
        image = image_data[0]
        image_title = image_data[1][0]

        run(encoder, decoder, word_map, rev_word_map, save_dir, top_k, top_p, image, image_title)
