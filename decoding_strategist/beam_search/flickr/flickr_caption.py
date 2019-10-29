import sys

from decoding_strategist.decoding_strategist_utils import visualization


sys.path.append('/home/mlspeech/gshalev/anaconda3/envs/python3_env/lib')
sys.path.append('/home/mlspeech/gshalev/gal/image_captioning')

from utils import *
from dataset_loader.dataloader import load
from decoding_strategist.beam_search.beam_search_pack_utils import *

import torch
import en_core_web_sm

import numpy as np


args = get_args()


def caption_image(encoder, decoder, image, word_map, beam_size=3):
    vocab_size = len(word_map)

    return beam_search_decode(encoder, image, beam_size, word_map, decoder, vocab_size)


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


def run(encoder, decoder, word_map, rev_word_map, save_dir, image_data):
    image = image_data[0].unsqueeze(0)
    image_title = image_data[1]
    image_title = min(image_title, key=len)

    seq, alphas, top_seq_total_scors, seq_sum, logits_list = caption_image(encoder,
                                                              decoder,
                                                              image,
                                                              word_map,
                                                              args.beam_size)

    alphas = torch.FloatTensor(alphas)

    visualize_att(image, seq, alphas, rev_word_map, top_seq_total_scors, save_dir, image_title, args.smooth)

    print('seq_sum: {}'.format(seq_sum))


if __name__ == '__main__':

    save_dir_name = '{}_{}'.format(args.beam_size, args.save_dir_name)
    model_path, save_dir = get_model_path_and_save_path(args, save_dir_name)

    # Load model
    encoder, decoder = get_models(model_path)

    # Create rev word map
    word_map, rev_word_map = get_word_map()

    dataloader = load('flicker', args.run_local, 1, 1)

    for ind, image_data in enumerate(dataloader):
        run(encoder, decoder, word_map, rev_word_map, save_dir, image_data)
