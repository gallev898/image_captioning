import sys


sys.path.append('/home/mlspeech/gshalev/anaconda3/envs/python3_env/lib')
sys.path.append('/home/mlspeech/gshalev/gal/image_cap2')

from PIL import Image
from scipy.misc import imread, imresize
from decoding_strategist.decoding_strategist_utils import *
from decoding_strategist.beam_search.beam_search_pack_utils import *

import os
import torch
import en_core_web_sm

import numpy as np
import torchvision.transforms as transforms


args = get_args()



def visualize_att(image_path, seq, alphas, rev_word_map, top_seq_total_scors, save_dir, image_name, smooth=True):
    image = image_path
    words = [rev_word_map[ind] for ind in seq]

    nlp = en_core_web_sm.load()
    doc = nlp(' '.join(words[1:-1]))
    pos = [x.pos_ for x in doc]
    pos.insert(0, '-')
    pos.insert(len(pos), '-')

    top_seq_total_scors_exp = np.exp(top_seq_total_scors)

    return visualization(image, alphas, words, pos, top_seq_total_scors, top_seq_total_scors_exp, smooth, save_dir,
                         image_name)


def run(encoder, decoder, word_map, rev_word_map, save_dir, image_path, image_title):

    img = imread(image_path)
    img = imresize(img, (256, 256))

    img = img.transpose(2, 0, 1)
    img = img / 255.
    img = torch.FloatTensor(img).to(device)

    transform = transforms.Compose([data_normalization])
    image = transform(img)  # (3, 256, 256)

    seq, alphas, top_seq_total_scors, seq_sum, logits_list = beam_search_decode(encoder, image.unsqueeze(0), args.beam_size, word_map, decoder)

    alphas = torch.FloatTensor(alphas)

    image = Image.open(image_path)
    image = image.resize([14 * 24, 14 * 24], Image.LANCZOS)

    visualize_att(image, seq, alphas, rev_word_map, top_seq_total_scors, save_dir, image_title, args.smooth)

    f = open(os.path.join(save_dir, 'seq_sum.txt'), 'a+')
    f.write('seq_sum: {}    for image id: {}\n'.format(seq_sum, image_title))
    print('seq_sum: {}'.format(seq_sum))


if __name__ == '__main__':

    save_dir_name = '{}_{}'.format(args.beam_size, args.save_dir_name)

    model_path, save_dir = get_model_path_and_save_path(args, save_dir_name)

    encoder, decoder = get_models(model_path)

    word_map, rev_word_map = get_word_map()

    if args.run_local:
        desktop_path = os.path.join(os.path.join(os.path.expanduser('~')), 'Desktop')
        dir = os.path.join(desktop_path, 'datasets/mscoco/val2014')
    else:
        dir = '/yoav_stg/gshalev/semantic_labeling/mscoco/val2014'

    for ind, filename in enumerate(os.listdir(dir)):
        img_path = os.path.join(dir, filename)
        run(encoder, decoder, word_map, rev_word_map, save_dir, img_path, filename)
