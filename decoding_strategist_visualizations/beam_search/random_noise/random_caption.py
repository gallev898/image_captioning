import sys


sys.path.append('/home/mlspeech/gshalev/anaconda3/envs/python3_env/lib')
sys.path.append('/home/mlspeech/gshalev/gal/image_cap2')

from PIL import Image
from decoding_strategist_visualizations.decoding_strategist_utils import *
from decoding_strategist_visualizations.beam_search.beam_search_pack_utils import *

import torch
import en_core_web_sm

import numpy as np
import torchvision.transforms as transforms


args = get_args()


def visualize_att(image, seq, alphas, rev_word_map, top_seq_total_scors, save_dir, image_name, smooth=True):
    image = Image.fromarray(image)
    image = image.resize([14 * 24, 14 * 24], Image.LANCZOS)

    words = [rev_word_map[ind] for ind in seq]

    nlp = en_core_web_sm.load()
    doc = nlp(' '.join(words[1:-1]))
    pos = [x.pos_ for x in doc]
    pos.insert(0, '-')
    pos.insert(len(pos), '-')

    top_seq_total_scors_exp = np.exp(top_seq_total_scors)

    return visualization(image, alphas, words, pos, top_seq_total_scors, top_seq_total_scors_exp, smooth, save_dir,
                         image_name)


def run(encoder, decoder, word_map, rev_word_map, save_dir, ind, beam_size):
    # OOD
    img1 = np.random.uniform(low=0., high=255., size=(256, 256, 3))
    img1 = np.uint8(img1)
    img = img1

    img = img.transpose(2, 0, 1)
    img = img / 255.
    img = torch.FloatTensor(img).to(device)

    transform = transforms.Compose([data_normalization])
    image = transform(img)  # (3, 256, 256)

    # Encode
    image = image.unsqueeze(0)  # (1, 3, 256, 256)

    (seq, alphas, top_seq_total_scors, seq_sum, logits_list), img = beam_search_decode(encoder, image, beam_size,
                                                                                       word_map, decoder), img1

    alphas = torch.FloatTensor(alphas)

    visualize_att(img, seq, alphas, rev_word_map, top_seq_total_scors, save_dir, 'random_{}'.format(ind), args.smooth)

    f = open(os.path.join(save_dir, 'seq_sum.txt'), 'a+')
    f.write('seq_sum: {}    for some random image\n'.format(seq_sum))
    print('seq_sum: {}'.format(seq_sum))


if __name__ == '__main__':
    save_dir_name = '{}_{}'.format(args.beam_size, args.save_dir_name)
    model_path, save_dir = get_model_path_and_save_path(args, save_dir_name)

    # Load model
    encoder, decoder = get_models(model_path)

    # Create rev word map
    word_map, rev_word_map = get_word_map()

    for ind in range(args.limit_ex):
        run(encoder, decoder, word_map, rev_word_map, save_dir, ind, args.beam_size)
