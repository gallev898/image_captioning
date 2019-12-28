import sys


sys.path.append('/home/mlspeech/gshalev/anaconda3/envs/python3_env/lib')
sys.path.append('/home/mlspeech/gshalev/gal/image_cap2')

from dataset_loader.datasets import CaptionDataset
from utils import *
from decoding_strategist_visualizations.decoding_strategist_utils import *
from decoding_strategist_visualizations.beam_search.beam_search_pack_utils import *

import torch
import en_core_web_sm

import numpy as np
import torchvision.transforms as transforms


args = get_args()

#############################################################
# from language_model.data import Corpus
# from language_model.get_prop_for_IC_sentence import get_sentence_prop

#
# data = '../../../language_model/data_dir'
# checkpoint = '../../../language_model/model.pt'
# cuda = False
# seed = 1111
# torch.manual_seed(seed)
# if torch.cuda.is_available():
#     if not cuda:
#         print("WARNING: You have a CUDA device, so you should probably run with --cuda")
#
# device = torch.device("cuda" if cuda else "cpu")
# with open(checkpoint, 'rb') as f:
#     lm_model = torch.load(f, map_location=torch.device(device))
# lm_model.eval()
# corpus = Corpus(data)
#

#############################################################

def visualize_att(image_path, seq, alphas, rev_word_map, top_seq_total_scors, save_dir, image_name, smooth=True):
    image = image_path.squeeze(0)
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
    #############################################################
    # prop_dif = open(os.path.join(save_dir, 'prop_dif.txt'), 'a+')
    # prop_dif.write('\nimage name: {}\n'.format(image_name))
    # lm_prop_data = get_sentence_prop(words, pos, lm_model, corpus, device)
    # lm_prop = [x[2] for x in lm_prop_data]
    # w = [x[0] for x in lm_prop_data]
    # for idx in range(len(pos)):
    #     if pos[idx] == 'NOUN':
    #         prop_dif.write('word: {}    image_caption prop: {}   LM prop: {}        {}      diff: {}    div:{}\n'.format(
    #             w[idx],
    #             top_seq_total_scors_exp[idx],
    #             lm_prop[idx],
    #             'image' if top_seq_total_scors_exp[idx] > lm_prop[idx] else 'LM',
    #             top_seq_total_scors_exp[idx] - lm_prop[idx],
    #             top_seq_total_scors_exp[idx] / lm_prop[idx]))
    #
    #         print('word: {}     image_caption prop: {}   LM prop: {}'.format(w[idx], top_seq_total_scors_exp[idx],
    #                                                                          lm_prop[idx]))
    #############################################################
    return visualization(image, alphas, words, pos, top_seq_total_scors, top_seq_total_scors_exp, smooth, save_dir,
                         image_name)


def run(encoder, decoder, word_map, rev_word_map, save_dir, image_path, image_name):
    seq, alphas, top_seq_total_scors, seq_sum, logits_list = beam_search_decode(encoder, image, args.beam_size,
                                                                                word_map, decoder)

    alphas = torch.FloatTensor(alphas)

    visualize_att(image_path[0], seq, alphas, rev_word_map, top_seq_total_scors, save_dir, image_name, args.smooth)

    f = open(os.path.join(save_dir, 'seq_sum.txt'), 'a+')
    f.write('seq_sum: {}    for image with caption: {}\n'.format(seq_sum, image_name))
    print('seq_sum: {}'.format(seq_sum))


if __name__ == '__main__':
    save_dir_name = '{}_{}'.format(args.beam_size, args.save_dir_name)
    model_path, save_dir = get_model_path_and_save_path(args, save_dir_name)

    # Load model
    encoder, decoder = get_models(model_path)

    # Create rev word map
    word_map, rev_word_map = get_word_map()

    test_loader = torch.utils.data.DataLoader(
        CaptionDataset('../../../output_folder', data_name, 'TEST', transform=transforms.Compose([data_normalization])),
        batch_size=1, shuffle=True, num_workers=1, pin_memory=True)

    for i, (image, caps, caplens, allcaps) in enumerate(test_loader):
        if i > 20:
            break
        caps_ = [rev_word_map[x.item()] for x in caps[0]]
        ind = caps_.index('<end>')
        name = ' '.join(caps_[1:ind])
        run(encoder, decoder, word_map, rev_word_map, save_dir, image, name)
