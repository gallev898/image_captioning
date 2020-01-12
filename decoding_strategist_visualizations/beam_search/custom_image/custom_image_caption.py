import sys



sys.path.append('/home/mlspeech/gshalev/anaconda3/envs/python3_env/lib')
sys.path.append('/home/mlspeech/gshalev/gal/image_cap2')
from language_model.word_language_model.data_holder import Corpus

from decoding_strategist_visualizations.decoding_strategist_utils import visualization
from decoding_strategist_visualizations.beam_search.beam_search_pack_utils import *

import torch
import en_core_web_sm

import numpy as np


args = get_args()


def load_lm_model():
    torch.manual_seed(1111)

    p = '../../../language_model/word_language_model/model.pt'
    model = torch.load(p, map_location=torch.device(device))
    model.eval()

    return model

def visualize_att(image, seq, alphas, rev_word_map, top_seq_total_scors, save_dir, image_name, smooth=True):
    image = image.squeeze(0)
    image = image.numpy()

    image = image.transpose((1, 2, 0))

    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)

    words = [rev_word_map[ind] for ind in seq]  # notice

    nlp = en_core_web_sm.load()
    doc = nlp(' '.join(words[1:-1]))
    pos = [x.pos_ for x in doc]  # notice
    pos.insert(0, '-')
    pos.insert(len(pos), '-')

    top_seq_total_scors_exp = np.exp(top_seq_total_scors)  # notice


    return visualization(image, alphas, words, pos, top_seq_total_scors, top_seq_total_scors_exp, smooth, save_dir,
                         image_name)


def run(encoder, decoder, word_map, rev_word_map, save_dir, image, image_title):
    seq, alphas, top_seq_total_scors, seq_sum, logits_list = beam_search_decode(encoder, image, args.beam_size,
                                                                                word_map, decoder)

    alphas = torch.FloatTensor(alphas)

    visualize_att(image, seq, alphas, rev_word_map, top_seq_total_scors, save_dir, image_title, args.smooth)

    f = open(os.path.join(save_dir, 'seq_sum.txt'), 'a+')
    f.write('seq_sum: {}    for: {}\n'.format(seq_sum, image_title))


if __name__ == '__main__':
    save_dir_name = '{}_{}'.format(args.beam_size, args.save_dir_name)
    model_path, save_dir = get_model_path_and_save_path(args, save_dir_name)
    encoder, decoder = get_models(model_path)
    word_map, rev_word_map = get_word_map()

    transform = transforms.Compose([
        transforms.Resize((336, 336)),
        transforms.ToTensor(),
        data_normalization
        ])

    from PIL import Image

    global desktop_path, lm_model, corpus
    desktop_path = os.path.join(os.path.join(os.path.expanduser('~')), 'Desktop')
    lm_model = load_lm_model()
    corpus = Corpus('../../../language_model/word_language_model/data_dir')

    # image_name = 'Fork.png'
    for image_name in os.listdir(os.path.join(desktop_path, 'custom_images')):
        # image_name = 'Cart.jpeg'
        print(image_name)
        image_path = 'custom_images/{}'.format(image_name)
        img = Image.open(os.path.join(desktop_path, image_path))

        try:
            image = transform(img).unsqueeze(0)
        except:
            continue

        run(encoder, decoder, word_map, rev_word_map, save_dir, image, image_name)
