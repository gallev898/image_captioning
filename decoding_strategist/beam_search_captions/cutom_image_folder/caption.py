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
parser.add_argument('--save_dir_name', type=str, default='beam_size')
parser.add_argument('--run_local', default=False, action='store_true')
parser.add_argument('--ood', default=False, action='store_true')
parser.add_argument('--all_data', default=False, action='store_true')
parser.add_argument('--limit_ex', type=int, default=1)
parser.add_argument('--beam_size', default=1, type=int)
parser.add_argument('--dont_smooth', dest='smooth', action='store_false', help='do not smooth alpha overlay')
args = parser.parse_args()

# global
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_name = 'coco_5_cap_per_img_5_min_word_freq'
filename = 'BEST_checkpoint_' + data_name + '.pth.tar'


def caption_image(encoder, decoder, image_path, word_map, beam_size=3):
    image = image_path

    vocab_size = len(word_map)

    return beam_search_decode(encoder, image, beam_size, word_map, decoder, vocab_size)


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


def run(encoder, decoder, word_map, rev_word_map, save_dir, image_path=None):
    seq, alphas, top_seq_total_scors, seq_sum = caption_image(encoder,
                                                              decoder,
                                                              image_path,
                                                              word_map,
                                                              args.beam_size)

    alphas = torch.FloatTensor(alphas)

    visualize_att(image_path, seq, alphas, rev_word_map, top_seq_total_scors, save_dir, args.smooth)

    print('seq_sum: {}'.format(seq_sum))


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
    save_dir_name = '{}_{}'.format(args.beam_size,args.save_dir_name)
    model_path, save_dir = get_model_path_and_save_path(args, save_dir_name)

    # Load model
    encoder, decoder = get_models(model_path)

    # Create rev word map
    word_map, rev_word_map = get_word_map()

    dataloader = get_data_loader()

    for ind, image in enumerate(dataloader):
        run(encoder, decoder, word_map, rev_word_map, save_dir, image_path=image)
