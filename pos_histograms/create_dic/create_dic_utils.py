
import sys


sys.path.append('/home/mlspeech/gshalev/gal/image_cap')
# sys.path.append('/home/mlspeech/gshalev/gal/image_captioning')
sys.path.append('/home/mlspeech/gshalev/anaconda3/envs/python3_env/lib')

import argparse

import torch
import os


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_name = 'coco_5_cap_per_img_5_min_word_freq'
filename = 'BEST_checkpoint_' + data_name + '.pth.tar'
pre_pros_data_dir = '../../output_folder'
noun_phrase_sum_of_log_prop = []
sentences_likelihood = []
pos_dic_obj = {}




def get_model_path_and_save_dir(args, save_dir_name):
    if args.run_local:
        desktop_path = os.path.join(os.path.join(os.path.expanduser('~')), 'Desktop')
        desktop_path = os.path.join(desktop_path, 'trained_models')
        model_path = os.path.join(desktop_path, os.path.join(args.model, filename))
        save_dir = save_dir_name
    else:
        model_path = "/yoav_stg/gshalev/image_captioning/{}/{}".format(args.model, filename)
        save_dir = "/yoav_stg/gshalev/image_captioning/{}/{}".format(args.model, save_dir_name)

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
        print('created dir: {}'.format(save_dir))

    save_dir = os.path.join(save_dir, args.model)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
        print('created dir: {}'.format(save_dir))

    print('model path: {}'.format(model_path))
    print('save dir: {}'.format(save_dir))
    return model_path, save_dir


def get_args():
    parser = argparse.ArgumentParser(description='Generate Caption')
    parser.add_argument('--model', type=str)
    parser.add_argument('--run_local', default=False, action='store_true')
    parser.add_argument('--debug', default=False, action='store_true')
    parser.add_argument('--data', default='random', type=str)
    parser.add_argument('--beam_size', default=10, type=int)
    parser.add_argument('--top_k', default=0, type=int)
    parser.add_argument('--top_p', default=0.0, type=float)
    args = parser.parse_args()
    return args

# create_dic_utils.py