import sys


sys.path.append('/home/mlspeech/gshalev/anaconda3/envs/python3_env/lib')
sys.path.append('/home/mlspeech/gshalev/gal/image_cap2')
# sys.path.append('/home/mlspeech/gshalev/gal/image_captioning')


import json
import os

import torch
import torchvision.transforms as transforms


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
desktop_path = os.path.join(os.path.join(os.path.expanduser('~')), 'Desktop')
data_name = 'coco_5_cap_per_img_5_min_word_freq'
filename = 'BEST_checkpoint_' + data_name + '.pth.tar'
word_map_file = '../../../output_folder/WORDMAP_' + data_name + '.json'
data_normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def get_multiple_models_path_and_save_path(args, save_dir_name):
    model_paths = []
    save_dirs = []
    for i in range(3):
        model = args.model1 if i == 0 else args.model2 if i == 1 else args.model3

        if args.run_local:
            desktop_path = os.path.join(os.path.join(os.path.expanduser('~')), 'Desktop')
            desktop_path = os.path.join(desktop_path, 'trained_models')
            model_path = os.path.join(desktop_path, os.path.join(model, filename))
            save_dir = "GIFs"
        else:
            model_path = "/yoav_stg/gshalev/image_captioning/{}/{}".format(model, filename)
            save_dir = "/yoav_stg/gshalev/image_captioning/{}/GIFs".format(model)

        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        save_dir = os.path.join(save_dir, model)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        save_dir = os.path.join(save_dir, save_dir_name)

        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        model_paths.append(model_path)
        save_dirs.append(save_dir)

    return model_paths, save_dirs


def get_model_path_and_save_path(args, save_dir_name):
    if args.run_local:
        desktop_path = os.path.join(os.path.join(os.path.expanduser('~')), 'Desktop')
        desktop_path = os.path.join(desktop_path, 'trained_models')
        model_path = os.path.join(desktop_path, os.path.join(args.model, filename))
        save_dir = "GIFs"
    else:
        model_path = "/yoav_stg/gshalev/image_captioning/{}/{}".format(args.model, filename)
        save_dir = "/yoav_stg/gshalev/image_captioning/{}/GIFs".format(args.model)

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    save_dir = os.path.join(save_dir, args.model)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    save_dir = os.path.join(save_dir, save_dir_name)

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    return model_path, save_dir


def get_models(model_path, device):
    checkpoint = torch.load(model_path, map_location=torch.device(device))
    decoder = checkpoint['decoder']
    decoder = decoder.to(device)
    decoder.eval()
    encoder = checkpoint['encoder']
    encoder = encoder.to(device)
    encoder.eval()

    return encoder, decoder


def get_word_map(run_local=True, map_file=None):
    if run_local:
        file = word_map_file
        if not None == map_file:
            file = map_file
    else:
        p = '/yoav_stg/gshalev/image_captioning/output_folder'
        file = os.path.join(p, 'WORDMAP_' + data_name + '.json')

    print('Loading word map from: {}'.format(file))
    with open(file, 'r') as j:
        word_map = json.load(j)
    rev_word_map = {v: k for k, v in word_map.items()}

    return word_map, rev_word_map
# utils.py
