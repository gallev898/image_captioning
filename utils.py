import json
import os

import torch
import torchvision.transforms as transforms


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
desktop_path = os.path.join(os.path.join(os.path.expanduser('~')), 'Desktop')
data_name = 'coco_5_cap_per_img_5_min_word_freq'
filename = 'BEST_checkpoint_' + data_name + '.pth.tar'
word_map_file = '../../../output_folder/WORDMAP_' + data_name + '.json'
data_normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def get_model_path_and_save_path(args, save_dir_name):

    if args.run_local:
        desktop_path = os.path.join(os.path.join(os.path.expanduser('~')), 'Desktop')
        model_path = os.path.join(desktop_path, os.path.join(args.model, filename))
        save_dir = "GIFs"
    else:
        model_path = "/yoav_stg/gshalev/image_captioning/{}/{}".format(args.model, filename)
        save_dir = "/yoav_stg/gshalev/image_captioning/{}/GIFs".format(args.model)

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    save_dir = os.path.join(save_dir, save_dir_name)

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    return model_path, save_dir


def get_models(model_path):

    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    decoder = checkpoint['decoder']
    decoder = decoder.to(device)
    decoder.eval()
    encoder = checkpoint['encoder']
    encoder = encoder.to(device)
    encoder.eval()

    return encoder, decoder


def get_word_map(map_file=None):

    file = word_map_file
    if not None == map_file:
        file = map_file

    with open(file, 'r') as j:
        word_map = json.load(j)
    rev_word_map = {v: k for k, v in word_map.items()}

    return word_map, rev_word_map


def encode(encoder, image, k, word_map):

    # Encode - return the output of resnet as 2048 channels
    encoder_out = encoder(image)  # (1, enc_image_size, enc_image_size, encoder_dim)
    enc_image_size = encoder_out.size(1)
    encoder_dim = encoder_out.size(3)

    # Flatten encoding
    encoder_out = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
    num_pixels = encoder_out.size(1) # num of pixels in each

    # We'll treat the problem as having a batch size of k
    encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)  # (k, num_pixels, encoder_dim)

    # Tensor to store top k previous words at each step; now they're just <start>
    k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device)  # (k, 1)

    # Tensor to store top k sequences; now they're just <start>
    seqs = k_prev_words  # (k, 1)
    seqs_scores = torch.FloatTensor([[0.]] * k).to(device)

    # Tensor to store top k sequences' scores; now they're just 0
    top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

    # Tensor to store top k sequences' alphas; now they're just 1s
    seqs_alpha = torch.ones(k, 1, enc_image_size, enc_image_size).to(device) # NOTICE: for visualization

    return encoder_out, enc_image_size, k_prev_words, seqs, seqs_scores, top_k_scores, seqs_alpha
