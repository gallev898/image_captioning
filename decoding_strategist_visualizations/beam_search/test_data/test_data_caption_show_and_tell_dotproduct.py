import sys


sys.path.append('/home/mlspeech/gshalev/anaconda3/envs/python3_env/lib')
sys.path.append('/home/mlspeech/gshalev/gal/image_cap2')

from dataset_loader.datasets import CaptionDataset
from utils import *
from decoding_strategist_visualizations.decoding_strategist_utils import *
from decoding_strategist_visualizations.beam_search.beam_search_pack_utils_show_and_tell import *

import torch
import en_core_web_sm

import numpy as np
import torchvision.transforms as transforms


args = get_args()
device = torch.device("cuda:{}".format(args.cuda) if torch.cuda.is_available() else "cpu")


def visualize_att(image_path, seq, rev_word_map):
    image = image_path.squeeze(0)
    image = image.numpy()

    image = image.transpose((1, 2, 0))

    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)

    plt.imshow(image)
    plt.show()
    words = [rev_word_map[ind] for ind in seq]
    print(' '.join(words))


def run(encoder, decoder, word_map, rev_word_map, image_path, image_name, representations):
    seq, _, top_seq_total_scors, seq_sum, logits_list = beam_search_decode(encoder, image, args.beam_size,
                                                                           word_map, decoder, device, representations)

    # alphas = torch.FloatTensor(alphas)
    print(top_seq_total_scors)
    visualize_att(image_path[0], seq, rev_word_map)

    f = open(os.path.join('.', 'seq_sum.txt'), 'a+')
    f.write('seq_sum: {}    for image with caption: {}\n'.format(seq_sum, image_name))
    print('seq_sum: {}'.format(seq_sum))


def get_models(model_path, device):
    checkpoint = torch.load(model_path, map_location=torch.device(device))
    decoder = checkpoint['decoder']
    decoder = decoder.to(device)
    decoder.eval()
    encoder = checkpoint['encoder']
    encoder = encoder.to(device)
    representations = checkpoint['representations']
    representations = representations.to(device)
    encoder.eval()

    return encoder, decoder, representations


def get_model_path_and_save_path(modelA, modelB):
    if args.run_local:
        desktop_path = os.path.join(os.path.join(os.path.expanduser('~')), 'Desktop')
        desktop_path = os.path.join(desktop_path, 'trained_models')
        modelA_path = os.path.join(desktop_path, os.path.join(modelA, filename))
        modelB_path = os.path.join(desktop_path, os.path.join(modelB, filename))

    return modelA_path, modelB_path


if __name__ == '__main__':
    # section: word map
    word_map, rev_word_map = get_word_map()

    # section: data loader
    test_loader = torch.utils.data.DataLoader(
        CaptionDataset('../../../output_folder', data_name, 'TEST', transform=transforms.Compose([data_normalization])),
        batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

    modelA_path, modelB_path = get_model_path_and_save_path(args.modelA, args.modelB)

    # Load model
    encoderA, decoderA, representationsA = get_models(modelA_path, device)
    encoderB, decoderB, representationsB = get_models(modelB_path, device)

    assert representationsA.requires_grad == True #train_show_and_tell_dotproduct
    assert representationsB.requires_grad == False #train_show_and_tell_fixed_dotproduct

    for i, (image, caps, caplens, allcaps) in enumerate(test_loader):
        if i > 50:
            break

        print('---------------------- G T ---------------------------------------')
        caps_ = [rev_word_map[x.item()] for x in caps[0]]
        ind = caps_.index('<end>')
        name = ' '.join(caps_[1:ind])
        print(name)

        print('----------------------model A ---------------------------------------')
        run(encoderA, decoderA, word_map, rev_word_map, image, name, representationsA)
        print('----------------------model B ---------------------------------------')
        run(encoderB, decoderB, word_map, rev_word_map, image, name, representationsB)
