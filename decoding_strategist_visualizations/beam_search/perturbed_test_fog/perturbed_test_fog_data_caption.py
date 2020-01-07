import sys

from dataset_loader.Pertubation import ImgAugTransformFog, ImgAugTransformSaltAndPepper, ImgAugTransformJpegCompression
from dataset_loader.datasets import CaptionDataset
import PIL

sys.path.append('/home/mlspeech/gshalev/anaconda3/envs/python3_env/lib')
sys.path.append('/home/mlspeech/gshalev/gal/image_cap2')

from utils import *
from decoding_strategist_visualizations.decoding_strategist_utils import *
from decoding_strategist_visualizations.beam_search.beam_search_pack_utils import *

import torch
import en_core_web_sm

import numpy as np
import torchvision.transforms as transforms


args = get_args()


def visualize_att(image_path, seq, alphas, rev_word_map, top_seq_total_scors, save_dir, image_name, file, smooth=True):
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
    file.write(' '.join(words)+'\n')

    nlp = en_core_web_sm.load()
    doc = nlp(' '.join(words[1:-1]))
    pos = [x.pos_ for x in doc]
    pos.insert(0, '-')
    pos.insert(len(pos), '-')

    plt.imshow(image)
    plt.show()
    # plt.savefig(os.path.join(save_dir, image_name))


def run(encoder, decoder, word_map, rev_word_map, save_dir, image_, image_name, file):
    seq, alphas, top_seq_total_scors, seq_sum, logits_list = beam_search_decode(encoder, image_, args.beam_size,
                                                                                word_map, decoder)

    alphas = torch.FloatTensor(alphas)

    visualize_att(image_[0], seq, alphas, rev_word_map, top_seq_total_scors, save_dir, image_name, file, args.smooth)

    print('seq_sum: {}'.format(seq_sum))


if __name__ == '__main__':
    save_dir_name = '{}_{}'.format(args.beam_size, args.save_dir_name)
    model_path, save_dir = get_model_path_and_save_path(args, save_dir_name)

    # Load model
    encoder, decoder = get_models(model_path)

    # Create rev word map
    word_map, rev_word_map = get_word_map()

    test_loader_pertubeded = torch.utils.data.DataLoader(
        CaptionDataset('../../../output_folder', data_name, 'TEST', transform=transforms.Compose([
            transforms.ToPILImage(),
            ImgAugTransformJpegCompression(),
            # ImgAugTransformSaltAndPepper(),
            lambda x: PIL.Image.fromarray(x),
            transforms.ToTensor(),
            data_normalization
            ])),
        batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        CaptionDataset('../../../output_folder', data_name, 'TEST', transform=transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            data_normalization
            ])),
        batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

    first_round = True
    for i, (t, tp) in enumerate(zip(test_loader, test_loader_pertubeded)):
        t_image, t_caps, t_caplens, t_allcaps = t
        tp_image, tp_caps, tp_caplens, tp_allcaps = tp

        if (i+1)%5 != 0:
            continue

        caps_0 = [rev_word_map[x.item()] for x in t_allcaps[0][0]]
        caps_1 = [rev_word_map[x.item()] for x in t_allcaps[0][1]]
        caps_2 = [rev_word_map[x.item()] for x in t_allcaps[0][2]]
        caps_3 = [rev_word_map[x.item()] for x in t_allcaps[0][3]]
        caps_4 = [rev_word_map[x.item()] for x in t_allcaps[0][4]]
        ind_0 = caps_0.index('<end>')
        ind_1 = caps_1.index('<end>')
        ind_2 = caps_2.index('<end>')
        ind_3 = caps_3.index('<end>')
        ind_4 = caps_4.index('<end>')
        name_0 = ' '.join(caps_0[1:ind_0])
        name_1 = ' '.join(caps_1[1:ind_1])
        name_2 = ' '.join(caps_2[1:ind_2])
        name_3 = ' '.join(caps_3[1:ind_3])
        name_4 = ' '.join(caps_4[1:ind_4])

        f = open(os.path.join(save_dir,'caps_{}'.format(i)), 'w')
        f.write('---------------------CAPS-----------------\n')
        f.write(name_0+'\n')
        f.write(name_1+'\n')
        f.write(name_2+'\n')
        f.write(name_3+'\n')
        f.write(name_4+'\n')
        f.write('---------------------T cap-----------------\n')
        run(encoder, decoder, word_map, rev_word_map, save_dir, t_image, 'img_{}'.format(i),f)
        f.write('---------------------TP cap-----------------\n')
        run(encoder, decoder, word_map, rev_word_map, save_dir, tp_image, 'img_per{}'.format(i),f)
        g =0
