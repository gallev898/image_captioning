import sys

from dataset_loader.Pertubation import ImgAugTransformSaltAndPepper, ImgAugTransformJpegCompression, \
    ImgAugTransformCartoon, ImgAugTransformSnow
from dataset_loader.custom_dataloader import Custom_Image_Dataset
from dataset_loader.dataloader import load

sys.path.append('/home/mlspeech/gshalev/anaconda3/envs/python3_env/lib')
sys.path.append('/home/mlspeech/gshalev/gal/image_cap2')

import PIL
from dataset_loader.datasets import CaptionDataset
from utils import *
from decoding_strategist_visualizations.decoding_strategist_utils import *
from decoding_strategist_visualizations.beam_search.beam_search_pack_utils import *
from standart_training.V_models_with_attention import Encoder, DecoderWithAttention
import torch
import en_core_web_sm

import numpy as np
import torchvision.transforms as transforms

args = get_args()
device = torch.device("cuda:{}".format(args.cuda) if torch.cuda.is_available() else "cpu")


def get_models(model_path, device):
    checkpoint = torch.load(model_path, map_location=torch.device(device))

    decoder = DecoderWithAttention(attention_dim=512,
                                   embed_dim=512,
                                   decoder_dim=512,
                                   vocab_size=9490,
                                   device=device)
    decoder.load_state_dict(checkpoint['decoder'])
    decoder = decoder.to(device)
    decoder.eval()

    encoder = Encoder()
    encoder.load_state_dict(checkpoint['encoder'])
    encoder = encoder.to(device)
    encoder.eval()

    return encoder, decoder


def visualize_att(image_path, seq, rev_word_map, save_dir,  img_idx):
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
    plt.savefig(os.path.join(save_dir, img_idx))
    f = open(os.path.join(save_dir, '{}.txt'.format(img_idx)), 'w')
    words = [rev_word_map[ind] for ind in seq]
    f.write(' '.join(words))
    print(words)
    # nlp = en_core_web_sm.load()
    # doc = nlp(' '.join(words[1:-1]))
    # pos = [x.pos_ for x in doc]
    # pos.insert(0, '-')
    # pos.insert(len(pos), '-')
    #
    # top_seq_total_scors_exp = np.exp(top_seq_total_scors)
    # return visualization(image, alphas, words, pos, top_seq_total_scors, top_seq_total_scors_exp, smooth, save_dir,
    #                      image_name)


def run(encoder, decoder, word_map, rev_word_map, save_dir, image_path, image_name, args, img_idx):
    # encoder, image, beam_size, word_map, decoder, device, args
    seq, alphas, top_seq_total_scors, seq_sum, logits_list = beam_search_decode(encoder, image_path, args.beam_size,
                                                                                word_map, decoder, device, args)

    # if args.show_all_beam:
    #     image = image_path.squeeze(0)
    #     image = image.numpy()
    #
    #     image = image.transpose((1, 2, 0))
    #
    #     # Undo preprocessing
    #     mean = np.array([0.485, 0.456, 0.406])
    #     std = np.array([0.229, 0.224, 0.225])
    #     image = std * image + mean
    #
    #     # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    #     image = np.clip(image, 0, 1)
    #     plt.imshow(image)
    #     plt.show()
    #
    #     for s, ss in zip(seq, top_seq_total_scors):
    #         words = [rev_word_map[ind] for ind in s]
    #         print(' '.join(words))
    #         print(ss)
    #         print(sum(ss))
    #     print('---------------')
    #     return

    alphas = torch.FloatTensor(alphas)

# image_path, seq, rev_word_map, save_dir,  img_idx
    visualize_att(image_path[0], seq, rev_word_map, save_dir,  img_idx)

    f = open(os.path.join(save_dir, 'seq_sum.txt'), 'a+')
    f.write('seq_sum: {}    for image with caption: {}\n'.format(seq_sum, image_name))
    print('seq_sum: {}'.format(seq_sum))


if __name__ == '__main__':
    save_dir = 'visualization_pngs'
    data_type = 'custom'

    # Create rev word map
    word_map, rev_word_map = get_word_map()

    t = [data_normalization]

    if data_type=='test':
        t = [data_normalization]

    if data_type == 'salt':
        t = [
            transforms.ToPILImage(),
            ImgAugTransformSaltAndPepper(),
            lambda x: PIL.Image.fromarray(x),
            transforms.ToTensor(),
            data_normalization
        ]
    if data_type == 'jpeg':
        t = [
            transforms.ToPILImage(),
            ImgAugTransformJpegCompression(),
            lambda x: PIL.Image.fromarray(x),
            transforms.ToTensor(),
            data_normalization
        ]
    if data_type == 'cartoon':
        t = [
            transforms.ToPILImage(),
            ImgAugTransformCartoon(),
            lambda x: PIL.Image.fromarray(x),
            transforms.ToTensor(),
            data_normalization
        ]
    if data_type == 'snow':
        t = [
            transforms.ToPILImage(),
            ImgAugTransformSnow(),
            lambda x: PIL.Image.fromarray(x),
            transforms.ToTensor(),
            data_normalization
        ]


    test_loader = torch.utils.data.DataLoader(
        CaptionDataset('../../output_folder', data_name, 'TEST', transform=transforms.Compose(t)),
        batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    if data_type == 'custom':
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor(),
            # data_normalization #notice: normalize happenes inside Custom_Image_Dataset
        ])
        path = '/Users/gallevshalev/Desktop/datasets/custom'
        data = Custom_Image_Dataset(path, transform, data_normalization)

        test_loader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=False, num_workers=1)


    model_path = '/Users/gallevshalev/Desktop/trained_models/run_8/NEW_BEST_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar'

    # Load model
    encoder, decoder = get_models(model_path, device)

    e = enumerate(test_loader)
    for i, image in e:
    # for i, (image, caps, caplens, allcaps) in e:
        if i > 100:
            break
        # [next(e, None) for _ in range(4)]
        # caps_ = [rev_word_map[x.item()] for x in caps[0]]
        # ind = caps_.index('<end>')
        # name = ' '.join(caps_[1:ind])
        run(encoder, decoder, word_map, rev_word_map, save_dir, image[0], '', args, data_type+' '+str(i))
        # run(encoder, decoder, word_map, rev_word_map, save_dir, image, name, args, data_type+' '+str(i))
