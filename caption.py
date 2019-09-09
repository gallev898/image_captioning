import sys


sys.path.append('/home/mlspeech/gshalev/gal/image_captioning')
sys.path.append('/home/mlspeech/gshalev/anaconda3/envs/python3_env/lib')

import os
import tqdm
import json
import torch
import random
import argparse
import skimage.transform

import numpy as np
import matplotlib.cm as cm
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.transforms as transforms

from PIL import Image
from scipy.misc import imread, imresize


# args
parser = argparse.ArgumentParser(description='Show, Attend, and Tell - Tutorial - Generate Caption')
parser.add_argument('--model', type=str)
parser.add_argument('--run_local', default=False, action='store_true')
parser.add_argument('--ood', default=False, action='store_true')
parser.add_argument('--all_data', default=False, action='store_true')
parser.add_argument('--limit_ex', type=int, default=1)
parser.add_argument('--ood_limit_ex', type=int, default=1)
parser.add_argument('--beam_size', default=5, type=int)
parser.add_argument('--dont_smooth', dest='smooth', action='store_false', help='do not smooth alpha overlay')
args = parser.parse_args()

# global
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_name = 'coco_5_cap_per_img_5_min_word_freq'
filename = 'BEST_checkpoint_' + data_name + '.pth.tar'


def caption_image_beam_search(encoder, decoder, image_path, word_map, beam_size=3):
    """
    Reads an image and captions it with beam search.

    :param encoder: encoder model
    :param decoder: decoder model
    :param image_path: path to image
    :param word_map: word map
    :param beam_size: number of sequences to consider at each decode-step
    :return: caption, weights for visualization
    """

    k = beam_size
    vocab_size = len(word_map)
    img1 = None

    # OOD
    if image_path is None:
        img1 = np.random.uniform(low=0., high=255., size=(256, 256, 3))
        img1 = np.uint8(img1)
        img = img1
    else:
        # Read image and process
        img = imread(image_path)
        if len(img.shape) == 2:
            img = img[:, :, np.newaxis]
            img = np.concatenate([img, img, img], axis=2)
        img = imresize(img, (256, 256))

    img = img.transpose(2, 0, 1)
    img = img / 255.
    img = torch.FloatTensor(img).to(device)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([normalize])
    image = transform(img)  # (3, 256, 256)


    # Encode
    image = image.unsqueeze(0)  # (1, 3, 256, 256)

    encoder_out = encoder(image)  # (1, enc_image_size, enc_image_size, encoder_dim)
    enc_image_size = encoder_out.size(1)
    encoder_dim = encoder_out.size(3)

    # Flatten encoding
    encoder_out = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
    num_pixels = encoder_out.size(1)

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
    seqs_alpha = torch.ones(k, 1, enc_image_size, enc_image_size).to(
        device)  # (k, 1, enc_image_size, enc_image_size) TODO: for visualization

    # Lists to store completed sequences, their alphas and scores
    complete_seqs = list()
    complete_seqs_alpha = list()
    complete_seqs_scores = list()
    complete_seqs_scores_for_all_steps = list()

    # Start decoding
    step = 1
    h, c = decoder.init_hidden_state(encoder_out)

    # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
    while True:
        embeddings = decoder.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)

        awe, alpha = decoder.attention(encoder_out,
                                       h)  # (s, encoder_dim), (s, num_pixels) האלפות אחרי סופטמקס  והוקטור האטנשיין

        alpha = alpha.view(-1, enc_image_size, enc_image_size)  # (s, enc_image_size, enc_image_size)

        gate = decoder.sigmoid(decoder.f_beta(h))  # gating scalar, (s, encoder_dim)# TODO: tihs is bla bla
        awe = gate * awe  # TODO: this is bla bla - motivated by dropout "like" - type of regularization, gate of [0,1] because of sigmoid

        concatination_of_input_and_att = torch.cat([embeddings, awe], dim=1)
        h, c = decoder.decode_step(concatination_of_input_and_att, (
            h, c))  # (s, decoder_dim)

        scores = decoder.fc(h)  # (s, vocab_size)
        scores = F.log_softmax(scores, dim=1)  # TODO: in our next work this is where we intervert
        scores_copy = scores.clone()

        # Add
        scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size) TODO: can look like aggregated "log"

        # For the first step, all k points will have the same scores (since same k previous words, h, c)
        if step == 1:
            top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
        else:
            # Unroll and find top scores, and their unrolled indices
            top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

        # Convert unrolled indices to actual indices of scores
        prev_word_inds = top_k_words / vocab_size  # (s)
        next_word_inds = top_k_words % vocab_size  # (s)

        # Add new words to sequences, alphas
        seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)
        seqs_scores = torch.cat([seqs_scores[prev_word_inds], scores_copy.view(-1)[top_k_words].unsqueeze(1)], dim=1)
        seqs_alpha = torch.cat([seqs_alpha[prev_word_inds], alpha[prev_word_inds].unsqueeze(1)],
                               dim=1)  # (s, step+1, enc_image_size, enc_image_size)

        # Which sequences are incomplete (didn't reach <end>)?
        incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                           next_word != word_map['<end>']]
        complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

        # Set aside complete sequences
        if len(complete_inds) > 0:
            complete_seqs.extend(seqs[complete_inds].tolist())
            complete_seqs_alpha.extend(seqs_alpha[complete_inds].tolist())
            complete_seqs_scores.extend(top_k_scores[complete_inds])
            complete_seqs_scores_for_all_steps.extend(seqs_scores[complete_inds].tolist())
        k -= len(complete_inds)  # reduce beam length accordingly

        # Proceed with incomplete sequences
        if k == 0:
            break
        seqs = seqs[incomplete_inds]
        seqs_alpha = seqs_alpha[incomplete_inds]
        h = h[prev_word_inds[incomplete_inds]]
        c = c[prev_word_inds[incomplete_inds]]
        encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
        top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
        k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

        # Break if things have been going on too long
        if step > 50:
            break
        step += 1

    i = complete_seqs_scores.index(max(complete_seqs_scores))
    seq_sum = round(max(complete_seqs_scores).item(), 4)
    try:
        exception_data_name = None
        assert round(np.array(complete_seqs_scores_for_all_steps[i]).sum(), 4) == seq_sum
    except AssertionError:
        print('------------EXCEPTION ACCRUED---------------')
        print('{} != {}'.format(round(np.array(complete_seqs_scores_for_all_steps[i]).sum(), 4), seq_sum))

        if not args.ood:
            exception_data_name = img_path.split('/')[-1].replace('.jpg', '')
            print('for : {}'.format(exception_data_name))

    top_seq_total_scors = complete_seqs_scores_for_all_steps[i]
    seq = complete_seqs[i]
    alphas = complete_seqs_alpha[i]

    return seq, alphas, top_seq_total_scors, seq_sum, exception_data_name, img1


def visualize_att(image_path, seq, alphas, rev_word_map, top_seq_total_scors, save_dir, ood_img=None, smooth=True):
    """
    Visualizes caption with weights at every word.

    Adapted from paper authors' repo: https://github.com/kelvinxu/arctic-captions/blob/master/alpha_visualization.ipynb

    :param image_path: path to image that has been captioned
    :param seq: caption
    :param alphas: weights
    :param rev_word_map: reverse word mapping, i.e. ix2word
    :param smooth: smooth weights?
    """
    if ood_img is not None and image_path is None:
        image = Image.fromarray(ood_img)
        image = image.resize([14 * 24, 14 * 24], Image.LANCZOS)

    else:
        image = Image.open(image_path)
        image = image.resize([14 * 24, 14 * 24], Image.LANCZOS)


    words = [rev_word_map[ind] for ind in seq]

    for t in range(len(words)):
        if t > 50:
            break
        plt.subplot(np.ceil(len(words) / 5.), 5, t + 1)

        plt.text(0, 1, '{} -\n {:.4f}'.format(words[t], top_seq_total_scors[t]), color='black', backgroundcolor='white',
                 fontsize=12)
        plt.imshow(image)
        current_alpha = alphas[t, :]
        if smooth:
            alpha = skimage.transform.pyramid_expand(current_alpha.numpy(), upscale=24, sigma=8)
        else:
            alpha = skimage.transform.resize(current_alpha.numpy(), [14 * 24, 14 * 24])
        if t == 0:
            plt.imshow(alpha, alpha=0)
        else:
            plt.imshow(alpha, alpha=0.8)
        plt.set_cmap(cm.Greys_r)
        plt.axis('off')

    plt.savefig('temp')
    plt.clf()
    return words, image
    # L = torch.load('{}/data'.format(save_dir))
    # TO SHOW IMAGE THATS INSIDE THE DIC
    # imgplot = plt.imshow(L['fig'])
    # plt.show()


def run(encoder, decoder, word_map, rev_word_map, save_dir, image_path=None, epoch=None):
    # Encode, decode with attention and beam search
    seq, alphas, top_seq_total_scors, seq_sum, edn, ood_img = caption_image_beam_search(encoder,
                                                                                        decoder,
                                                                                        image_path,
                                                                                        word_map,
                                                                                        args.beam_size)

    alphas = torch.FloatTensor(alphas)
    words, image = visualize_att(image_path, seq, alphas, rev_word_map, top_seq_total_scors, save_dir, ood_img,
                                 args.smooth)
    # Visualize caption and attention of best sequence
    print('seq_sum: {}'.format(seq_sum))
    if not args.run_local:
        img = mpimg.imread('temp.png')
        state = {'fig': img,
                 'cap': words,
                 'img': image,
                 'seq_scores': top_seq_total_scors,
                 'seq_sum': seq_sum,
                 'alphas': alphas}

        if args.ood:
            saved_to = '{}/{}'.format(save_dir, 'data_ood_num_{}'.format(epoch))
        else:
            saved_to = '{}/{}'.format(save_dir, 'data_{}'.format(image_path.split('/')[-1].replace('.jpg', '')))
        torch.save(state, saved_to)

    return edn


if __name__ == '__main__':
    # Creat save dir
    if args.run_local:
        model_path = filename
        if args.ood:
            save_dir = "OOD_GIFs"
        else:
            save_dir = "GIFs"
    else:
        model_path = "/yoav_stg/gshalev/image_captioning/{}/{}".format(args.model, filename)
        if args.ood:
            save_dir = "/yoav_stg/gshalev/image_captioning/{}/OOD_GIFs".format(args.model)
        else:
            save_dir = "/yoav_stg/gshalev/image_captioning/{}/GIFs".format(args.model)

    # create dir
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # Load model
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    decoder = checkpoint['decoder']
    decoder = decoder.to(device)
    decoder.eval()
    encoder = checkpoint['encoder']
    encoder = encoder.to(device)
    encoder.eval()

    # Load word map (word2ix)
    data_folder = 'output_folder'  # folder with data files saved by create_input_files.py
    word_map_file = os.path.join(data_folder, 'WORDMAP_' + data_name + '.json')

    # Create rev word map
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)
    rev_word_map = {v: k for k, v in word_map.items()}  # ix2word

    # Create input files
    if args.ood:
        for e in range(args.ood_limit_ex):
            run(encoder, decoder, word_map, rev_word_map, save_dir, epoch=e)
    else:
        if args.run_local:
            desktop_path = os.path.join(os.path.join(os.path.expanduser('~')), 'Desktop')
            dir = os.path.join(desktop_path, 'datasets/mscoco/val2014')
            filename = random.choice(os.listdir(dir))
            img_path = os.path.join(dir, filename)
        else:
            dir = '/yoav_stg/gshalev/semantic_labeling/mscoco/val2014'
            filename = random.choice(os.listdir(dir))
            img_path = os.path.join(dir, filename)

        exception_data_list = list()
        if args.all_data:
            for ind, filename in tqdm(enumerate(os.listdir(dir))):
                img_path = os.path.join(dir, filename)
                edn = run(encoder, decoder, word_map, rev_word_map, save_dir, image_path=img_path)
                if edn is not None:
                    exception_data_list.append(edn)
                if ind + 1 % 100 == 0:
                    print('have data for {} files'.format(ind))
            if len(exception_data_list) > 0:
                torch.save(exception_data_list, save_dir.replace('/GIFs', '/exception_data_list'))
        else:
            limit = 0
            for ind in range(args.limit_ex):
                filename = random.choice(os.listdir(dir))
                img_path = os.path.join(dir, filename)
                edn = run(encoder, decoder, word_map, rev_word_map, save_dir, image_path=img_path)
                if edn is not None:
                    exception_data_list.append(edn)
                if ind + 1 % 100 == 0:
                    print('have data for {} files'.format(ind))
