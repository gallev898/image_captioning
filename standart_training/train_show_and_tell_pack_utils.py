import sys

sys.path.append('/home/mlspeech/gshalev/gal/image_cap2')
sys.path.append('/home/mlspeech/gshalev/anaconda3/envs/python3_env/lib')
# sys.path.append('/home/mlspeech/gshalev/gal/image_captioning')

import os
import h5py
import json
import torch
import argparse

import numpy as np

from tqdm import tqdm
from random import seed, choice, sample
from collections import Counter


data_folder = '/Users/gallevshalev/PycharmProjects/image_captioning/output_folder/'  # folder with data files saved by create_input_files.py
data_name = 'coco_5_cap_per_img_5_min_word_freq'  # base name shared by data files




def init_embedding(embeddings):
    """
    Fills embedding tensor with values from the uniform distribution.

    :param embeddings: embedding tensor
    """
    bias = np.sqrt(3.0 / embeddings.size(1))
    torch.nn.init.uniform_(embeddings, -bias, bias)


def load_embeddings(emb_file, word_map):
    """
    Creates an embedding tensor for the specified word map, for loading into the model.

    :param emb_file: file containing embeddings (stored in GloVe format)
    :param word_map: word map
    :return: embeddings in the same order as the words in the word map, dimension of embeddings
    """

    # Find embedding dimension
    with open(emb_file, 'r') as f:
        emb_dim = len(f.readline().split(' ')) - 1

    vocab = set(word_map.keys())

    # Create tensor to hold embeddings, initialize
    embeddings = torch.FloatTensor(len(vocab), emb_dim)
    init_embedding(embeddings)

    # Read embedding file
    print("\nLoading embeddings...")
    for line in open(emb_file, 'r'):
        line = line.split(' ')

        emb_word = line[0]
        embedding = list(map(lambda t: float(t), filter(lambda n: n and not n.isspace(), line[1:])))

        # Ignore word if not in train_vocab
        if emb_word not in vocab:
            continue

        embeddings[word_map[emb_word]] = torch.FloatTensor(embedding)

    return embeddings, emb_dim


def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.

    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def save_checkpoint(data_name, epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer, decoder_optimizer,
                    bleu4, is_best, representations=None, runname=None):
    """
    Saves model checkpoint.

    :param data_name: base name of processed dataset
    :param epoch: epoch number
    :param epochs_since_improvement: number of epochs since last improvement in BLEU-4 score
    :param encoder: encoder model
    :param decoder: decoder model
    :param encoder_optimizer: optimizer to update encoder's weights, if fine-tuning
    :param decoder_optimizer: optimizer to update decoder's weights
    :param bleu4: validation BLEU-4 score for this epoch
    :param is_best: is this checkpoint the best so far?
    """
    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'bleu-4': bleu4,
             'representations': representations,
             'encoder': encoder,
             'decoder': decoder,
             'encoder_optimizer': encoder_optimizer,
             'decoder_optimizer': decoder_optimizer}
    filename = 'checkpoint_' + data_name + '.pth.tar'


    if runname is None:
        torch.save(state, filename)
        # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
        if is_best:
            torch.save(state, 'BEST_' + filename)
    else:

        if not os.path.exists("/yoav_stg/gshalev/image_captioning/{}".format(runname)):
            os.mkdir("/yoav_stg/gshalev/image_captioning/{}".format(runname))


        torch.save(state, "/yoav_stg/gshalev/image_captioning/{}/{}".format(runname, filename))
        # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
        if is_best:
            torch.save(state,  "/yoav_stg/gshalev/image_captioning/{}/BEST_{}".format(runname, filename))


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, shrink_factor):
    """
    Shrinks learning rate by a specified factor.

    :param optimizer: optimizer whose learning rate must be shrunk.
    :param shrink_factor: factor in interval (0, 1) to multiply learning rate with.
    """

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        lr_befor_update = param_group['lr']
        param_group['lr'] = param_group['lr'] * shrink_factor
        print('LR before: {} LR after: {}'.format(lr_befor_update, param_group['lr']))
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))


def accuracy(scores, targets, k):
    """
    Computes top-k accuracy, from predicted and true labels.

    :param scores: scores from the model
    :param targets: true labels
    :param k: k in top-k accuracy
    :return: top-k accuracy
    """

    batch_size = targets.size(0)
    top_scores, top_ind = scores.topk(k, 1, True, True)

    #if one of the five is the correct then -TRUE
    correct = top_ind.eq(targets.view(-1, 1).expand_as(top_ind)) # targets.unsqueeze(1) == targets.view(-1, 1)
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)

def get_args():
    parser = argparse.ArgumentParser(description='train')
    # general
    parser.add_argument('--runname', type=str)
    parser.add_argument('--batch_size', default=32, type=int)
    #cosine
    parser.add_argument('--cosine', default=False, action='store_true')
    # fixed
    parser.add_argument('--sphere', type=int, default=0)
    parser.add_argument('--normalize_f_x', default=False, action='store_true')
    parser.add_argument('--fixed', default=False, action='store_true')
    #unlikelihood
    parser.add_argument('--num_of_fake', default=16, type=int)
    parser.add_argument('--alpha', default=0.0, type=float)

    # replace
    parser.add_argument('--replace_type', type=str)

    # fine-tune
    parser.add_argument('--checkpoint', default=None, type=str)
    parser.add_argument('--fine_tune_encoder', default=False, action='store_true')
    parser.add_argument('--fine_tune_epochs', default=-1, type=int)

    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--debug', default=False, action='store_true')
    parser.add_argument('--learning_rate', default=-1, type=float)

    parser.add_argument('--run_local', default=False, action='store_true')
    args = parser.parse_args()
    return args
# pack_utils.py