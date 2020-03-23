import sys

sys.path.append('/home/mlspeech/gshalev/gal/image_cap2')
sys.path.append('/home/mlspeech/gshalev/anaconda3/envs/python3_env/lib')
from dataset_loader.datasets import CaptionDataset
from standart_training.models.fixed_models_no_attention import DecoderWithoutAttention, Encoder


from tqdm import *
from utils import *

import os
import torch
import base64
import argparse
import en_core_web_sm

import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms

# section: args
parser = argparse.ArgumentParser()
parser.add_argument('--run_local', default=False, action='store_true')
parser.add_argument('--beam_size', type=int, default=5)
parser.add_argument('--sphere', type=int, default=1)
parser.add_argument('--cuda', type=int, default=0)
parser.add_argument('--cosine', default=False, action='store_true')
parser.add_argument('--model', type=str)
args = parser.parse_args()

# section: data holders
noun_phrase_sum_of_log_prop = []
generated_sentences_likelihood = []
pos_dic_obj = {}
gt_metric_dic = {'annotations': list()}
hp_metric_dic = {'annotations': list()}

# section: device
device = torch.device("cuda:{}".format(args.cuda) if torch.cuda.is_available() else "cpu")
print('using cuda: {}', format(device))

# section: paths
coco_data_path = '/Users/gallevshalev/PycharmProjects/image_captioning/output_folder' if args.run_local else '/yoav_stg/gshalev/image_captioning/output_folder'


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
desktop_path = os.path.join(os.path.join(os.path.expanduser('~')), 'Desktop')
data_name = 'coco_5_cap_per_img_5_min_word_freq'
model_filename = 'NEW_BEST_checkpoint_' + data_name + '.pth.tar'
output_folder_path = os.path.join(os.path.expanduser('~'), 'PycharmProjects/image_captioning/output_folder')
word_map_file_name = 'WORDMAP_' + data_name + '.json'
word_map_file = os.path.join(output_folder_path, word_map_file_name)
data_normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
attention_dim = 512  # dimension of attention linear layers
decoder_dim = 512  # dimension of decoder RNN
emb_dim = 512  # dimension of word embeddings
dropout = 0.5

def get_model_path_and_save_dir(args):
    if args.run_local:
        desktop_path = os.path.join(os.path.expanduser('~'), 'Desktop')
        trained_models_path = os.path.join(desktop_path, 'trained_models')
        model_path = os.path.join(trained_models_path, os.path.join(args.model, model_filename))
        save_dir = 'inference_data'
    else:
        model_path = "/yoav_stg/gshalev/image_captioning/{}/{}".format(args.model, model_filename)
        save_dir = "/yoav_stg/gshalev/image_captioning/{}/inference_data".format(args.model)

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
        print('created dir: {}'.format(save_dir))

    if args.run_local:
        save_dir = os.path.join(save_dir, args.model)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
            print('created dir: {}'.format(save_dir))

    print('model path: {}'.format(model_path))
    print('save dir: {}'.format(save_dir))
    return model_path, save_dir



def get_models(model_path, device, vocab_size):
    checkpoint = torch.load(model_path, map_location=torch.device(device))

    decoder = DecoderWithoutAttention(attention_dim=attention_dim,
                                      embed_dim=emb_dim,
                                      decoder_dim=decoder_dim,
                                      vocab_size=vocab_size,
                                      device=device,
                                      dropout=dropout)
    decoder.load_state_dict(checkpoint['decoder'])
    decoder = decoder.to(device)
    decoder.eval()

    encoder = Encoder()
    encoder.load_state_dict(checkpoint['encoder'])
    encoder = encoder.to(device)
    encoder.eval()

    representations= checkpoint['representations']

    return encoder, decoder, representations


def get_word_map(args_):
    if args_.run_local:
        file = os.path.join(output_folder_path, word_map_file_name)
    else:
        server_path = '/yoav_stg/gshalev/image_captioning/output_folder'
        file = os.path.join(server_path, word_map_file_name)

    print('Loading word map from: {}'.format(file))
    with open(file, 'r') as j:
        word_map = json.load(j)
    rev_word_map = {v: k for k, v in word_map.items()}

    return word_map, rev_word_map
# utils.py

def encode(encoder_, image_, beam_size_, word_map_):

    # subsec: encode the img
    encoder_out = encoder_(image_)

    # notice: we treat the problem of beam size like batch size
    encoder_out = encoder_out.expand(beam_size_, 512)

    # subsec: data holders
    k_prev_words = torch.LongTensor([[word_map_['<start>']]] * beam_size_).to(device)
    seqs = k_prev_words
    seqs_scores = torch.FloatTensor([[0.]] * beam_size_).to(device)
    top_k_scores = torch.zeros(beam_size_, 1).to(device)  # (k, 1)

    return encoder_out, k_prev_words, seqs, seqs_scores, top_k_scores


def beam_search_decode(encoder_, decoder_, beam_size_, word_map_, image_, args_, representations_):
    vocab_size = len(word_map_)
    # subsec: encode
    encoder_out, k_prev_words, seqs, seqs_scores, top_k_scores = encode(encoder_, image_, beam_size_, word_map_)

    # subsec: data holders
    uncompleted_seq = 0
    complete_seqs = list()
    complete_seqs_scores = list()
    complete_seqs_scores_for_all_steps = list()
    complete_logits_list = list()
    logits_list = torch.zeros([beam_size_, 1]).to(device)

    # subsec: initialization
    step = 1
    h, c = decoder_.init_hidden_state(encoder_out)
    h = torch.zeros(h.shape).squeeze(1).to(device)
    c = torch.zeros(c.shape).squeeze(1).to(device)

    while True:
        embeddings = decoder_.embedding(k_prev_words).squeeze(1)
        if step == 1:  # notice: in the first step we feed the encoder outpot
            h, c = decoder.decode_step(encoder_out, (h, c))
        else:
            h, c = decoder.decode_step(embeddings, (h, c))

        if args_.cosine:
            h = F.normalize(h, dim=1, p=2)
            representations_ = F.normalize(representations_, dim=0, p=2)

        preds = torch.matmul(h, representations).to(device)
        preds *= args.sphere

        scores = preds
        logits = scores
        scores = F.log_softmax(scores, dim=1)
        scores_copy = scores.clone()

        scores = top_k_scores.expand_as(scores) + scores

        if step == 1:
            #notice: because this is the first step then the beam search starts here so we take 'scores[0]'
            top_k_scores, top_k_words = scores[0].topk(beam_size_, 0, True, True)
        else:
            top_k_scores, top_k_words = scores.view(-1).topk(beam_size_, 0, True, True)

        # Convert unrolled indices to actual indices of scores
        prev_word_inds = top_k_words / vocab_size  # (s)
        next_word_inds = top_k_words % vocab_size  # (s)

        logits = [logits[i][next_word_inds[i].item()] for i in range(beam_size_)]  # NOTICE
        logits = torch.stack(logits, dim=0).unsqueeze(1).to(device)  # NOTICE
        logits_list = torch.cat([logits_list[prev_word_inds], logits], dim=1)  # NOTICE

        # Add new words to sequences
        seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)
        seqs_scores = torch.cat([seqs_scores[prev_word_inds], scores_copy.view(-1)[top_k_words].unsqueeze(1)], dim=1)

        # Which sequences are incomplete (didn't reach <end>)?
        incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                           next_word != word_map['<end>']]
        complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

        # Set aside complete sequences
        if len(complete_inds) > 0:
            complete_logits_list.extend(logits_list[complete_inds].tolist())  # NOTICE
            complete_seqs.extend(seqs[complete_inds].tolist())
            complete_seqs_scores.extend(top_k_scores[complete_inds])
            complete_seqs_scores_for_all_steps.extend(seqs_scores[complete_inds].tolist())
        beam_size_ -= len(complete_inds)  # reduce beam length accordingly

        # Proceed with incomplete sequences
        if beam_size_ == 0:
            break
        seqs = seqs[incomplete_inds]
        h = h[prev_word_inds[incomplete_inds]]
        c = c[prev_word_inds[incomplete_inds]]
        encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
        top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
        k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

        # Break if things have been going on too long
        if step > 50:
            break
        step += 1

    if len(complete_seqs_scores) == 0:
        print('complete_seqs_scores is empty')
        uncompleted_seq += 1
        return seqs, None, None, None

    i = complete_seqs_scores.index(max(complete_seqs_scores))
    seq_sum = round(max(complete_seqs_scores).item(), 4)
    round_score = round(np.array(complete_seqs_scores_for_all_steps[i]).sum(), 4)
    try:
        assert round_score == seq_sum
    except AssertionError:
        print('------------EXCEPTION ACCRUED---------------')
        print('{} != {}'.format(round_score, seq_sum))

    top_seq_total_scors = complete_seqs_scores_for_all_steps[i]
    seq = complete_seqs[i]
    logits_list = complete_logits_list[i][1:-1]

    print(seq_sum)
    return seq, top_seq_total_scors, seq_sum, logits_list


def caption_image_beam_search(encoder_, decoder_, image_, word_map_, rev_word_map_, args_, representations_):
    beam_size = args_.beam_size
    seq_, top_seq_total_scors_, seq_sum_, logits_list_ = beam_search_decode(encoder_, decoder_, beam_size, word_map_,
                                                                            image_, args_, representations_)

    if seq_ == None:
        return None, None, None, None


    top_seq_total = top_seq_total_scors_[1:-1]
    ex_top_seq_total = np.exp(top_seq_total_scors_[1:-1])

    words = [rev_word_map_[ind] for ind in seq_]
    words = words[2:-1] #notice
    words_as_str = ' '.join(words)

    nlp = en_core_web_sm.load()
    doc = nlp(words_as_str)

    lats_i = 0
    np_idx = []
    words_copy = words
    original_str = words

    # Collect inexed of noun phrase
    for i in doc.noun_chunks:
        i_str = '{} '.format(i.string)
        if '<unk ' in i_str:
            i_str = i_str.replace('<unk ', '<unk> ')
        i_str = i_str.strip()

        print('*****')
        print(words_copy)
        print(i_str)

        try:
            inde = [words_copy.index(str(x)) + lats_i for x in i_str.split()]
        except ValueError:
            print(ValueError.args)
            return None, None, None, None

        np_idx.append(inde)
        print(inde)
        lats_i = inde[-1] + 1
        words_copy = list(original_str[lats_i:])

    idx_count = 0
    for w, token, score, exp, logit in zip(words, doc, top_seq_total, ex_top_seq_total, logits_list_):
        if token.pos_ in pos_dic_obj:
            pos_dic_obj[token.pos_]['exp'].append(exp)
            pos_dic_obj[token.pos_]['logits'].append(logit)
        else:
            pos_dic_obj[token.pos_] = {'exp': [exp],
                                       'logits': [logit]}
        for l in np_idx:
            if idx_count in l:
                l[l.index(idx_count)] = score
                break

        idx_count += 1
    for l in np_idx:
        noun_phrase_sum_of_log_prop.append(sum(l) * -1)

    return seq_, top_seq_total_scors_, seq_sum_, words


if __name__ == '__main__':

    if args.cosine and 'cosine' not in args.model:
        raise Exception('cosine flag is on but model is dotproduct')
    if not args.cosine and 'cosine' in args.model:
        raise Exception('cosine flag is off but cosine model given')

    print('Strating beam search : {}'.format(args.beam_size))

    # section: get word map
    word_map, rev_word_map = get_word_map(args)
    print('vocab size:{}'.format(len(word_map)))

    # section: get models
    model_path, save_dir = get_model_path_and_save_dir(args)
    encoder, decoder, representations = get_models(model_path, device, len(word_map))

    # section: dataloader
    test_loader = torch.utils.data.DataLoader(
        CaptionDataset(coco_data_path, data_name, 'TEST', transform=transforms.Compose([data_normalization])),
        batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
    print('len test_loader: {}'.format(len(test_loader)))

    # section: start inference
    enumerator = enumerate(test_loader)
    for bi, (image, caps, caplens, allcaps) in tqdm(enumerator):
        [next(enumerator, None) for _ in range(4)]

        # subsec: collect all current img caps
        for ci in range(allcaps.shape[1]):
            gt = [rev_word_map[ind.item()] for ind in allcaps[0][ci]][1:caplens[0][ci].item() - 1]
            gt_metric_dic['annotations'].append({u'image_id': bi, u'caption': gt})

        # subsec: move to device
        image = image.to(device)

        # subsec: run beam search
        seq_, top_seq_total_scors_, seq_sum_, words = caption_image_beam_search(encoder, decoder, image, word_map,
                                                                                rev_word_map, args, representations)
        if not None == words:
            hp_metric_dic['annotations'].append({u'image_id': bi, u'caption': words})
        if not None == seq_sum_:
            generated_sentences_likelihood.append((bi, seq_sum_))

        if args.run_local and bi == 50:
            break

    # section: save dic
    dic_name = 'pos_dic_beam_{}'.format(args.beam_size)
    print('dic name: {}'.format(dic_name))

    save_data_path = os.path.join(save_dir, dic_name)
    print('saving dic in: {}'.format(save_data_path))

    torch.save({'pos': pos_dic_obj,
                'noun_phrase_sum_of_log_prop': noun_phrase_sum_of_log_prop,
                'generated_sentences_likelihood': generated_sentences_likelihood},
               save_data_path)

    # section: seva metrics
    if args.run_local:
        metrics_save_dir = 'metrics'
    else:
        metrics_save_dir = "/yoav_stg/gshalev/image_captioning/{}/{}".format(args.model, 'metrics')

    metrics_result_file_name = 'metrics_results_beam_{}'.format(args.beam_size)
    save_data_path = os.path.join(save_dir, metrics_result_file_name)

    torch.save({'gt': gt_metric_dic, 'hyp': hp_metric_dic}, save_data_path)
    print('Saved metrics results in {}'.format(os.path.join(save_dir, metrics_result_file_name)))

# run_beam_search_for_fixed_models.py
