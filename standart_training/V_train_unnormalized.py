import sys

sys.path.append('/home/mlspeech/gshalev/gal/image_cap2')
sys.path.append('/home/mlspeech/gshalev/anaconda3/envs/python3_env/lib')

# sys.path.append('/home/mlspeech/gshalev/gal/image_captioning')
from utils import *

# wandb login a8c4526db3e8aa11d7b2674d7c257c58313b45ca
import time
import torch.optim
import torch.utils.data

import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

from torch import nn
from standart_training.pack_utils import *
from standart_training.V_models_with_attention import Encoder, DecoderWithAttention
from dataset_loader.datasets2 import *
# from dataset_loader.datasets import *
from torch.nn.utils.rnn import pack_padded_sequence
from nltk.translate.bleu_score import corpus_bleu

data_normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# parser = argparse.ArgumentParser(description='train')
# parser.add_argument('--runname', type=str)
# parser.add_argument('--cuda', type=int, default=0)
# parser.add_argument('--checkpoint', default=None, type=str)
# parser.add_argument('--fine_tune_encoder', default=False, action='store_true')
# parser.add_argument('--debug', default=False, action='store_true')
# parser.add_argument('--fine_tune_epochs', default=-1, type=int)
#
# parser.add_argument('--run_local', default=False, action='store_true')
# parser.add_argument('--batch_size', default=32, type=int)
# args = parser.parse_args()
# args = get_args()
parser = argparse.ArgumentParser(description='train')
# general
parser.add_argument('--runname', type=str)
parser.add_argument('--batch_size', default=32, type=int)
#cosine
parser.add_argument('--cosine', default=False, action='store_true')
# fixed
parser.add_argument('--sphere', type=int, default=0)
parser.add_argument('--scale', type=int, default=100)
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

if not args.run_local:
    import wandb

    wandb.init(project="image_captioning", name=args.runname, dir='/yoav_stg/gshalev/wandb')

# Data parameters !-CONST-!
# data_folder = '../output_folder'  # folder with data files saved by create_input_files.py
# data_name = 'coco_5_cap_per_img_5_min_word_freq'  # base name shared by data files

# Model parameters
emb_dim = 512  # dimension of word embeddings
attention_dim = 512  # dimension of attention linear layers
decoder_dim = 512  # dimension of decoder RNN
dropout = 0.5
device = torch.device(
    "cuda:{}".format(args.cuda) if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

# Training parameters
batch_size = 1 if args.run_local else args.batch_size
workers = 1  # for data-loading; right now, only 1 works with h5py
start_epoch = 0
epochs = 120  # number of epochs to train for (if early stopping is not triggered)
epochs_since_improvement = 0  # keeps track of number of epochs since there's been an improvement in validation BLEU
encoder_lr = 1e-4  # learning rate for encoder if fine-tuning
decoder_lr = 4e-4  # learning rate for decoder
grad_clip = 5.  # clip gradients at an absolute value of
alpha_c = 1.  # regularization parameter for 'doubly stochastic attention', as in the paper
best_bleu4 = 0.  # BLEU-4 score right now
print_freq = 100  # print training/validation stats every __ batches

data_folder = '../output_folder'  # folder with data files saved by create_input_files.py


def get_embeddings(embedding_size, vocab_size):
    word2vec_dictionary = dict()
    print('embeddings with sacle: {}'.format(args.scale))
    for cls_idx in range(vocab_size):
        v = np.random.randint(low=-args.scale, high=args.scale, size=embedding_size)
        # v = v / np.linalg.norm(v)
        word2vec_dictionary[cls_idx] = torch.from_numpy(v).float()

    w2v_matrix = torch.stack(list(word2vec_dictionary.values()), dim=1)
    return w2v_matrix


def main():
    global best_bleu4, epochs_since_improvement, start_epoch, data_name, word_map

    # sec: Read word map
    data_f = data_folder if args.run_local else '/yoav_stg/gshalev/image_captioning/output_folder'
    word_map_file = os.path.join(data_f, 'WORDMAP_' + data_name + '.json')
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)
    rev_word_map = {v: k for k, v in word_map.items()}
    print('load word map from: {} COMPLETED'.format(word_map_file))

    # sec: decoder
    decoder = DecoderWithAttention(attention_dim=attention_dim,
                                   embed_dim=emb_dim,
                                   decoder_dim=decoder_dim,
                                   vocab_size=len(word_map),
                                   device=device,
                                   dropout=dropout)

    decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                         lr=decoder_lr)

    # sec: encoder
    encoder = Encoder()
    encoder.fine_tune(True if args.fine_tune_encoder and args.fine_tune_epochs == 0 else False)
    encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                         lr=encoder_lr) if args.fine_tune_encoder and args.fine_tune_epochs == 0 else None

    # sec:  Move to GPU, if available
    decoder = decoder.to(device)
    encoder = encoder.to(device)

    # section: representation
    representations = get_embeddings(decoder_dim, len(word_map)).to(device)

    # section: not fixed
    if not args.fixed:
        representations.requires_grad = True

    if not args.fixed:
        decoder_optimizer.add_param_group({'params': representations})

    # sec: wandb
    if not args.run_local:
        wandb.watch(decoder)

    # sec: Loss function
    criterion = nn.CrossEntropyLoss().to(device)

    # sec: dataloaders
    train_loader = torch.utils.data.DataLoader(
        CaptionDataset(data_f, data_name, 'TRAIN', transform=transforms.Compose([data_normalization])),
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        CaptionDataset(data_f, data_name, 'VAL', transform=transforms.Compose([data_normalization])),
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)

    val_loader_for_val = torch.utils.data.DataLoader(
        CaptionDataset(data_f, data_name, 'VAL', transform=transforms.Compose([data_normalization])),
        batch_size=1, shuffle=True, num_workers=workers, pin_memory=True)

    # sec: Epochs
    for epoch in range(start_epoch, epochs):

        if epoch == 2:
            adjust_learning_rate(decoder_optimizer, 0.8)

        # sec: Decay learning rate if there is no improvement for 8 consecutive epochs, and terminate training after 20
        if epochs_since_improvement == 20:
            print('break after : epochs_since_improvement == 20')
            break

        if epochs_since_improvement > 0 and epochs_since_improvement % 8 == 0:
            print('adjust lr afetr : epochs_since_improvement > 0 and epochs_since_improvement % 8 == 0')
            adjust_learning_rate(decoder_optimizer, 0.8)

        # sec: train
        print('--------------111111111-----------Start train----------epoch-{}'.format(epoch))
        train(train_loader=train_loader,
              encoder=encoder,
              decoder=decoder,
              criterion=criterion,
              encoder_optimizer=encoder_optimizer,
              decoder_optimizer=decoder_optimizer,
              epoch=epoch,
              representations=representations)

        # sec: val with teacher forcing
        print('--------------2222222222-----------Start validation----------epoch-{}'.format(epoch))
        with torch.no_grad():
            recent_bleu4 = validate(val_loader=val_loader,
                                    encoder=encoder,
                                    decoder=decoder,
                                    criterion=criterion,
                                    representations=representations,
                                    rev_word_map=rev_word_map)

        print('9999999999999- recent blue {}'.format(recent_bleu4))
        print('--------------3333333333-----------Start val without teacher forcing----------epoch-{}'.format(epoch))
        # sec: val without teacher forsing
        with torch.no_grad():
            caption_image_beam_search(encoder, decoder, val_loader_for_val, word_map, rev_word_map, representations)
        print('!@#!@!#!#@!#@!#@ DONE WITH TRAIN VAL AND VAL WITHOUT TEACHER FORCING FOR EPOCH :{}'.format(epoch))

        # Check if there was an improvement
        is_best = recent_bleu4 > best_bleu4
        best_bleu4 = max(recent_bleu4, best_bleu4)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        # Save checkpoint
        save_checkpoint(data_name, epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer,
                        decoder_optimizer, recent_bleu4, is_best, representations, args.runname)


def train(train_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, epoch, representations):
    # train mode
    decoder.train()  # train mode (dropout and batchnorm is used)
    encoder.train()

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss (per word decoded)
    top5accs = AverageMeter()  # top5 accuracy

    start = time.time()

    # Batches
    for i, (imgs, caps, caplens) in enumerate(train_loader):
        if (args.run_local or args.debug) and i > 2:
            break

        data_time.update(time.time() - start)

        # sec: Move to GPU, if available
        imgs, caps, caplens = imgs.to(device), caps.to(device), caplens.to(device)

        # sec: Forward prop.
        imgs = encoder(imgs)
        scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens, args, representations)

        # sec: Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
        targets = caps_sorted[:, 1:]

        # sec: pack_padded_sequence
        scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data

        # sec: Calculate loss
        loss = criterion(scores, targets)

        # sec:Add doubly stochastic attention regularization
        loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

        # sec: Back prop.
        decoder_optimizer.zero_grad()
        if encoder_optimizer is not None:
            encoder_optimizer.zero_grad()
        loss.backward()

        # sec: Clip gradients - notice this is for preventing exploding grad not venishing!
        if grad_clip is not None:
            clip_gradient(decoder_optimizer, grad_clip)
            if encoder_optimizer is not None:
                clip_gradient(encoder_optimizer, grad_clip)

        # Update weights
        decoder_optimizer.step()
        if encoder_optimizer is not None:
            encoder_optimizer.step()

        # Keep track of metrics_roc_and_more
        top5 = accuracy(scores, targets, 5)

        losses.update(loss.item(), sum(decode_lengths))
        top5accs.update(top5, sum(decode_lengths))
        batch_time.update(time.time() - start)

        start = time.time()

        # Print status
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})'.format(epoch, i, len(train_loader),
                                                                          batch_time=batch_time,
                                                                          data_time=data_time, loss=losses,
                                                                          top5=top5accs))
            if not args.run_local:
                wandb.log({"Top-5 Accuracy": top5accs.avg,
                           "Test Loss": losses.avg})


def caption_image_beam_search(encoder, decoder, val_loader, word_map, rev_word_map, representations, beam_size=3):
    for i, (imgs, caps, caplens, allcaps) in enumerate(val_loader):
        if i > 100 or (args.debug and i > 2):
            break

        k = beam_size
        vocab_size = len(word_map)
        imgs = imgs.to(device)
        # # Encode
        encoder_out = encoder(imgs)  # (1, enc_image_size, enc_image_size, encoder_dim)
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
                                           h)  # (s, encoder_dim), (s, num_pixels)

            alpha = alpha.view(-1, enc_image_size, enc_image_size)  # (s, enc_image_size, enc_image_size)

            gate = decoder.sigmoid(decoder.f_beta(h))  # gating scalar, (s, encoder_dim)# TODO: tihs is bla bla
            awe = gate * awe  # TODO: this is bla bla - motivated by dropout "like" - type of regularization, gate of [0,1] because of sigmoid

            concatination_of_input_and_att = torch.cat([embeddings, awe], dim=1)
            h, c = decoder.decode_step(concatination_of_input_and_att, ( h, c))  # (s, decoder_dim)

            if args.cosine:
                h = F.normalize(h, dim=1, p=2)
                representations = F.normalize(representations, dim=0, p=2)
            scores = torch.matmul(h, representations).to(device)

            if args.sphere > 0:
                scores *= args.sphere

            scores = F.log_softmax(scores, dim=1)
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
            seqs_scores = torch.cat([seqs_scores[prev_word_inds], scores_copy.view(-1)[top_k_words].unsqueeze(1)],
                                    dim=1)
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

        if len(complete_seqs_scores) > 0 :
            i = complete_seqs_scores.index(max(complete_seqs_scores))

            seq = complete_seqs[i]

            words = [rev_word_map[ind] for ind in seq]

            print('5    ' + ' '.join(words))


def validate(val_loader, encoder, decoder, criterion, representations, rev_word_map):

    # eval mode
    decoder.eval()
    if encoder is not None:
        encoder.eval()

    # meter
    batch_time = AverageMeter()
    losses = AverageMeter()
    top5accs = AverageMeter()

    start = time.time()

    references = list()  # references (true captions) for calculating BLEU-4 score
    hypotheses = list()  # hypotheses (predictions)

    # explicitly disable gradient calculation to avoid CUDA memory error
    # solves the issue #57

    # Batches
    for i, (imgs, caps, caplens, allcaps) in enumerate(val_loader):
        # break after one epoch if debugging locally
        if (args.run_local or args.debug) and i > 2:
            break

        # Move to device, if available
        imgs = imgs.to(device)
        caps = caps.to(device)
        caplens = caplens.to(device)

        # Forward prop.
        if encoder is not None:
            imgs = encoder(imgs)

        scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens, args, representations)

        # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
        targets = caps_sorted[:, 1:]

        # Remove timesteps that we didn't decode at, or are pads
        # pack_padded_sequence is an easy trick to do this
        scores_copy = scores.clone()
        scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data

        # Calculate loss
        loss = criterion(scores, targets)

        # Add doubly stochastic attention regularization
        # We know the weights sum to 1 at a given timestep. But we also encourage
        # the weights at a single pixel p to sum to 1 across all timesteps T
        # This means we want the model to attend to every pixel over the course of generating
        # the entire sequence. Therefore, we try to minimize the difference between 1 and the sum of
        # a pixel's weights across all timesteps
        loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

        # Keep track of metrics_roc_and_more
        losses.update(loss.item(), sum(decode_lengths))
        top5 = accuracy(scores, targets, 5)
        top5accs.update(top5, sum(decode_lengths))
        batch_time.update(time.time() - start)

        start = time.time()

        if i % print_freq == 0:
            print('4    Validation: [{0}/{1}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})\t'.format(i, len(val_loader),
                                                                            batch_time=batch_time,
                                                                            loss=losses, top5=top5accs))

        # Store references (true captions), and hypothesis (prediction) for each image
        # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
        # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]
        # References
        allcaps = allcaps[sort_ind]  # because images were sorted in the decoder
        for j in range(allcaps.shape[0]):  # for each example
            img_caps = allcaps[j].tolist()
            img_captions = list(
                map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<pad>']}],
                    img_caps))  # remove <start> and pads
            references.append(img_captions)

        # Hypotheses
        # get for each example the max pred at each time step (batch size, max length caption)
        pred_values, preds_ind = torch.max(scores_copy, dim=2)
        preds_ind = preds_ind.tolist()
        temp_preds = list()

        # remove pads
        for j, p in enumerate(preds_ind):
            temp_preds.append(preds_ind[j][:decode_lengths[j]])
        preds_ind = temp_preds
        hypotheses.extend(preds_ind)

        assert len(references) == len(hypotheses)

        if (i + 1) % 300 == 0:
            print('-1   ************print captions***********')
            num_to_print = 0
            for h in hypotheses:
                if num_to_print < 100:
                    words = []
                    for w in h:
                        words.append(rev_word_map[w])
                    print('1    ' + ' '.join(words))
                    num_to_print += 1
                else:
                    break

            print('2    **************************************')

    # Calculate BLEU-4 scores
    bleu4 = corpus_bleu(references, hypotheses)

    print(
        '\n * LOSS - {loss.avg:.3f}, TOP-5 ACCURACY - {top5.avg:.3f}, BLEU-4 - {bleu}\n'.format(
            loss=losses,
            top5=top5accs,
            bleu=bleu4))

    return bleu4


if __name__ == '__main__':
    main()
# V_train_unnormalized.py
