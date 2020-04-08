import sys

sys.path.append('/home/mlspeech/gshalev/gal/image_cap2')
sys.path.append('/home/mlspeech/gshalev/anaconda3/envs/python3_env/lib')

from dataset_loader.datasets2 import CaptionDataset
from standart_training.V_fixed_models_no_attention import *
from standart_training.train_show_and_tell_pack_utils import *

from utils import *
import time
import torch.optim
import torch.utils.data

import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from nltk.translate.bleu_score import corpus_bleu

# section: settengs
data_normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
args = get_args()
print('ARGS: {}'.format(args))

# section: W&B
if not args.run_local:
    # wandb login a8c4526db3e8aa11d7b2674d7c257c58313b45ca
    import wandb

    wandb.init(project="image_captioning", name=args.runname, dir='/yoav_stg/gshalev/wandb')

# section: Model parameters
emb_dim = 300  # dimension of word embeddings
attention_dim = 300  # dimension of attention linear layers
decoder_dim = 300  # dimension of decoder RNN
dropout = 0.5
device = torch.device(
    "cuda:{}".format(args.cuda) if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

# section: Training parameters
batch_size = 3 if args.run_local else args.batch_size
workers = 1  # for data-loading; right now, only 1 works with h5py
start_epoch = 0
epochs = 120  # number of epochs to train for (if early stopping is not triggered)
epochs_since_improvement = 0  # keeps track of number of epochs since there's been an improvement in validation BLEU
encoder_lr = 1e-4  # learning rate for encoder if fine-tuning
decoder_lr = args.lr if args.lr > -1 else 4e-4  # learning rate for decoder
grad_clip = 5.  # clip gradients at an absolute value of
alpha_c = 1.  # regularization parameter for 'doubly stochastic attention', as in the paper
best_bleu4 = 0.  # BLEU-4 score right now
print_freq = 100  # print training/validation stats every __ batches
server_data_folder = '/Users/gallevshalev/PycharmProjects/image_captioning/output_folder/'
data_name = 'coco_5_cap_per_img_5_min_word_freq'


def get_embeddings(embedding_size, vocab_size):
    word2vec_dictionary = dict()
    for cls_idx in range(vocab_size):
        v = np.random.randint(low=-100, high=100, size=embedding_size)
        v = v / np.linalg.norm(v)
        word2vec_dictionary[cls_idx] = torch.from_numpy(v).float()

    w2v_matrix = torch.stack(list(word2vec_dictionary.values()), dim=1)
    return w2v_matrix


def main():
    global epochs_since_improvement, best_bleu4
    print('LR: {}'.format(args.lr))

    # section: word map
    if not args.run_local:
        data_f = '/yoav_stg/gshalev/image_captioning/output_folder'
    else:
        data_f = server_data_folder

    word_map_file = os.path.join(data_f, 'WORDMAP_' + data_name + '.json')
    print('word_map_file: {}'.format(word_map_file))

    print('loading word map from path: {}'.format(word_map_file))
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)
    print('load word map COMPLETED')

    rev_word_map = {v: k for k, v in word_map.items()}

    # section: representation
    representations = get_embeddings(decoder_dim, len(word_map)).to(device)

    # section: not fixed
    if not args.fixed:
        representations.requires_grad = True

    # section: decoder
    decoder = DecoderWithoutAttention(attention_dim=attention_dim,
                                      embed_dim=emb_dim,
                                      decoder_dim=decoder_dim,
                                      vocab_size=len(word_map),
                                      device=device,
                                      dropout=dropout, encoder_dim=300)

    decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                         lr=decoder_lr)
    # section: not fixed
    if not args.fixed:
        decoder_optimizer.add_param_group({'params': representations})

    # section: encoder
    encoder = Encoder(embeded_dim=300)
    # notice: fine to encoder
    encoder.fine_tune(True if args.fine_tune_encoder and args.fine_tune_epochs == 0 else False)
    encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                         lr=encoder_lr) if args.fine_tune_encoder and args.fine_tune_epochs == 0 else None

    # section: Move to GPU, if available
    decoder = decoder.to(device)
    encoder = encoder.to(device)

    # section: wandb
    if not args.run_local:
        wandb.watch(decoder)

    # section: Loss function
    criterion = nn.CrossEntropyLoss().to(device)

    # section: dataloaders
    train_loader = torch.utils.data.DataLoader(
        CaptionDataset(data_f, data_name, 'TRAIN', transform=transforms.Compose([data_normalization])),
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        CaptionDataset(data_f, data_name, 'VAL', transform=transforms.Compose([data_normalization])),
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)

    val_loader_for_val = torch.utils.data.DataLoader(
        CaptionDataset(data_f, data_name, 'VAL', transform=transforms.Compose([data_normalization])),
        batch_size=1, shuffle=True, num_workers=workers, pin_memory=True)

    # section: Epochs
    print('starting epochs')
    for epoch in range(start_epoch, epochs):

        # section: terminate training after 20 epochs without improvment
        if epochs_since_improvement == 20:
            print('break after : epochs_since_improvement == 20')
            break

        # section: fine tune encoder
        if epoch == args.fine_tune_epochs:
            print('fine tuning after epoch({}) == args.fine_tune_epochs({})'.format(epoch, args.fine_tune_epochs))
            encoder.fine_tune(args.fine_tune_encoder)
            encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                                 lr=encoder_lr)

        # section: adjust LR after 8 epochs without improvment
        if epochs_since_improvement > 0 and epochs_since_improvement % 8 == 0:
            print('!!!  ADJUST LR AFTER : epochs_since_improvement: {}'.format(epochs_since_improvement))
            adjust_learning_rate(decoder_optimizer, 0.8)

        # section: train
        print('--------------111111111-----------Start train----------epoch-{}'.format(epoch))
        train(train_loader=train_loader,
              encoder=encoder,
              decoder=decoder,
              criterion=criterion,
              encoder_optimizer=encoder_optimizer,
              decoder_optimizer=decoder_optimizer,
              epoch=epoch, representations=representations)

        # section: eval
        print('--------------2222222222-----------Start validation----------epoch-{}'.format(epoch))
        with torch.no_grad():
            recent_bleu4 = validate(val_loader=val_loader,
                                    encoder=encoder,
                                    decoder=decoder,
                                    criterion=criterion,
                                    rev_word_map=rev_word_map, representations=representations, word_map=word_map)

        print('9999999999999- recent blue {}'.format(recent_bleu4))
        print('--------------3333333333-----------Start val without teacher forcing----------epoch-{}'.format(epoch))
        with torch.no_grad():
            caption_image_beam_search(encoder, decoder, val_loader_for_val, word_map, rev_word_map, representations)
            print('!@#!@!#!#@!#@!#@ DONE WITH TRAIN VAL AND VAL WITHOUT TEACHER FORCING FOR EPOCH :{}'.format(epoch))

        # section: save model if there was an improvement
        is_best = recent_bleu4 > best_bleu4
        best_bleu4 = max(recent_bleu4, best_bleu4)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        save_checkpoint(data_name, epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer,
                        decoder_optimizer, recent_bleu4, is_best, representations=representations, runname=args.runname)


def train(train_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, epoch, representations):
    # section: train mode
    decoder.train()  # train mode (dropout and batchnorm is used)
    encoder.train()

    # section: metrics_roc_and_more
    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss (per word decoded)
    top5accs = AverageMeter()  # top5 accuracy
    start = time.time()

    # section: Batches
    for i, (imgs, caps, caplens) in enumerate(train_loader):

        # section: break if debug mode
        if (args.run_local or args.debug) and i > 2:
            break

        # section: metrics_roc_and_more
        data_time.update(time.time() - start)

        # section: move to device
        imgs, caps, caplens = imgs.to(device), caps.to(device), caplens.to(device)

        # section:  Forward prop.
        imgs = encoder(imgs)
        scores, targets, decode_lengths, _, sort_ind = decoder(imgs, caps, caplens, args, representations)

        # Remove timesteps that we didn't decode at, or are pads
        # pack_padded_sequence is an easy trick to do this
        # section: pack pad becauese of lstm
        scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data

        # section: Calculate loss
        loss = criterion(scores, targets)

        # section: calc grad
        decoder_optimizer.zero_grad()
        if encoder_optimizer is not None:
            encoder_optimizer.zero_grad()
        loss.backward()

        # section: Clip gradients - notice this is for preventing exploding grad not venishing!
        if grad_clip is not None:
            clip_gradient(decoder_optimizer, grad_clip)
            if encoder_optimizer is not None:
                clip_gradient(encoder_optimizer, grad_clip)

        # section: Update weights
        decoder_optimizer.step()
        if encoder_optimizer is not None:
            encoder_optimizer.step()

        # section: Keep track of metrics_roc_and_more
        top5 = accuracy(scores, targets, 5)
        losses.update(loss.item(), sum(decode_lengths))
        top5accs.update(top5, sum(decode_lengths))
        batch_time.update(time.time() - start)
        start = time.time()

        # section: Print status
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})'.format(epoch, i, len(train_loader),
                                                                          batch_time=batch_time,
                                                                          data_time=data_time, loss=losses,
                                                                          top5=top5accs))
            # section: W&B
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
        encoder_out = encoder(imgs).repeat(k, 1).to(device)  # (1, enc_image_size, enc_image_size, encoder_dim)

        # Tensor to store top k previous words at each step; now they're just <start>
        k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device)  # (k, 1)

        # Tensor to store top k sequences; now they're just <start>
        seqs = k_prev_words  # (k, 1)
        seqs_scores = torch.FloatTensor([[0.]] * k).to(device)

        # Tensor to store top k sequences' scores; now they're just 0
        top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

        # Lists to store completed sequences, their alphas and scores
        complete_seqs = list()
        complete_seqs_scores = list()
        complete_seqs_scores_for_all_steps = list()

        # Start decoding
        step = 1
        h, c = decoder.init_hidden_state(encoder_out)
        h = torch.zeros(h.shape).to(device)
        c = torch.zeros(c.shape).to(device)

        # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
        while True:
            embeddings = decoder.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)

            if step == 1:
                h, c = decoder.decode_step(encoder_out, (h, c))  # (s, decoder_dim)
            else:
                h, c = decoder.decode_step(embeddings, (h, c))  # (s, decoder_dim)

            if args.cosine:
                h = F.normalize(h, dim=1, p=2)
                representations = F.normalize(representations, dim=0, p=2)
            scores = torch.matmul(h, representations).to(device)

            if args.sphere > 0:
                scores *= args.sphere

            # scores = decoder.fc(h)  # (s, vocab_size)
            # scores = F.log_softmax(scores, dim=1)
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

            # Which sequences are incomplete (didn't reach <end>)?
            incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                               next_word != word_map['<end>']]
            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

            # Set aside complete sequences
            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])
                complete_seqs_scores_for_all_steps.extend(seqs_scores[complete_inds].tolist())
            k -= len(complete_inds)  # reduce beam length accordingly

            # Proceed with incomplete sequences
            if k == 0:
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

        if len(complete_seqs_scores) > 0:
            i = complete_seqs_scores.index(max(complete_seqs_scores))

            seq = complete_seqs[i]

            words = [rev_word_map[ind] for ind in seq]

            print('5    ' + ' '.join(words))


def validate(val_loader, encoder, decoder, criterion, rev_word_map, representations, word_map):
    # section: eval mode
    decoder.eval()
    encoder.eval()

    # section: meter
    batch_time = AverageMeter()
    losses = AverageMeter()
    top5accs = AverageMeter()
    start = time.time()

    references = list()  # references (true captions) for calculating BLEU-4 score
    hypotheses = list()  # hypotheses (predictions)

    # section: Batches
    for i, (imgs, caps, caplens, allcaps) in enumerate(val_loader):

        # section: break after one epoch if debugging locally
        if (args.run_local or args.debug) and i > 2:
            break

        # section: Move to device, if available
        imgs = imgs.to(device)
        caps = caps.to(device)
        caplens = caplens.to(device)

        # section: Forward prop.
        imgs = encoder(imgs)
        scores, caps_sorted, decode_lengths, _, sort_ind = decoder(imgs, caps, caplens, args, representations)

        # notice: Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
        targets = caps_sorted[:, 1:]

        # notice: Remove timesteps that we didn't decode at, or are pads
        # notice: pack_padded_sequence is an easy trick to do this
        scores_copy = scores.clone()
        scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data

        # section: Calculate loss
        loss = criterion(scores, targets)

        # Add doubly stochastic attention regularization
        # We know the weights sum to 1 at a given timestep. But we also encourage
        # the weights at a single pixel p to sum to 1 across all timesteps T
        # This means we want the model to attend to every pixel over the course of generating
        # the entire sequence. Therefore, we try to minimize the difference between 1 and the sum of
        # a pixel's weights across all timesteps
        # loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

        # section: Keep track of metrics_roc_and_more
        losses.update(loss.item(), sum(decode_lengths))
        top5 = accuracy(scores, targets, 5)
        top5accs.update(top5, sum(decode_lengths))
        batch_time.update(time.time() - start)
        start = time.time()

        # section: print
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

        # section: remove pads
        for j, p in enumerate(preds_ind):
            temp_preds.append(preds_ind[j][:decode_lengths[j]])
        preds_ind = temp_preds
        hypotheses.extend(preds_ind)

        assert len(references) == len(hypotheses)

        # section: print preds
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

    # section: Calculate BLEU-4 scores and print
    bleu4 = corpus_bleu(references, hypotheses)

    print(
        '\n * LOSS - {loss.avg:.3f}, TOP-5 ACCURACY - {top5.avg:.3f}, BLEU-4 - {bleu}\n'.format(
            loss=losses,
            top5=top5accs,
            bleu=bleu4))

    return bleu4


# if __name__ == '__main__':
main()

# V_train_fix_show_and_tell_300.py
