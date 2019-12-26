# import torch
# import argparse
#
# import numpy as np
# import torch.nn.functional as F
#
# from word_language_model.data_holder import Corpus
#
#
# parser = argparse.ArgumentParser(description='get_prop_for_IC_sentence')
#
# # Model parameters.
# parser.add_argument('--data', type=str, default='./data_dir', help='location of the data corpus')
# parser.add_argument('--checkpoint', type=str, default='model.pt', help='model checkpoint to use')
# parser.add_argument('--sentence', type=str, help='sentence to get prop for')
# parser.add_argument('--pos', type=str, help='sentence pos')
# parser.add_argument('--outf', type=str, default='generated.txt', help='output file for generated text')
# parser.add_argument('--seed', type=int, default=1111, help='random seed')
# parser.add_argument('--cuda', action='store_true', help='use CUDA')
# parser.add_argument('--temperature', type=float, default=1.0, help='temperature - higher will increase diversity')
# parser.add_argument('--log-interval', type=int, default=100, help='reporting interval')
# args = parser.parse_args()
#
# # Set the random seed manually for reproducibility.
# torch.manual_seed(args.seed)
# if torch.cuda.is_available():
#     if not args.cuda:
#         print("WARNING: You have a CUDA device, so you should probably run with --cuda")
#
# device = torch.device("cuda" if args.cuda else "cpu")
#
# with open(args.checkpoint, 'rb') as f:
#     model = torch.load(f, map_location=torch.device(device))
# model.eval()
#
# corpus = Corpus(args.data)
# ntokens = len(corpus.dictionary)
#
# is_transformer_model = hasattr(model, 'model_type') and model.model_type == 'Transformer'
# if not is_transformer_model:
#     hidden = model.init_hidden(1)
#
# sentence = ' '.join(args.sentence.split('_'))
# # pos = args.pos.split('_')
# tokens = sentence.split()
# lm_prop = [(1.0, sentence[0])]
#
# for idx in range(len(tokens) - 1):
#     input = torch.tensor([[corpus.dictionary.word2idx[tokens[idx]]]]).to(device)
#     output, hidden = model(input, hidden)
#     ######
#     # word_weights = output.squeeze().div(args.temperature).exp().cpu()
#     # word_idx = torch.multinomial(word_weights, 1)[0]
#     # print(corpus.dictionary.idx2word[word_idx])
#     #########
#     log_likelihood = F.log_softmax(output[0][0]).detach()
#     prop = np.exp(log_likelihood)
#
#     lm_prop.append((tokens[idx + 1], prop[corpus.dictionary.word2idx[tokens[idx + 1]]].item()))
#
# print(lm_prop)
