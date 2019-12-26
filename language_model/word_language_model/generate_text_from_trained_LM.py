import torch
import argparse
from language_model.word_language_model.data_holder import Corpus


# section: settings
parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 Language Model')

# Model parameters.
parser.add_argument('--data', type=str, default='./data_dir', help='location of the data corpus')
parser.add_argument('--checkpoint', type=str, default='./model.pt', help='model checkpoint to use')
parser.add_argument('--outf', type=str, default='generated.txt', help='output file for generated text')
parser.add_argument('--words', type=int, default='1000', help='number of words to generate')
parser.add_argument('--num_of_sentence_to_generate', type=int, default=10, help='number of sentences to generate')
parser.add_argument('--seed', type=int, default=1111, help='random seed')
parser.add_argument('--cuda', action='store_true', help='use CUDA')
parser.add_argument('--temperature', type=float, default=1.0, help='temperature - higher will increase diversity')
parser.add_argument('--log-interval', type=int, default=100, help='reporting interval')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda" if args.cuda else "cpu")

if args.temperature < 1e-3:
    parser.error("--temperature has to be greater or equal 1e-3")

with open(args.checkpoint, 'rb') as f:
    model = torch.load(f, map_location=torch.device(device))
model.eval()

# section: generate

corpus = Corpus(args.data)
ntokens = len(corpus.dictionary)

hidden = model.init_hidden(1)
input = torch.tensor([[corpus.dictionary.word2idx['<start>']]]).to(device)

with open(args.outf, 'w') as outf:
    with torch.no_grad():  # no tracking history
        for i in range(args.num_of_sentence_to_generate):
            end_word_idx = 0

            while end_word_idx != corpus.dictionary.word2idx['<end>']:
                output, hidden = model(input, hidden)
                word_weights = output.squeeze().div(args.temperature).exp().cpu()
                word_idx = torch.multinomial(word_weights, 1)[0]
                input.fill_(word_idx)
                end_word_idx = word_idx.item()

                word = corpus.dictionary.idx2word[word_idx]

                outf.write(word + ('\n' if end_word_idx == corpus.dictionary.word2idx['<end>'] else ' '))
