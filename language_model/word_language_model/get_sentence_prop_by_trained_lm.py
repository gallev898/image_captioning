import torch
import torch.nn.functional as F


def load_model(device):
    torch.manual_seed(1111)
    p = '../../../language_model/word_language_model/model.pt'
    model = torch.load(p, map_location=torch.device(device))
    model.eval()
    return model


def get_sen_prop(tokens, model, corpus, device):
    lm_prop = []

    hidden = model.init_hidden(1)

    for idx in range(len(tokens) - 1):
        input = torch.tensor([[corpus.dictionary.word2idx[tokens[idx]]]]).to(device)
        output, hidden = model(input, hidden)
        word_weights = output.squeeze().div(1.0).exp().cpu()
        word_idx = torch.multinomial(word_weights, 1)[0]
        # print(corpus.dictionary.idx2word[word_idx])

        log_likelihood = F.log_softmax(output[0][0], dim=0).detach()
        prop = log_likelihood #notice
        # prop = np.exp(log_likelihood)

        lm_prop.append((tokens[idx + 1], prop[corpus.dictionary.word2idx[tokens[idx + 1]]].item()))

    return lm_prop


