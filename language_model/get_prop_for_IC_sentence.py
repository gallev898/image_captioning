import torch

import numpy as np
import torch.nn.functional as F


def get_sentence_prop(sentence, pos, model, corpus, device):


    is_transformer_model = hasattr(model, 'model_type') and model.model_type == 'Transformer'
    if not is_transformer_model:
        hidden = model.init_hidden(1)

    tokens = sentence
    lm_prop = [(sentence[0], pos[0], 1.0)]

    for idx in range(len(tokens) - 1):
        input = torch.tensor([[corpus.dictionary.word2idx[tokens[idx]]]]).to(device)
        output, hidden = model(input, hidden)
        ######
        # word_weights = output.squeeze().div(args.temperature).exp().cpu()
        # word_idx = torch.multinomial(word_weights, 1)[0]
        # print(corpus.dictionary.idx2word[word_idx])
        #########
        log_likelihood = F.log_softmax(output[0][0]).detach()
        prop = np.exp(log_likelihood)

        lm_prop.append((tokens[idx + 1], pos[idx+1], prop[corpus.dictionary.word2idx[tokens[idx + 1]]].item()))

    print(lm_prop)
    return lm_prop
