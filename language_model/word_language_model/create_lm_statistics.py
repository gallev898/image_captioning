import sys


sys.path.append('/home/mlspeech/gshalev/gal/word_language_model')
sys.path.append('/home/mlspeech/gshalev/anaconda3/envs/python3_env/lib')

# coding: utf-8
import en_core_web_sm


adj_to_noun = {}
with open('data_dir/train.txt', 'r') as f:
    for cnt, line in enumerate(f):
        tokens = line.split()[1:-1]

        nlp = en_core_web_sm.load()
        doc = nlp(' '.join(tokens))
        pos = [x.pos_ for x in doc]

        d = 0
        for noun_chunk in doc.noun_chunks:
            noun_chunk_string = noun_chunk.string
            if '<' or '>' in noun_chunk_string:
                noun_chunk_string.replace('<', '').replace('>', '')

            noun_chunk_tokens = noun_chunk_string.split()
            noun_chunk_doc = nlp(noun_chunk_string)
            pos = [x.pos_ for x in noun_chunk_doc]
            if 'ADJ' in pos:
                for i, p in enumerate(pos):
                    if p == 'ADJ':
                        for ii in range(i + 1, len(pos)):
                            if pos[ii] == 'NOUN':
                                cur_adj = noun_chunk_tokens[i]
                                cur_noun = noun_chunk_tokens[ii]

                                if cur_adj in adj_to_noun:
                                    if cur_noun in adj_to_noun[cur_adj]:
                                        adj_to_noun[cur_adj][cur_noun] += 1
                                    else:
                                        adj_to_noun[cur_adj][cur_noun] = 1
                                else:
                                    adj_to_noun[cur_adj] = {cur_noun: 1}
                                    # for
                                    # adj_to_noun[noun_chunk_tokens[i]] = {}
                                    # [noun_chunk_tokens[i]]
                            elif pos[ii] == 'ADJ':
                                continue
                            else:
                                break

    f = p
    import json


    with open('result.json', 'w') as fp:
        json.dump(adj_to_noun, fp)
