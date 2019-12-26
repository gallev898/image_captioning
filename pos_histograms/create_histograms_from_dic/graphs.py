import matplotlib.pyplot as plt
import torch
import numpy as np
import os


if __name__ == '__main__':
    dic_name = 'pos_dic_test_top_p_0.9'
    model_name = 'run_7'

    dic_type = 'top_k' if 'top_k' in dic_name else 'top_p' if 'top_p' in dic_name else 'beam' if 'beam' in dic_name else 'none'
    exp_or_prop = 'prop' if dic_type == 'top_k' or dic_type == 'top_p' else 'exp'

    if not os.path.exists(model_name):
        os.mkdir(model_name)

    save_dir = os.path.join(model_name, dic_name)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    dic = torch.load('../create_dic_for_histograms/pos_dic/{}/{}'.format(model_name, dic_name))
    pos_dic = dic['pos']
    noun_phrase_sum_of_log_prop = dic['noun_phrase_sum_of_log_prop']
    sentence_likelihood = dic['sentence_likelihood']
    sentence_likelihood = [i for i in sentence_likelihood if i != None]

    plt.clf()
    _ = plt.hist(noun_phrase_sum_of_log_prop, normed=True,
                 bins=np.arange(min(noun_phrase_sum_of_log_prop), max(noun_phrase_sum_of_log_prop) + 0.1, 0.01),
                 color='purple')
    plt.xlabel('noun_phrase_sum_of_log_prop')
    plt.savefig(os.path.join(save_dir, plt.gca().get_xlabel()))

    plt.clf()
    _ = plt.hist(sentence_likelihood, normed=True,
                 bins=np.arange(min(sentence_likelihood), max(sentence_likelihood) + 0.1, 0.01),
                 color='purple')
    plt.xlabel('sentence_likelihood')
    plt.savefig(os.path.join(save_dir, plt.gca().get_xlabel()))

    avg = {}
    for k in pos_dic.keys():
        avg[k] = {exp_or_prop: np.average(pos_dic[k][exp_or_prop]),
                  # avg[k] = {'exp': np.average(pos_dic[k]['exp']),
                  'logits': np.average(pos_dic[k]['logits']),
                  'alphas_max': np.average(pos_dic[k]['alphas_max']),
                  'alphas_var': np.average(pos_dic[k]['alphas_var']),
                  'count': len(pos_dic[k][exp_or_prop])
                  }

        plt.clf()
        _ = plt.hist(pos_dic[k][exp_or_prop], normed=True, bins=np.arange(0.0, 1.1, 0.01), color='blue')
        plt.xlabel(k + ' ' + exp_or_prop)
        plt.savefig(os.path.join(save_dir, plt.gca().get_xlabel()))

        plt.clf()
        min_logit = min(pos_dic[k]['logits'])
        max_logit = max(pos_dic[k]['logits'])
        _ = plt.hist(pos_dic[k]['logits'], normed=True, bins=np.arange(min_logit, max_logit + 0.1, 0.1), color='red')
        plt.xlabel(k + ' logits')
        plt.savefig(os.path.join(save_dir, plt.gca().get_xlabel()))

    plt.clf()
    avg_exp = [avg[k][exp_or_prop] for k in list(avg.keys())]
    plt.bar(list(avg.keys()), avg_exp, color='pink')
    plt.title('exp avg')
    plt.savefig(os.path.join(save_dir, plt.gca().get_title()))

    plt.clf()
    avg_logits = [avg[k]['logits'] for k in list(avg.keys())]
    plt.bar(list(avg.keys()), avg_logits, color='yellow')
    plt.title('logits avg')
    plt.savefig(os.path.join(save_dir, plt.gca().get_title()))

    plt.clf()
    count = [avg[k]['count'] for k in list(avg.keys())]
    plt.bar(list(avg.keys()), count, color='green')
    plt.title('count')
    plt.savefig(os.path.join(save_dir, plt.gca().get_title()))

    plt.clf()
    alphas = [avg[k]['alphas_max'] for k in list(avg.keys())]
    plt.bar(list(avg.keys()), alphas, color='orange')
    plt.title('alphas max avg')
    plt.savefig(os.path.join(save_dir, plt.gca().get_title()))

    plt.clf()
    alphas = [avg[k]['alphas_var'] for k in list(avg.keys())]
    plt.bar(list(avg.keys()), alphas, color='brown')
    plt.title('alphas var avg')
    plt.savefig(os.path.join(save_dir, plt.gca().get_title()))

    d = 0